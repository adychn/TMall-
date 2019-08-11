# coding=utf-8

import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer

DATA_PATH = './data/data.txt'

POP_FEATURES = ['instance_id', 'item_property_list', 'user_id',
                'context_timestamp', 'context_id',
                'predict_category_property', 'date']

CATEGORICAL_FEATURES = ['item_id', 'item_brand_id', 'item_city_id', 'user_gender_id', 'user_occupation_id', 'shop_id']

CONTINUOUS_FEATURES = ['item_price_level', 'item_sales_level', 'item_collected_level',
                       'item_pv_level', 'user_age_level', 'user_star_level',
                       'shop_review_num_level', 'shop_review_positive_rate',
                       'shop_star_level', 'shop_score_service', 'shop_score_delivery',
                       'shop_score_description']

VECTOR_FEATURES = ['item_category_list']

feature_buf = []


def _pop_features(df):
    for feature in POP_FEATURES:
        df.pop(feature)


def _generate_continuous_features(df):
    """连续特征"""
    space = None
    for feature in CONTINUOUS_FEATURES:
        if feature in df.columns:
            val = df[feature].values.reshape(-1, 1)
            if space is None:
                space = val
            else:
                space = np.hstack((space, val))
            feature_buf.append(feature)
    print("continuous shape = ", space.shape)
    return space


def _generate_categorical_features(df, space):
    """离散特征"""
    oh_encoder = OneHotEncoder(sparse=True, categories='auto')
    for feature in CATEGORICAL_FEATURES:
        val = oh_encoder.fit_transform(df[feature].values.reshape((-1, 1))).toarray()
        space = np.hstack((space, val))

        for i in range(val.shape[1]):
            feature_buf.append('{}_{}'.format(feature, i))
    print("categorical shape = ", space.shape)
    return space


def _generate_vector_features(df, space):
    """向量特征"""
    print(space.shape)
    cv = CountVectorizer()
    for feature in VECTOR_FEATURES:
        val = cv.fit_transform(df[feature]).toarray()
        space = np.hstack((space, val))

        for i in range(val.shape[1]):
            feature_buf.append('{}_{}'.format(feature, i))
    print("vector shape = ", space.shape)
    return space


def _build_date_buf(date_pivot, left, right):
    date_buf = []
    for i in range(left, right):
        date = date_pivot + pd.Timedelta(i, unit='d')
        date_buf.append(date.strftime('%Y-%m-%d'))

    return date_buf


def _generate_historical_convrate(df):
    """历史转化率"""
    unique_date = df['date'].unique()
    col_list = [(['item_id'], 'item_convrate'), (['user_age_level', 'item_id'], 'age_item_convraterate')]

    data_buf = []
    for day in unique_date:
        date_pivot = pd.to_datetime(day)
        lag_days = _build_date_buf(date_pivot, -3, 0)

        target_df = df[df['date'].isin([day])]
        lag_df = df[df['date'].isin(lag_days)]

        if lag_df.shape[0] == 0 or target_df.shape[0] == 0:
            continue

        for cols, col_name in col_list:
            lag_g = lag_df.groupby(cols).is_trade.mean().reset_index()

            lag_cols = []
            lag_cols.extend(cols)
            lag_cols.append(col_name)

            lag_g.columns = lag_cols
            target_df = pd.merge(target_df, lag_g, on=cols, how='left').fillna(0)
        data_buf.append(target_df)

    hc_df = pd.concat(data_buf, axis=0).reset_index(drop=True)
    print('historical convrate shape', hc_df.shape)
    return hc_df


def _make_instant_feature(df):
    first, prev = -1, -1
    first_buf, prev_buf, fif_min_buf = [], [], []

    for row in df.itertuples():
        cur = row.context_timestamp

        if first == -1:
            first = row.context_timestamp

        first_buf.append(cur - first)

        if prev == -1:
            prev_buf.append(0)
            fif_min_buf.append(1)
        else:
            prev_buf.append(cur - prev)
            if cur - prev <= 15 * 60:
                fif_min_buf.append(fif_min_buf[-1] + 1)
            else:
                fif_min_buf.append(1)
        prev = cur

    df['first_to_now'] = first_buf
    df['prev_to_now'] = prev_buf
    df['recent_15_minutes'] = fif_min_buf

    return df[['instance_id', 'first_to_now', 'prev_to_now', 'recent_15_minutes']].reset_index(drop=True)


def _generate_instant_feature(df):
    """实时特征"""
    sorted_df = df.sort_values('context_timestamp')

    uig = sorted_df.groupby(['user_id', 'item_id'])
    ins_g = uig[['instance_id', 'context_timestamp']].apply(_make_instant_feature)

    ins_df = pd.merge(df, ins_g.reset_index(drop=True), on='instance_id')
    print('instant feature shape', ins_df.shape)
    return ins_df


def generate_dataset(enhance=False):
    df = pd.read_csv(DATA_PATH, sep=' ', nrows=100000)
    df['date'] = df['context_timestamp'].apply(lambda x: time.strftime('%Y-%m-%d', time.localtime(x)))

    if enhance:
        df = _generate_historical_convrate(df)
        df = _generate_instant_feature(df)
        CONTINUOUS_FEATURES.extend(['item_convrate', 'age_item_convraterate',
                                    'first_to_now', 'prev_to_now', 'recent_15_minutes'])

    y = df.pop('is_trade')
    _pop_features(df)
    space = _generate_continuous_features(df)
    space = _generate_categorical_features(df, space)
    X = _generate_vector_features(df, space)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    return X_train, X_test, y_train, y_test, feature_buf


if __name__ == '__main__':
    generate_dataset()
