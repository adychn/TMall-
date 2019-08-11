# coding=utf-8

import xgboost as xgb
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot as plt

import dataset


def train(X_train, X_test, y_train, y_test):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    param = {
        'max_depth': 6,
        'eta': 0.05,
        'silent': 1,
        'objective': 'binary:logistic'
    }
    param['nthread'] = 4
    param['eval_metric'] = ['auc', 'logloss']

    eval_list = [(dtest, 'eval'), (dtrain, 'train')]
    num_round = 1000
    early_stopping_rounds = 20
    model = xgb.train(params=param, dtrain=dtrain, evals=eval_list, num_boost_round=num_round,
              early_stopping_rounds=early_stopping_rounds)
    return model


def plot_feature_importance(model, feature_buf):
    feature_important = model.get_score(importance_type='weight')

    feature_important = sorted(feature_important.items(), key=lambda x: x[1], reverse=True)

    keys, vals = [], []
    for item in feature_important[:20]:
        keys.append(feature_buf[int(item[0][1:])])
        vals.append(item[1])

    print(dict(zip(keys, vals)))

    plt.bar(range(len(vals)), vals, width=0.5)
    plt.xticks(range(len(vals)), keys, rotation=90)
    plt.savefig('feature_importance.pdf', bbox_inches='tight', format='pdf')


if __name__ == '__main__':
    X_train, X_test, y_train, y_test, feature_buf = dataset.generate_dataset(enhance=False)
    model = train(X_train, X_test, y_train, y_test)

    pre = model.predict(xgb.DMatrix(X_test, label=None))
    auc = roc_auc_score(y_test, pre)
    print(auc)

    plot_feature_importance(model, feature_buf)
