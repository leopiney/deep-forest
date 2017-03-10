#
# Inpired by https://arxiv.org/abs/1702.08835 and https://github.com/STO-OTZ/my_gcForest/
#
import numpy as np
import random
import uuid

from sklearn.datasets import fetch_mldata
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from deep_forest import MGCForest


mnist = fetch_mldata('MNIST original', data_home='~/scikit-learn-datasets')
mnist.data.shape

print('Data: {}, target: {}'.format(mnist.data.shape, mnist.target.shape))

X_train, X_test, y_train, y_test = train_test_split(
    mnist.data,
    mnist.target,
    test_size=0.2,
    random_state=42,
)

X_train = X_train.reshape((len(X_train), 28, 28))
X_test = X_test.reshape((len(X_test), 28, 28))

#
# Limit the size of the dataset
#
# X_train = X_train[:1000]
# y_train = y_train[:1000]
# X_test = X_test[:1000]
# y_test = y_test[:1000]

print('X_train:', X_train.shape, X_train.dtype)
print('y_train:', y_train.shape, y_train.dtype)
print('X_test:', X_test.shape)
print('y_test:', y_test.shape)

mgc_forest = MGCForest(
    estimators_config={
        'mgs': [{
            'estimator_class': ExtraTreesClassifier,
            'estimator_params': {
                'n_estimators': 30,
                'min_samples_split': 21,
                'n_jobs': -1,
            }
        }, {
            'estimator_class': RandomForestClassifier,
            'estimator_params': {
                'n_estimators': 30,
                'min_samples_split': 21,
                'n_jobs': -1,
            }
        }],
        'cascade': [{
            'estimator_class': ExtraTreesClassifier,
            'estimator_params': {
                'n_estimators': 1000,
                'min_samples_split': 11,
                'max_features': 1,
                'n_jobs': -1,
            }
        }, {
            'estimator_class': ExtraTreesClassifier,
            'estimator_params': {
                'n_estimators': 1000,
                'min_samples_split': 11,
                'max_features': 'sqrt',
                'n_jobs': -1,
            }
        }, {
            'estimator_class': RandomForestClassifier,
            'estimator_params': {
                'n_estimators': 1000,
                'min_samples_split': 11,
                'max_features': 1,
                'n_jobs': -1,
            }
        }, {
            'estimator_class': RandomForestClassifier,
            'estimator_params': {
                'n_estimators': 1000,
                'min_samples_split': 11,
                'max_features': 'sqrt',
                'n_jobs': -1,
            }
        }]
    },
    stride_ratios=[1.0 / 4, 1.0 / 9, 1.0 / 16],
    verbose=True,
)

mgc_forest.fit(X_train, y_train)


y_pred = mgc_forest.predict(X_test)

print('Prediction shape:', y_pred.shape)
print(
    'Accuracy:', accuracy_score(y_test, y_pred),
    'F1 score:', f1_score(y_test, y_pred, average='weighted')
)
