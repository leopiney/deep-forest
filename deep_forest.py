#
# Inspired by https://arxiv.org/abs/1702.08835 and https://github.com/STO-OTZ/my_gcForest/
#
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict

from utils import create_logger, rolling_window


class MGCForest():
    """Multi-Grained Cascade Forest"""
    def __init__(self, estimator_class, estimator_params, stride_ratios=[0.25], folds=3):
        self.mgs_instances = [
            MultiGrainedScanner(
                estimator_class,
                estimator_params['mgs'],
                stride_ratio=stride_ratio,
                folds=folds
            )
            for stride_ratio in stride_ratios
        ]
        self.stride_ratios = stride_ratios

        self.c_forest = CascadeForest(estimator_class, estimator_params['cascade'])

    def fit(self, X, y):
        X_scanned = np.hstack([
            mgs.scan_fit(X, y)
            for mgs in self.mgs_instances
        ])

        self.c_forest.fit(X_scanned, y)

    def predict(self, X):
        scan_pred = np.hstack([
            mgs.scan_predict(X)
            for mgs in self.mgs_instances
        ])

        return self.c_forest.predict(scan_pred)

    def __repr__(self):
        return '<MGCForest {}>'.format(self.stride_ratios)


class MultiGrainedScanner():
    def __init__(
        self, estimator_class, estimator_params, stride_ratio=0.25, folds=3, verbose=False
    ):
        self.estimator_class = estimator_class
        self.estimator_params = estimator_params
        self.stride_ratio = stride_ratio
        self.folds = folds

        self.windows_estimators = []

        self.logger = create_logger(self, verbose)

    def slices(self, X, reshape_2d=False):
        self.logger.debug('Slicing X with shape {}'.format(X.shape))
        sample_shape = list(X[0].shape)

        window_shape = np.maximum(
            np.array([s * self.stride_ratio for s in sample_shape]), 1
        ).astype(np.int16)
        self.logger.debug('Got window shape: {}'.format(window_shape.shape))

        #
        # Calculate the windows that are going to be used and the total
        # number of new generated samples.
        #
        windows_count = [sample_shape[i] - window_shape[i] + 1 for i in range(len(sample_shape))]
        new_instances_total = np.prod(windows_count)

        self.logger.debug('Slicing {} windows.'.format(windows_count))

        #
        # For each sample, get all the windows with their values
        #
        newX = np.array([
            rolling_window(x, window_shape)
            for x in X
        ])

        #
        # Swap the 0 and 1 axis so as to get for each window, the value of each sample.
        #
        newX = np.swapaxes(newX, 0, 1)
        self.logger.info(
            'Scanning turned X ({}) into newX ({}). {} new instances were added '
            'per sample'.format(X.shape, newX.shape, new_instances_total)
        )

        return newX

    def scan_fit(self, X, y):
        self.logger.info('Scanning and fitting for X ({}) and y ({}) started'.format(
            X.shape, y.shape
        ))
        self.n_classes = np.unique(y).size
        sliced_X = self.slices(X)

        #
        # Create an estimator for each generated window
        #
        self.windows_estimators = []
        predictions = []
        for window_index, window_X in enumerate(sliced_X):
            estimators = [self.estimator_class(**params) for params in self.estimator_params]
            self.windows_estimators.append(estimators)

            self.logger.info('Training estimators for window #{} with shape {}'.format(
                window_index, window_X.shape
            ))

            for estimator_index, estimator in enumerate(estimators):
                self.logger.debug('Fitting window #{} with shape ({}) with estimator #{}'.format(
                    window_index, window_X.shape, estimator_index
                ))
                estimator.fit(window_X, y)

                #
                # Gets a prediction of sliced_X with shape (len(newX), n_classes).
                # The method `predict_proba` returns a vector of size n_classes.
                #
                self.logger.debug('Cross-validation with estimator #{}'.format(estimator_index))
                prediction = cross_val_predict(
                    estimator,
                    window_X,
                    y,
                    cv=self.folds,
                    method='predict_proba',
                    n_jobs=-1,
                )

                predictions.append(prediction)

        return np.hstack(predictions)

    def scan_predict(self, X):
        self.logger.info('Predicting X ({})'.format(X.shape))
        sliced_X = self.slices(X)
        return np.hstack([
            estimator
            .predict_proba(window_X)
            for window_X, window_estimators in zip(sliced_X, self.windows_estimators)
            for estimator in window_estimators
        ])

    def __repr__(self):
        return '<MultiGrainedScanner {}>'.format(self.stride_ratio)


class CascadeForest():

    def __init__(self, estimator_class, estimator_params, folds=3, verbose=False):
        self.estimator_class = estimator_class
        self.estimator_params = estimator_params
        self.folds = folds

        self.logger = create_logger(self, verbose)

    def fit(self, X, y):
        self.logger.info('Cascade fitting for X ({}) and y ({}) started'.format(X.shape, y.shape))
        self.classes = np.unique(y)
        self.level = 0
        self.levels = []
        self.max_score = None

        while True:
            self.logger.info('Level {}:: X with shape: {}'.format(self.level + 1, X.shape))
            estimators = [self.estimator_class(**params) for params in self.estimator_params]

            predictions = []
            for estimator in estimators:
                self.logger.debug('Fitting X ({}) and y ({}) with estimator {}'.format(
                    X.shape, y.shape, estimator
                ))
                estimator.fit(X, y)

                #
                # Gets a prediction of X with shape (len(X), n_classes)
                #
                prediction = cross_val_predict(
                    estimator,
                    X,
                    y,
                    cv=self.folds,
                    method='predict_proba',
                    n_jobs=-1,
                )

                predictions.append(prediction)

            self.logger.info('Level {}:: got all predictions'.format(self.level + 1))

            #
            # Stacks horizontally the predictions to each of the samples in X
            #
            X = np.hstack([X] + predictions)

            #
            # For each sample, compute the average of predictions of all the estimators, and take
            # the class with maximum score for each of them.
            #
            y_prediction = self.classes.take(
                np.array(predictions).mean(axis=0).argmax(axis=1)
            )

            score = accuracy_score(y, y_prediction)
            self.logger.info('Level {}:: got accuracy {}'.format(self.level + 1, score))
            if self.max_score is None or score > self.max_score:
                self.level += 1
                self.max_score = score
                self.levels.append(estimators)
            else:
                break

    def predict(self, X):
        for estimators in self.levels:

            predictions = [
                estimator.predict_proba(X)
                for estimator in estimators
            ]
            self.logger.info('Shape of predictions: {} shape of X: {}'.format(
                np.array(predictions).shape, X.shape
            ))
            X = np.hstack([X] + predictions)

        return self.classes.take(
            np.array(predictions).mean(axis=0).argmax(axis=1)
        )

    def __repr__(self):
        return '<CascadeForest {}>'.format(len(self.estimator_params))
