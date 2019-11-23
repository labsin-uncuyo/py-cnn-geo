import curses
import datetime
import gc
import multiprocessing
import types
import numpy as np
from functools import partial
from natsort import natsorted
from keras.wrappers.scikit_learn import KerasClassifier
from os import listdir, makedirs, open, O_CREAT
from os.path import exists, join, isfile
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.utils.multiclass import unique_labels
from config import SamplesConfig


class KerasBatchClassifier(KerasClassifier):

    def __init__(self, build_fn=None, verbose=0):
        super().__init__(build_fn=build_fn, verbose=verbose)
        # self.dataset = None
        # self.dataset_gt = None
        self.dir_name = None
        self.name = None
        self.store_path = None
        self.neighbors = None
        self.feature_groups = None
        self.paths = None
        self.center = None
        self.feature_extraction_fn = None
        self.n_features = None
        self.date_and_time = None
        self.final_estimators = None

    def fit(self, X, y, **kwargs):
        print('Fitting %d elements' % (X.shape[0]))

        global dataset
        global dataset_gt
        dataset = kwargs['dataset']
        dataset_gt = kwargs['dataset_gt']
        naming_args = kwargs['name_args']
        self.neighbors = kwargs['neighbors']
        self.paths = kwargs['paths']
        self.center = kwargs['center']
        # taken from keras.wrappers.scikit_learn.KerasClassifier.fit ###################################################
        if self.build_fn is None:
            self.model = self.__call__(**self.filter_sk_params(self.__call__))
        elif not isinstance(self.build_fn, types.FunctionType) and not isinstance(self.build_fn, types.MethodType):
            self.model = self.build_fn(**self.filter_sk_params(self.build_fn.__call__))
        else:
            self.model = self.build_fn(**self.filter_sk_params(self.build_fn))

        # loss_name = self.model.loss
        # if hasattr(loss_name, '__name__'):
        #    loss_name = loss_name.__name__

        '''if loss_name == 'binary_crossentropy' and len(y.shape) != 2:
            y = to_categorical(y)'''

        ### fit => fit_generator
        ################################################################################################################
        self.dir_name = self.build_name(naming_args, self.sk_params)
        self.date_and_time = '{date:%Y%m%d_%H%M%S}'.format(date=datetime.datetime.now())
        self.name = self.date_and_time + '.' + self.dir_name + ('-vec' if self.paths is not None else '-cir')

        self.store_path = join("../storage/rf-search/vec/", self.dir_name)
        if not exists(self.store_path):
            makedirs(self.store_path)

        print('\n-------------------------------------------------------------------------------\n'
              'Starting with file %s'
              '\n-------------------------------------------------------------------------------\n\n' % (self.name))

        X_train, X_val = self.divide_indexes(X)

        self.feature_groups = int(self.neighbors / 2) + 1

        estimator_step_size = 50
        n_estimators = self.sk_params.get('n_est')

        steps = int(n_estimators / estimator_step_size) + 1
        batch_size = int(X_train.shape[0] / steps) + 1

        if self.paths is not None:
            self.feature_extraction_fn = get_item_features_vector
            self.n_features = (self.paths.shape[0] + 1) * self.paths.shape[1]
        else:
            self.feature_extraction_fn = get_item_features
            self.n_features = dataset.shape[1] * self.feature_groups

        print('Starting training phase...')

        self.final_estimators = self.model.n_estimators

        print('Train progress: ', end='')

        for i in range(steps):
            print('{0}/{1} - '.format(i + 1, steps), end='')
            #self.progress_bar(steps, i + 1, 'Train', stdscr)
            start = (i) * batch_size
            end = (i + 1) * batch_size
            end = end if end < X_train.shape[0] else X_train.shape[0]

            X_preprocessed_train_bag = np.zeros(shape=(end - start, self.n_features),
                                                dtype=np.float32)
            Y_preprocessed_train_bag = np.zeros(shape=(end - start), dtype=np.uint8)
            pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
            X_preprocessed_train_bag = np.array(
                pool.map(
                    partial(self.feature_extraction_fn, feature_groups=self.feature_groups, neighbors=self.neighbors,
                            paths=self.paths, center=self.center), X_train[start:end, :]))
            Y_preprocessed_train_bag = np.array(pool.map(get_item_gt, X_train[start:end, :]))
            pool.close()
            pool.join()

            self.model.fit(X_preprocessed_train_bag, Y_preprocessed_train_bag)
            X_preprocessed_train_bag = None
            Y_preprocessed_train_bag = None
            gc.collect()

            # print('Storing model...')
            joblib.dump(self.model, join(self.store_path, self.name + '-{step:03d}'.format(step=i) + '.pkl'))

            self.model = None
            gc.collect()

            if kwargs['finish_earlier'] and i == 2:
                break

            gc.collect()

            if i + 1 != steps:
                if i + 1 != steps - 1:
                    self.model = self.build_fn(**self.filter_sk_params(self.build_fn))
                    # self.model.set_params(n_estimators=self.model.n_estimators + estimator_step_size)
                else:
                    estimators_left = int((X_train.shape[0] - end) / batch_size) + 1
                    self.model = self.build_fn(**self.filter_sk_params(self.build_fn))
                    self.model.set_params(n_estimators=estimators_left)
                self.final_estimators += self.model.n_estimators

        print('Finished!\n')

        X_preprocessed_train_bag = None
        Y_preprocessed_train_bag = None
        X_train = None
        gc.collect()

        print('Starting validation phase...')

        val_batch_size = int(X_val.shape[0] / steps) + 1

        random_trees_files = [f for f in listdir(self.store_path) if
                              not isfile(join(self.dir_name, f)) and f.startswith(self.date_and_time) and f.endswith(
                                  '.pkl')]
        random_trees_files = natsorted(random_trees_files, key=lambda y: y.lower())

        values_votes = np.zeros(shape=(X_val.shape[0], 2), dtype=np.float32)
        expected_val = np.zeros(shape=(X_val.shape[0],), dtype=np.uint8)

        print('Validation progress: ', end='')

        for i in range(steps):
            print('{0}/{1} - '.format(i + 1, steps), end='')
            start = (i) * val_batch_size
            end = (i + 1) * val_batch_size
            end = end if end < X_val.shape[0] else X_val.shape[0]

            X_partial_preprocessed_val_bag = np.zeros(shape=(end - start, self.n_features), dtype=np.float32)
            # expected_val[start:end] = np.zeros(shape=(end - start), dtype=np.uint8)
            pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
            X_partial_preprocessed_val_bag = np.array(
                pool.map(
                    partial(self.feature_extraction_fn, feature_groups=self.feature_groups, neighbors=self.neighbors,
                            paths=self.paths, center=self.center), X_val[start:end, :]))
            expected_val[start:end] = np.array(pool.map(get_item_gt, X_val[start:end, :]))
            pool.close()
            pool.join()

            for file_i, rf_filename in enumerate(random_trees_files):
                rf_full_filename = join(self.store_path, rf_filename)
                self.model = joblib.load(rf_full_filename)

                if file_i == 0:
                    batch_votes = np.multiply(self.model.predict_proba(X_partial_preprocessed_val_bag),
                                              self.model.n_estimators)
                else:
                    batch_votes = np.add(batch_votes,
                                         np.multiply(self.model.predict_proba(X_partial_preprocessed_val_bag),
                                                     self.model.n_estimators))

                self.model = None
                gc.collect()

            values_votes[start:end] = batch_votes

        print('Finished!')

        predicted_val = np.divide(values_votes, self.final_estimators)
        X_partial_preprocessed_val_bag = None
        batch_votes = None
        values_votes = None
        gc.collect()
        predicted_val = np.argmax(predicted_val, axis=1)

        val_acc = accuracy_score(expected_val, predicted_val)

        print('Validation score: {val_acc:.4f}\n'.format(val_acc=val_acc))

        print('Storing value accuracy...')
        open(join(self.store_path, self.name + '-{val_acc:.4f}'.format(val_acc=val_acc) + '.txt'), O_CREAT)

        return None

    def score(self, X, y, **kwargs):
        print('Scoring %d elements' % (X.shape[0]))

        estimator_step_size = 50
        n_estimators = self.sk_params.get('n_est')

        steps = int(n_estimators / estimator_step_size) + 1
        batch_size = int(X.shape[0] / steps) + 1

        predicted_test = np.zeros(shape=(X.shape[0],), dtype=np.uint8)
        expected_test = np.zeros(shape=(X.shape[0],), dtype=np.uint8)

        random_trees_files = [f for f in listdir(self.store_path) if
                              not isfile(join(self.dir_name, f)) and f.startswith(self.date_and_time) and f.endswith(
                                  '.pkl')]
        random_trees_files = natsorted(random_trees_files, key=lambda y: y.lower())

        print('Starting testing phase...')

        values_votes = np.zeros(shape=(X.shape[0], 2), dtype=np.float32)

        print('Test progress: ', end='')
        for i in range(steps):
            print('{0}/{1} - '.format(i + 1, steps), end='')
            start = (i) * batch_size
            end = (i + 1) * batch_size
            end = end if end < X.shape[0] else X.shape[0]

            X_partial_preprocessed_test_bag = np.zeros(shape=(end - start, self.n_features), dtype=np.float32)
            pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
            X_partial_preprocessed_test_bag = np.array(
                pool.map(
                    partial(self.feature_extraction_fn, feature_groups=self.feature_groups, neighbors=self.neighbors,
                            paths=self.paths, center=self.center), X[start:end, :]))
            expected_test[start:end] = np.array(pool.map(get_item_gt, X[start:end, :]))
            pool.close()
            pool.join()

            for file_i, rf_filename in enumerate(random_trees_files):
                rf_full_filename = join(self.store_path, rf_filename)
                self.model = joblib.load(rf_full_filename)

                if file_i == 0:
                    batch_votes = np.multiply(self.model.predict_proba(X_partial_preprocessed_test_bag),
                                              self.model.n_estimators)
                else:
                    batch_votes = np.add(batch_votes,
                                         np.multiply(self.model.predict_proba(X_partial_preprocessed_test_bag),
                                                     self.model.n_estimators))

                self.model = None
                gc.collect()

            values_votes[start:end] = batch_votes

        print('Finished!\n')

        predicted_test = np.divide(values_votes, self.final_estimators)
        X_partial_preprocessed_test_bag = None
        batch_votes = None
        values_votes = None
        gc.collect()
        predicted_test = np.argmax(predicted_test, axis=1)

        print('Calculating reports and metrics...')

        test_acc = accuracy_score(expected_test, predicted_test)
        print('Test score: {test_acc:.4f}'.format(test_acc=test_acc))

        cm = self.print_confusion_matrix(expected_test, predicted_test, np.array(['No Forest', 'Forest']))

        print(
            classification_report(expected_test, predicted_test, target_names=np.array(['no forest', 'forest'])))

        confmat_file = join(self.store_path, self.name + '.conf_mat.npz')

        print('Storing confusion matrix...')
        if not exists(confmat_file):
            np.savez_compressed(confmat_file, cm=cm)

        return test_acc

    def divide_indexes(self, indexes_file):
        split_1 = int(SamplesConfig.TEST_PERCENTAGE * indexes_file.shape[0])

        train_idxs = indexes_file[split_1:]
        validation_idxs = indexes_file[:split_1]

        return train_idxs, validation_idxs

    def build_name(self, naming_args, sk_params):

        arr = []
        for naming_arg in naming_args:
            arr.append((naming_arg, sk_params[naming_arg]))

        name = ''
        for i, item in enumerate(arr):
            if i != 0:
                name += '-'

            name += item[0] + '_' + str(item[1])

        return name

    def print_confusion_matrix(self, y_true, y_pred, classes,
                               normalize=False,
                               title=None):
        """
        This function prints the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if not title:
            if normalize:
                title = 'Normalized confusion matrix'
            else:
                title = 'Confusion matrix, without normalization'

        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        # Only use the labels that appear in the data
        classes = classes[unique_labels(y_true, y_pred)]
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)
        return cm

    def progress_bar(self, total, current, phase, stdscr):
        try:
            increments = 50
            percentual = ((current / total) * 100)
            i = int(percentual // (100 / increments))
            text = "\r{0} progress - {1}/{2} steps [{3: <{4}}] {5}%".format(phase, current, total, '=' * i,
                                                                            increments,
                                                                            "{0:.2f}".format(percentual))
            stdscr.addstr(0, 0, text)
            # print(text, end="\n" if percentual == 100 else "")
            stdscr.refresh()
        except curses.error as e:
            pass

    @property
    def history(self):
        return self.__history


def get_item_features(item, feature_groups=None, neighbors=None, paths=None, center=None):
    # Paths and center are not used in this function para we keep the parameter to maintain the same function definition
    # as get_item_features_vector and avoid an if statement being executed billion of times

    data_patch = dataset[item[0], :, item[1]: item[1] + neighbors, item[2]: item[2] + neighbors]

    np_feat = np.zeros(shape=(data_patch.shape[0] * feature_groups,), dtype=np.float32)
    patch_sliced = data_patch
    for i in range(feature_groups - 1):
        top_row = patch_sliced[:, 0, :]
        bottom_row = patch_sliced[:, -1, :]

        patch_sliced = patch_sliced[:, 1:-1, :]
        left_col = patch_sliced[:, :, 0]
        right_col = patch_sliced[:, :, -1]

        patch_sliced = patch_sliced[:, :, 1:-1]

        border_values = np.concatenate((left_col, right_col, top_row, bottom_row), axis=1)
        np_feat[(i * data_patch.shape[0]):((i + 1) * data_patch.shape[0])] = np.mean(border_values, axis=1)

    np_feat[(-1 * data_patch.shape[0]):] = np.reshape(patch_sliced, data_patch.shape[0])

    return np_feat


def get_item_gt(item):
    return dataset_gt[item[0], item[1], item[2]]


def get_item_features_vector(item, feature_groups=None, neighbors=None, paths=None, center=None):
    # Feature groups are not used in this function para we keep the parameter to maintain the same function definition
    # as get_item_features and avoid an if statement being executed billion of times

    data_patch = dataset[item[0], :, item[1]: item[1] + neighbors, item[2]: item[2] + neighbors]
    data_feat = np.array([np.mean([data_patch[:, g[0], g[1]] for g in f], axis=0) for f in paths], dtype=np.float32)

    data_center = data_patch[:, center, center]
    data_center = data_center.reshape(1, data_center.shape[0])
    data_feat = np.append(data_feat, data_center, axis=0)

    data_feat = data_feat.reshape(data_feat.shape[0] * data_feat.shape[1])
    return data_feat
