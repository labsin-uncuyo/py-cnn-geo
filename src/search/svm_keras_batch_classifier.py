import datetime
import gc
import multiprocessing
import types
import numpy as np
import _pickle as pkl
from functools import partial
from keras.wrappers.scikit_learn import KerasClassifier
from os import makedirs
from os.path import exists, join
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.utils.multiclass import unique_labels
from config import SamplesConfig, NetworkParameters
from entities.parameter_search_status import ParameterSearchStatus


class KerasBatchClassifier(KerasClassifier):

    def __init__(self, build_fn=None, verbose=0):
        super().__init__(build_fn=build_fn, verbose=verbose)
        # self.dataset = None
        # self.dataset_gt = None
        self.dir_name = None
        self.name = None
        self.store_path = None
        self.store_sub_path = None
        self.neighbors = None
        self.feature_groups = None
        self.paths = None
        self.center = None
        self.feature_extraction_fn = None
        self.n_features = None
        self.date_and_time = None
        self.final_estimators = None
        self.search_status = None
        self.total_kfold = None
        self.perform_execution = False


    def fit(self, X, y, **kwargs):
        global dataset
        global dataset_gt
        dataset = kwargs['dataset']
        dataset_gt = kwargs['dataset_gt']
        naming_args = kwargs['name_args']
        self.neighbors = kwargs['neighbors']
        self.paths = kwargs['paths']
        self.center = kwargs['center']
        self.store_sub_path = kwargs['store_sub_path']
        self.total_kfold = kwargs['total_kfold']

        self.dir_name = self.build_name(naming_args, self.sk_params)
        self.date_and_time = '{date:%Y%m%d_%H%M%S}'.format(date=datetime.datetime.now())
        self.name = self.date_and_time + '.' + self.dir_name + ('-vec' if self.paths is not None else '-cir')

        self.store_path = join(self.store_sub_path, self.dir_name)
        if not exists(self.store_path):
            makedirs(self.store_path)

        with open(join(self.store_sub_path, 'search-status.pkl'), 'rb') as input:
            self.search_status = pkl.load(input)

        if self.search_status is None:
            self.search_status = ParameterSearchStatus()

        self.search_status.set_current_parameter(self.dir_name)
        if self.search_status.get_current_kfold() == 0 or self.search_status.get_current_kfold() == self.total_kfold:
            self.search_status.set_current_kfold(1)
        else:
            self.search_status.set_current_kfold(self.search_status.get_current_kfold() + 1)

        if self.search_status.get_status() is ParameterSearchStatus.UNDER_EXECUTION:
            self.perform_execution = True
        if self.search_status.get_status() is ParameterSearchStatus.CONTINUE_PREVIOUS_FAIL:
            if self.search_status.get_current_parameter() == self.search_status.get_continue_parameter() \
                    and self.search_status.get_current_kfold() == self.search_status.get_continue_kfold():
                self.perform_execution = True
                self.search_status.set_status(ParameterSearchStatus.UNDER_EXECUTION)

        with open(join(self.store_sub_path, 'search-status.pkl'), 'wb') as output:
            pkl.dump(self.search_status, output, -1)

        print('\n-------------------------------------------------------------------------------\n'
              'File %s (kfold %s)'
              '\n-------------------------------------------------------------------------------\n\n' % (
              self.name, str(self.search_status.get_current_kfold())))

        if self.perform_execution:
            print('Fitting %d elements' % (X.shape[0]))
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

            X_train, X_val = self.divide_indexes(X)

            self.feature_groups = int(self.neighbors / 2) + 1

            estimator_step_size = 500
            batch_size = (NetworkParameters.BATCH_SIZE*2)

            steps = int(X_train.shape[0] / batch_size) + 1
            estimator_steps = int(steps / estimator_step_size) + 1

            if self.paths is not None:
                self.feature_extraction_fn = get_item_features_vector
                self.n_features = (self.paths.shape[0] + 1) * dataset.shape[1]
            else:
                self.feature_extraction_fn = get_item_features
                self.n_features = dataset.shape[1] * self.feature_groups

            print('Starting training phase...')

            #self.final_estimators = self.model.n_estimators

            print('Train progress: ', end='')

            epochs = 10
            for e in range(epochs):
                print('\n========== EPOCH %s ===========\n' % str(e + 1))

                np.random.shuffle(X_train)
                for i in range(estimator_steps):
                    print('{0}/{1} - '.format(i + 1, estimator_steps), end='')
                    start = (i * estimator_step_size) * batch_size
                    end = ((i + 1) * estimator_step_size) * batch_size
                    end = end if end < X_train.shape[0] else X_train.shape[0]

                    X_preprocessed_train_bag = np.zeros(shape=(end - start, self.n_features), dtype=np.float32)
                    Y_preprocessed_train_bag = np.zeros(shape=(end - start), dtype=np.uint8)

                    for pxt_i, pxt_item in enumerate(X_train[start:end, :]):
                        X_preprocessed_train_bag[pxt_i] = self.feature_extraction_fn(pxt_item, feature_groups=self.feature_groups,
                                                                                     neighbors=self.neighbors, paths=self.paths, center=self.center)
                        Y_preprocessed_train_bag[pxt_i] = get_item_gt(pxt_item)

                    # pool = multiprocessing.Pool(processes=int(multiprocessing.cpu_count()/2))
                    # X_preprocessed_train_bag = np.array(pool.map(partial(self.feature_extraction_fn, feature_groups=self.feature_groups, neighbors=self.neighbors, paths=self.paths, center=self.center), X_train[start:end,:]))
                    # Y_preprocessed_train_bag = np.array(pool.map(get_item_gt, X_train[start:end,:]))
                    # pool.close()
                    # pool.join()


                    self.model.partial_fit(X_preprocessed_train_bag, Y_preprocessed_train_bag, classes=[0,1])
                    X_preprocessed_train_bag = None
                    Y_preprocessed_train_bag = None

                    gc.collect()

                print('Epoch Finished!\n')


                X_partial_preprocessed_val_bag = np.zeros(shape=(X_val.shape[0], dataset.shape[1] * self.n_features), dtype=np.float32)
                expected_val = np.zeros(shape=(X_val.shape[0],), dtype=np.uint8)
                pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
                X_partial_preprocessed_val_bag = np.array(
                    pool.map(
                        partial(self.feature_extraction_fn, feature_groups=self.feature_groups,
                                neighbors=self.neighbors, paths=self.paths, center=self.center), X_val))
                expected_val = np.array(pool.map(get_item_gt, X_val))
                pool.close()
                pool.join()

                predicted_val = self.model.predict(X_partial_preprocessed_val_bag)

                val_acc = accuracy_score(expected_val, predicted_val)

                X_partial_preprocessed_val_bag = None
                expected_val = None
                predicted_val = None

                print('Validation score: {val_acc:.4f}\n'.format(val_acc=val_acc))

                # print('Storing model...')
                joblib.dump(self.model, join(self.store_path, self.name + '-epoch_{epoch:02d}-val_acc_{val_acc:.4f}'.format(epoch=e, val_acc=val_acc) + '.pkl'))

            X_train = None
            X_val = None
            gc.collect()

        else:
            print('Passing this FIT execution!\n')

        return None

    def score(self, X, y, **kwargs):
        if self.perform_execution:
            perform_test = False
            if self.search_status.get_current_test() == 0 or self.search_status.get_current_test() == 2:
                self.search_status.set_current_test(1)
                perform_test = True
            else:
                self.search_status.set_current_test(self.search_status.get_current_test() + 1)

            with open(join(self.store_sub_path, 'search-status.pkl'), 'wb') as output:
                pkl.dump(self.search_status, output, -1)

            if perform_test:
                print('Testing...')
                X_preprocessed_test_bag = np.zeros(shape=(X.shape[0], dataset.shape[1] * self.feature_groups),
                                                   dtype=np.float32)
                expected_test = np.zeros(shape=(X.shape[0],), dtype=np.uint8)

                pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
                X_preprocessed_test_bag = np.array(
                    pool.map(partial(self.feature_extraction_fn, feature_groups=self.feature_groups,
                                neighbors=self.neighbors, paths=self.paths, center=self.center), X))
                expected_test = np.array(pool.map(get_item_gt, X))
                pool.close()
                pool.join()
                # for item_i, test_item in enumerate(X_test):
                #    X_preprocessed_test_bag[item_i] = get_patch_features(
                #        dataset[test_item[0], :, test_item[1]: test_item[1] + neighbors, test_item[2]: test_item[2] + neighbors],
                #        feature_groups)
                #    Y_preprocessed_test_bag[item_i] = dataset_gt[test_item[0], test_item[1], test_item[2]]

                # print(clf.score(X_preprocessed_test_bag, Y_preprocessed_test_bag))

                predicted_test = self.model.predict(X_preprocessed_test_bag)

                print('Calculating reports and metrics...')

                test_acc = accuracy_score(expected_test, predicted_test)
                print('Test score: {test_acc:.4f}'.format(test_acc=test_acc))

                print('Storing value accuracy...')
                metrics_filename = join(self.store_path,
                                        self.name + '-score_{test_acc:.4f}'.format(test_acc=test_acc) + '.txt')

                cm = self.print_confusion_matrix(expected_test, predicted_test, np.array(['No Forest', 'Forest']))

                with open(metrics_filename, 'w') as output:
                    output.write(str(cm))

                class_report = classification_report(expected_test, predicted_test,
                                                     target_names=np.array(['no forest', 'forest']))

                print(class_report)

                with open(metrics_filename, 'a') as output:
                    output.write('\n\n' + str(class_report))

                confmat_file = join(self.store_path, self.name + '.conf_mat.npz')

                print('Storing confusion matrix...')
                if not exists(confmat_file):
                    np.savez_compressed(confmat_file, cm=cm)

                return test_acc
            else:
                print('Passing second SCORE execution!\n')
                return 0
        else:
            print('Passing this SCORE execution!\n')
            return 0

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
