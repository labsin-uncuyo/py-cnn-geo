from keras.callbacks import ModelCheckpoint
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential, Model
from keras.layers import Conv2D
from models.index_based_generator import IndexBasedGenerator
from entities.AccuracyHistory import AccuracyHistory
from config import NetworkParameters, SamplesConfig, DatasetConfig

import types
import copy
import datetime
import numpy as np
from os.path import exists
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.multiclass import unique_labels


class KerasBatchClassifier(KerasClassifier):

    def __init__(self, build_fn=None, verbose=0):
        super().__init__(build_fn=build_fn, verbose=verbose)
        self.dataset = None
        self.dataset_gt = None
        self.name = None

    def fit(self, X, y, **kwargs):
        print('Fitting %d elements' % (X.shape[0]))

        self.dataset = kwargs['dataset']
        self.dataset_gt = kwargs['dataset_gt']
        naming_args = kwargs['name_args']
        # taken from keras.wrappers.scikit_learn.KerasClassifier.fit ###################################################
        if self.build_fn is None:
            self.model = self.__call__(**self.filter_sk_params(self.__call__))
        elif not isinstance(self.build_fn, types.FunctionType) and not isinstance(self.build_fn, types.MethodType):
            self.model = self.build_fn(**self.filter_sk_params(self.build_fn.__call__))
        else:
            self.model = self.build_fn(**self.filter_sk_params(self.build_fn))

        loss_name = self.model.loss
        if hasattr(loss_name, '__name__'):
            loss_name = loss_name.__name__

        '''if loss_name == 'binary_crossentropy' and len(y.shape) != 2:
            y = to_categorical(y)'''

        ### fit => fit_generator
        fit_args = copy.deepcopy(self.filter_sk_params(Sequential.fit_generator))
        fit_args.update(kwargs)
        ################################################################################################################

        patch_size = self.get_padding(self.model.layers)
        offset = int(DatasetConfig.MAX_PADDING / 2) - int(patch_size / 2)

        train_idxs, val_idxs = self.divide_indexes(X)

        traingen = IndexBasedGenerator(batch_size=NetworkParameters.BATCH_SIZE, dataset=self.dataset,
                                       dataset_gt=self.dataset_gt, indexes=train_idxs, patch_size=patch_size,
                                       offset=offset)
        '''traingen = IndexBasedGenerator(batch_size=NetworkParameters.BATCH_SIZE, dataset=self.dataset,
                                       dataset_gt=self.dataset_gt, indexes=X, patch_size=patch_size,
                                       offset=offset)'''
        valgen = IndexBasedGenerator(batch_size=NetworkParameters.BATCH_SIZE, dataset=self.dataset,
                                     dataset_gt=self.dataset_gt, indexes=val_idxs, patch_size=patch_size, offset=offset)

        self.name = '{date:%Y%m%d_%H%M%S}'.format(date=datetime.datetime.now()) + '.' + self.build_name(naming_args, self.sk_params)

        # serialize model to JSON
        print('\n-------------------------------------------------------------------------------\n'
              'Starting with file %s'
              '\n-------------------------------------------------------------------------------\n\n' % (self.name))
        model_json = self.model.to_json()
        with open("../storage/search/" + self.name + ".json", "w") as json_file:
            json_file.write(model_json)

        accuracy_history = AccuracyHistory()
        #early_stopping = EarlyStopping(patience=5, verbose=5, mode="auto", monitor='acc')
        model_checkpoint = ModelCheckpoint(
            "../storage/search/" + self.name + ".weights.{epoch:02d}-{loss:.4f}-{acc:.4f}-{val_accuracy:.2f}.hdf5", monitor='val_accuracy',
            verbose=5, save_best_only=False, mode="auto")

        #callbacks = [accuracy_history, early_stopping, model_checkpoint]
        callbacks = [accuracy_history, model_checkpoint]

        # epochs = self.sk_params['epochs'] if 'epochs' in self.sk_params else 100

        fit_args.__delitem__('dataset')
        fit_args.__delitem__('dataset_gt')
        fit_args.__delitem__('name_args')

        self.__history = self.model.fit_generator(
            traingen,
            steps_per_epoch=int(train_idxs.shape[0] // NetworkParameters.BATCH_SIZE) + 1,
            validation_data=valgen,
            validation_steps=int(val_idxs.shape[0] // NetworkParameters.BATCH_SIZE) + 1,
            callbacks=callbacks,
            epochs=10,
            **fit_args
        )

        return self.__history

    def score(self, X, y, **kwargs):
        print('Scoring %d elements' % (X.shape[0]))

        kwargs = self.filter_sk_params(Sequential.evaluate_generator, kwargs)

        loss_name = self.model.loss
        if hasattr(loss_name, '__name__'):
            loss_name = loss_name.__name__
        # if loss_name == 'binary_crossentropy':# and len(y.shape) != 2:
        #    y = to_categorical(y)

        patch_size = self.get_padding(self.model.layers)
        offset = int(DatasetConfig.MAX_PADDING / 2) - int(patch_size / 2)

        evalgen = IndexBasedGenerator(batch_size=NetworkParameters.BATCH_SIZE, dataset=self.dataset,
                                      dataset_gt=self.dataset_gt,
                                      indexes=X, patch_size=patch_size, offset=offset)

        outputs = self.model.evaluate_generator(generator=evalgen,
                                                steps=int(X.shape[0] // NetworkParameters.BATCH_SIZE) + 1, **kwargs)

        predict_out = self.model.predict_generator(generator=evalgen,
                                                   steps=int(X.shape[0] // NetworkParameters.BATCH_SIZE) + 1,
                                                   use_multiprocessing=True, verbose=0)
        predict_out = np.argmax(predict_out, axis=1)

        expected_out = self.get_expected(X)

        cm = self.print_confusion_matrix(expected_out, predict_out, np.array(['No Forest', 'Forest']))

        print(classification_report(expected_out, predict_out, target_names=np.array(['no forest', 'forest'])))

        confmat_file = "../storage/search/" + self.name + ".conf_mat.npz"

        if not exists(confmat_file):
            np.savez_compressed(confmat_file, cm=cm)

        if type(outputs) is not list:
            outputs = [outputs]
        for name, output in zip(self.model.metrics_names, outputs):
            if name == 'acc':
                return output
        raise Exception('The model is not configured to compute accuracy. '
                        'You should pass `metrics=["accuracy"]` to '
                        'the `model.compile()` method.')

    def divide_indexes(self, indexes_file):
        split_1 = int(SamplesConfig.TEST_PERCENTAGE * indexes_file.shape[0])

        train_idxs = indexes_file[split_1:]
        validation_idxs = indexes_file[:split_1]

        return train_idxs, validation_idxs

    def get_padding(self, layers, ws=None):
        ws = ws if ws is not None else 1

        for layer_idx in range(len(layers) - 1, -1, -1):
            layer = layers[layer_idx]
            if type(layer) == Conv2D:
                padding = 0 if layer.padding == 'valid' else (
                    int(layer.kernel_size[0] / 2) if layer.padding == 'same' else 0)
                ws = ((ws - 1) * layer.strides[0]) - (2 * padding) + layer.kernel_size[0]
            if isinstance(layer, Model):
                ws = self.get_padding(layer.layers, ws)

        return ws

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

    def get_expected(self, indexes):
        expected = np.zeros(shape=(len(indexes),), dtype=np.uint8)
        for i, idx in enumerate(indexes):
            expected[i] = self.dataset_gt[idx[0], idx[1], idx[2]]

        return expected

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

        '''fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               # ... and label them with the respective list entries
               xticklabels=classes, yticklabels=classes,
               title=title,
               ylabel='True label',
               xlabel='Predicted label')

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        return ax, cm'''
        return cm

    @property
    def history(self):
        return self.__history


'''
    
unction NetworkBuilder.getWindowSize(net, ws)
   ws = ws or 1

   --for i = 1,#net.modules do
   for i = #net.modules,1,-1 do
      local module = net:get(i)
      if torch.typename(module) == 'cudnn.SpatialConvolution' or torch.typename(module) == 'cudnn.SpatialMaxPooling' then
         --ws = ws + module.kW - 1 - module.padW - module.padH
         ws = ((ws - 1) * module.dW) - (2 * module.padW) + module.kW
      end
      if module.modules then
         ws = NetworkBuilder.getWindowSize(module, ws)
      end
   end
   return ws
end
'''
