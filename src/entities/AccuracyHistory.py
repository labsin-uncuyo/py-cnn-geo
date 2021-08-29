import tensorflow.keras as keras

class AccuracyHistory(keras.callbacks.Callback):

    def on_train_begin(self, logs={}):
        self.acc = []
        self.loss = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('val_acc'))
        self.loss.append(logs.get('val_loss'))
