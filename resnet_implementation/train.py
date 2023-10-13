import tensorflow as tf
import keras

from keras.callbacks import ModelCheckpoint

from keras.datasets import cifar10

from model_resnet import ResidualModel

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') is not None and logs.get('accuracy') > 0.99:
            print("\nReached 99% accuracy so stopping training!")
            self.model.stop_training = True
        if logs.get('accuracy') is not None and logs.get('accuracy') < 0.01:
            print("\nVery low accuracy, something is wrong with the model!")
            self.model.stop_training = True

class Train:
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape
    
    def train_model(self):
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        
        x_validation = x_test[0:5000]
        y_validation = y_test[0:5000]

        x_test = x_test[5000:10000]
        y_test = y_test[5000:10000]
        
        x_train, x_test, x_validation = x_train/255.0, x_test/255.0, x_validation/255.0
        y_train, y_test, y_validation = y_train.flatten(), y_test.flatten(), y_validation.flatten()
        
        # creating the model
        modelClass = ResidualModel()
        model = modelClass.model(input_shape = self.input_shape, output_shape=self.output_shape)
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        callbacks = myCallback()
        
        history = model.fit(
            x_train,
            y_train,
            epochs=30,
            validation_data=(
                x_validation,
                y_validation
            ),
            callbacks=[callbacks]
        )
        
        model.save('saved_model_residual')