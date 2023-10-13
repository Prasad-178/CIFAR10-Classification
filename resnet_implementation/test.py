import tensorflow as tf
import keras

from keras.datasets import cifar10

from model_resnet import ResidualModel

class Test:
    def test_model(self):
        model = tf.keras.models.load_model('resnet_implementation/saved_model_residual')
        
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        
        x_validation = x_test[0:5000]
        y_validation = y_test[0:5000]
        
        x_test = x_test[5000:10000]
        y_test = y_test[5000:10000]
        
        x_train, x_test, x_validation = x_train/255.0, x_test/255.0, x_validation/255.0
        y_train, y_test, y_validation = y_train.flatten(), y_test.flatten(), y_validation.flatten()
        
        print(model.summary())
        
        scores = model.evaluate(x_test, y_test)
        
        print(f'test accuracy is {scores[1]}')