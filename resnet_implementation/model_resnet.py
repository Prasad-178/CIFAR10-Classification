import tensorflow as tf
import keras

from keras.models import Model
from keras.layers import Dense, MaxPooling2D, Conv2D, Flatten, BatchNormalization, Activation, GlobalAveragePooling2D, DepthwiseConv2D, Dropout, ReLU, Concatenate, InputLayer, add, Input

from keras.callbacks import ModelCheckpoint

from keras.optimizers import adam
import numpy as np

class ResidualModel:
    def model(self, input_shape, output_shape):
        input = Input(shape=(input_shape[0], input_shape[1], input_shape[2]))
        
        # block 1
        b1_conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same',
                          name='b1_conv1', kernel_initializer='normal')(input)
        b1_act1 = ReLU(name='b1_act1')(b1_conv1)
        b1_norm1 = BatchNormalization(name='b1_bn1')(b1_act1)
        
        b1_conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same',
                          name='b1_conv2', kernel_initializer='normal')(b1_norm1)
        b1_act2 = ReLU(name='b1_act2')(b1_conv2)
        b1_out = BatchNormalization(name='b1_bn2')(b1_act2)
        
        # block 2
        b2_conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same',
                          name='b2_conv1', kernel_initializer='normal')(b1_out)
        b2_act1 = ReLU(name='b2_act1')(b2_conv1)
        b2_norm1 = BatchNormalization(name='b2_bn1')(b2_act1)
        
        b2_add = add([b1_out, b2_norm1])
        
        b2_conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same',
                          name='b2_conv2', kernel_initializer='normal')(b2_add)
        b2_act2 = ReLU(name='b2_act2')(b2_conv2)
        b2_out = BatchNormalization(name='b2_bn2')(b2_act2)
        
        # block 3
        b3_conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same',
                          name='b3_conv1', kernel_initializer='normal')(b2_out)
        b3_act1 = ReLU(name='b3_act1')(b3_conv1)
        b3_norm1 = BatchNormalization(name='b3_bn1')(b3_act1)
        
        b3_add = add([b2_out, b3_norm1])
        
        b3_conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same',
                          name='b3_conv2', kernel_initializer='normal')(b3_add)
        b3_act2 = ReLU(name='b3_act2')(b3_conv2)
        b3_out = BatchNormalization(name='b3_bn2')(b3_act2)
        
        # block 4
        b4_conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same',
                          name='b4_conv1', kernel_initializer='normal')(b3_out)
        b4_act1 = ReLU(name='b4_act1')(b4_conv1)
        b4_norm1 = BatchNormalization(name='b4_bn1')(b4_act1)
        
        b4_add = add([b3_out, b4_norm1])
        
        b4_conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same',
                          name='b4_conv2', kernel_initializer='normal')(b4_add)
        b4_act2 = ReLU(name='b4_act2')(b4_conv2)
        b4_out = BatchNormalization(name='b4_bn2')(b4_act2)
        
        # block 5
        b5_avgp = GlobalAveragePooling2D(name='b5_avgp1')(b4_out)
        output = Dense(output_shape, name='output', activation='softmax',
                       kernel_initializer='he_uniform')(b5_avgp)
        
        # finalizing the model
        model = Model(input, output)
        
        model_json = model.to_json()
        with open("residual_cifar10.json", "w") as json_file:
            json_file.write(model_json)
        
        model.summary()
        return model
