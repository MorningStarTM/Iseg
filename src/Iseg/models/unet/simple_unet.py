import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model



class Unet:
    def __init__(self, num_filers:list, input_shape:tuple, n_class:int, activation:str, conv_block_size=1):
        self.num_filters = num_filers
        self.num_block =  len(num_filers)
        self.input_shape = input_shape
        self.conv_block_size = conv_block_size
        self.n_class = n_class
        self.activation = activation

    def conv_block(self, input, filters):
        x = layers.Conv2D(filters, 3, padding='same')(input)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        return x
    
    def encoder_block(self, input, filters:int, num_blocks=1):
        x = input
        for i in range(num_blocks):
            x = self.conv_block(x, filters)
        p = layers.MaxPool2D((2, 2))(x)
        return x, p
    

    def decoder_block(self, input, skip_features, num_filters:int, num_blocks=1):
        x = layers.Conv2DTranspose(num_filters, (2,2), strides=2, padding='same')(input)
        x = layers.Concatenate()([x, skip_features])
        for i in range(num_blocks):
            x = self.conv_block(x, num_filters)
        return x
    

    def build_unet(self):
        inputs = layers.Input(self.input_shape)
        skips = []
        x = inputs

        for i in range(self.num_block):
            s, x = self.encoder_block(x, filters=self.num_filters[i])
            skips.append(s)

        b = self.conv_block(x, self.num_filters[-1] * 2)
        print(b.shape)

        for i in range(self.num_block):
            b = self.decoder_block(b, skips[-(i+1)], self.num_filters[-(i+1)])

        outputs = layers.Conv2D(1, 1, padding='same', activation='sigmoid', name="classification_layer")(b)

        model = Model(inputs, outputs, name='U-Net')
        return model
    