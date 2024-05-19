import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import optimizers



class Unet:
    def __init__(self, num_filers:list, num_block:int, input_shape:tuple, n_class:int, activation:str, conv_block_size=1):
        self.num_filters = num_filers
        self.num_block: num_block
        self.input_shape = input_shape
        self.conv_block_size = conv_block_size
        self.n_class = n_class
        self.activation = activation

    def conv_block(input, filters):
        x = layers.Conv2D(filters, 3, padding='same')(input)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        return x
    
    def encoder_block(self, input, filters:int, num_blocks=1):
        x = input
        for _ in range(num_blocks):
            x = self.conv_block(x, filters)
        p = layers.MaxPool2D((2, 2))(x)
        return x, p
    

    def decoder_block(self, input, skip_features, num_filters:int, num_blocks=1):
        x = layers.Conv2DTranspose(num_filters, (2,2), strides=2, padding='same')(input)
        x = layers.Concatenate()([x, skip_features])
        for _ in range(num_blocks):
            x = self.conv_block(x, num_filters)
        return x
    

    def build_unet(self):
        inputs = layers.Input(self.input_shape)
        skips = []
        x = inputs

        for i in range(len(self.num_filters)):
            x, p = self.encoder_block(x, self.num_filters[i], num_blocks=self.conv_block_size)
            skips.append(x) 

        b = self.conv_block(p, self.num_filters[-1] * 2)


        skips = reversed(skips[:-1])  # Exclude the last skip connection from the encoder path
        for i, skip in enumerate(skips):
            b = self.decoder_block(b, skip, self.num_filters[-(i+2)], num_blocks=self.num_blocks)


        outputs = layers.Conv2D(self.n_class, 1, padding='same', activation=self.activation)(b)

        model = layers.Model(inputs, outputs, name='U-Net')
        return model
