import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model

class Unet:
    """
    This class for U-Net architecture.
    
    """
    def __init__(self, num_filters: list, input_shape: tuple, n_class: int, activation: str, conv_block_size=1):
        self.num_filters = num_filters
        self.num_block = len(num_filters)
        self.input_shape = input_shape
        self.conv_block_size = conv_block_size
        self.n_class = n_class
        self.activation = activation
        self.model = self.build_unet()  # Build the model during instantiation

    def conv_block(self, input, filters):
        x = layers.Conv2D(filters, 3, padding='same')(input)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        return x
    
    def encoder_block(self, input, filters: int, num_blocks=1):
        """
        This function buid encoder for unet

        Args:
            input (array)
            filter (int)
            num_blocks (int)

        Returns:
            x (array) with same shape of input shape
            p (array) with maxpooling applied shape
        """
        x = input
        for i in range(num_blocks):
            x = self.conv_block(x, filters)
        p = layers.MaxPool2D((2, 2))(x)
        return x, p

    def decoder_block(self, input, skip_features, num_filters: int, num_blocks=1):
        """
        This function for build decoder block
        
        Args:
            input 
        """
        x = layers.Conv2DTranspose(num_filters, (2, 2), strides=2, padding='same')(input)
        x = layers.Concatenate()([x, skip_features])
        for i in range(num_blocks):
            x = self.conv_block(x, num_filters)
        return x

    def build_unet(self):
        """
        This function will build the U-Net architecture.
        """
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

        outputs = layers.Conv2D(self.n_class, 1, padding='same', activation=self.activation, name="classification_layer")(b)

        model = Model(inputs, outputs, name='U-Net')
        return model
    

    def __call__(self):
        return self.build_unet()

    def train(self, train_ds, valid_ds, epochs=25, callbacks=None):
        """
        Function for train the model

        Args:
            train batch (tf.dataset)
            valid batch (tf.dataset)
        """
        self.model.compile(
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            metrics=["accuracy"]
        )
        self.model.fit(train_ds, epochs=epochs, validation_data=valid_ds, callbacks=callbacks)
    
    def summary(self):
        """
        This function returns summary of model
        """
        self.model.summary()

    def prediction(self, image):
        """
        This function for prediction
        
        Args:
            Image (as batch)

        Return:
            prediction (array)
        """
        self.model.predict(image)

        




class AttentionUNet(Model):
    def __init__(self, input_shape, n_filters, n_class, activation, conv_block_size=1):
        super(AttentionUNet, self).__init__()
        self.num_filters = n_filters
        self.num_block = len(n_filters)
        self.input_shape = input_shape
        self.conv_block_size = conv_block_size
        self.n_class = n_class
        self.activation = activation


    def conv_block(self, input_layer, num_filters):
        x = layers.Conv2D(num_filters, 3, padding='same')(input_layer)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        
        return x
    
    def attention_layer(self, g, s, num_filters):
        """
        This function make attention mechanism.

        Args:
            g (array) output of decoder
            s (array) skip connection array
        
        Return:
            out (array)
        
        """
        wg = layers.Conv2D(num_filters, 1, padding='same')(g)
        wg = layers.BatchNormalization()(wg)
        
        ws = layers.Conv2D(num_filters, 1, padding='same')(s)
        ws = layers.BatchNormalization()(ws)
        
        out = layers.Activation('relu')(wg + ws)
        out = layers.Conv2D(num_filters, 1, padding='same')(out)
        out = layers.Activation('sigmoid')(out)
        
        return out 
    

    def encoder_block(self, input, filters: int, num_blocks=1):
        """
        This function buid encoder for unet

        Args:
            input (array)
            filter (int)
            num_blocks (int)

        Returns:
            x (array) with same shape of input shape
            p (array) with maxpooling applied shape
        """
        x = input
        for i in range(num_blocks):
            x = self.conv_block(x, filters)
        p = layers.MaxPool2D((2, 2))(x)
        return x, p
    
    def decoder_block(self, input, skip_features, num_filters: int, num_blocks=1):
        """
        This function for build decoder block
        
        Args:
            input (array)
            skip feature (array)
            num_filters (int)
            num_blocks (int)

        Returns:
            x (array)
        """
        x = layers.UpSampling2D(interpolation='bilinear')(input)
        s = self.attention_layer(x, skip_features, num_filters)
        x = layers.Concatenate()([x, s])
        for i in range(num_blocks):
            x = self.conv_block(x, num_filters)
        return x
    

    def build_att_unet(self):
        """
        This function will build the Attention U-Net architecture.
        """
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

        outputs = layers.Conv2D(self.n_class, 1, padding='same', activation=self.activation, name="classification_layer")(b)

        model = Model(inputs, outputs, name='Attention U-Net')
        return model
    
