import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
from utils.metrics import Metrics
from tensorflow.keras.metrics import Recall, Precision
from tensorflow.keras.applications import VGG16, VGG19
from tensorflow.keras.applications.resnet50 import ResNet50

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
            metrics = [Metrics.dice_coef, Metrics.iou, Recall(), Precision()]
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

        



class AttentionUNet:
    def __init__(self, target_shape, n_filters:list, n_class:int, activation:str, conv_block_size=1):
        self.num_filters = n_filters
        self.num_block = len(n_filters)
        self.target_shape = target_shape
        self.conv_block_size = conv_block_size
        self.n_class = n_class
        self.activation = activation
        self.model = self.build_att_unet()


    def conv_block(self, input_layer, num_filters):
        x = layers.Conv2D(num_filters, 3, padding='same')(input_layer)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        
        return x
    
    def attention_layer(self, g, s, num_filters):
        """
        This function make attention mechanism.

        Args:
            g (tensor) output of decoder
            s (tensor) skip connection array
        
        Return:
            out (tensor)
        
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
            input (tensor)
            filter (int)
            num_blocks (int)

        Returns:
            x (tensor) with same shape of input shape
            p (tensor) with maxpooling applied shape
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
            input (tensor)
            skip feature (tensor)
            num_filters (int)
            num_blocks (int)

        Returns:
            x (tensor)
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
        inputs = layers.Input(shape=self.target_shape)
        skips = []
        x = inputs

        for i in range(self.num_block):
            s, x = self.encoder_block(x, filters=self.num_filters[i], num_blocks=2)
            skips.append(s)

        b = self.conv_block(x, self.num_filters[-1] * 2)
        print(b.shape)

        for i in range(self.num_block):
            b = self.decoder_block(b, skips[-(i+1)], self.num_filters[-(i+1)], num_blocks=2)

        outputs = layers.Conv2D(self.n_class, 1, padding='same', activation=self.activation, name="classification_layer")(b)

        model = Model(inputs, outputs, name='Attention U-Net')
        return model
    

    def __call__(self):
        return self.build_att_unet()
    
    def summary(self):
        """
        This function returns summary of model
        """
        self.model.summary()

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
            metrics = [Metrics.dice_coef, Metrics.iou, Recall(), Precision()]
        )
        self.model.fit(train_ds, epochs=epochs, validation_data=valid_ds, callbacks=callbacks)


    def prediction(self, image):
        """
        This function for prediction
        
        Args:
            Image (as batch)

        Return:
            prediction (array)
        """
        self.model.predict(image)



class ResUNet:
    def __init__(self, target_shape, n_filters:list, n_class:int, activation:str, conv_block_size=1):
        self.num_filters = n_filters
        self.num_block = len(n_filters)
        self.target_shape = target_shape
        self.conv_block_size = conv_block_size
        self.n_class = n_class
        self.activation = activation
        self.model = self.build_att_unet()


    def batch_norm_relu(self, inputs):
        """
        This function for create natch normalization 
        Args:
            input (tensor)

        Return:
            x (tensor)
        """
        x = layers.BatchNormalization()(inputs)
        x = layers.Activation("relu")(x)
        return x
    
    def residual_block(self, inputs, num_filters, strides=1):
        """
        This function to build residual connection

        Args:
            inputs (tensor)
            num_filters (int)
        
        Returns:
            x (tensor)
        
        """
        x = self.batch_norm_relu(inputs)
        x = layers.Conv2D(num_filters, (3,3), padding='same', strides=strides)(x)
        x = self.batch_norm_relu(x)
        x = layers.Conv2D(num_filters, (3,3), padding='same', strides=1)(x)
        
        short = layers.Conv2D(num_filters, 1, padding='same', strides=strides)(inputs)
        x = layers.Add()([x, short])
        return x
    

    
    def decoder_block(self, input, skip_features, num_filters: int):
        """
        This function for build decoder block
        
        Args:
            input (tensor)
            skip feature (tensor)
            num_filters (int)
            num_blocks (int)

        Returns:
            x (tensor)
        """
        x = layers.UpSampling2D((2,2))(input)
        x = layers.Concatenate()([x, skip_features])
        x = self.residual_block(x, num_filters, strides=1)
        return x
    

    def build_att_unet(self):
        """
        This function will build the Attention U-Net architecture.
        """
        inputs = layers.Input(shape=self.target_shape)
        skips = []
        x = inputs
        x = layers.Conv2D(self.num_filters[0], (3,3), padding='same', strides=1)(inputs)
        x = self.batch_norm_relu(x)
        x = layers.Conv2D(self.num_filters[0], (3,3), padding='same', strides=1)(inputs)
        
        s = layers.Conv2D(self.num_filters[0], 1, padding='same')(inputs)
        x = layers.Concatenate()([x,s])
        skips.append(x)


        for i in range(1, self.num_block):
            x = self.residual_block(x, num_filters=self.num_filters[i], strides=2)
            skips.append(x)
        
        print(skips)
        b = self.residual_block(x, self.num_filters[-1] * 2, strides=2)
        print(b.shape)

        for i in range(self.num_block):
            b = self.decoder_block(b, skips[-(i+1)], self.num_filters[-(i+1)])

        outputs = layers.Conv2D(self.n_class, 1, padding='same', activation=self.activation, name="classification_layer")(b)

        model = Model(inputs, outputs, name='Attention U-Net')
        return model
    

    def __call__(self):
        return self.build_att_unet()
    
    def summary(self):
        """
        This function returns summary of model
        """
        self.model.summary()

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
            metrics = [Metrics.dice_coef, Metrics.iou, Recall(), Precision()]
        )
        self.model.fit(train_ds, epochs=epochs, validation_data=valid_ds, callbacks=callbacks)


    def prediction(self, image):
        """
        This function for prediction
        
        Args:
            Image (as batch)

        Return:
            prediction (array)
        """
        self.model.predict(image)



class Double_Unet:
    """
    This class build double u-net. currently this architecture supports VGG16 and VGG19
    """
    def __init__(self, target_shape, n_filters:list, n_class:int, model_name:str):
        self.num_filters = n_filters
        self.num_block = len(n_filters)
        self.target_shape = target_shape
        self.n_class = n_class
        self.model_name = model_name
        self.model = None

    def squeeze_excite_block(inputs, ratio=8):
        """
        This is function for build Squeeze and Excitation (SE) mechanism

        Args:
            input (tensor)
        
        Return:
            x (tensor)
        """
        init = inputs       
        channel_axis = -1
        filters = init.shape[channel_axis]
        se_shape = (1, 1, filters)

        se = layers.GlobalAveragePooling2D()(init)     
        se = layers.Reshape(se_shape)(se)
        se = layers.Dense(filters//ratio, activation="relu", use_bias=False)(se)
        se = layers.Dense(filters, activation="sigmoid", use_bias=False)(se)

        x = layers.Multiply()([inputs, se])
        return x
    

    def ASPP(self, x, filters):
        """
        This function for build Atrous Spatial Pyramid Pooling.

        Args:
            x (tensor) :features
            filters (int)

        Return :
            y (tensor)
        """
        shape = x.shape
        
        y1 = layers.AveragePooling2D(pool_size=(shape[1], shape[2]))(x)
        y1 = layers.Conv2D(filters, 1, padding='same')(y1)
        y1 = layers.BatchNormalization()(y1)
        y1 = layers.Activation("relu")(y1)
        y1 = layers.UpSampling2D((shape[1], shape[2]), interpolation='bilinear')(y1)
        
        y2 = layers.Conv2D(filters, 1, dilation_rate=1, padding='same', use_bias=False)(x)
        y2 = layers.BatchNormalization()(y2)
        y2 = layers.Activation("relu")(y2) 
        
        y3 = layers.Conv2D(filters, 3, dilation_rate=6, padding='same', use_bias=False)(x)
        y3 = layers.BatchNormalization()(y3)
        y3 = layers.Activation("relu")(y3) 
        
        y4 = layers.Conv2D(filters, 3, dilation_rate=12, padding='same', use_bias=False)(x)
        y4 = layers.BatchNormalization()(y4)
        y4 = layers.Activation("relu")(y4) 
        
        y5 = layers.Conv2D(filters, 3, dilation_rate=18, padding='same', use_bias=False)(x)
        y5 = layers.BatchNormalization()(y5)
        y5 = layers.Activation("relu")(y5) 
        
        y = layers.Concatenate()([y1, y2, y3, y4, y5])
        
        y = layers.Conv2D(filters, 1, dilation_rate=1, padding='same', use_bias=False)(y)
        y = layers.BatchNormalization()(y)
        y = layers.Activation("relu")(y) 
        
        return y
    

    def encoder_1(self, inputs):
        """
        This function build encoder. currently supports vgg model architectures.
        Args:
            inputs (tensor)
        
        Returns:
            output (tensor)
            skip_connection (tensor)
        """

        skip_connections = []

        if self.model_name == "vgg16":
            model = VGG19(include_top=False, weights="imagenet", input_tensor=inputs)
            names = ["block1_conv2", "block2_conv2", "block3_conv4", "block4_conv4"]
        elif self.model_name == "vg19":
            model = VGG16(include_top=False, weights="imagenet", input_tensor=inputs)
            names = ["block1_conv2", "block2_conv2", "block3_conv3", "block4_conv3"]


        
        for name in names:
            skip_connections.append(model.get_layer(name).output)

        output = model.layers[-2].output
        return output, skip_connections
    

    def decoder_1(self, inputs, skip_connections):
        """
        This function build decoder block

        Args:
            inputs (tensor)
            skip connection(tensor)

        Return:
            x (tensor)
        """
        num_filters = [256, 128, 64, 32]
        skip_connections.reverse()

        x = inputs
        for i, f in enumerate(num_filters):
            x = layers.UpSampling2D((2, 2), interpolation="bilinear")(x)
            x = layers.Concatenate()([x, skip_connections[i]])
            x = self.conv_block(x, f)

        return x
    

    def decoder_2(self, inputs, skip_1, skip_2):
        """
        This function build decoder block

        Args:
            inputs (tensor)
            skip_1 (tensor)
            skip_2 (tensor)
        
        Returns:
            x (tensor)
        """
        num_filters = [256, 128, 64, 32]
        skip_2.reverse()

        x = inputs
        for i, f in enumerate(num_filters):
            x = layers.UpSampling2D((2, 2), interpolation="bilinear")(x)
            x = layers.Concatenate()([x, skip_1[i], skip_2[i]])
            x = self.conv_block(x, f)

        return x
    
    def output_block(inputs):
        """
        This function for helping to final layer of architecture

        Args:
            inputs (tensor)

        Return:
            x (tensor)
        """
        x = layers.Conv2D(1, 1, padding="same")(inputs)
        x = layers.Activation("sigmoid")(x)
        return x
    
    def build_double_unet(self, input_shape):
        """
        This function contain multiple component (ASPP, encoder, decoder) for build double U-Net architecture.
        """
        inputs = layers.Input(input_shape)
        x, skip_1 = self.encoder_1(inputs)
        x = self.ASPP(x, 64)
        x = self.decoder_1(x, skip_1)
        outputs1 = self.output_block(x)
        
        x = layers.Multiply()([inputs, outputs1])
        
        x, skip_2 = self.encoder_2(x)
        x = self.ASPP(x, 64)
        x = self.decoder_2(x, skip_1, skip_2)
        outputs2 = self.output_block(x)
        
        x = layers.Concatenate()([outputs1, outputs2])
        outputs = layers.Conv2D(1, 1, padding='same', activation='sigmoid')(x)
        
        model = Model(inputs, outputs)
        return model
    

    