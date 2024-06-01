import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50





class DeepLabV3Plus(Model):
    def __init__(self, target_size, num_classes=13):
        super(DeepLabV3Plus, self).__init__()
        self.target_size = target_size
        self.num_classes = num_classes
        self.model = self.build_model()
    
    def ASPP(self, inputs):
        shape = inputs.shape
        y_pool = layers.AveragePooling2D(pool_size=(shape[1], shape[2]))(inputs)
        y_pool = layers.Conv2D(filters=256, kernel_size=1, padding='same', use_bias=False)(y_pool)
        y_pool = layers.BatchNormalization()(y_pool)
        y_pool = layers.Activation("relu")(y_pool)
        y_pool = layers.UpSampling2D((shape[1], shape[2]), interpolation="bilinear")(y_pool)
        
        y_1 = layers.Conv2D(filters=256, kernel_size=1, padding='same', use_bias=False)(inputs)
        y_1 = layers.BatchNormalization()(y_1)
        y_1 = layers.Activation("relu")(y_1)
        
        y_6 = layers.Conv2D(filters=256, kernel_size=3, dilation_rate=6, padding='same', use_bias=False)(inputs)
        y_6 = layers.BatchNormalization()(y_6)
        y_6 = layers.Activation("relu")(y_6) 
        
        y_12 = layers.Conv2D(filters=256, kernel_size=3, dilation_rate=12, padding='same', use_bias=False)(inputs)
        y_12 = layers.BatchNormalization()(y_12)
        y_12 = layers.Activation("relu")(y_12) 
        
        y_18 = layers.Conv2D(filters=256, kernel_size=3, dilation_rate=18, padding='same', use_bias=False)(inputs)
        y_18 = layers.BatchNormalization()(y_18)
        y_18 = layers.Activation("relu")(y_18)  
        
        y = layers.Concatenate()([y_pool, y_1, y_6, y_12, y_18])
        
        y = layers.Conv2D(filters=256, kernel_size=1, padding='same', use_bias=False)(y)  # Fixed the misuse of inputs
        y = layers.BatchNormalization()(y)
        y = layers.Activation("relu")(y)
        
        return y

    def build_model(self):
        inputs = layers.Input(self.target_size)
        
        base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=inputs)
        
        image_features = base_model.get_layer('conv4_block6_out').output
        x_a = self.ASPP(image_features)
        x_a = layers.UpSampling2D((4, 4), interpolation='bilinear')(x_a)
        
        # Get low-level features
        x_b = base_model.get_layer('conv2_block2_out').output
        x_b = layers.Conv2D(filters=48, kernel_size=1, padding='same', use_bias=False)(x_b)
        x_b = layers.BatchNormalization()(x_b)
        x_b = layers.Activation('relu')(x_b)
        
        x = layers.Concatenate()([x_a, x_b])
        
        x = layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        x = layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        x = layers.UpSampling2D((4, 4), interpolation='bilinear')(x)
        
        # Output
        x = layers.Conv2D(self.num_classes, (1, 1), name="output_layer")(x)
        x = layers.Activation("softmax")(x)  # Changed to softmax for multi-class segmentation
        
        # Model
        model = Model(inputs=inputs, outputs=x)
        return model
    
    def call(self, inputs):
        return self.build_model()(inputs)
    

    def summary(self):
        return self.model.summary()
