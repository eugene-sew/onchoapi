import os
import numpy as np
import tensorflow as tf
from .effnetv2_model import EffNetV2Model

class FineTuningModel:
    def __init__(self, model_name, pretrained_ckpt, debug=True):
        self.model = None
        self.debug = debug

        # Load EfficientNetV2 model
        self.base_model = EffNetV2Model(model_name, include_top=False)
        input_shape = (None, None, 3)

        # Initialize base model
        self.base_model(tf.keras.Input(shape=input_shape), training=True, with_endpoints=False)

        # Load weights from checkpoint if provided
        if pretrained_ckpt:
            if tf.io.gfile.isdir(pretrained_ckpt):
                pretrained_ckpt = tf.train.latest_checkpoint(pretrained_ckpt)
            self.base_model.load_weights(pretrained_ckpt)
            print(f"--- Loaded weights from {pretrained_ckpt}")

    def show_base_model_layers(self):
        print("--- EfficientNetV2 Layers")
        if self.debug:
            for i, layer in enumerate(self.base_model.layers):
                print(f"--- i: {i}, name: {layer.name}, trainable: {layer.trainable}")

    def build(self, image_size, num_classes, fine_tuning, trainable_layers_ratio=0.3):
        num_layers = len(self.base_model.layers)
        print(f"--- num_layers: {num_layers}")

        if fine_tuning:
            non_trainable_layers_ratio = 1.0 - trainable_layers_ratio
            non_trainable_max_layers = int(num_layers * non_trainable_layers_ratio)
            print(f"--- non_trainable_max_layers: {non_trainable_max_layers}")

            # Make a portion of the layers trainable for fine-tuning
            self.base_model.trainable = True
            for i, layer in enumerate(self.base_model.layers):
                if i < non_trainable_max_layers:
                    layer.trainable = False
                else:
                    layer.trainable = True
        else:
            # Freeze the base model for transfer learning
            self.base_model.trainable = False

        self.show_base_model_layers()

        input_shape = [image_size, image_size, 3]
        dropout_rate = 0.2  # Adjust as needed or pass it as an argument

        # Create a Sequential model
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.InputLayer(input_shape=input_shape, name='image', dtype=tf.float32),
            self.base_model,
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Dense(num_classes, name='predictions', kernel_regularizer=tf.keras.regularizers.l2(0.0001))
        ])

        self.show_customized_model_layers()
        return self.model

    def show_customized_model_layers(self):
        print("--- Customized EfficientNetV2 Layers")
        if self.debug:
            for i, layer in enumerate(self.model.layers):
                print(f"--- i: {i}, name: {layer.name}, trainable: {layer.trainable}")
