import tensorflow as tf
from tensorflow import keras
from keras import layers
# from keras.preprocessing.image import load_img
from keras.utils import load_img
from tensorflow.python.client import device_lib

import os
import datetime
from matplotlib import pyplot as plt
import numpy as np

# init
print(device_lib.list_local_devices())

########## CLASSES ##########
class CellDataset(keras.utils.Sequence):
    def __init__(self, path: str, num_images: int, batch_size: int):
        self.path = path
        self.batch_size = batch_size
        self.num_images = num_images

    def __getitem__(self, idx):
        start = idx * self.batch_size
        images = []
        masks = []
        for i in range(start, start + self.batch_size):
            img_path = self.path + 'images/' + str(i) + '.jpg'
            mask_path = self.path + 'masks/' + str(i) + '.jpg'
            img = tf.keras.utils.img_to_array(load_img(img_path))[:,:,0]
            mask = tf.keras.utils.img_to_array(load_img(mask_path))[:,:,0] / 255
            images.append(img)
            masks.append(mask)
        
        images = np.stack(images)
        masks = np.stack(masks)

        return images, masks

    def __len__(self):
        return self.num_images // self.batch_size
    
class TrainConfig:
    def __init__(self, batch_size, learning_rate, n_epochs, n_transformer_blocks):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.n_transformer_blocks = n_transformer_blocks

########## FUNCTIONS ##########
def create_vit_classifier(input_shape=(520, 704, 1), number_of_transformer_blocks=1):
    inputs = layers.Input(shape=input_shape)
    
    # Augment data
    # augmented = data_augmentation(inputs)
    
    # Create patches (downsampling)
    encoded_patches = layers.Conv2D(2, 2, strides=2, padding="same", activation='relu')(inputs)
    encoded_patches = layers.Conv2D(2, 2, strides=2, padding="same", activation='relu')(encoded_patches)
    encoded_patches = layers.Conv2D(2, 2, strides=2, padding="same", activation='relu')(encoded_patches)
    encoded_patches = layers.Conv2D(2, 2, strides=2, padding="same", activation='relu')(encoded_patches)

    # Create multiple layers of the Transformer block.
    for _ in range(number_of_transformer_blocks):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=2, key_dim=2, dropout=0.1
        )(x1, x1)
        encoded_patches = attention_output

        # Skip connection 1.
        # x2 = layers.Add()([attention_output, encoded_patches])
        # # Layer normalization 2.
        # x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # # MLP.
        # # x3 = layers.Dense(2, activation=tf.nn.gelu)(x3)
        # x3 = layers.Dropout(0.1)(x3)
        # # Skip connection 2.
        # encoded_patches = layers.Add()([x3, x2])

    # Upsampling
    encoded_patches = layers.UpSampling2D(2)(encoded_patches)
    encoded_patches = layers.Conv2D(2, 4,padding="same", activation='relu')(encoded_patches)
    encoded_patches = layers.UpSampling2D(2)(encoded_patches)
    encoded_patches = layers.Conv2D(2, 4,padding="same", activation='relu')(encoded_patches)
    encoded_patches = layers.UpSampling2D(2)(encoded_patches)
    encoded_patches = layers.Conv2D(2, 4,padding="same", activation='relu')(encoded_patches)
    encoded_patches = layers.UpSampling2D(2)(encoded_patches)
    encoded_patches = tf.keras.layers.Resizing(520, 704, interpolation="bilinear", crop_to_aspect_ratio=False)(encoded_patches)

    # Output
    outputs = layers.Conv2D(1, 4, padding="same", activation=tf.nn.sigmoid)(encoded_patches)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

def sample_dataset_image(dataset_path):
    a = CellDataset(dataset_path + "val/", 500, 10)

    i, m = next(iter(a))
    print(i.shape, m.shape)

    plt.imshow(i[7], cmap='gray', vmin=0, vmax=255)
    plt.show()
    plt.imshow(m[7]*255, cmap='gray', vmin=0, vmax=255)
    plt.show()

def train_model(dataset_path, weights_path, log_path, config: TrainConfig, bool_tensorboard=False):
    keras.backend.clear_session()
    model = create_vit_classifier(number_of_transformer_blocks=config.n_transformer_blocks)
    model.summary()

    train_data = CellDataset(dataset_path + "train/", num_images=3253, batch_size=config.batch_size)
    validation_data = CellDataset(dataset_path + "train/", num_images=570, batch_size=config.batch_size)

    # compile model
    model.compile(optimizer="rmsprop", loss="mean_squared_error")
    # model.compile(optimizer=keras.optimizers.Adam(learning_rate=config.learning_rate),
                #   loss=keras.losses.BinaryCrossentropy(from_logits=True))

    callbacks = [
        keras.callbacks.ModelCheckpoint(weights_path, save_best_only=True),
    ]

    if bool_tensorboard:
        # tensorboard
        log_dir = log_path + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        callbacks.append(tensorboard_callback)

    model.fit(train_data, epochs=config.n_epochs, validation_data=validation_data, callbacks=callbacks)
    
    return model

if __name__ == '__main__':
    # configure training
    config = TrainConfig(batch_size=4,
                             learning_rate=0.0001,
                             n_epochs=50,
                             n_transformer_blocks=1)
    
    # load data
    dataset_path = "/nethome/mlamsey3/Documents/Coursework/cs7643-project/dataset/"
    weights_path = "/nethome/mlamsey3/Documents/Coursework/cs7643-project/weights/"
    log_path = "/nethome/mlamsey3/Documents/Coursework/cs7643-project/logs/fit/"

    config_string = f"batch_size={config.batch_size}_lr={config.learning_rate}_epochs={config.n_epochs}_transformer_blocks={config.n_transformer_blocks}"
    weights_path = weights_path + config_string + "/"

    train_data = CellDataset(dataset_path + "train/", 3253, config.batch_size)
    validation_data = CellDataset(dataset_path + "train/", 570, config.batch_size)

    if not os.path.exists(weights_path):
        os.makedirs(weights_path)

    # toggle whether to train
    # if False:        
    #     model = train_model(dataset_path, weights_path, log_path, config, bool_tensorboard=True)
    model = keras.models.load_model(weights_path)
    validation_preds = model.predict(validation_data)

    input, actual = validation_data[0]
    plt.imshow(actual[2]*255, cmap='gray', vmin=0, vmax=255)
    plt.title("Actual")
    plt.show()

    temp = validation_preds[2]
    temp[temp >= 0.5] = 1
    temp[temp < 0.5] = 0
    example = (temp*255).astype('uint8')
    plt.imshow(example, cmap='gray', vmin=0, vmax=255)
    plt.title("Predicted")
    plt.show()

    plt.imshow(actual[3]*255, cmap='gray', vmin=0, vmax=255)
    plt.title("Actual")
    plt.show()

    temp = validation_preds[3]
    temp[temp >= 0.5] = 1
    temp[temp < 0.5] = 0
    example = (temp*255).astype('uint8')
    plt.imshow(example, cmap='gray', vmin=0, vmax=255)
    plt.title("Predicted")
    plt.show()