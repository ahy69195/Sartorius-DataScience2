import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import os
import glob as gb
import matplotlib.pyplot as plt
import cv2
from PIL import Image

from skimage.io import imread, imshow
from skimage.transform import resize

def main():
    train_path = './data/images'
    masks_path = './data/masks'

    IMG_WIDTH = 256
    IMG_HEIGHT = 256
    IMG_CHANNELS = 3

    X = next(os.walk(train_path))[2]
    y = next(os.walk(masks_path))[2]

    X_ids = X[:-10]
    y_ids = y[:-10]

    print(y_ids)

    X_train = np.zeros((len(X_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.float32)
    y_train = np.zeros((len(y_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)

    for n, id in enumerate(X_ids):
        # Load image with keras
        image = tf.keras.preprocessing.image.load_img(train_path + '/' + id, target_size=(IMG_HEIGHT, IMG_WIDTH))
        # convert the image to array with keras and slice to crop
        input_arr = tf.keras.preprocessing.image.img_to_array(image)[90:450, 150:406]
        image = tf.keras.preprocessing.image.array_to_img(input_arr, ).resize((256, 256))
        X_train[n] = np.array(image)

    for n, id_ in enumerate(y_ids):
        # Load image with keras
        image = tf.keras.preprocessing.image.load_img(masks_path + '/' + id_, target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                      color_mode='grayscale')
        input_arr = tf.keras.preprocessing.image.img_to_array(image)[90:450, 150:406]
        image = tf.keras.preprocessing.image.array_to_img(input_arr, ).resize((256, 256))
        y_train[n] = np.array(image)[:, :, np.newaxis]

    # show the data with matplotlib after preprocessing
    plt.imshow(tf.keras.preprocessing.image.array_to_img(X_train[510]))
    plt.show()
    plt.imshow(tf.keras.preprocessing.image.array_to_img(y_train[510]))
    plt.show()

    # create the model
    inputs = tf.keras.layers.Input((256, 256, 3))
    s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)

    # Contraction path
    c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
    c1 = tf.keras.layers.Dropout(0.1)(c1)
    c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

    c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = tf.keras.layers.Dropout(0.1)(c2)
    c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

    c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = tf.keras.layers.Dropout(0.2)(c3)
    c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)

    c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = tf.keras.layers.Dropout(0.2)(c4)
    c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = tf.keras.layers.Dropout(0.3)(c5)
    c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

    # Expansive path
    u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = tf.keras.layers.concatenate([u6, c4])
    c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = tf.keras.layers.Dropout(0.2)(c6)
    c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

    u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = tf.keras.layers.concatenate([u7, c3])
    c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = tf.keras.layers.Dropout(0.2)(c7)
    c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

    u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = tf.keras.layers.concatenate([u8, c2])
    c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = tf.keras.layers.Dropout(0.1)(c8)
    c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

    u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
    c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = tf.keras.layers.Dropout(0.1)(c9)
    c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    results = model.fit(X_train, y_train, validation_split=0.1, batch_size=16, epochs=10)


    plt.plot(model.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    import random

    test_id = random.choice(X_ids[-10:])
    print(test_id)
    img = tf.keras.preprocessing.image.load_img(f'./data/images/{test_id}', target_size=(IMG_HEIGHT, IMG_WIDTH))
    input_array = tf.keras.preprocessing.image.img_to_array(img)
    input_array = np.array([input_array])
    predictions = model.predict(input_array)

    Image.open(f'./data/images/{test_id}').resize((256, 256))
    plt.imshow(tf.keras.preprocessing.image.array_to_img(np.squeeze(predictions)[:, :, np.newaxis]))
    Image.open(f'./data/masks/{test_id}').resize((256, 256))



if __name__ == '__main__':
    main()

