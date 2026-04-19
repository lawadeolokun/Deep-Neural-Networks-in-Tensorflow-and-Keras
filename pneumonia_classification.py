from __future__ import print_function

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np

# =========================
# SETTINGS
# =========================
batch_size = 16
epochs = 10
fine_tune_epochs = 5

img_width = 160
img_height = 160
img_channels = 3

fit = True

train_dir = '/Users/lawadeolokun/Downloads/chest_xray/train'
test_dir = '/Users/lawadeolokun/Downloads/chest_xray/test'

with tf.device('/cpu:0'):

    # =========================
    # DATASETS
    # =========================
    train_ds, val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        seed=123,
        validation_split=0.2,
        subset='both',
        image_size=(img_height, img_width),
        batch_size=batch_size,
        shuffle=True
    )

    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_dir,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        shuffle=True
    )

    class_names = train_ds.class_names
    print("Class Names:", class_names)
    num_classes = len(class_names)

    # =========================
    # DATA AUGMENTATION
    # =========================
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.05),
    ])

    # =========================
    # PRETRAINED MODEL
    # =========================
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(img_height, img_width, img_channels),
        include_top=False,
        weights='imagenet'
    )

    base_model.trainable = False  # freeze first

    # =========================
    # MODEL
    # =========================
    inputs = keras.Input(shape=(img_height, img_width, img_channels))

    x = layers.Rescaling(1./255)(inputs)
    x = data_augmentation(x)

    x = base_model(x, training=False)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.0003),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    # =========================
    # CALLBACKS
    # =========================
    earlystop_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )

    save_callback = tf.keras.callbacks.ModelCheckpoint(
        "pneumonia_transfer.keras",
        save_best_only=True
    )

    # =========================
    # CLASS WEIGHTS
    # =========================
    class_weight = {
        0: 1.0,   # BACTERIAL
        1: 1.5,   # NORMAL
        2: 1.5    # VIRAL
    }

    # =========================
    # TRAIN (STAGE 1)
    # =========================
    if fit:
        print("\n=== Stage 1: Training classifier head ===\n")

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=[earlystop_callback, save_callback],
            class_weight=class_weight
        )

        # =========================
        # 🔥 FINE-TUNING (STAGE 2)
        # =========================
        print("\n=== Stage 2: Fine-tuning ===\n")

        base_model.trainable = True

        # Freeze early layers, train top layers
        for layer in base_model.layers[:-30]:
            layer.trainable = False

        model.compile(
            optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=1e-5),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        history_fine = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=fine_tune_epochs,
            callbacks=[earlystop_callback],
            class_weight=class_weight
        )

    else:
        model = tf.keras.models.load_model("pneumonia_transfer.keras")

    # =========================
    # EVALUATE
    # =========================
    score = model.evaluate(test_ds)
    print("Test accuracy:", score[1])

    # =========================
    # PLOT ACCURACY
    # =========================
    if fit:
        plt.figure()
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(['Train', 'Validation'])
        plt.show()

    # =========================
    # PREDICTIONS
    # =========================
    test_batch = test_ds.take(1)
    plt.figure(figsize=(10, 10))

    for images, labels in test_batch:
        for i in range(6):
            ax = plt.subplot(2, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))

            prediction = model.predict(tf.expand_dims(images[i], 0), verbose=0)
            predicted_class = class_names[np.argmax(prediction)]
            confidence = 100 * np.max(prediction)

            plt.title(
                f"Actual: {class_names[labels[i]]}\nPredicted: {predicted_class} {confidence:.2f}%"
            )
            plt.axis("off")

    plt.show()