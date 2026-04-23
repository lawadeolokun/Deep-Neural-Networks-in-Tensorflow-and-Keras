from __future__ import print_function

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
import cv2
from collections import Counter
from sklearn.metrics import classification_report

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

# =========================
# FINAL STABLE GRAD-CAM
# =========================
def make_gradcam_heatmap(img_array, model):

    base_model = model.get_layer("mobilenetv2_1.00_160")

    # Extract classifier layers safely
    gap_layer = None
    bn_layer = None
    dense_layer = None
    output_layer = None

    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.GlobalAveragePooling2D):
            gap_layer = layer
        elif isinstance(layer, tf.keras.layers.BatchNormalization):
            bn_layer = layer
        elif isinstance(layer, tf.keras.layers.Dense) and layer.units == 128:
            dense_layer = layer
        elif isinstance(layer, tf.keras.layers.Dense) and layer.units == 3:
            output_layer = layer

    # Manual preprocessing (same as training)
    x = img_array / 255.0

    with tf.GradientTape() as tape:

        # Forward pass through base model
        conv_outputs = base_model(x, training=False)
        tape.watch(conv_outputs)

        # Forward pass through classifier
        x2 = gap_layer(conv_outputs)
        x2 = bn_layer(x2)
        x2 = dense_layer(x2)
        predictions = output_layer(x2)

        class_index = tf.argmax(predictions[0])
        loss = predictions[:, class_index]

    # Compute gradients
    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    heatmap = tf.maximum(heatmap, 0)
    heatmap /= (tf.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy()


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

    base_model.trainable = False

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
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
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

    # =========================
    # CLASS WEIGHTS
    # =========================
    class_weight = {
        0: 1.0,
        1: 1.5,
        2: 1.5
    }

    # =========================
    # TRAIN
    # =========================
    if fit:
        print("\n=== Stage 1 ===\n")

        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=[earlystop_callback],
            class_weight=class_weight
        )

        print("\n=== Stage 2 ===\n")

        base_model.trainable = True

        for layer in base_model.layers[:-30]:
            layer.trainable = False

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=fine_tune_epochs,
            callbacks=[earlystop_callback],
            class_weight=class_weight
        )

    # =========================
    # EVALUATE
    # =========================
    score = model.evaluate(test_ds)
    print("Test accuracy:", score[1])

    # =========================
    # CLASS DISTRIBUTION
    # =========================
    labels_list = []
    for _, y in train_ds.unbatch():
        labels_list.append(int(y))

    print("Class distribution:", Counter(labels_list))

    # =========================
    # CLASSIFICATION REPORT
    # =========================
    y_true = []
    y_pred = []

    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(preds, axis=1))

    print(classification_report(y_true, y_pred, target_names=class_names))

    # =========================
    # GRAD-CAM VISUALIZATION
    # =========================
    test_batch = test_ds.take(1)
    plt.figure(figsize=(10, 10))

    for images, labels in test_batch:
        for i in range(6):

            img = images[i].numpy().astype("uint8")

            img_array = tf.expand_dims(images[i], axis=0)

            heatmap = make_gradcam_heatmap(img_array, model)

            heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

            superimposed_img = heatmap * 0.4 + img

            prediction = model.predict(img_array, verbose=0)
            predicted_class = class_names[np.argmax(prediction)]
            confidence = 100 * np.max(prediction)

            plt.subplot(2, 3, i + 1)
            plt.imshow(superimposed_img.astype("uint8"))
            plt.title(
                f"Actual: {class_names[labels[i]]}\nPredicted: {predicted_class} {confidence:.2f}%"
            )
            plt.axis("off")

    plt.tight_layout()
    plt.show()