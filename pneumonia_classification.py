from __future__ import print_function

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
import cv2
from sklearn.metrics import classification_report


batch_size = 16
epochs = 10
fine_tune_epochs = 5

img_width = 160
img_height = 160
img_channels = 3

fit = False #make fit false if you do not want to train the network again
train_dir = '/Users/lawadeolokun/Downloads/chest_xray/train'
test_dir = '/Users/lawadeolokun/Downloads/chest_xray/test'

# GRAD-CAM Function
def make_gradcam_heatmap(img_array, model):

    base_model = model.get_layer("mobilenetv2_1.00_160")

    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.GlobalAveragePooling2D):
            gap_layer = layer
        elif isinstance(layer, tf.keras.layers.BatchNormalization):
            bn_layer = layer
        elif isinstance(layer, tf.keras.layers.Dense) and layer.units == 128:
            dense_layer = layer
        elif isinstance(layer, tf.keras.layers.Dense) and layer.units == 3:
            output_layer = layer

    x = img_array

    with tf.GradientTape() as tape:
        conv_outputs = base_model(x, training=False)
        tape.watch(conv_outputs)

        x2 = gap_layer(conv_outputs)
        x2 = bn_layer(x2)
        x2 = dense_layer(x2)
        predictions = output_layer(x2)

        class_index = tf.argmax(predictions[0])
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap /= (tf.reduce_max(heatmap) + 1e-8)

    heatmap = heatmap.numpy()
    heatmap = np.where(heatmap > 0.6, heatmap, 0)

    return heatmap

with tf.device('/cpu:0'):

   #create training,validation and test datatsets
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

    # Data Augmentation
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.05),
    ])

    # Model
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(img_height, img_width, img_channels),
        include_top=False,
        weights='imagenet'
    )

    base_model.trainable = False

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

    #  Loads model when not training
    if not fit:
        print("Loading saved model...")
        model = tf.keras.models.load_model("pneumonia.keras")

    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    # Callback for best model
    save_callback = tf.keras.callbacks.ModelCheckpoint(
        "pneumonia.keras",
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )

    # TRAIN
    if fit:
        print("\n Stage 1 \n")

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=[save_callback]
        )

        print("\n Stage 2 (Fine-tuning) \n")

        base_model.trainable = True

        for layer in base_model.layers[:-30]:
            layer.trainable = False

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=fine_tune_epochs,
            callbacks=[save_callback]
        )

    # Evaluate
    score = model.evaluate(test_ds)
    print("Test accuracy:", score[1])

    # Plot Training
    if fit:
        plt.figure(figsize=(12,5))

        plt.subplot(1,2,1)
        plt.plot(history.history['accuracy'], label='Train')
        plt.plot(history.history['val_accuracy'], label='Validation')
        plt.title('Accuracy')
        plt.legend()


        plt.tight_layout()
        plt.show()

    # Classification Report
    y_true = []
    y_pred = []

    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(preds, axis=1))

    print(classification_report(y_true, y_pred, target_names=class_names))

    # GRAD-CAM
    test_batch = test_ds.take(1)
    plt.figure(figsize=(10, 10))

    for images, labels in test_batch:
        for i in range(6):

            img = images[i].numpy().astype("uint8")
            img_array = tf.expand_dims(images[i], axis=0)

            heatmap = make_gradcam_heatmap(img_array, model)

            heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
            heatmap = cv2.GaussianBlur(heatmap, (11, 11), 0)

            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

            superimposed_img = cv2.addWeighted(img, 0.7, heatmap, 0.3, 0)

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