import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
MODEL_DIR   = os.path.join(PROJECT_DIR, "models")
INPUT_DIR   = os.path.join(PROJECT_DIR, "input_images")
os.makedirs(MODEL_DIR, exist_ok=True)


IMG_HEIGHT   = 32
IMG_WIDTH    = 128
IMG_CHANNELS = 1

CHARSET = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ" \
          r"+-*/=()[]{}^_\frac\int\sum\sqrt. "
NUM_CLASSES = len(CHARSET) + 1  

EPOCHS        = 50
BATCH_SIZE    = 2
VAL_SPLIT     = 0.2
MAX_LABEL_LEN = 32

print(f"Character set size: {len(CHARSET)}")
print(f"Number of classes (with CTC blank): {NUM_CLASSES}")

GROUND_TRUTH = {
    "img1":  r"x̄ = (x_1 + x_2 + ... + x_n) / n = Σ(i=1 to n) x_i / n",
    "img2":  r"μ = Σ(i=1 to N) x_i * f(x_i) = Σ(i=1 to N) x_i / N",
    "img3":  r"s^2 = Σ(i=1 to n)(x_i - x̄)^2 / (n - 1)",
    "img4":  r"Σ(i=1 to 8)(x_i - x̄)^2 = 1.60",
    "img5":  r"s = √0.2286 = 0.48 pounds",
    "img6":  r"s^2 = Σ(i=1 to n)(x_i - x̄)^2 / (n - 1)",
    "img7":  r"Σ(i=1 to n)(x_i^2 + x̄^2 - 2*x̄*x_i) / (n - 1)",
    "img8":  r"o^2 = Σ(i=1 to N)(x_i - μ)^2 / N",
    "img9":  r"n < 8 or 10",
    "img10": r"r_xy = Σ(i=1 to n) y_i(x_i - x̄) / [ Σ(i=1 to n)(y_i - ȳ)^2 * Σ(i=1 to n)(x_i - x̄)^2 ]^(1/2",
    "img11": r"(j - 0.5) / n = P(Z ≤ z_j) = Φ",
    "img12": r"(j - 0.5) / n = 0.05, Φ(z_j)",
    "img13": r"Σ(i=1 to n) (x_i - a)^2",
    "img14": r"y_i = a + b*x_i",
    "img15": r"(j - 0.5) / 10",
}


def build_crnn(img_height, img_width, img_channels, num_classes):
    inputs = keras.Input(shape=(img_height, img_width, img_channels), name="image_input")

    x = keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu")(inputs)
    x = keras.layers.MaxPooling2D((2, 2))(x)

    x = keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)

    x = keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = keras.layers.MaxPooling2D((2, 1))(x)

    new_shape = (img_width // 4, (img_height // 8) * 128)
    x = keras.layers.Reshape(target_shape=new_shape)(x)

    x = keras.layers.Dense(256, activation="relu")(x)
    x = keras.layers.Dropout(0.25)(x)
    x = keras.layers.Dense(128, activation="relu")(x)

    outputs = keras.layers.Dense(num_classes, activation="softmax", name="output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="CRNN_OCR")
    return model


def encode_label(text, charset):
    return [charset.index(c) for c in text if c in charset]


def load_dataset(input_dir, ground_truth, charset, img_height, img_width):
    images = []
    labels = []

    for key, latex in ground_truth.items():
        path = os.path.join(input_dir, f"{key}.jpg")
        if not os.path.exists(path):
            print(f"[WARNING] {path} not found — skipping.")
            continue

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[WARNING] Cannot read {path} — skipping.")
            continue

        img = cv2.fastNlMeansDenoising(img, h=30)
        img = cv2.adaptiveThreshold(
            img, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        img = cv2.resize(img, (img_width, img_height))
        img = img.astype(np.float32) / 255.0
        img = img.reshape(img_height, img_width, 1)

        encoded = encode_label(latex, charset)
        if len(encoded) == 0:
            print(f"[WARNING] Label for {key} has no encodable characters — skipping.")
            continue

        images.append(img)
        labels.append(encoded)
        print(f"  Loaded: {key}.jpg  →  {latex[:40]}...")

    return np.array(images), labels


def augment_image(img):
    variants = [img]

    blurred = cv2.GaussianBlur(img.reshape(IMG_HEIGHT, IMG_WIDTH), (3, 3), 0)
    variants.append(blurred.reshape(IMG_HEIGHT, IMG_WIDTH, 1).astype(np.float32) / 255.0)

    noisy = (img * 255).astype(np.uint8).reshape(IMG_HEIGHT, IMG_WIDTH)
    h, w = noisy.shape
    num_salt = int(0.01 * h * w)
    sy, sx = np.random.randint(0, h, num_salt), np.random.randint(0, w, num_salt)
    noisy[sy, sx] = 255
    py, px = np.random.randint(0, h, num_salt), np.random.randint(0, w, num_salt)
    noisy[py, px] = 0
    variants.append(noisy.reshape(IMG_HEIGHT, IMG_WIDTH, 1).astype(np.float32) / 255.0)

    low = cv2.convertScaleAbs((img * 255).astype(np.uint8), alpha=0.6, beta=0)
    variants.append(low.reshape(IMG_HEIGHT, IMG_WIDTH, 1).astype(np.float32) / 255.0)

    return variants


def ctc_loss_fn(y_true, y_pred):
    batch_size  = tf.shape(y_pred)[0]
    time_steps  = tf.shape(y_pred)[1]

    input_length = tf.fill([batch_size, 1], time_steps)
    input_length = tf.cast(input_length, tf.int32)

    label_length = tf.reduce_sum(
        tf.cast(tf.not_equal(y_true, -1), tf.int32),
        axis=1, keepdims=True
    )

    y_true_clean = tf.maximum(y_true, 0)

    loss = tf.keras.backend.ctc_batch_cost(
        y_true_clean,
        y_pred,
        input_length,
        label_length
    )
    return loss


print("\nBuilding CRNN model...")
model = build_crnn(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, NUM_CLASSES)
model.summary()


print("\nLoading dataset from input_images/...")
images, labels = load_dataset(INPUT_DIR, GROUND_TRUTH, CHARSET, IMG_HEIGHT, IMG_WIDTH)

if len(images) == 0:
    print("[ERROR] No images loaded. Add equation images to input_images/ and update GROUND_TRUTH.")
    exit(1)

print(f"\nLoaded {len(images)} image(s). Applying augmentation...")

aug_images = []
aug_labels = []
for img, lbl in zip(images, labels):
    variants = augment_image(img)
    for v in variants:
        aug_images.append(v)
        aug_labels.append(lbl)

aug_images = np.array(aug_images)
print(f"Dataset size after augmentation: {len(aug_images)} samples")

padded_labels = np.full((len(aug_labels), MAX_LABEL_LEN), -1, dtype=np.int32)
for i, lbl in enumerate(aug_labels):
    length = min(len(lbl), MAX_LABEL_LEN)
    padded_labels[i, :length] = lbl[:length]

split = max(1, int(len(aug_images) * (1 - VAL_SPLIT)))
X_train, X_val = aug_images[:split], aug_images[split:]
y_train, y_val = padded_labels[:split], padded_labels[split:]

print(f"Training samples  : {len(X_train)}")
print(f"Validation samples: {len(X_val)}")


print("\nCompiling model...")
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=ctc_loss_fn
)

callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(MODEL_DIR, "best_weights.keras"),
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        verbose=1,
        min_lr=1e-6
    ),
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
]

print(f"\nTraining for up to {EPOCHS} epochs (early stopping enabled)...\n")
history = model.fit(
    X_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_val, y_val) if len(X_val) > 0 else None,
    callbacks=callbacks,
    verbose=1
)

print("\nTraining complete!")
print(f"  Final train loss : {history.history['loss'][-1]:.4f}")
if "val_loss" in history.history:
    print(f"  Final val loss   : {history.history['val_loss'][-1]:.4f}")


KERAS_MODEL_PATH = os.path.join(MODEL_DIR, "equation_model.keras")
print(f"\nSaving trained Keras model to: {KERAS_MODEL_PATH}")
model.save(KERAS_MODEL_PATH)

print("\nConverting to TFLite...")

def representative_dataset():
    for img in aug_images[:min(100, len(aug_images))]:
        sample = img.reshape(1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS).astype(np.float32)
        yield [sample]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
]

tflite_model = converter.convert()

TFLITE_PATH = os.path.join(MODEL_DIR, "equation_model.tflite")
with open(TFLITE_PATH, "wb") as f:
    f.write(tflite_model)

size_kb = os.path.getsize(TFLITE_PATH) / 1024
print(f"\nTFLite model saved to : {TFLITE_PATH}")
print(f"Model size            : {size_kb:.1f} KB")
print("\nConversion complete!")
print("Transfer models/equation_model.tflite to your Raspberry Pi.")

CHARSET_PATH = os.path.join(MODEL_DIR, "charset.txt")
with open(CHARSET_PATH, "w") as f:
    f.write(CHARSET)
print(f"Charset saved to      : {CHARSET_PATH}")
print("Also transfer charset.txt to your Raspberry Pi.")
