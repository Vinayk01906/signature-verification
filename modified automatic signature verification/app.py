from flask import Flask, request, jsonify, render_template
import os
import numpy as np
import tensorflow as tf
import cv2
from werkzeug.utils import secure_filename
from sklearn.model_selection import train_test_split

app = Flask(__name__)
MODEL_PATH = "signature_model.h5"
DATASET_PATH = r"C:\Users\kumar\Downloads\archive (1)\Dataset_Signature_Final\Dataset\dataset4"

def load_data(data_dir, img_size=(150, 150)):
    images, labels = [], []
    for label, class_name in enumerate(["genuine", "forged"]):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.exists(class_dir):
            continue
        for img_file in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, img_size)
            img = img / 255.0
            img = np.repeat(img[..., np.newaxis], 3, axis=-1)
            images.append(img)
            labels.append(label)
    return np.array(images), np.array(labels)

def train_and_save_model():
    X, y = load_data(DATASET_PATH)
    if X.size == 0:
        return "No images found in dataset. Check dataset path."
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
    base_model.trainable = False
    
    input_layer = tf.keras.layers.Input(shape=(150, 150, 3))
    x = base_model(input_layer, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.models.Model(inputs=input_layer, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))
    model.save(MODEL_PATH)
    return "Model trained and saved successfully."

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/train", methods=["POST"])
def train_model():
    message = train_and_save_model()
    return jsonify({"message": message})

@app.route("/verify", methods=["POST"])
def verify_signature():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        genuine_file = request.files["genuine"]
        forged_file = request.files["forged"]
        
        genuine_img = cv2.imdecode(np.frombuffer(genuine_file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
        forged_img = cv2.imdecode(np.frombuffer(forged_file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
        
        if genuine_img is None or forged_img is None:
            return jsonify({"error": "Invalid image files."}), 400
        
        genuine_img = cv2.resize(genuine_img, (150, 150)) / 255.0
        forged_img = cv2.resize(forged_img, (150, 150)) / 255.0
        
        genuine_img = np.repeat(genuine_img[..., np.newaxis], 3, axis=-1)
        forged_img = np.repeat(forged_img[..., np.newaxis], 3, axis=-1)
        
        prediction = model.predict(np.array([genuine_img, forged_img]))[0][0]
        result = "Genuine" if prediction > 0.5 else "Forged"
        
        return jsonify({"score": float(prediction), "result": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)