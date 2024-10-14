import numpy as np
import cv2
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

# Paths
model_path = '/Users/krishilparikh/Desktop/Proj/handTextpredictor/handwriting_model_cnn_lstm.h5'
label_encoder_path = '/Users/krishilparikh/Desktop/Proj/handTextpredictor/label_encoder.npy'

# Load the model and label encoder
def load_model_and_encoder():
    model = tf.keras.models.load_model(model_path)
    label_classes = np.load(label_encoder_path, allow_pickle=True)
    label_encoder = LabelEncoder()
    label_encoder.classes_ = label_classes
    return model, label_encoder

# Preprocess an input image
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (64, 64))
    img = img.astype('float32') / 255.0
    img = img.reshape(1, 64, 64, 1)  # Reshape to match model input
    return img

# Function to predict the handwritten word from an input image
def predict_word(image_path):
    model, label_encoder = load_model_and_encoder()
    img = preprocess_image(image_path)
    print(f"Processed image shape (after reshaping): {img.shape}")

    predictions = model.predict(img)
    print(f"Raw predictions: {predictions}")

    predicted_label = np.argmax(predictions, axis=1)[0]
    print(f"Predicted label (index): {predicted_label}")

    if predicted_label >= len(label_encoder.classes_):
        print("Error: Predicted label index out of bounds.")
        return "Unknown"

    predicted_word = label_encoder.inverse_transform([predicted_label])[0]
    return predicted_word

def predict_words(image_paths):
    model, label_encoder = load_model_and_encoder()
    predicted_words = []
    
    for image_path in image_paths:
        img = preprocess_image(image_path)
        print(f"Processed image shape (after reshaping): {img.shape}")

        predictions = model.predict(img)
        print(f"Raw predictions for {image_path}: {predictions}")

        predicted_label = np.argmax(predictions, axis=1)[0]
        print(f"Predicted label (index) for {image_path}: {predicted_label}")

        if predicted_label >= len(label_encoder.classes_):
            print("Error: Predicted label index out of bounds.")
            predicted_words.append("Unknown")
            continue

        predicted_word = label_encoder.inverse_transform([predicted_label])[0]
        predicted_words.append(predicted_word)

    return predicted_words

image_paths = [
    '/Users/krishilparikh/Desktop/Proj/handTextpredictor/Screenshot 2024-10-14 at 10.22.58 PM.png'
]
predicted_words = predict_words(image_paths)
print('Predicted words for batch:', predicted_words)
