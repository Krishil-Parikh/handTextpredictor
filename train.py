import pandas as pd
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import models, layers

csv_file_path = '/Users/krishilparikh/Desktop/Proj/handTextpredictor/lab.csv'
images_dir = '/Users/krishilparikh/Desktop/Proj/handTextpredictor/database'
model_path = '/Users/krishilparikh/Desktop/Proj/handTextpredictor/handwriting_model_cnn_lstm.h5'
label_encoder_path = '/Users/krishilparikh/Desktop/Proj/handTextpredictor/label_encoder.npy'
df = pd.read_csv(csv_file_path)
print(df.head())
def load_images(image_ids, image_dir):
    images = []
    labels = []
    for index, row in df.iterrows():
        image_id = row['word_id']
        img_path = os.path.join(image_dir, f"{image_id}.png")
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (64, 64))
            images.append(img)
            labels.append(row['grammatical_tag'])
    return np.array(images), np.array(labels)

X, y = load_images(df['word_id'].values, images_dir)
X = X.astype('float32') / 255.0
X = X.reshape(-1, 64, 64, 1)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
np.save(label_encoder_path, label_encoder.classes_)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

print(f"Training data shape: {X_train.shape}, Training labels shape: {y_train.shape}")
print(f"Testing data shape: {X_test.shape}, Testing labels shape: {y_test.shape}")

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Reshape((36, 128)),
    layers.LSTM(128, return_sequences=False),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(np.unique(y_encoded)), activation='softmax')  # Output layer for classification
])


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100 , batch_size=32)
model.save(model_path)
print("Model saved!")
