import cv2
import numpy as np

# model loading
import tensorflow as tf
from tensorflow.keras.models import load_model
model = load_model('/content/drive/MyDrive/best_model.keras')

def preprocess_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0  # Rescale
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Example prediction
image_path = #image path
processed_image = preprocess_image(image_path)
prediction = model.predict(processed_image)

if prediction[0][0] > 0.5:
    print("Fake")
else:
    print("Real")
