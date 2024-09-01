from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Define the path to the saved model
saved_model_path = r'C:\Users\hp\Desktop\cotton_model\modelx.h5'

# Load the saved model
model = load_model(saved_model_path)

# Define the path to your test image(s)
test_image_path =r'C:\Users\hp\Desktop\cotton\cotton\curl_virus\curl00.jpg'  # Replace with your actual test image path

# Load and preprocess the test image
img = image.load_img(test_image_path, target_size=(224, 224))  # Ensure target_size matches your model's expected input size
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = x / 255.0  # Rescale pixel values to [0, 1], assuming you rescaled during training

# Make predictions
predictions = model.predict(x)

# Assuming you have a list of class labels corresponding to your model's output
class_labels = ['bacterial_blight', 'curl_virus', 'fussarium_wilt', 'healthy']  # Replace with your actual class labels

# Get the predicted class label
predicted_class = class_labels[np.argmax(predictions)]

# Print the predicted class
print(f"Predicted class: {predicted_class}")