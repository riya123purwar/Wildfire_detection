import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the trained model
model = load_model('forest_fire_detection_model.h5')


def predict_fire(image_path):
    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize pixel values

    # Perform inference using the model
    prediction = model.predict(img_array)

    if prediction > 0.5:
        return "The image does not contains fire."
    else:
        return "The image  contain fire."

# Path to the input image you want to test
input_image_path = '/content/fire_0014.jpg'

# Test the model with the input image
result = predict_fire(input_image_path)
print(result)

# Display the number of fire and no fire images
print("Number of Fire Images:", len(fire_images))
print("Number of No Fire Images:", len(no_fire_images))

from google.colab import drive
drive.mount('/content/drive')
