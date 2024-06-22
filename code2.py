# import os
# import numpy as np
# import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt


model = load_model('forest_fire_detection_model.h5')

# Directory containing testing images
test_images_dir = '/content/drive/MyDrive/Forest Fire Dataset/Testing'


fire_images = []


for image_name in os.listdir(test_images_dir):

    image_path = os.path.join(test_images_dir, image_name)


    img = image.load_img(image_path, target_size=(128, 128))#preprocess the image
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize pixel values

    # Perform inference using the model
    prediction = model.predict(img_array)

    # Assuming binary classification: 1 for fire, 0 for no fire
    if prediction > 0.5:

        fire_images.append((image_path, img))

# Print paths of images containing forest fire
print("Forest Fire Images:")
# for image_path in fire_images:
#     print(image_path)


# Display fire images along with their paths
for image_path, img in fire_images:
    plt.figure()
    plt.imshow(img)
    plt.title(image_path)
    plt.axis('off')
    plt.show()
