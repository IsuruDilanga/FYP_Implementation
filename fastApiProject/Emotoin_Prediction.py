from uuid import uuid4
import os
import numpy as np
import cv2
from keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from PIL import Image

from keras.applications.vgg19 import preprocess_input
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.preprocessing import image
from PIL import Image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
import cv2
import time
import numpy as np
import time
from lime.lime_image import LimeImageExplainer

import cv2
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model, load_model

import cv2
import numpy as np
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from tensorflow.keras.models import Model, load_model
from lime.lime_image import LimeImageExplainer
from PIL import Image
from tensorflow.keras.layers import GlobalAveragePooling2D, Reshape, Dense

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input

# Load the saved model
model = load_model('/Users/isurudissanayake/Documents/Data/DATA_SET/Feature-Extraction/VGG16/VGG16Model.h5')

# Load the image you want to classify
img_path = '/path/to/your/input/image.jpg'  # Replace with the actual path to the image
img = image.load_img(img_path, target_size=(224, 224))  # Resize to match model input shape

# Preprocess the image
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)

# Make prediction using the loaded model
prediction = model.predict(img_array)

# Get prediction value and interpret as ASD or not ASD
prediction_value = prediction[0][0]  # Extract the first value from the prediction array
if prediction_value > 0.5:
  prediction_class = 'ASD'
else:
  prediction_class = 'Not ASD'

print("Prediction value:", prediction_value)
print("Predicted class:", prediction_class)