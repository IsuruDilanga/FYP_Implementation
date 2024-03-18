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

import cv2
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.models import load_model

def predict_asd_vgg16(image_path):

    model_path = '/Users/isurudissanayake/Documents/Data/DATA_SET/Feature-Extraction/VGG16/VGG16Model.h5'
    target_size = (224, 224)

    base_model = VGG16(weights='imagenet', include_top=True)
    x = base_model.get_layer('block5_pool').output

    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    img = np.array(Image.open(image_path).resize(target_size))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR
    img = np.expand_dims(img, axis=0)  # Add a batch dimension
    img = preprocess_input(img)  # Preprocess the image

    img_scaled = img / 255.0

    prediction = model.predict(img)[0][0]  # Access the first element for ASD probability

    rounded_prediction = round(prediction, 2)
    print(f"Predicted probability: {rounded_prediction:.2f}")

    if rounded_prediction > 0.5:
        print(f"Predicted ASD with probability: {rounded_prediction:.2f}")

        return rounded_prediction
    else:
        print(f"Predicted non-ASD with probability: {1 - rounded_prediction:.2f}")
        return rounded_prediction


def predict_asd_vgg19(image_path):

    model_path = '/Users/isurudissanayake/Documents/Data/DATA_SET/Feature-Extraction/VGG19/VGG19Model.h5'
    target_size = (224, 224)

    base_model = VGG19(weights='imagenet', include_top=True)

    x = base_model.get_layer('block5_pool').output

    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    img = cv2.imread(image_path)
    img = cv2.resize(img, target_size)  # Resize the image to (224, 224)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert BGR to RGB
    img = np.expand_dims(img, axis=0)  # Add a batch dimension
    img = preprocess_input(img)  # Preprocess the image

    img_scaled = img / 255.0

    prediction = model.predict(img)[0][0]  # Access the first element for ASD probability

    rounded_prediction = round(prediction, 2)
    print(f"Predicted probability: {rounded_prediction:.2f}")

    if rounded_prediction > 0.5:
        print(f"Predicted ASD with probability: {rounded_prediction:.2f}")

        return rounded_prediction
    else:
        print(f"Predicted non-ASD with probability: {1 - rounded_prediction:.2f}")
        return rounded_prediction


def predict_asd_ResNet50(image_path):
    model_path = '/Users/isurudissanayake/Documents/Data/DATA_SET/Feature-Extraction/ResNet50/ResNet50Model.h5'
    target_size = (224, 224)

    base_model = ResNet50(weights='imagenet', include_top=True)
    x = base_model.get_layer('avg_pool').output

    x = Dense(255, activation='relu')(x)
    x = Reshape((1, 1, 255))(x)  # Add this line to reshape the output of GlobalAveragePooling2D
    x = GlobalAveragePooling2D()(x)
    prediction = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=prediction)

    img = cv2.imread(image_path)
    img = cv2.resize(img, target_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    img_scaled = img / 255

    prediction = model.predict(img)[0][0]  # Access the first element for ASD probability
    print("prediction: {:.5f}".format(prediction))

    rounded_prediction = round(prediction, 2)
    print(f"Predicted probability: {rounded_prediction:.2f}")

    if rounded_prediction > 0.5:
        print(f"Predicted ASD with probability: {rounded_prediction:.2f}")

        return rounded_prediction
    else:
        print(f"Predicted non-ASD with probability: {1 - rounded_prediction:.2f}")
        return rounded_prediction


def predict_asd_ResNet50V2(image_path):
    model_path = '/Users/isurudissanayake/Documents/Data/DATA_SET/Feature-Extraction/ResNet50V2/ResNet50V2Model.h5'
    target_size = (224, 224)

    base_model = ResNet50V2(weights='imagenet', include_top=True)

    x = base_model.get_layer('avg_pool').output

    x = Dense(255, activation='relu')(x)
    x = Reshape((1, 1, 255))(x)  # Add this line to reshape the output of GlobalAveragePooling2D
    x = GlobalAveragePooling2D()(x)
    prediction = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=prediction)

    img = cv2.imread(image_path)
    img = cv2.resize(img, target_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    img_scaled = img / 255

    prediction = model.predict(img)[0][0]  # Access the first element for ASD probability
    print("prediction: {:.5f}".format(prediction))

    rounded_prediction = round(prediction, 2)
    print(f"Predicted probability: {rounded_prediction:.2f}")

    if rounded_prediction > 0.5:
        print(f"Predicted ASD with probability: {rounded_prediction:.2f}")

        return rounded_prediction
    else:
        print(f"Predicted non-ASD with probability: {1 - rounded_prediction:.2f}")
        return rounded_prediction


def predict_asd_InceptionV3(image_path):
    model_path = '/Users/isurudissanayake/Documents/Data/DATA_SET/Feature-Extraction/InceptionV3/InceptionV3Model.h5'
    target_size = (299, 299)

    # Load the pre-trained ResNet50 model
    base_model = InceptionV3(weights='imagenet', include_top=True)

    x = base_model.get_layer('avg_pool').output

    x = Dense(255, activation='relu')(x)
    x = Reshape((1, 1, 255))(x)  # Add this line to reshape the output of GlobalAveragePooling2D
    x = GlobalAveragePooling2D()(x)
    prediction = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=prediction)

    img = cv2.imread(image_path)
    img = cv2.resize(img, target_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    img_scaled = img / 255

    prediction = model.predict(img)[0][0]  # Access the first element for ASD probability
    print("prediction: {:.5f}".format(prediction))

    rounded_prediction = round(prediction, 2)
    print(f"Predicted probability: {rounded_prediction:.2f}")

    if rounded_prediction > 0.5:
        print(f"Predicted ASD with probability: {rounded_prediction:.2f}")

        return rounded_prediction
    else:
        print(f"Predicted non-ASD with probability: {1 - rounded_prediction:.2f}")
        return rounded_prediction


# def predict_asd_InceptionV3(image_path):
#     model_path = '/Users/isurudissanayake/Documents/Data/DATA_SET/Feature-Extraction/InceptionV3/InceptionV3Model.h5'
#     target_size = (299, 299)
#
#     # Load the pre-trained ResNet50 model
#     base_model = InceptionV3(weights='imagenet', include_top=True)
#
#     # Create a new model that takes the input of ResNet50 and outputs the desired layer
#     model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)
#
#     # Process the input image
#     img = cv2.imread(image_path)
#     img = cv2.resize(img, target_size)
#     img = preprocess_input(np.expand_dims(image.img_to_array(img), axis=0))
#
#     # Extract features using the full ResNet50 model
#     features = model.predict(img)  # Use the new model
#
#     # Reshape features to match the expected input shape of trained_model
#     features_reshaped = np.reshape(features, (1, 2048))  # Reshape to (1, 7, 7, 512)
#
#     # Load the trained model
#     trained_model = load_model(model_path)
#
#     # Predict ASD probability using the trained model and extracted features
#     prediction = trained_model.predict(features_reshaped)[0][0]  # Access the first element for ASD probability
#     print("prediction: ", prediction)
#     print("prediction: {:.5f}".format(prediction))
#
#     rounded_prediction = round(prediction, 2)
#     print(f"Predicted probability: {rounded_prediction:.2f}")
#
#     if rounded_prediction > 0.5:
#         print(f"Predicted ASD with probability: {rounded_prediction:.2f}")
#
#         return rounded_prediction
#     else:
#         print(f"Predicted non-ASD with probability: {1 - rounded_prediction:.2f}")
#         return rounded_prediction


# import cv2
# import time
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import MinMaxScaler
# from skimage.segmentation import mark_boundaries
# from keras.applications.vgg16 import preprocess_input
# from tensorflow.keras.applications.vgg16 import VGG16
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
# from PIL import Image
# from pathlib import Path
# from IPython.display import display
# from lime.lime_image import LimeImageExplainer
#
# image_path = "/Users/isurudissanayake/Documents/Data/DATA_SET/ASD/0579.jpg"
#
# model_path = '/Users/isurudissanayake/Documents/Data/DATA_SET/Feature-Extraction/VGG16/VGG16Model.h5'
# target_size = (224, 224)
#
# base_model = VGG16(weights='imagenet', include_top=True)
# x = base_model.get_layer('block5_pool').output
#
# x = GlobalAveragePooling2D()(x)
# x = Dense(1024, activation='relu')(x)
# predictions = Dense(1, activation='sigmoid')(x)
#
# model = Model(inputs=base_model.input, outputs=predictions)
#
# img = np.array(Image.open(image_path).resize(target_size))
# img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR
# img = np.expand_dims(img, axis=0)  # Add a batch dimension
# img = preprocess_input(img)  # Preprocess the image
#
# img_scaled = img / 255.0
#
# prediction = model.predict(img)[0][0]  # Access the first element for ASD probability
#
# rounded_prediction = round(prediction, 2)
# print(f"Predicted probability: {rounded_prediction:.2f}")
#
# if rounded_prediction > 0.5:
#     print(f"Predicted ASD with probability: {rounded_prediction:.2f}")
#
#     start_time = time.time()
#     explainer = LimeImageExplainer()
#
#     # Call the function to generate and visualize explanation
#     explanation = explainer.explain_instance(img_scaled[0], model.predict, top_labels=1, hide_color=0,
#                                              num_samples=10000, random_seed=42)
#
#     # Visualize the explanation using matplotlib
#     temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5,
#                                                 hide_rest=False)
#
#     # Resize the explanation mask to match the original image dimensions
#     mask = cv2.resize(mask, (target_size[0], target_size[1]), interpolation=cv2.INTER_NEAREST)
#
#     # Convert the mask to the original image mode
#     original_image = Image.open(image_path)
#     original_width, original_height = original_image.size
#     original_mode = original_image.mode
#
#     # Overlay the explanation mask on the original image
#     mask = cv2.resize(mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
#     original_image = np.array(original_image)
#     original_image[mask > 0.5] = (0, 255, 0)
#
#     # Display the original image with the explanation mask
#     display(Image.fromarray(original_image))
#     end_time = time.time()
#     elapsed_time = end_time - start_time  # Calculate elapsed time in seconds
#
#     print("Generate time to XAI:", elapsed_time, "seconds")
# else:
#     print(f"Predicted non-ASD with probability: {1 - rounded_prediction:.2f}")