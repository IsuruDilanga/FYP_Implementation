import numpy as np
import cv2
from keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from PIL import Image

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



# import numpy as np
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