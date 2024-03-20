from uuid import uuid4
import os
import tensorflow as tf
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

folder_path = '/Users/isurudissanayake/DataspellProjects/FYP_Implementation/aunite/src/CaptureImages'

vgg16Model_predict = None
img_scaled_vgg16 = None
vgg16GradCam_img_original = None
vgg16GradCam_img_for_model = None
vgg16GradCam_model = None
target_size_vgg16 = (244, 244)
def predict_asd_vgg16(image_path):
    global vgg16Model_predict
    global img_scaled_vgg16
    global vgg16GradCam_img_original
    global vgg16GradCam_img_for_model
    global vgg16GradCam_model
    model_path = '/Users/isurudissanayake/Documents/Data/DATA_SET/Feature-Extraction/VGG16/VGG16Model.h5'
    target_size = (224, 224)

    base_model = VGG16(weights='imagenet', include_top=True)
    x = base_model.get_layer('block5_pool').output

    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    vgg16GradCam_model = model

    img = np.array(Image.open(image_path).resize(target_size))
    vgg16GradCam_img_original = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR
    img = np.expand_dims(img, axis=0)  # Add a batch dimension
    img = preprocess_input(img)  # Preprocess the image
    vgg16GradCam_img_for_model = preprocess_input(np.expand_dims(image.img_to_array(img), axis=0))

    img_scaled_vgg16 = img / 255.0

    vgg16Model_predict = model.predict
    prediction = model.predict(img)[0][0]  # Access the first element for ASD probability

    rounded_prediction = round(prediction, 2)
    print(f"Predicted probability: {rounded_prediction:.2f}")

    if rounded_prediction > 0.5:
        print(f"Predicted ASD with probability: {rounded_prediction:.2f}")

        return rounded_prediction
    else:
        print(f"Predicted non-ASD with probability: {1 - rounded_prediction:.2f}")
        return rounded_prediction


def request_GradCam_vgg16():
    print("request_GradCam_vgg16")
    global vgg16GradCam_img_original
    global vgg16GradCam_img_for_model
    global vgg16GradCam_model

    # Generate class activation heatmap
    heatmap = generate_grad_cam(vgg16GradCam_model, vgg16GradCam_img_for_model, 'block5_conv3')

    # Resize the heatmap to match the original image
    heatmap = cv2.resize(heatmap, (vgg16GradCam_img_original.shape[1], vgg16GradCam_img_original.shape[0]))

    # Convert the heatmap to the RGB color map
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Superimpose the heatmap on the original image
    superimposed_img = cv2.addWeighted(vgg16GradCam_img_original, 0.6, heatmap, 0.4, 0)

    # Save the modified image to the specified location
    save_folder = '/Users/isurudissanayake/DataspellProjects/FYP_Implementation/aunite/public/ASD_Prediction/GradCAM/GradCAM_VGG16'

    unique_filename = str(uuid4()) + '.jpg'
    output_image_path = os.path.join(save_folder, unique_filename)
    output_image = Image.fromarray(superimposed_img)
    output_image.save(output_image_path)

    returnOutput = './ASD_Prediction/GradCAM/GradCAM_VGG16/' + unique_filename

    return returnOutput


def request_Lime_vgg16(image_path):

    print("request_Lime_vgg16")
    global vgg16Model_predict
    global img_scaled_vgg16

    explainer = LimeImageExplainer()

    # Call the function to generate and visualize explanation
    explanation = explainer.explain_instance((img_scaled_vgg16)[0], vgg16Model_predict, top_labels=1, hide_color=0,
                                             num_samples=10000, random_seed=42)

    # Visualize the explanation using matplotlib
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5,
                                                hide_rest=False)

    # Resize the explanation mask to match the original image dimensions
    mask = cv2.resize(mask, (target_size_vgg16[0], target_size_vgg16[1]), interpolation=cv2.INTER_NEAREST)

    # Convert the mask to the original image mode
    original_image = Image.open(image_path)
    original_width, original_height = original_image.size
    original_mode = original_image.mode

    # Overlay the explanation mask on the original image
    mask = cv2.resize(mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
    original_image = np.array(original_image)
    original_image[mask > 0.5] = (0, 255, 0)

    # Save the modified image to the specified location
    save_folder = '/Users/isurudissanayake/DataspellProjects/FYP_Implementation/aunite/public/ASD_Prediction/LIME/LIME_VGG16'


    unique_filename = str(uuid4()) + '.jpg'
    output_image_path = os.path.join(save_folder, unique_filename)
    output_image = Image.fromarray(original_image)
    output_image.save(output_image_path)

    returnOutput = './ASD_Prediction/LIME/LIME_VGG16/' + unique_filename

    return returnOutput

    # display(Image.fromarray(original_image))


vgg19Model_predict = None
img_scaled_vgg19 = None
target_size_vgg19 = (244, 244)

def predict_asd_vgg19(image_path):
    global vgg19Model_predict
    global img_scaled_vgg19
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

    img_scaled_vgg19 = img / 255.0

    vgg19Model_predict = model.predict
    prediction = model.predict(img)[0][0]  # Access the first element for ASD probability

    rounded_prediction = round(prediction, 2)
    print(f"Predicted probability: {rounded_prediction:.2f}")

    if rounded_prediction > 0.5:
        print(f"Predicted ASD with probability: {rounded_prediction:.2f}")

        return rounded_prediction
    else:
        print(f"Predicted non-ASD with probability: {1 - rounded_prediction:.2f}")
        return rounded_prediction


def request_Lime_vgg19(image_path):
    global vgg19Model_predict
    global img_scaled_vgg19

    explainer = LimeImageExplainer()

    # Generate an explanation for the prediction using the explainer object
    explanation = explainer.explain_instance(img_scaled_vgg19[0], vgg19Model_predict, top_labels=1, hide_color=0,
                                             num_samples=10000, random_seed=42)

    # Visualize the explanation using matplotlib
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5,
                                                hide_rest=False)

    # Resize the explanation mask to match the original image dimensions
    mask = cv2.resize(mask, (target_size_vgg19[0], target_size_vgg19[1]), interpolation=cv2.INTER_NEAREST)

    # Convert the mask to the original image mode
    original_image = Image.open(image_path)
    # original_image = original_image.convert("L")  # Convert the original image to grayscale
    original_width, original_height = original_image.size
    original_mode = original_image.mode

    # Overlay the explanation mask on the original image
    mask = cv2.resize(mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
    original_image = np.array(original_image)
    original_image[mask > 0.5] = (0, 255, 0)

    # Save the modified image to the specified location
    save_folder = '/Users/isurudissanayake/DataspellProjects/FYP_Implementation/aunite/public/ASD_Prediction/LIME/LIME_VGG19'

    unique_filename = str(uuid4()) + '.jpg'
    output_image_path = os.path.join(save_folder, unique_filename)
    output_image = Image.fromarray(original_image)
    output_image.save(output_image_path)

    returnOutput = './ASD_Prediction/LIME/LIME_VGG19/' + unique_filename

    return returnOutput


resNet50Model_predict = None
img_scaled_ResNet50 = None
target_size_ResNet50 = (244, 244)
def predict_asd_ResNet50(image_path):
    global resNet50Model_predict
    global img_scaled_ResNet50

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

    img_scaled_ResNet50 = img / 255

    resNet50Model_predict = model.predict
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


def request_Lime_ResNet50(image_path):
    global resNet50Model_predict
    global img_scaled_ResNet50

    explainer = LimeImageExplainer()

    # Generate an explanation for the prediction using the explainer object
    explanation = explainer.explain_instance(img_scaled_ResNet50[0], resNet50Model_predict, top_labels=1, hide_color=0,
                                             num_samples=10000, random_seed=42)

    # Visualize the explanation using matplotlib
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5,
                                                hide_rest=False)

    # Resize the explanation mask to match the original image dimensions
    mask = cv2.resize(mask, (target_size_ResNet50[0], target_size_ResNet50[1]), interpolation=cv2.INTER_NEAREST)

    # Convert the mask to the original image mode
    original_image = Image.open(image_path)
    # original_image = original_image.convert("L")  # Convert the original image to grayscale
    original_width, original_height = original_image.size
    original_mode = original_image.mode

    # Overlay the explanation mask on the original image
    mask = cv2.resize(mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
    original_image = np.array(original_image)
    original_image[mask > 0.5] = (0, 255, 0)

    # Save the modified image to the specified location
    save_folder = '/Users/isurudissanayake/DataspellProjects/FYP_Implementation/aunite/public/ASD_Prediction/LIME/LIME_ResNet50'

    unique_filename = str(uuid4()) + '.jpg'
    output_image_path = os.path.join(save_folder, unique_filename)
    output_image = Image.fromarray(original_image)
    output_image.save(output_image_path)

    returnOutput = './ASD_Prediction/LIME/LIME_ResNet50/' + unique_filename

    return returnOutput


resNet50V2Model_predict = None
img_scaled_ResNet50V2 = None
target_size_ResNet50V2 = (244, 244)
def predict_asd_ResNet50V2(image_path):
    global resNet50V2Model_predict
    global img_scaled_ResNet50V2
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

    img_scaled_ResNet50V2 = img / 255

    resNet50V2Model_predict = model.predict
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


def request_Lime_ResNet50V2(image_path):
    global resNet50V2Model_predict
    global img_scaled_ResNet50V2

    explainer = LimeImageExplainer()

    # Generate an explanation for the prediction using the explainer object
    explanation = explainer.explain_instance(img_scaled_ResNet50V2[0], resNet50V2Model_predict, top_labels=1, hide_color=0,
                                             num_samples=10000, random_seed=42)

    # Visualize the explanation using matplotlib
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5,
                                                hide_rest=False)

    # Resize the explanation mask to match the original image dimensions
    mask = cv2.resize(mask, (target_size_ResNet50V2[0], target_size_ResNet50V2[1]), interpolation=cv2.INTER_NEAREST)

    # Convert the mask to the original image mode
    original_image = Image.open(image_path)
    # original_image = original_image.convert("L")  # Convert the original image to grayscale
    original_width, original_height = original_image.size
    original_mode = original_image.mode

    # Overlay the explanation mask on the original image
    mask = cv2.resize(mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
    original_image = np.array(original_image)
    original_image[mask > 0.5] = (0, 255, 0)

    # Save the modified image to the specified location
    save_folder = '/Users/isurudissanayake/DataspellProjects/FYP_Implementation/aunite/public/ASD_Prediction/LIME/LIME_ResNet50V2'

    unique_filename = str(uuid4()) + '.jpg'
    output_image_path = os.path.join(save_folder, unique_filename)
    output_image = Image.fromarray(original_image)
    output_image.save(output_image_path)

    returnOutput = './ASD_Prediction/LIME/LIME_ResNet50V2/' + unique_filename

    return returnOutput


InceptionV3Model_predict = None
img_scaled_InceptionV3 = None
target_size_InceptionV3 = (299, 299)
def predict_asd_InceptionV3(image_path):
    global InceptionV3Model_predict
    global img_scaled_InceptionV3
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

    img_scaled_InceptionV3 = img / 255

    InceptionV3Model_predict = model.predict
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


def request_Lime_InceptionV3(image_path):
    global InceptionV3Model_predict
    global img_scaled_InceptionV3

    explainer = LimeImageExplainer()

    # Generate an explanation for the prediction using the explainer object
    explanation = explainer.explain_instance(img_scaled_InceptionV3[0], InceptionV3Model_predict, top_labels=1, hide_color=0,
                                             num_samples=10000, random_seed=42)

    # Visualize the explanation using matplotlib
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5,
                                                hide_rest=False)

    # Resize the explanation mask to match the original image dimensions
    mask = cv2.resize(mask, (target_size_InceptionV3[0], target_size_InceptionV3[1]), interpolation=cv2.INTER_NEAREST)

    # Convert the mask to the original image mode
    original_image = Image.open(image_path)
    # original_image = original_image.convert("L")  # Convert the original image to grayscale
    original_width, original_height = original_image.size
    original_mode = original_image.mode

    # Overlay the explanation mask on the original image
    mask = cv2.resize(mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
    original_image = np.array(original_image)
    original_image[mask > 0.5] = (0, 255, 0)

    # Save the modified image to the specified location
    save_folder = '/Users/isurudissanayake/DataspellProjects/FYP_Implementation/aunite/public/ASD_Prediction/LIME/LIME_InceptionV3'

    unique_filename = str(uuid4()) + '.jpg'
    output_image_path = os.path.join(save_folder, unique_filename)
    output_image = Image.fromarray(original_image)
    output_image.save(output_image_path)

    returnOutput = './ASD_Prediction/LIME/LIME_InceptionV3/' + unique_filename

    return returnOutput


def generate_grad_cam(model, img_array, layer_name):
    # Create a model that maps the input image to the desired layer's output
    grad_model = Model(inputs=model.input, outputs=(model.get_layer(layer_name).output, model.output))

    # Compute the gradient of the predicted class with respect to the output feature map of the given layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        predicted_class_output = preds[:, 0]  # ASD class index assuming ASD class is the first one

    grads = tape.gradient(predicted_class_output, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]

    # Compute the heatmap
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, last_conv_layer_output), axis=-1)
    heatmap = np.maximum(heatmap, 0)  # ReLU on the heatmap
    heatmap /= np.max(heatmap)  # Normalize

    return heatmap