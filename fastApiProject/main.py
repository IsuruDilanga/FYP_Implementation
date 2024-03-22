import base64

from fastapi import FastAPI, File, UploadFile, HTTPException
import os
from fastapi.middleware.cors import CORSMiddleware
from ASD_Prediction import *
from Emotoin_Prediction import *

from fastapi.responses import JSONResponse
from typing import Optional

app = FastAPI()

# Allow all origins and methods for CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

folder_path = '/Users/isurudissanayake/DataspellProjects/FYP_Implementation/aunite/src/CaptureImages'


@app.get("/predictASD")
async def predict_asd(filepath: str, selected_model: Optional[str] = None):
    global image_path

    try:
        print(f"Filepath: {filepath}")
        print(f"Selected Model: {selected_model}")

        input_file_path = '/Users/isurudissanayake/DataspellProjects/FYP_Implementation/aunite/src/CaptureImages/' + filepath
        image_path = input_file_path

        if selected_model == "VGG16":
            result = predict_asd_vgg16(input_file_path)
        elif selected_model == "VGG19":
            result = predict_asd_vgg19(input_file_path)
        elif selected_model == "ResNet50":
            result = predict_asd_ResNet50(input_file_path)
        elif selected_model == "ResNet50V2":
            result = predict_asd_ResNet50V2(input_file_path)
        elif selected_model == "InceptionV3":
            result = predict_asd_InceptionV3(input_file_path)

        print(f"Result: {result}")
        rounded_result = round(result, 2)
        if rounded_result > 0.5:
            return JSONResponse(
                content={"message": "Prediction successful", "prediction": float(rounded_result), "isASD": True})
        else:
            return JSONResponse(
                content={"message": "Prediction successful", "prediction": float(1 - rounded_result), "isASD": False})
        # return JSONResponse(content={"message": "Prediction successful", "prediction": float(round(rounded_result, 2))})
    except Exception as e:
        return JSONResponse(content={"error": f"Failed to predict ASD: {str(e)}"})


@app.get("/predictEmotoin")
async def predict_Emotoin(filepath: str, selected_model: Optional[str] = None):
    try:
        print(f"Filepath: {filepath}")
        print(f"Selected Model: {selected_model}")

        input_file_path = '/Users/isurudissanayake/DataspellProjects/FYP_Implementation/aunite/src/CaptureImages/' + filepath
        image_path = input_file_path

        if selected_model == "VGG16":
            emotion, probability = predict_emotion_vgg16(input_file_path)
        elif selected_model == "VGG19":
            emotion, probability = predict_emotion_vgg19(input_file_path)
        elif selected_model == "ResNet50":
            emotion, probability = predict_emotion_resNet50(input_file_path)
        elif selected_model == "ResNet50V2":
            emotion, probability = predict_emotion_resNet50V2(input_file_path)
        elif selected_model == "InceptionV3":
            emotion, probability = predict_emotion_inceptionV3(input_file_path)

        print(f"emotion: {emotion}")
        print(f"probability: {probability}")

        return JSONResponse(content={"message": "Emotion prediction successful", "emotion": emotion, "probability": float(probability)})
    except Exception as e:
        return JSONResponse(content={"error": f"Failed to predict ASD: {str(e)}"})


@app.get("/emotion-xai-lime")
async def emotion_explain_lime(filepath: str, selected_model: Optional[str] = None):
    try:
        print(f"Filepath: {filepath}")
        print(f"Selected Model: {selected_model}")
        input_file_path = '/Users/isurudissanayake/DataspellProjects/FYP_Implementation/aunite/src/CaptureImages/' + filepath
        image_path = input_file_path
        print(f"Image Path: {image_path}")

        if selected_model == "VGG16":
            print("VGG16")
            xai_lime_path = request_lime_emotion_vgg16(image_path)
        elif selected_model == "VGG19":
            print("VGG19")
            xai_lime_path = request_lime_emotion_vgg19(image_path)
        elif selected_model == "ResNet50":
            print("ResNet50")
            xai_lime_path = request_lime_emotion_resNet50(image_path)
        elif selected_model == "ResNet50V2":
            print("ResNet50V2")
            xai_lime_path = request_lime_emotion_resNet50V2(image_path)
        elif selected_model == "InceptionV3":
            print("InceptionV3")
            xai_lime_path = request_lime_emotion_inceptionV3(image_path)

        print(f"XAI LIME Path: {xai_lime_path}")
        return JSONResponse(content={"message": "LIME explanation successful", "xai_lime_path": xai_lime_path,
                                     "image_path": image_path})
    except Exception as e:
        return JSONResponse(content={"error": f"Failed to explain LIME: {str(e)}"})


@app.get("/emotion-xai-gradcam")
async def explain_lime(filepath: str, selected_model: Optional[str] = None):
    try:
        print(f"Filepath: {filepath}")
        print(f"Selected Model: {selected_model}")
        input_file_path = '/Users/isurudissanayake/DataspellProjects/FYP_Implementation/aunite/src/CaptureImages/' + filepath
        image_path = input_file_path
        print(f"Image Path: {image_path}")

        if selected_model == "VGG16":
            print("VGG16")
            xai_gradCam_path = request_emotion_gradcam_vgg16()
        elif selected_model == "VGG19":
            print("VGG19")
            xai_gradCam_path = request_gradcam_emotion_vgg19()
        elif selected_model == "ResNet50":
            print("ResNet50")
            xai_gradCam_path = request_gradcam_emotion_resNet50()
        elif selected_model == "ResNet50V2":
            print("ResNet50V2")
            xai_gradCam_path = request_gradcam_emotion_resNet50V2()
        elif selected_model == "InceptionV3":
            print("InceptionV3")
            xai_gradCam_path = request_gradcam_emotion_inceptionV3()

        print(f"XAI GradCam Path: {xai_gradCam_path}")
        return JSONResponse(content={"message": "GradCAM explanation successful", "xai_gradCAM_path": xai_gradCam_path,
                                     "image_path": image_path})
    except Exception as e:
        return JSONResponse(content={"error": f"Failed to explain GradCAM: {str(e)}"})


@app.get("/xai-lime")
async def explain_lime(filepath: str, selected_model: Optional[str] = None):
    try:
        print(f"Filepath: {filepath}")
        print(f"Selected Model: {selected_model}")
        input_file_path = '/Users/isurudissanayake/DataspellProjects/FYP_Implementation/aunite/src/CaptureImages/' + filepath
        image_path = input_file_path
        print(f"Image Path: {image_path}")

        if selected_model == "VGG16":
            print("VGG16")
            xai_lime_path = request_Lime_vgg16(image_path)
        elif selected_model == "VGG19":
            print("VGG19")
            xai_lime_path = request_Lime_vgg19(image_path)
        elif selected_model == "ResNet50":
            print("ResNet50")
            xai_lime_path = request_Lime_ResNet50(image_path)
        elif selected_model == "ResNet50V2":
            print("ResNet50V2")
            xai_lime_path = request_Lime_ResNet50V2(image_path)
        elif selected_model == "InceptionV3":
            print("InceptionV3")
            xai_lime_path = request_Lime_InceptionV3(image_path)

        print(f"XAI LIME Path: {xai_lime_path}")
        return JSONResponse(content={"message": "LIME explanation successful", "xai_lime_path": xai_lime_path,
                                     "image_path": image_path})
    except Exception as e:
        return JSONResponse(content={"error": f"Failed to explain LIME: {str(e)}"})


@app.get("/xai-gradcam")
async def explain_lime(filepath: str, selected_model: Optional[str] = None):
    try:
        print(f"Filepath: {filepath}")
        print(f"Selected Model: {selected_model}")
        input_file_path = '/Users/isurudissanayake/DataspellProjects/FYP_Implementation/aunite/src/CaptureImages/' + filepath
        image_path = input_file_path
        print(f"Image Path: {image_path}")

        if selected_model == "VGG16":
            print("VGG16")
            xai_gradCam_path = request_GradCam_vgg16()
        elif selected_model == "VGG19":
            print("VGG19")
            xai_gradCam_path = request_GradCam_vgg19()
        elif selected_model == "ResNet50":
            print("ResNet50")
            xai_gradCam_path = request_GradCam_ResNet50()
        elif selected_model == "ResNet50V2":
            print("ResNet50V2")
            xai_gradCam_path = request_GradCam_ResNet50V2()
        elif selected_model == "InceptionV3":
            print("InceptionV3")
            xai_gradCam_path = request_GradCam_InceptionV3()

        print(f"XAI GradCam Path: {xai_gradCam_path}")
        return JSONResponse(content={"message": "GradCAM explanation successful", "xai_gradCAM_path": xai_gradCam_path,
                                     "image_path": image_path})
    except Exception as e:
        return JSONResponse(content={"error": f"Failed to explain GradCAM: {str(e)}"})


@app.post("/save_image")
async def save_image(image: UploadFile = File(...)):
    if not image.content_type.startswith('image'):
        raise HTTPException(status_code=415, detail="Unsupported Media Type. Please provide an image file.")

    save_folder = '/Users/isurudissanayake/DataspellProjects/FYP_Implementation/aunite/src/CaptureImages'  # Update with your desired save folder path
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Generate a unique filename using UUID
    unique_filename = str(uuid4()) + '.jpg'
    image_path = os.path.join(save_folder, unique_filename)

    with open(image_path, "wb") as image_file:
        content = await image.read()
        image_file.write(content)
    print(f"Filename: {unique_filename}");
    return {"message": "Image saved successfully", "filename": unique_filename}


@app.post("/upload_image")
async def upload_image(image: UploadFile = File(...)):
    try:
        if not image.content_type.startswith('image'):
            raise HTTPException(status_code=415, detail="Unsupported Media Type. Please provide an image file.")

        save_folder = '/Users/isurudissanayake/DataspellProjects/FYP_Implementation/aunite/src/CaptureImages'  # Update with your desired save folder path
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        # Generate a unique filename using UUID
        unique_filename = str(uuid4()) + '.jpg'
        image_path = os.path.join(save_folder, unique_filename)

        with open(image_path, "wb") as image_file:
            content = await image.read()
            image_file.write(content)

        print(f"Filename: {unique_filename}")
        return {"message": "Image saved successfully", "filename": unique_filename}

    except Exception as e:
        return {"error": f"Failed to upload image: {str(e)}"}
