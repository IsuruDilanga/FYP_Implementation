import base64

from fastapi import FastAPI, File, UploadFile, HTTPException
import os
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from uuid import uuid4
from VGG16_ASD_Prediction import predict_asd_vgg16;
from fastapi.responses import JSONResponse

app = FastAPI()

# Allow all origins and methods for CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}

@app.get("/predictASD")
async def predict_asd(filepath: str):
    try:
        print(f"Filepath: {filepath}")
        input_file_path = '/Users/isurudissanayake/DataspellProjects/FYP_Implementation/aunite/src/CaptureImages/' + filepath
        result = predict_asd_vgg16(input_file_path)
        print(f"Result: {result}")
        rounded_result = round(result, 2)
        if rounded_result > 0.5:
            return JSONResponse(content={"message": "Prediction successful", "prediction": float(rounded_result), "isASD": True})
        else:
            return JSONResponse(content={"message": "Prediction successful", "prediction": float(1-rounded_result), "isASD": False})
        # return JSONResponse(content={"message": "Prediction successful", "prediction": float(round(rounded_result, 2))})
    except Exception as e:
        return JSONResponse(content={"error": f"Failed to predict ASD: {str(e)}"})

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


