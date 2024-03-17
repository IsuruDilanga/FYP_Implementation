import base64

from fastapi import FastAPI, File, UploadFile, HTTPException
import os
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from uuid import uuid4

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


