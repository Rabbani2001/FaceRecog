from fastapi import FastAPI
from deepface import DeepFace
import base64
import numpy as np
import cv2
import os

app = FastAPI()

DB_PATH = "faces"

print("🔄 Loading model...")
model = DeepFace.build_model("Facenet")
print("✅ Model loaded")

def base64_to_image(base64_str):
    image_data = base64_str.split(",")[1]
    image_bytes = base64.b64decode(image_data)
    np_arr = np.frombuffer(image_bytes, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)


@app.post("/identify")
def identify_face(data: dict):
    try:
        img = base64_to_image(data["image"])

        temp_path = "temp.jpg"
        cv2.imwrite(temp_path, img)

        result = DeepFace.find(
            img_path=temp_path,
            db_path=DB_PATH,
            model_name="Facenet",
            detector_backend="opencv",
            enforce_detection=False
        )

        if len(result) > 0 and len(result[0]) > 0:
            match = result[0].iloc[0]
            identity_path = match["identity"]
            teacher_name = os.path.basename(identity_path).split(".")[0]

            return {"teacher": teacher_name}

        return {"teacher": None}

    except Exception as e:
        return {"error": str(e)}