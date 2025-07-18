from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
import uvicorn
import asyncio

import numpy as np
import cv2
import torch
from PIL import Image
from io import BytesIO
from typing import Dict

app = FastAPI()


from irsl_instance_segmentation.inference_detectron2 import InferenceDetectron2
from irsl_instance_segmentation.ram_grounding_sam import RAMGroundingSegmentAnything
model_maskrcnn = None
model_rgs = None

@app.on_event("startup")
async def load_dnn_models():
    global model_maskrcnn, model_rgs
    model_maskrcnn = InferenceDetectron2()
    model_rgs = RAMGroundingSegmentAnything()


# --- 画像予測処理 ---
semaphore = asyncio.Semaphore(1)

@app.post("/{model_name}/predict")
async def predict(model_name: str,
                   file: UploadFile = File(...),
                   text_prompt: str = Form(None),
                   ):
    if model_name not in ["maskrcnn", "ram-grounded-sam"]:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found.")
    async with semaphore:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
        image_np = np.array(image)

        if model_name == 'maskrcnn':
            results = model_maskrcnn.inference(image_np)
            return JSONResponse(content={"predictions": results})
        else:
            results = model_rgs.inference(image_np)
            return JSONResponse(content={"predictions": results})

@app.get("/ping")
def ping():
    return {"status": "ok"}

@app.get("/models")
def list_models():
    return {"available_models": list(model_registry.keys())}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8008, reload=True)
