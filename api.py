import argparse
import io
import os
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from omegaconf import OmegaConf
from chat import DOLPHIN
from demo_page_edit import process_single_image
from utils.utils import setup_output_dirs
import uvicorn

app = FastAPI()
model = None
save_dir = "results_api"
max_batch_size = 4

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.on_event("startup")
async def startup_event():
    global model
    config = OmegaConf.load("./config/Dolphin.yaml")
    model = DOLPHIN(config)
    setup_output_dirs(save_dir)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """
    Accepts an image file, processes it using the DOLPHIN model,
    and returns the recognition results in JSON format.
    """
    if not model:
        return {"error": "Model is not loaded yet"}

    try:
        # Read image from upload
        contents = await file.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Process the single image
        image_name = os.path.splitext(file.filename)[0]
        json_path, recognition_results = process_single_image(
            pil_image, model, save_dir, image_name, max_batch_size
        )

        if json_path:
            print(f"API results saved to: {json_path}")

        return {"filename": file.filename, "results": recognition_results}

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)