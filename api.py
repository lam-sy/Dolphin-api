import argparse
import io
import os
import glob
import copy
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from omegaconf import OmegaConf
from chat import DOLPHIN
from demo_page_edit import (
    process_single_image,
    process_document,
    parse_html_to_json,
    save_outputs_with_html_parsing,
    save_combined_pdf_results_with_html_parsing
)
from utils.utils import setup_output_dirs, convert_pdf_to_images
import uvicorn

app = FastAPI()
model = None
save_dir = "results_api"
max_batch_size = 4

@app.get("/")
async def get_api_info():
    """
    Returns information about all available API endpoints.
    """
    endpoints = {
        "api_name": "Dolphin Document Processing API",
        "description": "API for processing documents and images using the DOLPHIN model",
        "available_endpoints": [
            {
                "path": "/",
                "method": "GET",
                "description": "Get API information and list of all available endpoints",
                "parameters": "None"
            },
            {
                "path": "/health",
                "method": "GET",
                "description": "Health check endpoint",
                "parameters": "None"
            },
            {
                "path": "/predict/",
                "method": "POST",
                "description": "Process a single image file using the DOLPHIN model",
                "parameters": "file: UploadFile (image file)",
                "supported_formats": ["jpg", "jpeg", "png"],
                "returns": "JSON with filename and recognition results"
            },
            {
                "path": "/predict_document/",
                "method": "POST",
                "description": "Process a document (image or PDF) using the DOLPHIN model with HTML parsing",
                "parameters": "file: UploadFile (document file)",
                "supported_formats": ["jpg", "jpeg", "png", "pdf"],
                "returns": "JSON with filename and structured recognition results"
            },
            {
                "path": "/parse_html/",
                "method": "POST",
                "description": "Parse HTML content into structured JSON format",
                "parameters": "JSON body with 'html_text' field",
                "example_body": {"html_text": "<html>...</html>"},
                "returns": "JSON with parsed content structure"
            },
            {
                "path": "/batch_predict/",
                "method": "POST",
                "description": "Process all supported document files in a directory",
                "parameters": "JSON body with 'directory_path' field",
                "supported_formats": ["jpg", "jpeg", "png", "pdf"],
                "example_body": {"directory_path": "/path/to/documents"},
                "returns": "JSON with batch processing results for all files"
            }
        ],
        "usage_examples": {
            "single_image": "curl -X POST -F 'file=@image.jpg' http://localhost:8000/predict/",
            "document": "curl -X POST -F 'file=@document.pdf' http://localhost:8000/predict_document/",
            "html_parsing": "curl -X POST -H 'Content-Type: application/json' -d '{\"html_text\":\"<p>Sample HTML</p>\"}' http://localhost:8000/parse_html/",
            "batch_processing": "curl -X POST -H 'Content-Type: application/json' -d '{\"directory_path\":\"/path/to/docs\"}' http://localhost:8000/batch_predict/"
        },
        "interactive_docs": "Visit /docs for interactive API documentation",
        "model_status": "Check /health for API status"
    }
    return endpoints

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
        raise HTTPException(status_code=503, detail="Model is not loaded yet")

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
        raise HTTPException(status_code=500, detail=str(e))


# @app.post("/predict_document/")
# async def predict_document(file: UploadFile = File(...)):
#     """
#     Accepts a document file (image or PDF), processes it using the DOLPHIN model,
#     and returns the recognition results in JSON format with HTML parsing.
#     Supports both single images and multi-page PDFs.
#     """
#     if not model:
#         raise HTTPException(status_code=503, detail="Model is not loaded yet")

#     try:
#         # Check file extension
#         file_ext = os.path.splitext(file.filename)[1].lower()
#         supported_exts = ['.jpg', '.jpeg', '.png', '.pdf']
        
#         if file_ext not in supported_exts:
#             raise HTTPException(
#                 status_code=400,
#                 detail=f"Unsupported file type: {file_ext}. Supported types: {supported_exts}"
#             )

#         # Save uploaded file temporarily
#         temp_path = os.path.join(save_dir, "temp", file.filename)
#         os.makedirs(os.path.dirname(temp_path), exist_ok=True)
        
#         contents = await file.read()
#         with open(temp_path, "wb") as f:
#             f.write(contents)

#         try:
#             # Process the document
#             json_path, recognition_results = process_document(
#                 document_path=temp_path,
#                 model=model,
#                 save_dir=save_dir,
#                 max_batch_size=max_batch_size
#             )

#             if json_path:
#                 print(f"API results saved to: {json_path}")

#             return {"filename": file.filename, "results": recognition_results}

#         finally:
#             # Clean up temporary file
#             if os.path.exists(temp_path):
#                 os.remove(temp_path)

#     except HTTPException:
#         raise
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


# @app.post("/parse_html/")
# async def parse_html(html_content: dict):
#     """
#     Parses HTML content into structured JSON format.
#     Expects a JSON body with 'html_text' field.
#     """
#     try:
#         html_text = html_content.get('html_text', '')
#         if not html_text:
#             raise HTTPException(status_code=400, detail="html_text field is required")
        
#         parsed_content = parse_html_to_json(html_text)
#         return {"parsed_content": parsed_content}

#     except HTTPException:
#         raise
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# @app.post("/batch_predict/")
# async def batch_predict(directory_path: dict):
#     """
#     Processes all supported document files in a directory.
#     Expects a JSON body with 'directory_path' field.
#     """
#     if not model:
#         raise HTTPException(status_code=503, detail="Model is not loaded yet")

#     try:
#         dir_path = directory_path.get('directory_path', '')
#         if not dir_path:
#             raise HTTPException(status_code=400, detail="directory_path field is required")
        
#         if not os.path.isdir(dir_path):
#             raise HTTPException(status_code=400, detail=f"Directory does not exist: {dir_path}")

#         # Collect Document Files (images and PDFs)
#         file_extensions = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG", ".pdf", ".PDF"]
        
#         document_files = []
#         for ext in file_extensions:
#             document_files.extend(glob.glob(os.path.join(dir_path, f"*{ext}")))
#         document_files = sorted(document_files)

#         if not document_files:
#             return {"message": "No supported document files found in directory", "results": []}

#         batch_results = []
#         total_files = len(document_files)
#         print(f"Processing {total_files} files from directory: {dir_path}")

#         # Process All Document Files
#         for i, file_path in enumerate(document_files):
#             try:
#                 print(f"Processing file {i+1}/{total_files}: {file_path}")
                
#                 json_path, recognition_results = process_document(
#                     document_path=file_path,
#                     model=model,
#                     save_dir=save_dir,
#                     max_batch_size=max_batch_size,
#                 )

#                 batch_results.append({
#                     "filename": os.path.basename(file_path),
#                     "file_path": file_path,
#                     "status": "success",
#                     "results": recognition_results,
#                     "json_path": json_path
#                 })

#             except Exception as e:
#                 print(f"Error processing {file_path}: {str(e)}")
#                 batch_results.append({
#                     "filename": os.path.basename(file_path),
#                     "file_path": file_path,
#                     "status": "error",
#                     "error": str(e)
#                 })

#         return {
#             "directory": dir_path,
#             "total_files": total_files,
#             "processed_files": len(batch_results),
#             "results": batch_results
#         }

#     except HTTPException:
#         raise
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)