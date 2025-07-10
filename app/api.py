"""
FastAPI REST API for Chest X-ray Pneumonia Detection
"""
import io
import os
from typing import Optional, List
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import uvicorn

app = FastAPI(
    title="Chest X-ray Pneumonia Detection API",
    description="API for detecting pneumonia in chest X-ray images",
    version="1.0.0"
)

# Global model variable
model: Optional[tf.keras.Model] = None

def load_model() -> tf.keras.Model:
    """Load the pre-trained model"""
    global model
    if model is None:
        model_path = os.path.join(os.path.dirname(__file__), "chest_xray_model.keras")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        model = tf.keras.models.load_model(model_path)
    return model

def preprocess_image(image_bytes):
    """Preprocess uploaded image for model prediction"""
    try:
        # Open image and convert to RGB
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Resize to model input size
        image = image.resize((224, 224))
        
        # Convert to numpy array and normalize
        img_array = np.array(image) / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model()

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Chest X-ray Pneumonia Detection API", "status": "healthy"}

@app.get("/health")
async def health_check():
    """Detailed health check"""
    try:
        model = load_model()
        return {
            "status": "healthy",
            "model_loaded": model is not None,
            "model_input_shape": str(model.input_shape) if model else None
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)}
        )

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict pneumonia from chest X-ray image
    
    Returns:
        - prediction: "Normal" or "Pneumonia"
        - confidence: probability score
    """
    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read and preprocess image
        image_bytes = await file.read()
        processed_image = preprocess_image(image_bytes)
        
        # Make prediction
        model_instance = load_model()
        prediction_prob = model_instance.predict(processed_image)[0][0]
        
        # Convert to label
        prediction_label = "Pneumonia" if prediction_prob > 0.5 else "Normal"
        confidence = float(prediction_prob) if prediction_prob > 0.5 else float(1 - prediction_prob)
        
        return {
            "prediction": prediction_label,
            "confidence": round(confidence, 4),
            "raw_probability": round(float(prediction_prob), 4),
            "filename": file.filename
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/batch_predict")
async def batch_predict(files: List[UploadFile] = File(...)):
    """
    Predict pneumonia for multiple chest X-ray images
    """
    if len(files) > 10:  # Limit batch size
        raise HTTPException(status_code=400, detail="Maximum 10 files allowed per batch")
    
    results = []
    
    for file in files:
        if not file.content_type or not file.content_type.startswith("image/"):
            results.append({
                "filename": file.filename,
                "error": "File must be an image"
            })
            continue
            
        try:
            image_bytes = await file.read()
            processed_image = preprocess_image(image_bytes)
            
            model_instance = load_model()
            prediction_prob = model_instance.predict(processed_image)[0][0]
            
            prediction_label = "Pneumonia" if prediction_prob > 0.5 else "Normal"
            confidence = float(prediction_prob) if prediction_prob > 0.5 else float(1 - prediction_prob)
            
            results.append({
                "filename": file.filename,
                "prediction": prediction_label,
                "confidence": round(confidence, 4),
                "raw_probability": round(float(prediction_prob), 4)
            })
            
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e)
            })
    
    return {"results": results}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
