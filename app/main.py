import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array
import os
from typing import Tuple, Optional

def load_model(model_path: Optional[str] = None) -> tf.keras.Model:
    """Load the pre-trained model"""
    if model_path is None:
        model_path = os.path.join(os.path.dirname(__file__), "chest_xray_model.keras")
    model = tf.keras.models.load_model(model_path)
    return model

def predict_image(img_path: str, model: Optional[tf.keras.Model] = None) -> Tuple[str, float]:
    """Predict pneumonia from a chest X-ray image"""
    if model is None:
        model = load_model()
    
    # Load and preprocess image
    img = load_img(img_path, target_size=(224, 224))
    x = img_to_array(img)
    x = x / 255.0
    x = np.expand_dims(x, axis=0)
    
    # Make prediction
    pred = model.predict(x)[0][0]
    label = "Pneumonia" if pred > 0.5 else "Normal"
    return label, float(pred)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python main.py <image_path>")
        exit(1)
    
    img_path = sys.argv[1]
    if not os.path.exists(img_path):
        print(f"Error: Image file '{img_path}' not found")
        exit(1)
    
    try:
        model = load_model()
        label, prob = predict_image(img_path, model)
        print(f"Prediction: {label} (probability: {prob:.4f})")
    except Exception as e:
        print(f"Error: {e}")
        exit(1)
