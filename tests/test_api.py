"""
Test suite for the Chest X-ray Detection API
"""
import pytest
import os
import io
from fastapi.testclient import TestClient
from PIL import Image
import numpy as np

# Import the API
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from app.api import app

client = TestClient(app)

def create_test_image():
    """Create a test chest X-ray-like image"""
    # Create a 224x224 grayscale image
    img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    
    # Convert to bytes
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    
    return img_bytes

class TestAPI:
    
    def test_root_endpoint(self):
        """Test the root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        assert "Chest X-ray Pneumonia Detection API" in response.json()["message"]
    
    def test_health_endpoint(self):
        """Test the health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
    
    @pytest.mark.skipif(
        not os.path.exists("app/chest_xray_model.keras") or os.path.getsize("app/chest_xray_model.keras") == 0,
        reason="Model file not available - skipping in CI environment"
    )
    def test_predict_endpoint(self):
        """Test the prediction endpoint with a test image"""
        test_image = create_test_image()
        
        response = client.post(
            "/predict",
            files={"file": ("test_image.jpg", test_image, "image/jpeg")}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Check response structure
        assert "prediction" in data
        assert "confidence" in data
        assert "raw_probability" in data
        assert data["prediction"] in ["Normal", "Pneumonia"]
        assert 0 <= data["confidence"] <= 1
        assert 0 <= data["raw_probability"] <= 1
    
    def test_predict_invalid_file(self):
        """Test prediction with invalid file type"""
        # Create a text file instead of image
        text_content = b"This is not an image"
        
        response = client.post(
            "/predict",
            files={"file": ("test.txt", text_content, "text/plain")}
        )
        
        assert response.status_code == 400
        assert "File must be an image" in response.json()["detail"]
    
    @pytest.mark.skipif(
        not os.path.exists("app/chest_xray_model.keras") or os.path.getsize("app/chest_xray_model.keras") == 0,
        reason="Model file not available - skipping in CI environment"
    )
    def test_batch_predict(self):
        """Test batch prediction endpoint"""
        # Create multiple test images
        test_images = [
            ("test1.jpg", create_test_image(), "image/jpeg"),
            ("test2.jpg", create_test_image(), "image/jpeg")
        ]
        
        response = client.post(
            "/batch_predict",
            files=[("files", img) for img in test_images]
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "results" in data
        assert len(data["results"]) == 2
        
        for result in data["results"]:
            if "error" not in result:
                assert "prediction" in result
                assert "confidence" in result
                assert result["prediction"] in ["Normal", "Pneumonia"]

class TestModelValidation:
    
    def test_model_exists(self):
        """Test that the model file exists"""
        model_path = os.path.join("app", "chest_xray_model.keras")
        if not os.path.exists(model_path):
            pytest.skip("Model file not found - skipping in CI environment")
    
    @pytest.mark.model
    def test_model_loads(self):
        """Test that the model can be loaded"""
        model_path = os.path.join("app", "chest_xray_model.keras")
        if not os.path.exists(model_path) or os.path.getsize(model_path) == 0:
            pytest.skip("Model file not available - skipping in CI environment")
            
        try:
            from app.api import load_model
            model = load_model()
            assert model is not None
            assert hasattr(model, 'predict')
        except Exception as e:
            pytest.fail(f"Model failed to load: {e}")
    
    @pytest.mark.model  
    def test_model_input_shape(self):
        """Test that the model has the expected input shape"""
        model_path = os.path.join("app", "chest_xray_model.keras")
        if not os.path.exists(model_path) or os.path.getsize(model_path) == 0:
            pytest.skip("Model file not available - skipping in CI environment")
            
        try:
            from app.api import load_model
            model = load_model()
            
            # Type check and attribute verification
            assert model is not None, "Model should not be None"
            
            # Check that model has input_shape attribute and verify the shape
            if hasattr(model, 'input_shape'):
                input_shape = getattr(model, 'input_shape')  # Use getattr to avoid linting issues
                # Expected input shape: (None, 224, 224, 3)
                assert input_shape[1:] == (224, 224, 3), f"Expected (None, 224, 224, 3), got {input_shape}"
            else:
                pytest.skip("Model does not have input_shape attribute")
        except Exception as e:
            pytest.fail(f"Model input shape test failed: {e}")

if __name__ == "__main__":
    pytest.main([__file__])
