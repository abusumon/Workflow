# Chest X-ray Pneumonia Detection - CI/CD Pipeline

This project implements a CI/CD pipeline for a chest X-ray pneumonia detection model.

## Project Structure
```
├── app/                    # Main application
│   ├── main.py            # Inference script
│   ├── api.py             # REST API server
│   └── chest_xray_model.keras  # Pre-trained model
├── model/                 # Training scripts
│   └── train.py           # Training script (for retraining)
├── tests/                 # Test suite
├── docker/                # Docker configuration
├── .github/workflows/     # GitHub Actions CI/CD
├── config/                # Configuration files
└── requirements.txt       # Dependencies
```

## CI/CD Pipeline Features

### 1. Continuous Integration (CI)
- Automated testing on every commit
- Model validation and performance checks
- Code quality and security scanning
- Docker image building

### 2. Continuous Deployment (CD)
- Automated deployment to staging/production
- Model serving via REST API
- Health checks and monitoring
- Rollback capabilities

### 3. ML-Specific Pipeline
- Model versioning and tracking
- Data validation
- Performance monitoring
- A/B testing for model updates

## Quick Start

### For Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/

# Start local inference server
python app/api.py
```

### For Production
```bash
# Build and run with Docker
docker build -t chest-xray-detector .
docker run -p 8000:8000 chest-xray-detector
```

## API Usage

```bash
# Test prediction endpoint
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@chest_xray_image.jpg"
```