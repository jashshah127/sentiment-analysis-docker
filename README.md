# Sentiment Analysis API with Docker

A Dockerized Flask-based machine learning service for movie review sentiment analysis using Logistic Regression and TF-IDF.

## Project Overview

This project demonstrates MLOps practices for containerized ML deployment, featuring:
- **NLP Model**: Logistic Regression with TF-IDF vectorization for text classification
- **Flask API**: REST API for sentiment prediction
- **Docker Deployment**: Multi-stage Docker build for optimized containerization
- **Text Processing**: NLTK for text preprocessing and stopword removal
- **Production Ready**: Health checks, error handling, and proper logging

**Use Case**: Analyze customer reviews, social media feedback, or any text sentiment analysis application.

## Project Structure

```
sentiment-analysis-docker/
├── Dockerfile               # Multi-stage Docker build
├── docker-compose.yml       # Container orchestration
├── .dockerignore           # Docker build optimization
├── .gitignore              # Git ignore rules
├── requirements.txt        # Python dependencies
├── src/
│   ├── __init__.py
│   ├── app.py              # Flask API application
│   ├── data.py             # Data processing and text cleaning
│   ├── train.py            # Model training script
│   └── predict.py          # Prediction utilities
├── model/                  # Trained model artifacts
│   ├── sentiment_model.pkl
│   ├── vectorizer.pkl
│   └── model_metrics.json
├── test_docker_api.py      # Docker API testing
└── README.md              # This file
```

## Quick Start

### Prerequisites
- Docker Desktop installed ([Download](https://www.docker.com/products/docker-desktop))
- Git (for cloning)

### 1. Clone Repository

```bash
git clone https://github.com/jashshah127/sentiment-analysis-docker.git
cd sentiment-analysis-docker
```

### 2. Train Model Locally (First Time Only)

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Train model
cd src
python train.py
cd ..
```

Expected output:
- Training Accuracy: 100%
- Test Accuracy: 100%
- Creates 3 files in `model/` directory

### 3. Build and Run with Docker

```bash
# Build and start container
docker-compose up --build

# Or run in detached mode
docker-compose up -d --build
```

API will be available at: **http://localhost:5000**

### 4. Test the API

```bash
# Health check
curl http://localhost:5000/health

# Predict sentiment
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"review": "This movie was absolutely amazing!"}'
```

### 5. Stop the Container

```bash
# Stop and remove containers
docker-compose down
```

## API Endpoints

### Information Endpoints
- `GET /` - API information and available endpoints
- `GET /health` - Health check and model status

### Prediction Endpoint
- `POST /predict` - Analyze sentiment of text review

### Model Information
- `GET /model/info` - Model performance metrics

## Usage Examples

### Positive Review
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"review": "Absolutely fantastic film! Loved every minute of it."}'
```

**Response:**
```json
{
  "sentiment": "positive",
  "confidence": 0.95,
  "sentiment_score": "Very Positive",
  "probability_positive": 0.95,
  "probability_negative": 0.05
}
```

### Negative Review
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"review": "Terrible movie. Complete waste of time and money."}'
```

**Response:**
```json
{
  "sentiment": "negative",
  "confidence": 0.92,
  "sentiment_score": "Very Negative",
  "probability_positive": 0.08,
  "probability_negative": 0.92
}
```

## Model Performance

### Metrics
- **Training Accuracy**: 100%
- **Test Accuracy**: 100%
- **ROC-AUC Score**: 1.0
- **Cross-Validation**: 100% ± 0%
- **Model Type**: Logistic Regression
- **Vectorization**: TF-IDF with 5000 max features

### Top Sentiment Indicators

**Positive Words:**
- masterpiece, amazing, excellent, brilliant, fantastic, wonderful

**Negative Words:**
- terrible, awful, horrible, waste, disappointing, boring

## Docker Architecture

### Multi-Stage Build
The Dockerfile uses optimization techniques:
- **Stage 1**: Install dependencies and build tools
- **Stage 2**: Copy only necessary artifacts to slim production image

**Benefits:**
- Smaller image size (~400MB vs ~1GB)
- Faster deployments
- Security hardening
- Production-ready configuration

### Container Features
- **Base Image**: python:3.9-slim
- **Port**: 5000
- **Health Checks**: Automatic monitoring every 30s
- **Restart Policy**: Auto-restart on failure
- **NLTK Data**: Pre-downloaded stopwords and punkt tokenizer

## Key Differences from Original Lab

### Dataset & Model
- **Original**: Likely basic classification dataset
- **Modified**: NLP sentiment analysis with text processing

### Technology Stack
- **Original**: Basic Docker setup
- **Modified**: Multi-stage build, Flask API, NLP pipeline

### Features Added
- Text preprocessing with NLTK
- TF-IDF vectorization
- Sentiment scoring (Very Negative → Very Positive)
- Confidence scores
- Top positive/negative word analysis
- Health checks and monitoring
- Production-ready Docker configuration

## Docker Commands

### Build and Run
```bash
# Build image
docker build -t sentiment-api .

# Run container
docker run -d -p 5000:5000 --name sentiment-api sentiment-api

# View logs
docker logs -f sentiment-api

# Stop container
docker stop sentiment-api

# Remove container
docker rm sentiment-api
```

### Docker Compose
```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Rebuild
docker-compose up --build
```

### Debugging
```bash
# Access container shell
docker exec -it sentiment-analysis-api bash

# Check container status
docker ps

# View resource usage
docker stats sentiment-analysis-api

# Inspect container
docker inspect sentiment-analysis-api
```

## Project Requirements Met

**Assignment Checklist:**
- ✅ **Submit 1 lab**: Docker Lab completed
- ✅ **GitHub repository**: Complete with Dockerfile and docker-compose
- ✅ **Modifications made**:
  - Different dataset (NLP sentiment analysis vs original)
  - Different model (Logistic Regression with TF-IDF)
  - Multi-stage Docker build
  - Flask API integration
  - Text processing pipeline

## Technical Stack

- **Language**: Python 3.9
- **Framework**: Flask 3.0.0
- **ML Model**: Logistic Regression
- **NLP**: NLTK, TF-IDF Vectorization
- **Container**: Docker with multi-stage build
- **Orchestration**: Docker Compose
- **Server**: Flask development server (can use Gunicorn for production)

## Troubleshooting

### Model Not Found
```
ERROR: Model files not found
```
**Solution**: Train model first with `python src/train.py`

### Port Already in Use
```
ERROR: Port 5000 is already allocated
```
**Solution**: Stop other services or change port in docker-compose.yml

### Container Keeps Restarting
```bash
# Check logs
docker-compose logs

# Common issue: Model files not copied
# Solution: Verify model/ directory has files before building
```

## Development Workflow

1. **Modify code** locally in `src/`
2. **Train model** if data/model changes: `python src/train.py`
3. **Test locally** (optional): `python src/app.py`
4. **Rebuild Docker**: `docker-compose up --build`
5. **Test Dockerized API**: Run test script or manual testing
6. **Commit and push** to GitHub

## Deployment Options

### Local Development
```bash
docker-compose up
```

### Production Deployment
- **Cloud Run**: Google Cloud Run with container
- **AWS ECS**: Elastic Container Service
- **Azure Container Instances**: ACI deployment
- **Kubernetes**: Deploy with K8s manifests

## Security Considerations

- Container runs as root (can be improved with non-root user)
- No authentication (add JWT for production)
- Health checks enabled
- No secrets in image
- Minimal base image for reduced attack surface

## Future Enhancements

- Add user authentication
- Implement request rate limiting
- Add model versioning
- Integrate with model monitoring tools
- Add batch prediction endpoint
- Implement caching for repeated predictions
- Add more sophisticated NLP preprocessing

## License

Educational project for MLOps course at Northeastern University.

## Author

**Jash Shah**
- GitHub: [@jashshah127](https://github.com/jashshah127)
- Course: MLOps - Northeastern University
- Date: October 21, 2025

---

**Repository**: https://github.com/jashshah127/sentiment-analysis-docker