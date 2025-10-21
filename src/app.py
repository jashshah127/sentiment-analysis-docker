"""
Flask API for Sentiment Analysis
"""
from flask import Flask, request, jsonify
import logging
from datetime import datetime
from predict import get_predictor, SentimentPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global predictor
predictor = None

# Load model on startup
try:
    predictor = get_predictor()
    logger.info("Model loaded successfully on startup")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    predictor = None

@app.route('/', methods=['GET'])
def home():
    """Root endpoint"""
    return jsonify({
        'message': 'Welcome to the Sentiment Analysis API',
        'version': '1.0.0',
        'framework': 'Flask',
        'model': 'Logistic Regression with TF-IDF',
        'use_case': 'Movie Review Sentiment Analysis',
        'endpoints': {
            '/': 'API information',
            '/health': 'Health check',
            '/predict': 'Analyze sentiment (POST)',
            '/model/info': 'Model information'
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy' if predictor is not None else 'unhealthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': predictor is not None,
        'api_version': '1.0.0'
    })

@app.route('/predict', methods=['POST'])
def predict_sentiment():
    """
    Analyze sentiment of movie review
    
    Expected JSON:
    {
        "review": "This movie was amazing!"
    }
    """
    if predictor is None:
        return jsonify({'error': 'Model not loaded'}), 503
    
    try:
        data = request.get_json()
        
        if not data or 'review' not in data:
            return jsonify({'error': 'Missing "review" field in request'}), 400
        
        review_text = data['review']
        
        if not review_text or not isinstance(review_text, str):
            return jsonify({'error': 'Review must be a non-empty string'}), 400
        
        # Make prediction
        result = predictor.predict_single(review_text)
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/model/info', methods=['GET'])
def get_model_info():
    """Get model information"""
    if predictor is None:
        return jsonify({'error': 'Model not loaded'}), 503
    
    try:
        info = predictor.get_model_info()
        return jsonify(info), 200
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)