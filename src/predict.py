"""
Prediction utilities for Sentiment Analysis
"""
import joblib
import json
import os
import re
from typing import Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentPredictor:
    """Class to handle sentiment predictions"""
    
    def __init__(self, model_dir="model"):
        self.model_dir = model_dir
        self.model = None
        self.vectorizer = None
        self.model_metrics = None
        self._load_model_components()
    
    def _load_model_components(self):
        """Load model, vectorizer, and metrics"""
        try:
            # Load model
            model_path = os.path.join(self.model_dir, "sentiment_model.pkl")
            self.model = joblib.load(model_path)
            logger.info("Model loaded successfully")
            
            # Load vectorizer
            vectorizer_path = os.path.join(self.model_dir, "vectorizer.pkl")
            self.vectorizer = joblib.load(vectorizer_path)
            logger.info("Vectorizer loaded successfully")
            
            # Load metrics
            metrics_path = os.path.join(self.model_dir, "model_metrics.json")
            with open(metrics_path, 'r') as f:
                self.model_metrics = json.load(f)
            logger.info("Model metrics loaded successfully")
            
        except FileNotFoundError as e:
            logger.error(f"Model file not found: {e}")
            raise FileNotFoundError(
                "Model files not found. Please train the model first by running train.py"
            )
        except Exception as e:
            logger.error(f"Error loading model components: {e}")
            raise
    
    def clean_text(self, text):
        """Clean text for prediction"""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = ' '.join(text.split())
        return text
    
    def predict_single(self, review_text: str) -> Dict:
        """
        Predict sentiment for a single review
        
        Args:
            review_text: Raw review text
            
        Returns:
            Dictionary with prediction results
        """
        # Clean text
        cleaned_text = self.clean_text(review_text)
        
        # Vectorize
        X = self.vectorizer.transform([cleaned_text])
        
        # Predict
        prediction = self.model.predict(X)[0]
        probability = self.model.predict_proba(X)[0]
        
        # Prepare result
        result = {
            'original_text': review_text,
            'cleaned_text': cleaned_text,
            'sentiment': 'positive' if prediction == 1 else 'negative',
            'confidence': float(max(probability)),
            'probability_negative': float(probability[0]),
            'probability_positive': float(probability[1]),
            'sentiment_score': self._get_sentiment_score(probability[1])
        }
        
        return result
    
    def _get_sentiment_score(self, positive_prob):
        """Convert probability to sentiment score"""
        if positive_prob < 0.3:
            return "Very Negative"
        elif positive_prob < 0.45:
            return "Negative"
        elif positive_prob < 0.55:
            return "Neutral"
        elif positive_prob < 0.7:
            return "Positive"
        else:
            return "Very Positive"
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        return {
            'model_type': self.model_metrics.get('model_type'),
            'training_accuracy': self.model_metrics.get('train_accuracy'),
            'test_accuracy': self.model_metrics.get('test_accuracy'),
            'roc_auc_score': self.model_metrics.get('roc_auc_score'),
            'cv_score': f"{self.model_metrics.get('cv_mean', 0):.4f} ± {self.model_metrics.get('cv_std', 0):.4f}",
            'training_date': self.model_metrics.get('training_date'),
            'vocabulary_size': self.model_metrics.get('vocabulary_size'),
            'model_parameters': self.model_metrics.get('model_parameters', {})
        }

# Global predictor instance
_predictor_instance = None

def get_predictor() -> SentimentPredictor:
    """Get or create predictor instance"""
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = SentimentPredictor()
    return _predictor_instance