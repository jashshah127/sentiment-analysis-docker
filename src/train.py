"""
Model training for Sentiment Analysis
"""
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import cross_val_score
import logging
import json
from datetime import datetime
from data import SentimentDataProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentModelTrainer:
    """Class to handle model training and evaluation"""
    
    def __init__(self, model_dir="../model"):
        self.model_dir = model_dir
        self.model = None
        self.vectorizer = None
        self.model_metrics = {}
        
        os.makedirs(self.model_dir, exist_ok=True)
    
    def train_model(self, C=1.0, max_iter=1000, random_state=42):
        """
        Train the sentiment analysis model
        
        Args:
            C: Inverse of regularization strength
            max_iter: Maximum iterations
            random_state: Random seed
        """
        logger.info("Starting model training...")
        
        # Prepare data
        data_processor = SentimentDataProcessor()
        data = data_processor.prepare_data()
        
        # Extract components
        X_train = data['X_train']
        X_test = data['X_test']
        y_train = data['y_train']
        y_test = data['y_test']
        self.vectorizer = data['vectorizer']
        
        # Initialize and train model
        self.model = LogisticRegression(
            C=C,
            max_iter=max_iter,
            random_state=random_state,
            solver='liblinear'
        )
        
        logger.info("Training Logistic Regression model...")
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        self._evaluate_model(X_train, X_test, y_train, y_test)
        
        logger.info("Model training completed!")
    
    def _evaluate_model(self, X_train, X_test, y_train, y_test):
        """Evaluate model performance"""
        
        # Predictions
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        
        # Probabilities for ROC-AUC
        y_test_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        roc_auc = roc_auc_score(y_test, y_test_proba)
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5)
        
        # Store metrics
        self.model_metrics = {
            'train_accuracy': float(train_accuracy),
            'test_accuracy': float(test_accuracy),
            'roc_auc_score': float(roc_auc),
            'cv_mean': float(cv_scores.mean()),
            'cv_std': float(cv_scores.std()),
            'vocabulary_size': len(self.vectorizer.vocabulary_),
            'training_date': datetime.now().isoformat(),
            'model_type': 'LogisticRegression',
            'model_parameters': {
                'C': self.model.C,
                'max_iter': self.model.max_iter,
                'solver': self.model.solver
            }
        }
        
        # Print results
        logger.info(f"Training Accuracy: {train_accuracy:.4f}")
        logger.info(f"Test Accuracy: {test_accuracy:.4f}")
        logger.info(f"ROC-AUC Score: {roc_auc:.4f}")
        logger.info(f"CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_test_pred,
                                  target_names=['Negative', 'Positive']))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_test_pred)
        print("\nConfusion Matrix:")
        print(f"True Negatives: {cm[0][0]}, False Positives: {cm[0][1]}")
        print(f"False Negatives: {cm[1][0]}, True Positives: {cm[1][1]}")
        
        # Top positive and negative words
        feature_names = self.vectorizer.get_feature_names_out()
        top_positive = sorted(zip(self.model.coef_[0], feature_names), reverse=True)[:10]
        top_negative = sorted(zip(self.model.coef_[0], feature_names))[:10]
        
        print("\nTop 10 Positive Words:")
        for coef, word in top_positive:
            print(f"  {word}: {coef:.4f}")
        
        print("\nTop 10 Negative Words:")
        for coef, word in top_negative:
            print(f"  {word}: {coef:.4f}")
    
    def save_model(self):
        """Save trained model, vectorizer, and metrics"""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        # Save model
        model_path = os.path.join(self.model_dir, "sentiment_model.pkl")
        joblib.dump(self.model, model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Save vectorizer
        vectorizer_path = os.path.join(self.model_dir, "vectorizer.pkl")
        joblib.dump(self.vectorizer, vectorizer_path)
        logger.info(f"Vectorizer saved to {vectorizer_path}")
        
        # Save metrics
        metrics_path = os.path.join(self.model_dir, "model_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(self.model_metrics, f, indent=2)
        logger.info(f"Metrics saved to {metrics_path}")

def main():
    """Main training function"""
    trainer = SentimentModelTrainer()
    trainer.train_model(C=1.0, max_iter=1000, random_state=42)
    trainer.save_model()
    
    print("\n" + "="*50)
    print("Model training completed successfully!")
    print("Files saved in 'model/' directory:")
    print("- sentiment_model.pkl")
    print("- vectorizer.pkl")
    print("- model_metrics.json")
    print("="*50)

if __name__ == "__main__":
    main()