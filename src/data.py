"""
Data loading and preprocessing for Sentiment Analysis
"""
import pandas as pd
import numpy as np
import re
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import logging

# Download NLTK data
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
except:
    pass

from nltk.corpus import stopwords

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentDataProcessor:
    """Class to handle sentiment analysis data processing"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.stop_words = set(stopwords.words('english'))
    
    def create_synthetic_data(self, n_samples=2000):
        """
        Create synthetic movie review dataset
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            DataFrame with reviews and sentiments
        """
        logger.info(f"Creating synthetic movie review dataset with {n_samples} samples...")
        np.random.seed(42)
        
        # Positive review templates
        positive_words = [
            'excellent', 'amazing', 'wonderful', 'fantastic', 'brilliant',
            'outstanding', 'superb', 'great', 'loved', 'perfect',
            'incredible', 'masterpiece', 'enjoyed', 'best', 'awesome'
        ]
        
        positive_phrases = [
            'This movie was {}!',
            'Absolutely {} film!',
            'I {} this movie!',
            'One of the {} movies ever!',
            'Such a {} experience!',
            'The acting was {}!',
            'A {} masterpiece!',
            'Highly recommend, {}!'
        ]
        
        # Negative review templates
        negative_words = [
            'terrible', 'awful', 'horrible', 'disappointing', 'boring',
            'waste', 'poor', 'bad', 'worst', 'dull',
            'painful', 'mediocre', 'forgettable', 'tedious', 'unwatchable'
        ]
        
        negative_phrases = [
            'This movie was {}.',
            'Absolutely {} film.',
            'I hated this {} movie.',
            'One of the {} movies ever.',
            'Such a {} waste of time.',
            'The acting was {}.',
            'A {} disaster.',
            'Do not recommend, {}.'
        ]
        
        reviews = []
        sentiments = []
        
        # Generate positive reviews (50%)
        for _ in range(n_samples // 2):
            phrase = np.random.choice(positive_phrases)
            word = np.random.choice(positive_words)
            review = phrase.format(word)
            # Add some variety
            if np.random.random() > 0.5:
                extra = np.random.choice(positive_words)
                review += f" Really {extra}!"
            reviews.append(review)
            sentiments.append(1)  # Positive
        
        # Generate negative reviews (50%)
        for _ in range(n_samples // 2):
            phrase = np.random.choice(negative_phrases)
            word = np.random.choice(negative_words)
            review = phrase.format(word)
            # Add some variety
            if np.random.random() > 0.5:
                extra = np.random.choice(negative_words)
                review += f" Completely {extra}."
            reviews.append(review)
            sentiments.append(0)  # Negative
        
        df = pd.DataFrame({
            'review': reviews,
            'sentiment': sentiments
        })
        
        # Shuffle
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        logger.info(f"Dataset created with {len(df)} samples")
        logger.info(f"Positive reviews: {(df['sentiment'] == 1).sum()}")
        logger.info(f"Negative reviews: {(df['sentiment'] == 0).sum()}")
        
        return df
    
    def clean_text(self, text):
        """
        Clean and preprocess text
        
        Args:
            text: Raw text string
            
        Returns:
            Cleaned text string
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def prepare_data(self, test_size=0.2, random_state=42):
        """
        Prepare data for training
        
        Args:
            test_size: Proportion for test set
            random_state: Random seed
            
        Returns:
            Dictionary with train/test splits and vectorizer
        """
        # Load data
        df = self.create_synthetic_data()
        
        # Clean text
        logger.info("Cleaning text data...")
        df['cleaned_review'] = df['review'].apply(self.clean_text)
        
        # Split data
        X_train_text, X_test_text, y_train, y_test = train_test_split(
            df['cleaned_review'], 
            df['sentiment'],
            test_size=test_size,
            random_state=random_state,
            stratify=df['sentiment']
        )
        
        # Vectorize text
        logger.info("Vectorizing text with TF-IDF...")
        X_train = self.vectorizer.fit_transform(X_train_text)
        X_test = self.vectorizer.transform(X_test_text)
        
        logger.info("Data preprocessing completed")
        logger.info(f"Training set: {X_train.shape}")
        logger.info(f"Test set: {X_test.shape}")
        logger.info(f"Vocabulary size: {len(self.vectorizer.vocabulary_)}")
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'vectorizer': self.vectorizer
        }