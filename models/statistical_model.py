# models/statistical_model.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import re
from collections import Counter

class StatisticalChecker:
    def __init__(self):
        # Initialize multiple vectorizers for different features
        self.word_vectorizer = TfidfVectorizer(ngram_range=(1, 2), analyzer='word')
        self.char_vectorizer = TfidfVectorizer(ngram_range=(2, 4), analyzer='char')
        
        # Initialize multiple models
        self.spelling_model = MultinomialNB()
        self.grammar_model = RandomForestClassifier(n_estimators=100)
        
        # Training data - Expanded dataset
        self.train_texts = [
            "நான் பள்ளிக்கு செல்கிறேன்",
            "நாங்கள் பள்ளிக்கு செல்கிறோம்",
            "அவன் பள்ளிக்கு செல்கிறான்",
            "அவள் பள்ளிக்கு செல்கிறாள்",
            "நான் வீட்டிற்கு செல்கிறேன்",
            "நாங்கள் கடைக்கு செல்கிறோம்"
        ]
        
        # Labels for different error types
        self.spelling_labels = [1, 1, 1, 1, 1, 1]  # 1 for correct, 0 for incorrect
        self.grammar_labels = [1, 1, 1, 1, 1, 1]
        
        # Train the models
        self._train_models()
        
        # Error patterns for post-processing
        self.error_patterns = {
            'common_errors': {
                r'\w+கிறான்\s+\w+கிறாள்': 'Inconsistent gender agreement',
                r'\w+கிறேன்\s+\w+கிறார்': 'Inconsistent honorific usage',
                r'[அ-ஔ]\s+[ா-ௌ]': 'Invalid vowel mark placement'
            },
            'context_rules': [
                (r'நான்.*கிறார்கள்', 'subject_verb_mismatch'),
                (r'நாங்கள்.*கிறான்', 'number_mismatch')
            ]
        }

    def _train_models(self):
        # Prepare features
        word_features = self.word_vectorizer.fit_transform(self.train_texts)
        char_features = self.char_vectorizer.fit_transform(self.train_texts)
        
        # Combine features
        combined_features = np.hstack([
            word_features.toarray(),
            char_features.toarray()
        ])
        
        # Train models
        self.spelling_model.fit(combined_features, self.spelling_labels)
        self.grammar_model.fit(combined_features, self.grammar_labels)

    def _extract_features(self, text):
        # Extract word and character features
        word_feats = self.word_vectorizer.transform([text])
        char_feats = self.char_vectorizer.transform([text])
        
        # Combine features
        return np.hstack([
            word_feats.toarray(),
            char_feats.toarray()
        ])

    def _analyze_patterns(self, text):
        errors = []
        # Check common error patterns
        for pattern, msg in self.error_patterns['common_errors'].items():
            if re.search(pattern, text):
                errors.append(('statistical', f'Pattern error: {msg}', text))
        
        # Check context rules
        for pattern, error_type in self.error_patterns['context_rules']:
            if re.search(pattern, text):
                errors.append(('statistical', f'Context error: {error_type}', text))
        
        return errors

    def check_text(self, text):
        try:
            features = self._extract_features(text)
            
            # Get model predictions
            spelling_pred = self.spelling_model.predict_proba(features)[0]
            grammar_pred = self.grammar_model.predict_proba(features)[0]
            
            errors = []
            
            # Check spelling confidence
            if spelling_pred[0] < 0.8:  # Less than 80% confidence for correct spelling
                errors.append(('statistical', f'Possible spelling errors (confidence: {spelling_pred[1]:.2%})', text))
            
            # Check grammar confidence
            if grammar_pred[0] < 0.8:  # Less than 80% confidence for correct grammar
                errors.append(('statistical', f'Possible grammar errors (confidence: {grammar_pred[1]:.2%})', text))
            
            # Add pattern-based errors
            pattern_errors = self._analyze_patterns(text)
            errors.extend(pattern_errors)
            
            return errors
        except Exception as e:
            return [('error', str(e), text)]