# models/deep_learning_model.py
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
import warnings
from typing import List, Tuple
import re

class DeepLearningChecker:
    def __init__(self):
        # Suppress warnings during model initialization
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            
            # Initialize tokenizer and model for MLM
            try:
                self.tokenizer = AutoTokenizer.from_pretrained("ai4bharat/IndicBERTv2-MLM-only")
                self.model = AutoModelForMaskedLM.from_pretrained("ai4bharat/IndicBERTv2-MLM-only")
                self.model.eval()  # Set to evaluation mode
            except Exception as e:
                print(f"Model initialization error: {str(e)}")
                self.tokenizer = None
                self.model = None
            
            # Load common Tamil patterns
            self._load_tamil_patterns()
            
            # Load Tamil word dictionary
            self._load_tamil_dictionary()

    def _load_tamil_dictionary(self):
        """Load basic Tamil dictionary"""
        self.tamil_dict = {
            'நான்': 'pronoun',
            'நீ': 'pronoun',
            'அவன்': 'pronoun',
            'அவள்': 'pronoun',
            'அது': 'pronoun',
            'நாங்கள்': 'pronoun',
            'செல்': 'verb_root',
            'படி': 'verb_root',
            'கிறேன்': 'verb_suffix',
            'கிறான்': 'verb_suffix',
            'கிறாள்': 'verb_suffix',
            'கிறது': 'verb_suffix',
            'கிறோம்': 'verb_suffix',
            'பள்ளி': 'noun',
            'வீடு': 'noun'
        }

    def _load_tamil_patterns(self):
        """Load common Tamil language patterns"""
        self.tamil_patterns = {
            'grammar': [
                (r'நான்.*கிறது', 'First person with neuter verb - use கிறேன் instead'),
                (r'நாங்கள்.*கிறான்', 'Plural subject with singular verb - use கிறோம் instead'),
                (r'அவர்.*கிறேன்', 'Third person honorific with first person verb - use கிறார் instead'),
                (r'நீங்கள்.*கிறான்', 'Honorific subject with informal verb - use கிறீர்கள் instead')
            ],
            'spelling': [
                (r'சல்', 'செல்'),
                (r'பதில்', 'பதிவு'),
                (r'எங்க', 'எங்கே')
            ],
            'spacing': [
                (r'\w+க்கு\w+', 'Add space before க்கு'),
                (r'\w+யில்\w+', 'Add space before யில்'),
                (r'\w+உடன்\w+', 'Add space before உடன்')
            ]
        }

    def _assess_word_probability(self, text: str) -> List[Tuple[str, float]]:
        """Assess the probability of each word using MLM"""
        if self.model is None or self.tokenizer is None:
            return []

        try:
            words = text.split()
            word_scores = []

            for i, word in enumerate(words):
                # Create input with current word masked
                masked_text = ' '.join(words[:i] + ['[MASK]'] + words[i+1:])
                inputs = self.tokenizer(masked_text, return_tensors='pt', padding=True, truncation=True)
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    predictions = outputs.logits
                    
                # Get the probability of the actual word
                masked_index = (inputs.input_ids == self.tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
                if len(masked_index) > 0:
                    word_id = self.tokenizer.convert_tokens_to_ids(word)
                    word_prob = torch.softmax(predictions[0, masked_index], dim=-1)[0, word_id].item()
                    word_scores.append((word, word_prob))

            return word_scores
        except Exception as e:
            print(f"Word probability assessment error: {str(e)}")
            return []

    def _check_patterns(self, text: str) -> List[Tuple[str, str, str]]:
        """Check text against Tamil patterns"""
        errors = []
        
        for error_type, patterns in self.tamil_patterns.items():
            for pattern, correction in patterns:
                matches = re.finditer(pattern, text)
                for match in matches:
                    if error_type == 'spelling':
                        errors.append(('spelling', f'Suggestion: Replace "{match.group()}" with "{correction}"', text))
                    elif error_type == 'grammar':
                        errors.append(('grammar', correction, text))
                    else:
                        errors.append(('format', correction, text))
        
        return errors

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        return [s.strip() for s in re.split('[.!?।]', text) if s.strip()]

    def check_text(self, text: str) -> List[Tuple[str, str, str]]:
        """Check text for errors using MLM and pattern matching"""
        try:
            errors = []
            sentences = self._split_sentences(text)
            
            for sentence in sentences:
                # Check using patterns
                pattern_errors = self._check_patterns(sentence)
                errors.extend(pattern_errors)
                
                # Check word probabilities if model is available
                word_scores = self._assess_word_probability(sentence)
                for word, prob in word_scores:
                    if prob < 0.1:  # Low probability threshold
                        errors.append(('spelling', f'Unusual word detected: "{word}" (confidence: {prob:.2%})', sentence))
            
            return errors if errors else []
            
        except Exception as e:
            return [('error', f'Error in analysis: {str(e)}', text)]

    def get_correction_suggestions(self, text: str) -> List[str]:
        """Get correction suggestions for the text"""
        suggestions = []
        try:
            sentences = self._split_sentences(text)
            
            for sentence in sentences:
                # Get pattern-based suggestions
                for error_type, patterns in self.tamil_patterns.items():
                    for pattern, correction in patterns:
                        if re.search(pattern, sentence):
                            if error_type == 'spelling':
                                suggestions.append(f"Spelling suggestion: Replace '{pattern}' with '{correction}'")
                            elif error_type == 'grammar':
                                suggestions.append(f"Grammar suggestion: {correction}")
                            else:
                                suggestions.append(f"Format suggestion: {correction}")
                
                # Add MLM-based suggestions if available
                word_scores = self._assess_word_probability(sentence)
                for word, prob in word_scores:
                    if prob < 0.1:
                        suggestions.append(f"Unusual word detected: '{word}' might need review")
            
            return suggestions if suggestions else ["No specific corrections needed."]
            
        except Exception as e:
            return [f"Error generating suggestions: {str(e)}"]