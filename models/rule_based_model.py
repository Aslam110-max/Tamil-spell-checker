import re
from indicnlp.tokenize.indic_tokenize import trivial_tokenize
from collections import defaultdict

class RuleBasedChecker:
    def __init__(self):
        # Load Tamil dictionary
        self.tamil_words = self._load_tamil_dictionary()
        self.grammar_rules = {
            'subject_verb_agreement': [
                (r'நான்.*கிறார்கள்', 'First person singular with plural verb'),
                (r'நான்.*கிறார்', 'First person singular with formal verb'),
                (r'நான்.*கிறது', 'First person with neuter verb'),
                (r'நாங்கள்.*கிறான்', 'First person plural with singular verb'),
                (r'நாங்கள்.*கிறாள்', 'First person plural with feminine singular'),
                (r'நீ.*கிறார்கள்', 'Second person singular with plural verb'),
                (r'நீங்கள்.*கிறான்', 'Second person plural with singular verb'),
                (r'ஆசிரியர்.*கிறான்', 'Honorific subject with informal verb')
            ],
            'spelling_patterns': [
                (r'சல்', 'Possible misspelling of செல்'),
                (r'பதில்', 'Possible misspelling of பதிவு'),
                (r'எங்க', 'Possible misspelling of எங்கே')
            ],
            'word_spacing': [
                (r'\w+க்கு\w+', 'Missing space before க்கு'),
                (r'\w+யில்\w+', 'Missing space before யில்'),
                (r'\w+உடன்\w+', 'Missing space before உடன்')
            ]
        }

    def _load_tamil_dictionary(self) -> dict:
        """
        Load Tamil words from a dictionary file or use a basic predefined dictionary.
        
        Returns:
            dict: A dictionary of Tamil words with their parts of speech.
        """
        basic_dictionary = {
            'நான்': 'pronoun',
            'நீ': 'pronoun',
            'நாங்கள்': 'pronoun',
            'பள்ளி': 'noun',
            'பள்ளிக்கு': 'noun',
            'செல்கிறேன்': 'verb',
            'செல்கிறான்': 'verb',
            'செல்கிறாள்': 'verb',
            'செல்கிறது': 'verb',
            'படிக்கிறோம்': 'verb',
            'பாடல்': 'noun',
            'பாடுகிறேன்': 'verb',
            'ஆசிரியர்': 'noun',
            'நல்ல': 'adjective',
            'பாடங்களை': 'noun',
            'கற்றுக்': 'verb',
            'கொடுக்கிறார்': 'verb'
        }
        
        try:
            # Load dictionary from file if available
            with open("data/tamil_dictionary.txt", "r", encoding="utf-8") as file:
                for line in file:
                    if line.strip():
                        parts = line.strip().split(',')
                        if len(parts) >= 2:
                            basic_dictionary[parts[0]] = parts[1]
        except FileNotFoundError:
            pass  # Use the basic dictionary if file is not found
        
        return basic_dictionary

    def split_sentences(self, text: str) -> list:
       
        sentences = re.split(r'[.!?।]', text)
        return [s.strip() for s in sentences if s.strip()]

    def check_spelling(self, text: str) -> list:
        
        errors = []
        words = trivial_tokenize(text)
        
        for word in words:
            # Check spelling patterns
            for pattern, msg in self.grammar_rules['spelling_patterns']:
                if re.match(pattern, word):
                    errors.append(('spelling', msg, word))
            
            # Check against dictionary
            if word not in self.tamil_words and not any(char.isdigit() for char in word):
                errors.append(('spelling', f'Unknown word: {word}', word))
            
            # Check word spacing
            for pattern, msg in self.grammar_rules['word_spacing']:
                if re.match(pattern, word):
                    errors.append(('spelling', msg, word))
        
        return errors

    def check_grammar(self, text: str) -> list:
        """
        Check for grammatical errors in the text.
        
        Args:
            text (str): T  he text to check.
        
        Returns:
            list: A list of grammatical errors.
        """
        errors = []
        sentences = self.split_sentences(text)
        
        for sentence in sentences:
            # Check subject-verb agreement
            for pattern, error_msg in self.grammar_rules['subject_verb_agreement']:
                if re.search(pattern, sentence):
                    errors.append(('grammar', error_msg, sentence))
        
        return errors

    def check_text(self, text: str) -> list:
        """
        Check the text for both spelling and grammar errors.
        
        Args:
            text (str): The text to check.
        
        Returns:
            list: A list of errors found in the text.
        """
        try:
            # Perform spelling checks
            spelling_errors = self.check_spelling(text)
            
            # Perform grammar checks
            grammar_errors = self.check_grammar(text)
            
            # Combine all errors
            all_errors = spelling_errors + grammar_errors
            
            return all_errors if all_errors else []
        
        except Exception as e:
            return [('error', f'Error in text analysis: {str(e)}', text)]
