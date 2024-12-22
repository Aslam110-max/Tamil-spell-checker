# test_samples.py

test_cases = [
    {
        "title": "Spelling Errors Sample",
        "text": "நான பள்ளிக்கு சென்றேன்",  # நான் is correct
        "expected_correction": "நான் பள்ளிக்கு சென்றேன்"
    },
    {
        "title": "Vowel Mark Error",
        "text": "புத்தகத்தெ படித்தேன்",  # புத்தகத்தை is correct
        "expected_correction": "புத்தகத்தை படித்தேன்"
    },
    {
        "title": "Grammar Error - Subject-Verb Agreement",
        "text": "அவர்கள் வருகிறான்",  # வருகிறார்கள் is correct
        "expected_correction": "அவர்கள் வருகிறார்கள்"
    },
    {
        "title": "Mixed Errors",
        "text": "நாங்கள் சினிமாவுக்கு போனோம்",  # திரைப்படத்திற்கு is more formal
        "expected_correction": "நாங்கள் திரைப்படத்திற்கு போனோம்"
    },
    {
        "title": "Complex Sentence",
        "text": "அவன் நல்ல புத்தகத்தெ எழுதினான் ஆனா யாரும் படிக்கவில்லை",
        "expected_correction": "அவன் நல்ல புத்தகத்தை எழுதினான் ஆனால் யாரும் படிக்கவில்லை"
    }
]

# Test function
def run_tests(checker):
    print("Running Tamil Text Checker Tests\n")
    for i, test in enumerate(test_cases, 1):
        print(f"Test Case {i}: {test['title']}")
        print(f"Input Text: {test['text']}")
        
        # Get corrections
        spell_corrections = checker.spell_check(test['text'])
        grammar_errors = checker.check_grammar(test['text'])
        
        print("\nSpelling Corrections:")
        if spell_corrections:
            for original, correction in spell_corrections:
                print(f"- {original} → {correction}")
        else:
            print("No spelling errors found")
            
        print("\nGrammar Suggestions:")
        if grammar_errors:
            for error in grammar_errors:
                print(f"- Type: {error['type']}")
                print(f"  Original: {error['original']}")
                print(f"  Suggestion: {error['suggestion']}")
        else:
            print("No grammar errors found")
            
        print("\nExpected Output:", test['expected_correction'])
        print("-" * 50 + "\n")

if __name__ == "__main__":
    from main import TamilChecker
    checker = TamilChecker()
    run_tests(checker)