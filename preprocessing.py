import os
import re
from tamil import utf8

# Path to the extracted Wiki files
extracted_folder = "extracted"

# Output file for cleaned Tamil corpus
output_file = "tamil_corpus.txt"

def is_tamil_word(word):
    """Check if a word contains Tamil characters."""
    tamil_regex = r"[\u0B80-\u0BFF]+"
    return re.search(tamil_regex, word)

def preprocess_file(file_path):
    """Read and preprocess a single file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    # Tokenize and filter Tamil words
    words = utf8.get_words(content)
    tamil_words = [word for word in words if is_tamil_word(word)]
    return tamil_words

def preprocess_corpus(folder_path):
    """Process all files in the extracted folder."""
    tamil_words = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            tamil_words.extend(preprocess_file(file_path))
    return tamil_words

# Preprocess the extracted folder
tamil_words = preprocess_corpus(extracted_folder)

# Save the cleaned corpus
with open(output_file, 'w', encoding='utf-8') as f:
    f.write('\n'.join(tamil_words))

print(f"Cleaned Tamil corpus saved to {output_file}")
