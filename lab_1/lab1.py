import re
from tqdm import tqdm
import nltk
from nltk.tokenize import word_tokenize
import pymorphy3

# Uncomment the following line if running for the first time
# nltk.download('punkt')

def process_text(file_path):
    # Reading the file content
    with open(file_path, 'r', encoding='utf-8') as file:
        text_content = file.read()
    
    # Tokenizing the text into individual words
    tokens = word_tokenize(text_content)

    # Initializing the morphological analyzer
    morph_analyzer = pymorphy3.MorphAnalyzer()

    # Initialize variables for result collection and word tracking
    matched_pairs = []
    previous_word = {'word': '', 'POS': '', 'gender': '', 'number': '', 'case': '', 'index': -2}

    # Loop through each word token with progress bar
    for i in tqdm(range(len(tokens))):

        current_word = morph_analyzer.parse(tokens[i])[0]  # Get the morphological parsing of the word
        current_tag = current_word.tag
        
        # Check if the word is either a noun or an adjective
        if current_tag.POS in ['NOUN', 'ADJF']:
            # Check for agreement in gender, number, and case between consecutive words
            if (i - previous_word['index'] == 1 and previous_word['gender'] == current_tag.gender and
                previous_word['number'] == current_tag.number and previous_word['case'] == current_tag.case and
                previous_word['POS'] != current_tag.POS):
                
                # Append the matching word pair to results
                matched_pairs.append(f"{previous_word['word']} {current_word.normal_form}")
            
            # Update the details of the current word for future comparison
            previous_word.update({
                'word': current_word.normal_form,
                'POS': current_tag.POS,
                'gender': current_tag.gender,
                'number': current_tag.number,
                'case': current_tag.case,
                'index': i
            })

    # Remove duplicate pairs
    unique_pairs = list(set(matched_pairs))

    # Return the result
    return unique_pairs

def main():
    file_path = 'D:\\ИПМКН\\Семестр_7\\Обработка_естественного_языка\\NLP\\lab_1\\file_text.txt'  # You can change this path if needed
    results = process_text(file_path)

    # Output the final results
    print('\nFound matching word pairs:')
    for pair in results:
        print(pair)

if __name__ == "__main__":
    main()
