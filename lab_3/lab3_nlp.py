import re
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from rnnmorph.predictor import RNNMorphPredictor
import nltk


def extract_tag_attributes(tag_string):
    """Extract gender, number, and case from the tag string."""
    gender = 'masc' if 'm' in tag_string else 'fem' if 'f' in tag_string else 'neut' if 'n' in tag_string else None
    number = 'sing' if 'sg' in tag_string else 'plur' if 'pl' in tag_string else None
    case = 'nomn' if 'nom' in tag_string else 'gent' if 'gen' in tag_string else 'datv' if 'dat' in tag_string else \
           'accs' if 'acc' in tag_string else 'ablt' if 'abl' in tag_string else 'loct' if 'loc' in tag_string else None
    return gender, number, case

def process_text(file_path):
    # Read the file content
    with open(file_path, 'r', encoding='utf-8') as file:
        text_content = file.read()
    
    # Tokenize the text into individual words
    tokens = word_tokenize(text_content)
    
    # Initialize the RNNMorph predictor
    predictor = RNNMorphPredictor(language="ru")
    
    # Initialize variables for result collection and word tracking
    matched_pairs = []
    previous_word = {'word': '', 'POS': '', 'gender': '', 'number': '', 'case': '', 'index': -2}

    # Loop through each word token with progress bar
    for i in tqdm(range(len(tokens))):
        # Predict the morphological attributes of the current word
        parsed = predictor.predict([tokens[i]])[0]
        
        # Extract attributes from the tag string
        gender, number, case = extract_tag_attributes(parsed.tag)
        
        # Check if the word is either a noun or an adjective
        if parsed.pos in ['NOUN', 'ADJ']:
            # Check for agreement in gender, number, and case between consecutive words
            if (i - previous_word['index'] == 1 and
                previous_word['gender'] == gender and
                previous_word['number'] == number and
                previous_word['case'] == case and
                previous_word['POS'] != parsed.pos):
                
                # Append the matching word pair to results
                matched_pairs.append(f"{previous_word['word']} {parsed.normal_form}")
            
            # Update the details of the current word for future comparison
            previous_word.update({
                'word': parsed.normal_form,
                'POS': parsed.pos,
                'gender': gender,
                'number': number,
                'case': case,
                'index': i
            })

    # Remove duplicate pairs
    unique_pairs = list(set(matched_pairs))

    # Return the result
    return unique_pairs

def main():
    # file_path = 'D:\\ИПМКН\\Семестр_7\\Обработка_естественного_языка\\NLP\\lab_1\\file_text.txt' 
    file_path = 'file_text.txt' 
    results = process_text(file_path)

    # Output the final results
    print('\nFound matching word pairs:')
    for pair in results:
        print(pair)

if __name__ == "__main__":
    main()
    
