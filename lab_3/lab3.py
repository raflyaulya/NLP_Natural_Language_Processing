import re
from tqdm import tqdm
import nltk
from nltk.tokenize import word_tokenize
from rnnmorph.predictor import RNNMorphPredictor
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # This will only show errors



# Uncomment the following line if running for the first time
# nltk.download('punkt')

def process_text(file_path):
    # Reading the file content
    with open(file_path, 'r', encoding='utf-8') as file:
        text_content = file.read()
    
    # Tokenizing the text into individual words
    tokens = word_tokenize(text_content)

    # Initializing the RNNMorph predictor
    predictor = RNNMorphPredictor(language='ru')

    # Initializing variables for result collection and word tracking
    matched_pairs = []
    previous_word = {'word': '', 'POS': '', 'gender': '', 'number': '', 'case': '', 'index': -2}

    # Predicting morphology for all tokens
    predictions = predictor.predict(tokens)

    # Loop through each word token with progress bar
    for i in tqdm(range(len(predictions))):
        current_word = predictions[i]
        
        # Check if the word is either a noun or an adjective
        if current_word.pos in ['NOUN', 'ADJF']:
            # Check for agreement in gender, number, and case between consecutive words
            if (i - previous_word['index'] == 1 and previous_word['gender'] == current_word.gender and
                previous_word['number'] == current_word.number and previous_word['case'] == current_word.case and
                previous_word['POS'] != current_word.pos):
                
                # Append the matching word pair to results
                matched_pairs.append(f"{previous_word['word']} {current_word.normal_form}")
            
            # Update the details of the current word for future comparison
            previous_word.update({
                'word': current_word.normal_form,
                'POS': current_word.pos,
                'gender': current_word.gender,
                'number': current_word.number,
                'case': current_word.case,
                'index': i
            })

    # Remove duplicate pairs
    unique_pairs = list(set(matched_pairs))

    # Return the result
    return unique_pairs

def main():
    # file_path = '/mnt/data/file_text.txt'  # Path to the file you uploaded
    file_path = 'D:\\ИПМКН\\Семестр_7\\Обработка_естественного_языка\\NLP\\lab_3\\file_text.txt'  # You can change this path if needed
    results = process_text(file_path)

    # Output the final results
    print('\nFound matching word pairs:')
    for pair in results:
        print(pair)

if __name__ == "__main__":
    main()


# from rnnmorph.predictor import RNNMorphPredictor
# predictor = RNNMorphPredictor(language='ru')

# res = predictor.predict(['История', 'Прекрасного', 'денька', '.'])

# print(res[1].normal_form)
# print()
# print(res[1].pos)
# print()
# print(res[1].tag)



#     nice! tapi output yg keluar malah kek gini:
# Found matching word pairs:
# результат многолетний
# денежный обращение
# затяжной проблема
# предыдущий исследование
# предыдущий работа
# последующий исследование
# аспект стоимостный
# стоимостный теория
# дальнейший углубление
# ключевой момент
# новый вывод
# данный труд
# первый публикация
# новый источник
# многолетний исследование
# исторический аспект
# детальный освещение
# настоящий работа
# момент предыдущий
# первый раздел
# опубликованный исследование
# изложенный идея

# gw pengennya hasilnya kek gini:
# Found matching word pairs:
# настоящий работа
# дальнейший углубление
# денежный обращение
# первый раздел


# INI HASILNYA KENAPA KEK GINI? NGGAK BEDA SIH, ADA JG KESAMAANNYA, TAPI APAKAH LO BISA 
# MENJELASKAN KENAPA AGAK "SEDIKIT" BERBEDA ?? ? 
# tolong dong diperbaiki atau apa gitu?