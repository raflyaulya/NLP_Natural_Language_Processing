import gensim 
import re

# target 2 words: ['столик', 'кабинет']

def main():
    # Определяем список слов (положительный контекст), для которых хотим найти похожие слова
    positive = ['стул_NOUN', 'офис_NOUN']

    file_src = 'D:\\file_cbow.txt'
    
    # Загружаем модель word2vec с помощью KeyedVectors из библиотеки gensim
    word2vec = gensim.models.KeyedVectors.load_word2vec_format(file_src, binary=False)

    # Используем модель для поиска 10 самых похожих слов (topn=10)
    # на основе списка положительных слов (т.е. 'стул_NOUN' и 'офис_NOUN')
    dist = word2vec.most_similar(positive=positive, topn=10)

    # Компилируем регулярное выражение для поиска слов, оканчивающихся на '_NOUN'
    # Шаблон '(.*)_NOUN' захватывает слово перед '_NOUN'
    pat = re.compile('(.*)_NOUN')

    # Проходим по списку похожих слов и их оценок схожести
    for i in dist:
        e = pat.match(i[0])
        if e is not None:  
            print(e.group(1))

if __name__ == '__main__':
    main()
