import json
import string
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import wordnet as wn
import spacy
from collections import defaultdict
from nltk.corpus import stopwords


nltk.download('stopwords')
stop_words = set(stopwords.words('russian'))

nlp = spacy.load("ru_core_news_sm")

nltk.download('wordnet')
nltk.download('omw-1.4')


def process_responses(responses):
    word_dict = defaultdict(list)

    for response in responses:
        doc = nlp(response)
        main_word = None

        for token in doc:
            base_form = token.lemma_.lower()

            if base_form not in stop_words:
                if base_form not in word_dict:
                    word_dict[base_form].append(token)
                    if not main_word:
                        main_word = base_form

        if main_word:
            print(f"Главное слово в ответе '{response}': {main_word}")

    return list(word_dict.keys())


russian_stopwords = [
    'и', 'в', 'во', 'не', 'что', 'это', 'как', 'да', 'он', 'она',
    'они', 'мы', 'вы', 'я', 'с', 'на', 'к', 'у', 'от', 'за', 'то',
    'кто', 'для', 'или', 'так', 'по', 'как', 'да', 'то', 'все',
    'этот', 'такой', 'такое', 'много', 'все', 'каждый', 'где',
    'как', 'когда', 'почему', 'что', 'чем', 'зачем', 'тот', 'другой'
]


english_stopwords = [
    'and', 'the', 'is', 'in', 'to', 'a', 'that', 'it', 'of', 'on',
    'for', 'with', 'as', 'are', 'at', 'by', 'this', 'an', 'be',
    'from', 'not', 'but', 'or', 'which', 'who', 'when', 'where',
    'why', 'how', 'so', 'all', 'any', 'some', 'many', 'few'
]


def preprocess_text(text, language='russian'):
    text = text.lower()

    text = text.translate(str.maketrans('', '', string.punctuation))

    if language == 'russian':
        stop_words = set(russian_stopwords)
    elif language == 'english':
        stop_words = set(english_stopwords)
    else:
        stop_words = set()

    filtered_words = [word for word in text.split() if word not in stop_words]
    filtered_words = process_responses(filtered_words)

    return ' '.join(filtered_words)


def load_data_from_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data['responses']


def count_words(responses):
    words = []
    for response in responses:
        if any(char.isalpha() for char in response):  # Проверяем наличие букв
            if any(char in string.ascii_letters for char in response):
                words.extend(preprocess_text(response, 'english').split())
            else:
                words.extend(preprocess_text(response, 'russian').split())
    return Counter(words)


def generate_wordcloud(word_counts):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_counts)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()


def get_synonyms(word):
    synonyms = set()
    for syn in wn.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())  # добавляем синонимы
    return synonyms


def remove_synonyms(words):
    unique_words = set()
    processed_words = set()

    for word in words:
        if word not in processed_words:
            synonyms = get_synonyms(word)
            unique_words.add(word)
            processed_words.update(synonyms)

    return unique_words


file_path = 'responses.json'
responses = load_data_from_json(file_path)
word_counts = count_words(responses)


generate_wordcloud(word_counts)
