import nltk
from nltk.corpus import stopwords


en_stopwords = set(stopwords.words('english'))


def remove_stopwords(message):

    words = nltk.word_tokenize(message)
    tokens = [word.lower() for word in words if word.lower() not in en_stopwords]
    return ' '.join(tokens)
