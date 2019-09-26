import nltk
from nltk.corpus import stopwords
from typing import Union, Set, List


"""
Loading the stopwords requires reading from local files, which should be done only once, rather than 
every time we need to check if a word is a stopword or not. 
"""

en_stopwords = set(stopwords.words('english'))  # preloading stopword list


def preprocess(message: str, remove_stopwords: bool = True) -> List[str]:
    """
    Split message into tokenize.
    Optional: remove stopwords

    :param message: the message to preprocess
    :param remove_stopwords: whether to remove stopwords
    :return: a list of tokens after preprocessing
    """

    # TODO: implement this function
    words = nltk.word_tokenize(message)
    if remove_stopwords:
        tokens = [word.lower() for word in words if word.lower() not in en_stopwords]
        return tokens
    else:
        pass
