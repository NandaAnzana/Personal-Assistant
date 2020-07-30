import nltk
from nltk.stem.lancaster import LancasterStemmer
import json

nltk.download('punkt')


def make_list(x):
    stemmer = LancasterStemmer()

    with open(x) as f:
        data = json.load(f)

    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intens"]:
        for pattern in intent["patterns"]:
            wrds= nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(pattern)
            docs_y.append(intent["Tag"])
        if data["Tag"] not in labels:
            labels.append(intent["Tag"])

    words = [stemmer.stem(w.lower()) for w in words]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    return words, labels, docs_x, docs_y