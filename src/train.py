import nltk
from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow as tf
import json

with open("intens.json") as f:
    data = json.load(f)

print(data)