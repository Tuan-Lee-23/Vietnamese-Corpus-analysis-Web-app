import sys
from time import time
from Corpus import Corpus

import matplotlib.pyplot as plt

corpus = Corpus()
dirr = "resources/corpus_mini.txt"
corpus.read(dirr)

corpus.preprocess()
corpus.train_word2vec()

print(corpus.find_similar('thị xã'))
corpus.ner()







