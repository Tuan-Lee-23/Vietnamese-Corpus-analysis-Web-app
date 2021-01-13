import pandas as pd
from time import time
import numpy as np
import re
import csv

from nltk import ngrams
from gensim.models import Word2Vec


from underthesea import word_tokenize
from underthesea import sent_tokenize
from underthesea import ner
from underthesea import pos_tag
import pickle


class Corpus:
    def __init__(self):
        self.data = []  #original data
        self.data_pos_tagged = []  # pos_tagging and ner
        self.data_word_segment = [] # sentence with word segmentation 
        self.data_sent_segment = [] # sentence with sentence segmentation
        self.vocab = [] # list of vocabulary
        self.stopwords = {} # list of stopwords with count value
        self.w2v_model = {} # word2vec model

    def read(self, dirr):

        corpus = []
        with open(dirr, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                temp = re.split(r'\t+', line)
                corpus.append(temp[0])

        self.data = corpus

    def preprocess(self):

        def clean_html(raw_html):
            cleanr = re.compile('<.*?>')
            cleantext = re.sub(cleanr, '', raw_html)
            return cleantext

        def normalize_unicode(txt):
            def loaddicchar():
                dic = {}
                char1252 = 'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ'.split(
                    '|')
                charutf8 = "à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ".split(
                    '|')
                for i in range(len(char1252)):
                    dic[char1252[i]] = charutf8[i]
                return dic

            def convert_unicode(txt):
                dicchar = loaddicchar()
                return re.sub(
                    r'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ',
                    lambda x: dicchar[x.group()], txt)
            
            return convert_unicode(txt)

        def sentence_tokenize(data):
            data_test = list(map(sent_tokenize, data))
            self.data = np.hstack(data_test).tolist()

        def remove_spec_chars(data):
            self.data_sent_segment =  [re.sub(r'[^,.\w\s]', '', sen) for sen in data]

        def remove_nums(txt):
            result = re.sub(r'\d+', '', txt)
            return result 

        def word_tokenize_w_remove_stopwords():
            stop_words = set()
            with open("resources/stop_words.txt", encoding = 'utf-8') as f:
                lines = f.readlines()
                for line in lines:
                    stop_words.add(line.rstrip('\n'))

            for sen in self.data_sent_segment:
                word_seg = word_tokenize(sen)
                temp = []
                for word in word_seg:
                    word_lower = word.lower()
                    if word_lower not in stop_words and word_lower != '.' and word_lower != ',' and word_lower != '...':
                        temp.append(word_lower)
                    else:
                        if word != '.' and word != ',' and word != '...':
                            if word_lower not in self.stopwords:  # add to list stopwords
                                self.stopwords[word_lower] = 0
                            else:
                                self.stopwords[word_lower] += 1  # count stopwords
                self.data_word_segment.append(temp)   # append to data.word_segment
            # self.data_word_segment.remove(',')
            # self.data_word_segment.remove('.')

            

        # clean html
        self.data = list(map(clean_html, self.data))
        
        # chuẩn hoá bảng mã
        self.data = list(map(normalize_unicode, self.data))

        # tokenize sentence (data_sent_segment)
        sentence_tokenize(self.data)

        # remove punctuation and special chars (data_sent_segment)
        remove_spec_chars(self.data)

        # remove numbers (data_sent_segment)
        self.data_sent_segment = list(map(remove_nums, self.data_sent_segment))

        # Create dictionary (dict)
        word_tokenize_w_remove_stopwords()

    # Train word2vector
    def train_word2vec(self):
            w2v_model = Word2Vec(size= 500)
            t = time()
            w2v_model.build_vocab(sentences = self.data_word_segment, progress_per=10000)
            print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))

            t = time()
            w2v_model.train(sentences = self.data_word_segment, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)
            print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))
            w2v_model.init_sims(replace=True) # lower memory used
            with open('drafts/w2v.pik', 'wb') as f:
                pickle.dump(w2v_model, f, -1)
                print("Word2Vec model was saved successfully")


    def read_word2vec(self):
        with open('drafts/w2v.pik', 'rb') as f:
            w2v_model = pickle.load(f)
            corpus_vocabs = dict()
            for item in w2v_model.wv.vocab:
                corpus_vocabs[item]= w2v_model.wv.vocab[item].count

            corpus_vocabs_sorted=dict(sorted(corpus_vocabs.items(), key=lambda x: x[1],reverse=True))
            
            self.vocab = corpus_vocabs_sorted
            self.w2v_model = w2v_model

    # Find top 10 synonyms words
    def find_similar(self, word):
        try:
            temp = self.w2v_model.wv.most_similar(word)
        except KeyError:
            return []
        return temp

    # RUN TOO SLOW WITH 30k
    # Phân loại từ, chunk, nhận diện tên riêng
    def ner(self):
        def POS_Chunk_ner(txt):
            return ner(txt)
        # CARE FULLLLLLLLLLLLLLLLLLLLLLLLLL
        # self.data_pos_tagged = list(map(POS_Chunk_ner, self.data_sent_segment))
        for i, sen in enumerate(self.data_sent_segment):
            if i % 1000 == 0:
                print("--",i, '/', len(self.data_sent_segment),"--")
            self.data_pos_tagged.append(POS_Chunk_ner(sen))
        
        # with open('ner.txt', 'w', encoding = 'utf-8') as fp:
        #     for j, sen in enumerate(self.data_pos_tagged):
        #         for i, item in enumerate(sen):
        #             fp.write("(" + item[0] + "," + item[1] + "," + item[2] + "," + item[3] + ")")
        #             if i < len(sen) - 1:
        #                 fp.write(",")
        #             fp.write("\n")
        #         fp.write("\n")
        with open('drafts/ner.pik', 'wb') as f:
            pickle.dump(self.data_pos_tagged, f, -1)
        print("NER was saved successfully")

    def read_ner(self):
        with open('drafts/ner.pik', 'rb') as f:
            self.data_pos_tagged = pickle.load(f)

    # Search
    def isIn(self, sentence, word):
        n = len(word.split())

        ngrams_temp = ngrams(sentence.split(), n)
        for i, grams in enumerate(ngrams_temp):
            if grams == tuple(word.split()):
                return (True, i)
        
        return (False, -1)

    # morpheme and word search
    # def search_ambiguous(self, word):
    #     result = []
    #     for i, sen in enumerate(self.data_sent_segment):
    #         temp = self.isIn(sen, word)
    #         if temp[0]:
    #             result.append((self.data[i], temp[1]))
    #     return result
    
    # morpheme and word search
    # case: 0 --> non-case,  1: --> case sensitive
    def search_ambiguous(self, word, case = 0):
        if case:
            ex = r'' + word
            result = [i for i in self.data if re.search(ex, i)]
        else:
            ex = r'' + word.lower()
            result = [i for i in self.data if re.search(ex, i.lower())]
        return result


    # output: 
    # 0: matched sentence
    # 1: position of search_word in sentence
    # 2: matched word
    def search_by_pos(self, search_word, pos):
        result = set()
        for i, sen in enumerate(self.data_pos_tagged):
            if sen: 
                for word in sen:
                    if search_word.lower() in word[0].lower() and pos == word[1]:
                        temp = self.isIn(self.data[i], search_word)
                        result.add((self.data[i], temp[1], word[0]))
        return result

    # output: 
    # 0: sentence
    # 1: position of search_word in sentence
    # 2: word matched
    def search_by_ner(self, search_word, pos):
        result = set()
        for i, sen in enumerate(self.data_pos_tagged):
            if sen: 
                for word in sen:
                    if search_word.lower() in word[0].lower() and pos in word[3]:
                        temp = self.isIn(self.data[i], search_word)
                        result.add((self.data[i], temp[1], word[0]))
        return result


# if __name__ == "__main__":
    # dirr = "resources/vie-vn_web_2015_30K-sentences.txt"
    # dirr = "resources/corpus_mini.txt"

    # initialize Corpus
    # corpus = Corpus()

    # read corpus
    # corpus.read(dirr)

    # Preprocessing
    # corpus.preprocess()

    # Search ambiguous
    # test = corpus.search_ambiguous("Phương tiện", 1)
    # for sen in test:
    #     print(sen)

    # Ner (POS + chunk + NER)
    # corpus.ner()
    # corpus.read_ner()

    # print(corpus.data_pos_tagged[111])

    # Search by pos tag
    # for s in corpus.search_by_pos("phương", 'N'):
    #     print(s)

    # Vocab
    # print(corpus.stopwords)
