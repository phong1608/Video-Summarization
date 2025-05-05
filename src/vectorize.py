import numpy as np 
from gensim.models import KeyedVectors
from pyvi import ViTokenizer



class Vectorize:
    def __init__(self,w2v_path='data/vi.vec'):
        self.w2v = KeyedVectors.load_word2vec_format(w2v_path)
        
    def vectorize(self,sentences):
        vocab =self.w2v.key_to_index
        X = []
        for sentence in sentences:
            sentence_tokenized = ViTokenizer.tokenize(sentence)
            words = sentence_tokenized.split(" ")
            sentence_vec = np.zeros((100))
            for word in words:
                if word in vocab:
                    sentence_vec += self.w2v[word]
            X.append(sentence_vec)
        return X


