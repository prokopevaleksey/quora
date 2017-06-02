from pyemd import emd
import numpy as np
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import euclidean_distances
import tqdm 
import pickle

with open('id2q.pkl','rb') as f:
    id2q = pickle.load(f)

with open('q2id.pkl','rb') as f:
    q2id = pickle.load(f)

nlp = spacy.load('en_core_web_md')

N = len(id2q)

parse = lambda x: nlp(str(x)) #Python 3 renamed the unicode type to str, the old str type has been replaced by bytes
to_str = lambda x: " ".join([str(word) for word in parse(x)])  
vectorize = lambda x: parse(x).vector



def wmd(x,y):
    x = x.lower()
    y = y.lower()
    vect = CountVectorizer().fit([x, y]) 
    v_1, v_2 = vect.transform([x, y])

    v_1 = v_1.toarray().ravel()                                                                                                            
    v_2 = v_2.toarray().ravel()

    v_1 = v_1.astype(np.double)
    v_2 = v_2.astype(np.double)

    v_1 /= v_1.sum()
    v_2 /= v_2.sum()

    W_ = [vectorize(w) for w in vect.get_feature_names()]  
    D_ = euclidean_distances(W_)    

    D_ = D_.astype(np.double)
    D_ /= D_.max()  # just for comparison purposes
    return emd(v_1, v_2, D_)

