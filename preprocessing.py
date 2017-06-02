import pandas as pd
import spacy
import tqdm

nlp = spacy.load('en_core_web_md')

data = pd.DataFrame.from_csv('train.csv')

N = len(data)

parse = lambda x: nlp(str(x))
to_str = lambda x: " ".join([str(word) for word in parse(x)])                                                                               

q2id = {}
id2q = {}
is_dupl = {}

all_questions = []

q1 = []
q2 = []
labels = []

for i in tqdm.tqdm(range(N)):
    q1.append(to_str(data['question1'][i]))        
    q2.append(to_str(data['question2'][i]))
    labels.append(int(data['is_duplicate'][i]))    


