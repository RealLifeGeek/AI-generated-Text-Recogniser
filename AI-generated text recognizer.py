import numpy as np
import math

# The network computes the probability of docs to be AI or human generated.
# Docs are in this code the lines of the variable "text"
# The model is NOT TRAINED. There is still the database of AI and human generated posts missing.
# Regarding the warning above - the output will be currently always uncertain.

text = '''Humpty Dumpty sat on a wall
Humpty Dumpty had a great fall
all the king's horses and all the king's men
couldn't put Humpty together again'''

def hidden_activation(z):
    return np.maximum(0, z)

def output_activation(z):
    return 1 / (1 + np.exp(-z))

def forward_pass(tfidf_vector):
    w0 = np.random.randn(len(tfidf_vector), 2)  
    w1 = np.random.randn(2, 2)
    w2 = np.random.randn(2, 1)
    
    b0 = np.zeros(2)
    b1 = np.zeros(2)
    b2 = np.zeros(1)  

    h1_in = np.dot(tfidf_vector, w0) + b0
    h1_out = hidden_activation(h1_in)

    h2_in = np.dot(h1_out, w1) + b1
    h2_out = hidden_activation(h2_in)

    out_in = np.dot(h2_out, w2) + b2
    out = output_activation(out_in)

    return out

def main(text):
    docs = [line.lower().split() for line in text.splitlines()]
    vocabulary = list(set(word for doc in docs for word in doc))
    
    tf = {}
    df = {}
    N = len(docs)
    for word in vocabulary:
        tf[word] = [doc.count(word) / len(doc) for doc in docs]
        df[word] = sum(word in doc for doc in docs) / N

    tfidf = []
    for doc_index, doc in enumerate(docs):
        tfidf_vector = [tf[word][doc_index] * math.log(1 / df[word], 10) if df[word] > 0 else 0 for word in vocabulary]
        tfidf.append(tfidf_vector)

    return tfidf

tfidf_vectors = main(text)
for tfidf_vector in tfidf_vectors:
    output = forward_pass(np.array(tfidf_vector))
    print(f"Prediction: {output}")
