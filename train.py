import pickle
import argparse
from pyexpat import features
import sys
import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import regex


def createParser ():
    parser = argparse.ArgumentParser()
    parser.add_argument('files', nargs=1)
    parser.add_argument('plagiat1', nargs=1)
    parser.add_argument('plagiat2', nargs=1)
    parser.add_argument('--model')
    return parser

def text_preprocessing(s):
    return " ".join(regex.split('\W+', regex.sub('[^a-z ]', ' ', s.lower())))

def create_corpus(script_list):
    corpus = set()
    for script in script_list:
        with open(script, 'r') as f:
            corpus.update(set(filter(None, text_preprocessing(f.read()).split(" "))))
    return corpus

def extract_features(files, plagiat1, plagiat2):
    vectorizer = CountVectorizer()
    corpus = list(create_corpus(files + plagiat1 + plagiat2))
    vectorizer.fit(corpus)
    with open("vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    X = np.empty(shape=(3 * len(files), len(vectorizer.vocabulary_)))
    for i in range(len(files)):
        with open(files[i], 'r') as init_f, \
                open(plagiat1[i], 'r') as plagiat1_f, \
                open(plagiat2[i], 'r') as plagiat2_f:
            X_init_script = \
                vectorizer.transform([text_preprocessing(init_f.read())]).toarray()
            X_plagiat1 = \
                vectorizer.transform([text_preprocessing(plagiat1_f.read())]).toarray()
            X_plagiat2 = \
                vectorizer.transform([text_preprocessing(plagiat2_f.read())]).toarray()
            features1 = X_init_script + X_plagiat1
            features2 = X_init_script + X_plagiat2
            X[2 * i, :] = features1
            X[2 * i + 1, :] = features2
    for i in range(len(files)):
        with open(files[i], 'r') as f1, \
            open(files[(i + 8) % len(files)]) as f2:
            X1 = vectorizer.transform([text_preprocessing(f1.read())]).toarray()
            X2 = vectorizer.transform([text_preprocessing(f2.read())]).toarray()
            X[2 * len(files) + i] = X1 + X2
    return X


if __name__ == '__main__':
    parser = createParser()
    namespace = parser.parse_args(sys.argv[1:])
    files = os.listdir(namespace.files[0])
    plagiat1 = os.listdir(namespace.plagiat1[0])
    plagiat2 = os.listdir(namespace.plagiat2[0])
    for i in range(len(files)):
        files[i] = namespace.files[0] + '/' + files[i] #only for Linux!
        plagiat1[i] = namespace.plagiat1[0]+ '/' + plagiat1[i]
        plagiat2[i] = namespace.plagiat2[0]+ '/' + plagiat2[i]
    X = extract_features(files, plagiat1, plagiat2)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    y = np.array(([1] * (2 * len(files))) + ([0] * len(files)))
    model = LogisticRegression(tol=1e-1)
    model.fit(X, y)
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)