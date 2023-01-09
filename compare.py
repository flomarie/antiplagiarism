import argparse
import sys
import pickle
import train


def createParser ():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', nargs=1)
    parser.add_argument('scores', nargs=1)
    parser.add_argument('--model')
    return parser

def extract_features(script1, script2):
    X = None
    with open("vectorizer.pkl", "rb") as f, \
            open(script1, 'r') as f1, \
            open(script2, 'r') as f2:
        vectorizer = pickle.load(f)
        X1 = vectorizer.transform([train.text_preprocessing(f1.read())]).toarray()
        X2 = vectorizer.transform([train.text_preprocessing(f2.read())]).toarray()
        X = X1 + X2
    return X

if __name__ == '__main__':
    parser = createParser()
    namespace = parser.parse_args(sys.argv[1:])
    with open(namespace.input[0], 'r') as input_file, \
            open(namespace.scores[0], 'w') as scores_file, \
            open(namespace.model, "rb") as file_model:
        model = pickle.load(file_model)
        for line in input_file:
            script1, script2 = line.split()
            score = model.predict_proba(extract_features(script1, script2))[0]
            scores_file.write(str(score[0]) + '\n')