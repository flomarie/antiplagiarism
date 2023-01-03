import argparse
import sys
import pickle


def createParser ():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', nargs=1)
    parser.add_argument('scores', nargs=1)
    parser.add_argument('--model')
    return parser

def compare_scripts(model, script1, script2):
    return model.predict(script1, script2)

if __name__ == '__main__':
    parser = createParser()
    namespace = parser.parse_args(sys.argv[1:])
    with open(namespace.input[0], 'r') as input_file, \
            open(namespace.scores[0], 'w') as scores_file, \
            open(namespace.model, "rb") as file_model:
        model = pickle.load(file_model)
        for line in input_file:
            script1, script2 = line.split()
            score = compare_scripts(model, script1, script2)
            scores_file.write(str(score) + '\n')