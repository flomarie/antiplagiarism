import pickle
import argparse
import sys
import os
import ast

def createParser ():
    parser = argparse.ArgumentParser()
    parser.add_argument('files', nargs=1)
    parser.add_argument('plagiat1', nargs=1)
    parser.add_argument('plagiat2', nargs=1)
    parser.add_argument('--model')
    return parser

def extract_features(init_script, plagiat):
    init_script_tree = ast.dump(ast.parse(init_script))
    plagiat_tree = ast.dump(ast.parse(plagiat))


if __name__ == '__main__':
    parser = createParser()
    namespace = parser.parse_args(sys.argv[1:])
    files = os.listdir(namespace.files[0])
    plagiat1 = os.listdir(namespace.plagiat1[0])
    plagiat2 = os.listdir(namespace.plagiat2[0])
    #with open("model.pkl", "wb") as f:
        #pickle.dump(model, f)