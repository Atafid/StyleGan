from src.functions import train_new_model, train_existing_model, init_generation
import os
import sys

if (__name__ == '__main__'):
    train = input("Train model (y/n) : ")

    if (train == "y"):
        new = input("New model (y/n) : ")

        if (new == "y"):
            train_new_model()
        else:
            train_existing_model()

    else:
        sys.stdout = open(os.devnull, 'w')
        root, canvas = init_generation()
