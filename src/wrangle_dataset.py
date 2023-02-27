import pandas as pd
import numpy as np
import librosa
import os

path = 'dataset/IRMAS_Training_Data/'


def read_data(path_to_root):
    categories = os.listdir(path_to_root)

    for category in categories:
        if category == '.DS_Store':
            continue

        path_to_category = os.path.join(path_to_root, category)
        files = os.listdir(path_to_category)

        for file in files:
            if file == '.DS_Store':
                continue

            path_to_file = os.path.join(path_to_category, file)
            print(path_to_file)

    # print('Categories: ', categories)


def main():
    read_data(path)


if __name__ == '__main__':
    main()
