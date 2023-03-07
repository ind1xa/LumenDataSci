from pydub import AudioSegment
import pandas as pd
import numpy as np
import librosa
import soundfile as sf
import os
import itertools as it
from tqdm import tqdm

path = 'dataset/IRMAS_Training_Data/'
instuments = ['cel', 'cla', 'flu', 'gac', 'gel',
              'org', 'pia', 'sax', 'tru', 'vio', 'voi']
all_files = []


def merge_wav(path1, path2, delay, name1, name2, id):

    sound1 = AudioSegment.from_wav(path1).normalize()
    sound2 = AudioSegment.from_wav(path2).normalize()

    # #normalize
    # sound1 = sound1 - sound1.dBFS
    # sound2 = sound2 - sound2.dBFS

    overflow = len(sound1) - delay*1000

    sound2 = sound2.overlay(sound1, position=delay*1000)
    sound2 = sound2.overlay(sound1[-overflow:], position=0)

    sound2.export('overlaysNorm/overlaid' + str(id) + name1 + name2+ '.wav', format='wav')

def get_names(path_to_root):
    categories = os.listdir(path_to_root)
    print("ok")
    for category in categories:
        if category == '.DS_Store':
            continue

        path_to_category = os.path.join(path_to_root, category)
        files = os.listdir(path_to_category)

        for file in files:
            if file == '.DS_Store':
                continue

            path_to_file = os.path.join(path_to_category, file)
            #print(path_to_file)

            file = file.split(']')
            for i in range(0, len(file)):
                file[i] = file[i][-3:]
                if file[i] not in instuments:
                    file[i] = ''
                else:
                    file[i] = '[' + file[i] + ']'

            file = ''.join(file)

            pair = [path_to_file, file]
            all_files.append(pair)

            #print(file)


def main():
    print("ok")
    get_names(path)
    counter = 0
    while (counter < len(all_files)):
        random_number1 = np.random.randint(0, len(all_files)-1)
        random_number2 = np.random.randint(0, len(all_files)-1)
        random_float = np.random.uniform(0, 3)
        if (all_files[random_number1][1] != all_files[random_number2][1]):
            merge_wav(all_files[random_number1][0], all_files[random_number2][0], random_float, all_files[random_number1][1], all_files[random_number2][1], counter)
            counter += 1
    print(counter)
    #merge_wav('example/130__[cel][nod][cla]0040__1.wav', 'example/[pia][cla]1308__3.wav', 1.5)

if __name__ == '__main__':
    main()