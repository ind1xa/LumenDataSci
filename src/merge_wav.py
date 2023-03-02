from pydub import AudioSegment
import pandas as pd
import numpy as np
import librosa
import soundfile as sf
import os
import itertools as it
from tqdm import tqdm

path = 'dataset/IRMAS_Training_Data/'

def merge_wav(sounds, delays):
    final = AudioSegment.silent(duration=10*1000)

    #sound2 = AudioSegment.from_wav('example/094__[sax][dru][jaz_blu]1703__3.wav')
    #sound1 = AudioSegment.from_wav('example/130__[cel][nod][cla]0040__1.wav')

    for sound, delay in sounds, delays:
        final = final.overlay(sound, position=delay*1000)

    sound.export('src/example/overlaid.wav', format='wav')

def read_data(path_to_root):
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
            print(path_to_file)


def main():
    print("ok")
    audio = read_data(path)

if __name__ == '__main__':
    main()