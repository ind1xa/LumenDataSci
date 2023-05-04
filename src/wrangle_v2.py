from collections import defaultdict
import json
import random
import string
import zipfile
import pandas as pd
import numpy as np
import librosa
import soundfile as sf
import os
import itertools as it
from tqdm import tqdm
import soundfile as sf
from multiprocessing import Manager, Pool

import sqlite3

path = '/Users/nitkonitkic/Documents/LumenDataSci/dataset/IRMAS_Training_Data'


instuments = ['cel', 'cla', 'flu', 'gac', 'gel',
              'org', 'pia', 'sax', 'tru', 'vio', 'voi']


def scale_volume(audio, factor=1.0):
    return factor * audio


def shift_audio(audio, sampling_rate, shift_percent):
    shift = int(sampling_rate * shift_percent)
    return np.roll(audio, shift)


def stretch_audio(audio, rate=1.0):
    input_length = len(audio)
    audio = librosa.effects.time_stretch(y=audio, rate=rate)

    if len(audio) > input_length:
        max_offset = len(audio) - input_length
        offset = np.random.randint(max_offset)
        audio = audio[offset:(input_length + offset)]
    elif input_length > len(audio):
        max_offset = input_length - len(audio)
        offset = np.random.randint(max_offset)
        audio = np.pad(audio, (offset, input_length -
                       len(audio) - offset), "constant")
    else:
        pass

    return audio


def pitch_shift(audio, sampling_rate, n_steps):
    return librosa.effects.pitch_shift(y=audio, sr=sampling_rate, n_steps=n_steps)


def add_noise(audio, stddev):
    noise = np.random.normal(0, stddev, len(audio))
    return audio + noise


def add_blanking(audio, sampling_rate, n_blanks):
    blank_length = int(sampling_rate * 0.1)
    blank = np.zeros(blank_length)

    for i in range(n_blanks):
        index = np.random.randint(len(audio) - blank_length)
        audio[index:index + blank_length] = blank

    return audio


def merge_audio(audio1, audio2):
    return audio1 + audio2


def audio_to_mfcc(audio, sampling_rate):
    mfcc = librosa.feature.mfcc(y=audio, sr=sampling_rate, n_mfcc=130)
    return mfcc


def mfcc_to_audio(mfcc, sampling_rate):
    audio = librosa.feature.inverse.mfcc_to_audio(mfcc, sr=sampling_rate)
    return audio


def get_all_files(path):

    for ins in tqdm(instuments):
        files = os.listdir(os.path.join(path, ins))

        for file in files:
            audio, sampling_rate = librosa.load(
                os.path.join(path, ins, file), sr=22050)
            assert sampling_rate == 22050
            assert len(audio) == sampling_rate * 3

            yield ins, audio


def get_random_augmented(all_files):

    volume_scaling = random.uniform(0.5, 1.5)
    shift = random.uniform(-0.2, 0.2)
    stretch = random.uniform(0.8, 1.2)
    pitch = random.randint(-2, 2)
    noise = random.uniform(0, 0.02)
    blank = random.randint(0, 3)

    label, audio = random.choice(all_files)

    audio = scale_volume(audio, volume_scaling)
    audio = shift_audio(audio, 22050, shift)
    audio = stretch_audio(audio, stretch)
    audio = pitch_shift(audio, 22050, pitch)
    audio = add_noise(audio, noise)
    audio = add_blanking(audio, 22050, blank)

    assert len(audio) == 22050 * 3

    return label, audio


def get_random_sample(all_files):
    labels = set()

    base_label, base = get_random_augmented(all_files)
    labels.add(base_label)

    while random.random() < 0.5:
        label, audio = get_random_augmented(all_files)

        base = merge_audio(base, audio)
        labels.add(label)

    return labels, base


def runner(all_files):

    l = []
    for _ in tqdm(range(1000)):
        labels, audio = get_random_sample(all_files)

        labels_str = '_'.join(labels)

        assert len(audio) == 22050 * 3

        mfcc = audio_to_mfcc(audio, 22050)

        mfcc = mfcc.astype(np.float32)
        assert mfcc.shape == (128, 130), mfcc.shape

        # print(labels_str, mfcc.shape)

        l.append((labels_str, mfcc.tobytes()))

    with sqlite3.connect('data.db') as conn:
        conn.execute(
            'CREATE TABLE IF NOT EXISTS data (id INTEGER PRIMARY KEY, labels TEXT, audio BLOB)')

        c = conn.cursor()
        c.executemany('INSERT INTO data (labels, audio) VALUES (?, ?)', l)

        conn.commit()
        print('added 1000 samples')


def main():
    all_files = list(get_all_files(path))
    runner(all_files)


if __name__ == '__main__':
    main()
