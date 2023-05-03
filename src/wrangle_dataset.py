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


path = 'dataset/IRMAS_Training_Data'


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


def audio_to_mfcc(audio, sampling_rate):
    mfcc = librosa.feature.mfcc(y=audio, sr=sampling_rate, n_mfcc=100)
    return mfcc


def mfcc_to_audio(mfcc, sampling_rate):
    audio = librosa.feature.inverse.mfcc_to_audio(mfcc, sr=sampling_rate)
    return audio


instuments = ['cel', 'cla', 'flu', 'gac', 'gel',
              'org', 'pia', 'sax', 'tru', 'vio', 'voi']


def generate_all_data(path_to_root):
    categories = os.listdir(path_to_root)

    all_files = []
    all_files_labels = []

    for category in categories:
        if category == '.DS_Store':
            continue

        path_to_category = os.path.join(path_to_root, category)
        files = os.listdir(path_to_category)

        for file in files:
            if file == '.DS_Store':
                continue

            path_to_file = os.path.join(path_to_category, file)
            all_files.append(path_to_file)

            label = np.zeros(len(instuments))

            for i, instrument in enumerate(instuments):
                if instrument in file:
                    label[i] = 1.

            all_files_labels.append(label)

    volume_scaling = [0.4, 1.0, 1.5]
    shift = [-0.2, 0.0, 0.2]
    stretch = [0.8, 1.0, 1.2]
    pitch = [-2, 0, 2]
    noise = [0.0, 0.01, 0.02]
    blank = [0, 1, 2]

    for file, label in tqdm(zip(all_files, all_files_labels)):
        audio, sampling_rate = librosa.load(file)

        for vol, sh, st, p, n, b in it.product(volume_scaling, shift, stretch, pitch, noise, blank):

            audio2 = scale_volume(audio, vol)
            audio2 = shift_audio(audio2, sampling_rate, sh)
            audio2 = stretch_audio(audio2, st)
            audio2 = pitch_shift(audio2, sampling_rate, p)
            audio2 = add_noise(audio2, n)
            audio2 = add_blanking(audio2, sampling_rate, b)

            mfcc = audio_to_mfcc(audio2, sampling_rate)

            yield mfcc, label


def read_data_generator(path_to_root):
    categories = os.listdir(path_to_root)

    all_files = []
    all_files_labels = []

    for category in categories:
        if category == '.DS_Store':
            continue

        path_to_category = os.path.join(path_to_root, category)
        files = os.listdir(path_to_category)

        for file in files:
            if file == '.DS_Store':
                continue

            path_to_file = os.path.join(path_to_category, file)
            all_files.append(path_to_file)

            label = np.zeros(len(instuments))

            for i, instrument in enumerate(instuments):
                if instrument in file:
                    label[i] = 1.

            all_files_labels.append(label)

    volume_scaling = [0.7, 1.0]
    shift_percent = [-0.2, 0.25]
    stretch_rate = [0.8, 1.0, 1.2]
    pitch_shift_steps = [-2, 0, 2]

    while True:

        BATCH = 100

        choice = np.random.choice(len(all_files), BATCH, replace=False)

        # print(all_files[0])

        batch_files = [librosa.load(all_files[i]) for i in choice]

        # print([batch_files[i][1] for i in range(BATCH)])

        assert all([batch_files[i][1] == 22050 for i in range(BATCH)])
        assert all([batch_files[i][0].shape == (66150,) for i in range(BATCH)])

        batch_files = [batch_files[i][0] for i in range(BATCH)]
        batch_labels = [all_files_labels[i] for i in choice]

        volume_scaling_choice = np.random.choice(volume_scaling, BATCH)
        shift_percent_choice = np.random.choice(shift_percent, BATCH)
        stretch_rate_choice = np.random.choice(stretch_rate, BATCH)
        pitch_shift_steps_choice = np.random.choice(pitch_shift_steps, BATCH)

        audio_stddevs = np.array([np.std(batch_files[i])
                                 for i in range(BATCH)])
        audio_stddev_choice = np.random.choice(audio_stddevs, BATCH)

        audio_stddev_choice *= audio_stddevs

        input_ = []
        output_ = []

        for i in range(BATCH):
            batch_files[i] = scale_volume(
                batch_files[i], volume_scaling_choice[i])
            batch_files[i] = shift_audio(
                batch_files[i], 22050, shift_percent_choice[i])
            batch_files[i] = stretch_audio(
                batch_files[i], stretch_rate_choice[i])
            batch_files[i] = pitch_shift(
                batch_files[i], 22050, pitch_shift_steps_choice[i])
            batch_files[i] = add_noise(
                batch_files[i], audio_stddev_choice[i])

            batch_files[i] = audio_to_mfcc(batch_files[i], 22050)

            input_.append(batch_files[i])
            output_.append(batch_labels[i])

        input_ = np.array(input_, dtype=np.float32)
        output_ = np.array(output_, dtype=np.float32)

        for a, b in zip(input_, output_):
            yield a, b

        # yield input_, output_


d = defaultdict(lambda: 0)


def main():

    cnt = 0

    l = []

    for mfcc, label in tqdm(read_data_generator("../dataset/IRMAS_Training_Data")):
        # save mfcc as npz

        rnd_str = ''.join(random.choices(
            string.ascii_uppercase + string.digits, k=5))

        has_instruments = [
            instuments[i]
            for i, x in enumerate(label)
            if x >= 0.5
        ]

        # np.savez_compressed(
        #     f"./data_out/{rnd_str}_{'_'.join(has_instruments)}.npz", mfcc)

        for ins in has_instruments:
            d[ins] += 1

        l.append(
            [f"{rnd_str}_{'_'.join(has_instruments)}", mfcc])

        # debug output from mfcc to audio

        # audio = mfcc_to_audio(mfcc, 22050)

        # save audio as wav

        # librosa.output.write_wav(
        #     "./debug/" + f"{rnd_str}_{'_'.join(has_instruments)}.wav", audio, 22050)

        # sf.write(
        #     "../debug/" + f"{rnd_str}_{'_'.join(has_instruments)}.wav", audio, 22050)

        if cnt % 10000 == 0:
            l = np.array(l)

            print("saving", cnt)

            np.savez_compressed(
                f"../data_out/{cnt}.npz", l)

            json.dump(d, open(f"../data_out/{cnt}.json", "w"))

            l = []

        cnt += 1


if __name__ == '__main__':
    main()
