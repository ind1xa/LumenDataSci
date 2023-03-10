import random
import string
import pandas as pd
import numpy as np
import librosa
import soundfile as sf
import os
import itertools as it
from tqdm import tqdm

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

    volume_scaling = [0.4, 1.0, 1.5]
    shift_percent = [-0.33, 0.25, 0.5]
    stretch_rate = [0.5, 1.0, 1.4]
    pitch_shift_steps = [-2, 0, 1]

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

        yield input_, output_

    # for category in categories:
    #     if category == '.DS_Store':
    #         continue

    #     path_to_category = os.path.join(path_to_root, category)
    #     files = os.listdir(path_to_category)

    #     for file in files:
    #         if file == '.DS_Store':
    #             continue

    #         path_to_file = os.path.join(path_to_category, file)
    #         print(path_to_file)

    #         audio, sampling_rate = librosa.load(path_to_file)

    #         volume_scaling = [0.4, 1.0, 1.5]
    #         shift_percent = [-0.33, 0.25, 0.5]
    #         stretch_rate = [0.5, 1.0, 1.4]
    #         pitch_shift_steps = [-2, 0, 1]

    #         audio_stddev = np.std(audio)
    #         noise_stddev = [0.0, 0.1 * audio_stddev, 0.2 * audio_stddev]

    #         n_blanks = [0, 1, 2]

    #         product = it.product(volume_scaling, shift_percent, stretch_rate,
    #                              pitch_shift_steps, noise_stddev, n_blanks)

    #         sf.write('example/test_original.wav', audio, sampling_rate)

    #         for p in tqdm(product, total=len(volume_scaling) * len(shift_percent) * len(stretch_rate) * len(pitch_shift_steps) * len(noise_stddev) * len(n_blanks)):
    #             factor, shift, stretch, pitch, noise, n_blank = p

    #             audio2 = audio

    #             if factor != 1.0:
    #                 audio2 = scale_volume(audio, factor)

    #             if shift != 0:
    #                 audio2 = shift_audio(audio2, sampling_rate, shift)

    #             if stretch != 1.0:
    #                 audio2 = stretch_audio(audio2, stretch)

    #             if pitch != 0:
    #                 audio2 = pitch_shift(audio2, sampling_rate, pitch)

    #             if noise > 0:
    #                 audio2 = add_noise(audio2, noise)

    #             if n_blank > 0:
    #                 audio2 = add_blanking(audio2, sampling_rate, n_blank)

    #             assert len(audio) == len(audio2)

    #             mfcc = audio_to_mfcc(audio2, sampling_rate)
    #             # print('MFCC: ', mfcc)

    #             all_audio.append(mfcc)

    #             # print('MFCC shape: ', mfcc.shape)

    #             # audio_debug = mfcc_to_audio(mfcc, sampling_rate)

    #             # save audio
    #             # sf.write(f"example/test_{factor}_{shift}_{stretch}_{pitch}_{noise}_{n_blank}.wav",
    #             #          audio_debug, sampling_rate)

    #         #     scaled_audio = scale_volume(audio, factor)

    #         #     print('Scaled audio: ', scaled_audio)

    #         yield np.array(all_audio)

    # # save all_audio as parquet

    # # print('Categories: ', categories)


def main():
    for mfcc, label in generate_all_data("dataset/IRMAS_Training_Data"):
        # save mfcc as npz

        rnd_str = ''.join(random.choices(
            string.ascii_uppercase + string.digits, k=5))

        has_instruments = [
            instuments[i]
            for i, x in enumerate(label)
            if x >= 0.5
        ]
        
        np.savez_compressed(
            f"./dataset/output/{rnd_str}_{'_'.join(has_instruments)}.npz", mfcc)


if __name__ == '__main__':
    main()
