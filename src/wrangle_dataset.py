import pandas as pd
import numpy as np
import librosa
import soundfile as sf
import os
import itertools as it
from tqdm import tqdm

path = 'dataset/IRMAS_Training_Data/'


def add_blanking(audio, n_blank=1, percent=0.1):
    for _ in range(n_blank):
        audio_length = len(audio)
        blank_length = int(audio_length * percent)
        blank = np.zeros(blank_length)
        index = np.random.randint(audio_length - blank_length)
        audio[index:index + blank_length] = blank

    return audio


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


def read_data(path_to_root):
    categories = os.listdir(path_to_root)
    all_audio = []

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

            audio, sampling_rate = librosa.load(path_to_file)

            volume_scaling = [0.4, 1.0, 1.5]
            shift_percent = [-0.33, 0.25, 0.5]
            stretch_rate = [0.5, 1.0, 1.4]
            pitch_shift_steps = [-2, 0, 1]

            audio_stddev = np.std(audio)
            noise_stddev = [0.0, 0.1 * audio_stddev, 0.3 * audio_stddev]

            n_blanks = [0, 1, 2]

            product = it.product(volume_scaling, shift_percent, stretch_rate,
                                 pitch_shift_steps, noise_stddev, n_blanks)

            sf.write('example/test_original.wav', audio, sampling_rate)

            for p in tqdm(product, total=len(volume_scaling) * len(shift_percent) * len(stretch_rate) * len(pitch_shift_steps) * len(noise_stddev) * len(n_blanks)):
                factor, shift, stretch, pitch, noise, n_blank = p

                audio2 = audio

                if factor != 1.0:
                    audio2 = scale_volume(audio, factor)

                if shift != 0:
                    audio2 = shift_audio(audio2, sampling_rate, shift)

                if stretch != 1.0:
                    audio2 = stretch_audio(audio2, stretch)

                if pitch != 0:
                    audio2 = pitch_shift(audio2, sampling_rate, pitch)

                if noise > 0:
                    audio2 = add_noise(audio2, noise)

                if n_blank > 0:
                    audio2 = add_blanking(audio2, sampling_rate, n_blank)

                assert len(audio) == len(audio2)

                mfcc = audio_to_mfcc(audio2, sampling_rate)
                # print('MFCC: ', mfcc)

                audio_debug = mfcc_to_audio(mfcc, sampling_rate)

                # save audio
                sf.write(f"example/test_{factor}_{shift}_{stretch}_{pitch}_{noise}_{n_blank}.wav",
                         audio_debug, sampling_rate)

            #     scaled_audio = scale_volume(audio, factor)

            #     print('Scaled audio: ', scaled_audio)

            exit()

    # print('Categories: ', categories)


def main():
    audio = read_data(path)
    insert_noise(audio)


if __name__ == '__main__':
    main()
