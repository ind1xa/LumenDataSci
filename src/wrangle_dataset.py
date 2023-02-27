import pandas as pd
import numpy as np
import librosa
import soundfile as sf
import os
import itertools as it

path = 'dataset/IRMAS_Training_Data/'


def scale_volume(audio, factor=1.0):
    return factor * audio


def shift_audio(audio, sampling_rate, shift_percent):
    shift = int(sampling_rate * shift_percent)
    return np.roll(audio, shift)


def stretch_audio(audio, rate=1.0):
    input_length = len(audio)
    audio = librosa.effects.time_stretch(audio, rate)

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
    return librosa.effects.pitch_shift(audio, sampling_rate, n_steps)


def add_noise(audio, stddev):
    noise = np.random.normal(0, stddev, len(audio))
    return audio + noise


def add_reverb(audio, sampling_rate, delay=0.5, decay=0.5):
    return librosa.effects.reverb(audio, delay, decay)


def add_echo(audio, sampling_rate, delay=0.5, decay=0.5):
    return librosa.effects.echo(audio, sampling_rate, delay, decay)


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

            volume_scaling = [0.2, 0.5, 1.0, 1.5]
            shift_percent = [-0.5, -0.25, 0.25, 0.5]
            stretch_rate = [0.5, 0.75, 1.0, 1.25, 1.5]
            pitch_shift_steps = [-2, -1, 0, 1, 2]

            audio_stddev = np.std(audio)
            noise_stddev = [0.0, 0.1 * audio_stddev, 0.2 * audio_stddev,
                            0.3 * audio_stddev, 0.4 * audio_stddev]

            reverb_delay = [0.5, 1.0, 1.5, 2.0]
            reverb_decay = [0.5, 1.0, 1.5, 2.0]

            echo_delay = [0.5, 1.0, 1.5, 2.0]
            echo_decay = [0.5, 1.0, 1.5, 2.0]

            n_blanks = [0, 1, 2, 3, 4]

            #     scaled_audio = scale_volume(audio, factor)

            #     print('Scaled audio: ', scaled_audio)

            mfcc = audio_to_mfcc(audio, sampling_rate)
            print('MFCC: ', mfcc)

            audio_debug = mfcc_to_audio(mfcc, sampling_rate)

            # save audio
            sf.write('test.wav', audio_debug, sampling_rate)
            sf.write('test_original.wav', audio, sampling_rate)

            exit()

    # print('Categories: ', categories)


def main():
    read_data(path)


if __name__ == '__main__':
    main()
