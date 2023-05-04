from collections import defaultdict
import math
import os
import random
import soundfile as sf
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow_hub as hub
import librosa
from tqdm import trange


yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
yamnet_model = hub.load(yamnet_model_handle)


def main():
    # testing_wav_file_name = tf.keras.utils.get_file('miaow_16k.wav',
    #                                                 'https://storage.googleapis.com/audioset/miaow_16k.wav',
    #                                                 cache_dir='./',
    #                                                 cache_subdir='test_data')

    class_map_path = yamnet_model.class_map_path().numpy().decode('utf-8')
    class_names = list(pd.read_csv(class_map_path)['display_name'])

    # testing_wav_data = load_wav_16k_mono(testing_wav_file_name)

    # scores, embeddings, spectrogram = yamnet_model(testing_wav_data)
    # class_scores = tf.reduce_mean(scores, axis=0)
    # top_class = tf.math.argmax(class_scores)
    # inferred_class = class_names[top_class]

    # print(f'The main sound is: {inferred_class}')
    # print(f'The embeddings shape: {embeddings.shape}')

    d = defaultdict(list)

    PATH = "/Users/nitkonitkic/Documents/LumenDataSci/dataset/IRMAS_Training_Data"
    classes = ['pia', 'voi', 'tru', 'sax', 'org',
               'cla', 'gac', 'vio', 'flu', 'gel', 'cel']

    for x in classes:
        if x == '.DS_Store':
            continue
        for y in os.listdir(PATH + '/' + x):
            if y == '.DS_Store':
                continue
            d[x].append(y)

    l = []

    label_inv = {v: k for k, v in enumerate(classes)}

    for k, v in d.items():
        for x in v:
            filename = PATH + '/' + k + '/' + x
            target = label_inv[k]

            l.append((filename, target))

    l = np.array(l)
    l = l[np.random.permutation(len(l))]

    # filenames = filenames[:100]

    embeddings = []
    labels = []

    for i in range(len(l)):
        wav_data = load_wav_16k_mono(l[i][0])
        scores, _embeddings, spectrogram = yamnet_model(wav_data)

        # unbatch _embeddings
        _embeddings = _embeddings.numpy().flatten()

        embeddings.append(_embeddings)

        # print(i, l[i][0], l[i][1])
        labels.append(int(l[i][1]))

    for _ in trange(5):
        for i in range(len(l)):
            wav_data = load_wav_16k_mono_augmented(l[i][0])
            scores, _embeddings, spectrogram = yamnet_model(wav_data)

            # unbatch _embeddings
            _embeddings = _embeddings.numpy().flatten()

            embeddings.append(_embeddings)

            # print(i, l[i][0], l[i][1])
            labels.append(int(l[i][1]))

            # save file

            # sf.write(str(i) + '.wav', wav_data, 16000)

            # exit()

    embeddings = np.array(embeddings)
    labels = np.array(labels)

    # print(embeddings.shape)
    # print(labels.shape)

    # exit()

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(6 * 1024,)),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(11)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(
                      from_logits=True),
                  metrics=['accuracy'])

    # on every epoch, save the model weights
    checkpointer = tf.keras.callbacks.ModelCheckpoint(
        filepath='checkpoints', verbose=1)

    csv_logger = tf.keras.callbacks.CSVLogger('training.log')

    model.fit(embeddings, labels, epochs=10, validation_split=0.2,
              callbacks=[checkpointer, csv_logger])

    model.save('model_final.h5')


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


def load_wav_16k_mono(filename):
    wav, sr = librosa.load(filename, sr=16000)

    return wav


def load_wav_16k_mono_augmented(filename):
    wav, sr = librosa.load(filename, sr=16000)

    rand_scale = random.uniform(0.7, 1.2)
    rand_shift = random.uniform(-0.2, 0.2)
    rand_stretch = random.uniform(0.8, 1.2)
    rand_pitch = random.uniform(-2, 2)
    rand_noise = random.uniform(0, 0.05)

    wav = scale_volume(wav, rand_scale)
    wav = shift_audio(wav, sr, rand_shift)
    wav = stretch_audio(wav, rand_stretch)
    wav = pitch_shift(wav, sr, rand_pitch)
    wav = add_noise(wav, rand_noise)

    return wav


if __name__ == '__main__':
    main()
