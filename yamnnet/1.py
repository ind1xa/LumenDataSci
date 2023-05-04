from collections import defaultdict
import math
import os
import random

import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow_hub as hub
import librosa

import matplotlib.pyplot as plt

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

    filenames = []
    targets = []
    folds = []

    label_inv = {v: k for k, v in enumerate(classes)}

    for k, v in d.items():
        for x in v:
            filenames.append(PATH + '/' + k + '/' + x)
            targets.append(label_inv[k])
            folds.append(random.randint(0, 9))

    main_ds = tf.data.Dataset.from_tensor_slices((filenames, targets, folds))

    main_ds = main_ds.map(load_wav_for_map)

    main_ds = main_ds.shuffle(10000)

    # extract embedding
    main_ds = main_ds.map(extract_embedding).unbatch()

    cached_ds = main_ds.cache()

    train_ds = cached_ds.filter(lambda embedding, label, fold: fold < 9)
    val_ds = cached_ds.filter(lambda embedding, label, fold: fold == 9)

    def remove_fold_column(embedding, label, fold): return (embedding, label)

    train_ds = train_ds.map(remove_fold_column)
    val_ds = val_ds.map(remove_fold_column)

    train_ds = train_ds.shuffle(10000).batch(
        32).prefetch(tf.data.experimental.AUTOTUNE)
    val_ds = val_ds.cache().batch(32).prefetch(tf.data.AUTOTUNE)

    # use only 200 samples for validation

    # val_ds = val_ds.take(200)

    # print one sample

    # # model

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1024,), dtype=tf.float32,
                              name='input_embedding'),
        tf.keras.layers.Dense(128, activation='relu'),

        tf.keras.layers.Dense(len(classes))

    ], name='audioset_classifier')

    model.summary()

    # plot model

    tf.keras.utils.plot_model(model, show_shapes=True)
    plt.show()

    exit()

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True),
        metrics=['accuracy'])

    no_samples = len(filenames)

    model.fit(train_ds, epochs=100,
              steps_per_epoch=math.ceil(no_samples * 0.8 / 32),
              validation_steps=math.ceil(no_samples * 0.2 / 32),
              validation_data=val_ds)

    model.save('model.h5')


def load_wav_for_map(filename, label, fold):
    return load_wav_16k_mono(filename), label, fold


def extract_embedding(wav_data, label, fold):
    scores, embeddings, spectrogram = yamnet_model(wav_data)
    num_embeddings = tf.shape(embeddings)[0]
    return (embeddings,
            tf.repeat(label, num_embeddings),
            tf.repeat(fold, num_embeddings)
            )


def load_wav_16k_mono(filename):
    file_contents = tf.io.read_file(filename)
    wav, sample_rate = tf.audio.decode_wav(
        file_contents,
        desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    # wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)

    # wav = librosa.resample(
    #     y=wav.numpy(), orig_sr=sample_rate.numpy(), target_sr=16000)
    # assert sample_rate == 16000

    return wav


if __name__ == '__main__':
    main()
