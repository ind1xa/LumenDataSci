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
from tqdm import tqdm, trange

import json

yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
yamnet_model = hub.load(yamnet_model_handle)

val_data = "/Users/nitkonitkic/Documents/LumenDataSci/dataset/IRMAS_Validation_Data"


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)  # only difference


model = tf.keras.models.load_model('model_final.h5')


def get_score(no_ext_path):
    # random_file = random.choice(os.listdir(val_data))
    # no_extension = ".".join(random_file.split('.')[:-1])

    y, sr = librosa.load(no_ext_path + '.wav', sr=16000)

    classes = ['pia', 'voi', 'tru', 'sax', 'org',
               'cla', 'gac', 'vio', 'flu', 'gel', 'cel']

    label_inv = {v: k for k, v in enumerate(classes)}

    # cut out first 3 seconds

    THRESHOLD = 0.9

    detected = []

    for start in range(0, len(y), 3 * sr):

        y = y[start: start + 3 * sr]

        if len(y) < 3 * sr:
            continue

        scores, _embeddings, spectrogram = yamnet_model(y)

        _embeddings = _embeddings.numpy().flatten()

        # print(embeddings.shape)
        # print(labels.shape)

        # exit()

        # laod model

        # # evaluate model

        prediction = model.predict(np.array([_embeddings]))

        # print(prediction)

        # tanh
        pred_classes = np.tanh(prediction[0])

        print(pred_classes)

        # print(pred_classes)

        for ins, x in enumerate(pred_classes):

            if x > THRESHOLD:
                detected.append(classes[ins])

    real_classes = open(no_ext_path + '.txt').read().split('\n')

    real_classes = [
        s.strip() for s in real_classes if s.strip() != ''
    ]

    # print(detected)
    # print(real_classes)

    return (
        [label_inv[x] for x in detected],
        [label_inv[x] for x in real_classes]
    )


def main():

    l = []

    for x in os.listdir(val_data):
        if x == '.DS_Store':
            continue
        no_ext = ".".join(x.split('.')[:-1])
        l.append(no_ext)

    for_scoring = []
    for x in tqdm(l):
        det, real = get_score(val_data + '/' + x)

        for_scoring.append((x, det, real))

    json.dump(for_scoring, open('for_scoring.json', 'w'))


if __name__ == '__main__':
    main()
