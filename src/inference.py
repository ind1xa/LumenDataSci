import random
import librosa
import tensorflow as tf
import numpy as np
import os
import keras

import soundfile as sf

PATH = "/Users/nitkonitkic/Documents/LumenDataSci/dataset/IRMAS_Validation_Data/(02) dont kill the whale-1.wav"


instuments = ['cel', 'cla', 'flu', 'gac', 'gel',
              'org', 'pia', 'sax', 'tru', 'vio', 'voi']


def audio_to_mfcc(audio, sampling_rate):
    mfcc = librosa.feature.mfcc(y=audio, sr=sampling_rate, n_mfcc=100)
    return mfcc


def mfcc_to_audio(mfcc, sampling_rate):
    audio = librosa.feature.inverse.mfcc_to_audio(mfcc, sr=sampling_rate)
    return audio


def main():

    model = tf.keras.models.load_model("model.h5")

    sound, sampling_rate = librosa.load(PATH)
    sound = librosa.resample(y=sound, orig_sr=sampling_rate, target_sr=22050)

    # trim to 3 seconds

    sound = sound[:22050 * 3]

    mfcc = audio_to_mfcc(sound, 22050)

    # infer

    mfcc = np.expand_dims(mfcc, axis=0)

    prediction = model.predict(mfcc)

    for ins, pred in zip(instuments, prediction[0]):
        print(ins, pred)


if __name__ == '__main__':
    main()
