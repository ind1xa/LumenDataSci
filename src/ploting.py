import random
import librosa
import tensorflow as tf
import numpy as np
import os
import keras
import matplotlib.pyplot as plt
import soundfile as sf

PATH = "../dataset/IRMAS_Training_Data/"

instuments = ['cel', 'cla', 'flu', 'gac', 'gel',
              'org', 'pia', 'sax', 'tru', 'vio', 'voi']
def main():

    for inst in instuments:
        fig, ax =  plt.subplots(nrows=1, ncols=3, sharex=True, figsize=(10, 4))
        dir_list = os.listdir("../dataset/IRMAS_Training_Data/"+inst)
        i = np.random.randint(0, len(dir_list))

        sound, sampling_rate = librosa.load(PATH + inst + "/" + dir_list[i])
        sound = librosa.resample(y=sound, orig_sr=sampling_rate, target_sr=22050)
        sound = sound[:22050 * 3]
        
        librosa.display.waveshow(sound, sr=22050, ax=ax[0])

        mfcc = librosa.feature.mfcc(y=sound, sr=22050) 
        librosa.display.specshow(mfcc, x_axis='time', ax=ax[1])        

        melspectogram = librosa.feature.melspectrogram(y=sound, sr=22050)

        librosa.display.specshow(librosa.power_to_db(melspectogram), x_axis='time', y_axis='log', ax=ax[2])
        plt.savefig(f'../data/images/{inst}_sample_img.png')
        plt.show()

if __name__ == '__main__':
    main()