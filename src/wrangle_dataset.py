import pandas as pd
import numpy as np
import librosa
import os

path = 'dataset/IRMAS_Training_Data/'

def add_blanking(all_audio):
    for audio in all_audio:
        audio_length = len(audio)
        gap_length = np.random.randint(0, audio_length)
        gap_start = np.random.randint(0, audio_length - gap_length)
        audio[gap_start:gap_start + gap_length] = 0
    


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

            print('Audio: ', audio)
            all_audio.append(audio)
            exit()
    
    return all_audio

    # print('Categories: ', categories)


def main():
    audio = read_data(path)
    insert_noise(audio)



if __name__ == '__main__':
    main()
