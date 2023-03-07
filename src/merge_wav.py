from pydub import AudioSegment
import pandas as pd
import numpy as np
import librosa
import soundfile as sf
import os
import itertools as it
from tqdm import tqdm

path = 'dataset/IRMAS_Training_Data/'
instuments = ['cel', 'cla', 'flu', 'gac', 'gel',
              'org', 'pia', 'sax', 'tru', 'vio', 'voi']
all_files = []


def merge_wav(path1, path2, delay, name1, name2, id):

    sound1 = AudioSegment.from_wav(path1).normalize()
    sound2 = AudioSegment.from_wav(path2).normalize()

    overflow = len(sound1) - delay*1000

    sound2 = sound2.overlay(sound1, position=delay*1000)
    sound2 = sound2.overlay(sound1[-overflow:], position=0)

    sound2.export('overlaysNorm/2overlaid' + str(id) + name1 + name2 + '.wav', format='wav')

def mrege_wav_3_files(path1, path2, path3, delay1, delay2, name1, name2, name3, id):
    
        sound1 = AudioSegment.from_wav(path1).normalize()
        sound2 = AudioSegment.from_wav(path2).normalize()
        sound3 = AudioSegment.from_wav(path3).normalize()
    
        overflow1 = len(sound1) - delay1*1000
        overflow2 = len(sound2) - delay2*1000
    
        sound3 = sound3.overlay(sound1, position=delay1*1000)
        sound3 = sound3.overlay(sound1[-overflow1:], position=0)
    
        sound3 = sound3.overlay(sound2, position=delay2*1000)
        sound3 = sound3.overlay(sound2[-overflow2:], position=0)
    
        sound3.export('overlaysNorm/3overlaid' + str(id) + name1 + name2 + name3 + '.wav', format='wav')

def merge_wav_4_files(path1, path2, path3, path4, delay1, delay2, delay3, name1, name2, name3, name4, id):
    
        sound1 = AudioSegment.from_wav(path1).normalize()
        sound2 = AudioSegment.from_wav(path2).normalize()
        sound3 = AudioSegment.from_wav(path3).normalize()
        sound4 = AudioSegment.from_wav(path4).normalize()
    
        overflow1 = len(sound1) - delay1*1000
        overflow2 = len(sound2) - delay2*1000
        overflow3 = len(sound3) - delay3*1000
    
        sound4 = sound4.overlay(sound1, position=delay1*1000)
        sound4 = sound4.overlay(sound1[-overflow1:], position=0)
    
        sound4 = sound4.overlay(sound2, position=delay2*1000)
        sound4 = sound4.overlay(sound2[-overflow2:], position=0)
    
        sound4 = sound4.overlay(sound3, position=delay3*1000)
        sound4 = sound4.overlay(sound3[-overflow3:], position=0)
    
        sound4.export('overlaysNorm/4overlaid' + str(id) + name1 + name2 + name3 + name4 + '.wav', format='wav')

def merge_wav_5_files(path1, path2, path3, path4, path5, delay1, delay2, delay3, delay4, name1, name2, name3, name4, name5, id):
        
            sound1 = AudioSegment.from_wav(path1).normalize()
            sound2 = AudioSegment.from_wav(path2).normalize()
            sound3 = AudioSegment.from_wav(path3).normalize()
            sound4 = AudioSegment.from_wav(path4).normalize()
            sound5 = AudioSegment.from_wav(path5).normalize()
        
            overflow1 = len(sound1) - delay1*1000
            overflow2 = len(sound2) - delay2*1000
            overflow3 = len(sound3) - delay3*1000
            overflow4 = len(sound4) - delay4*1000
        
            sound5 = sound5.overlay(sound1, position=delay1*1000)
            sound5 = sound5.overlay(sound1[-overflow1:], position=0)
        
            sound5 = sound5.overlay(sound2, position=delay2*1000)
            sound5 = sound5.overlay(sound2[-overflow2:], position=0)
        
            sound5 = sound5.overlay(sound3, position=delay3*1000)
            sound5 = sound5.overlay(sound3[-overflow3:], position=0)
        
            sound5 = sound5.overlay(sound4, position=delay4*1000)
            sound5 = sound5.overlay(sound4[-overflow4:], position=0)
        
            sound5.export('overlaysNorm/5overlaid' + str(id) + name1 + name2 + name3 + name4 + name5 + '.wav', format='wav')
    

def get_names(path_to_root):
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
            #print(path_to_file)

            file = file.split(']')
            for i in range(0, len(file)):
                file[i] = file[i][-3:]
                if file[i] not in instuments:
                    file[i] = ''
                else:
                    file[i] = '[' + file[i] + ']'

            file = ''.join(file)

            pair = [path_to_file, file]
            all_files.append(pair)

            #print(file)


def main():
    get_names(path)
    counter = 0

    while (counter < len(all_files) * 4):
        random_number1 = np.random.randint(0, len(all_files)-1)
        random_number2 = np.random.randint(0, len(all_files)-1)
        random_number3 = np.random.randint(0, len(all_files)-1)
        random_number4 = np.random.randint(0, len(all_files)-1)
        random_number5 = np.random.randint(0, len(all_files)-1)
        random_float = np.random.uniform(0, 3)
        random_float2 = np.random.uniform(0, 3)
        random_float3 = np.random.uniform(0, 3)
        random_float4 = np.random.uniform(0, 3)
        if (all_files[random_number1][1] != all_files[random_number2][1]):
            merge_wav(all_files[random_number1][0], all_files[random_number2][0], random_float, all_files[random_number1][1], all_files[random_number2][1], counter)
            counter += 1
        if (all_files[random_number1][1] != all_files[random_number2][1] and all_files[random_number1][1] != all_files[random_number3][1] and all_files[random_number2][1] != all_files[random_number3][1]):
            mrege_wav_3_files(all_files[random_number1][0], all_files[random_number2][0], all_files[random_number3][0], random_float, random_float2, all_files[random_number1][1], all_files[random_number2][1], all_files[random_number3][1], counter)
            counter += 1
        if (all_files[random_number1][1] != all_files[random_number2][1] and all_files[random_number1][1] != all_files[random_number3][1] and all_files[random_number2][1] != all_files[random_number3][1] and all_files[random_number1][1] != all_files[random_number4][1] and all_files[random_number2][1] != all_files[random_number4][1] and all_files[random_number3][1] != all_files[random_number4][1]):
            merge_wav_4_files(all_files[random_number1][0], all_files[random_number2][0], all_files[random_number3][0], all_files[random_number4][0], random_float, random_float2, random_float3, all_files[random_number1][1], all_files[random_number2][1], all_files[random_number3][1], all_files[random_number4][1], counter)
            counter += 1
        if (all_files[random_number1][1] != all_files[random_number2][1] and all_files[random_number1][1] != all_files[random_number3][1] and all_files[random_number2][1] != all_files[random_number3][1] and all_files[random_number1][1] != all_files[random_number4][1] and all_files[random_number2][1] != all_files[random_number4][1] and all_files[random_number3][1] != all_files[random_number4][1] and all_files[random_number1][1] != all_files[random_number5][1] and all_files[random_number2][1] != all_files[random_number5][1] and all_files[random_number3][1] != all_files[random_number5][1] and all_files[random_number4][1] != all_files[random_number5][1]):
            merge_wav_5_files(all_files[random_number1][0], all_files[random_number2][0], all_files[random_number3][0], all_files[random_number4][0], all_files[random_number5][0], random_float, random_float2, random_float3, random_float4, all_files[random_number1][1], all_files[random_number2][1], all_files[random_number3][1], all_files[random_number4][1], all_files[random_number5][1], counter)
            counter += 1
    print(counter)
    #merge_wav('example/130__[cel][nod][cla]0040__1.wav', 'example/[pia][cla]1308__3.wav', 1.5)

if __name__ == '__main__':
    main()