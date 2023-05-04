import sqlite3
import librosa
import soundfile as sf
import numpy as np
from uuid import uuid4


def mfcc_to_audio(mfcc, sampling_rate):
    audio = librosa.feature.inverse.mfcc_to_audio(mfcc, sr=sampling_rate)
    return audio


def uuid():
    return str(uuid4())


def main():
    with sqlite3.connect('data.db') as conn:
        c = conn.cursor()
        c.execute('SELECT * FROM data ORDER BY RANDOM() LIMIT 1;')

        for row in c.fetchall():
            instruments = row[1].split('_')
            mfcc = np.frombuffer(row[2], dtype=np.float32)
            mfcc = mfcc.reshape((128, 130))

            audio = mfcc_to_audio(mfcc, 22050)
            sf.write(f"{uuid()}_{row[1]}.wav", audio, 22050)


if __name__ == '__main__':
    main()
