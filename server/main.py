from fastapi import File, UploadFile, FastAPI
import numpy as np
import tensorflow as tf
import uvicorn

import librosa

app = FastAPI()


SECRET = "znj"

model = tf.keras.models.load_model("model.h5")

instuments = ['cel', 'cla', 'flu', 'gac', 'gel',
              'org', 'pia', 'sax', 'tru', 'vio', 'voi']


def audio_to_mfcc(audio, sampling_rate):
    mfcc = librosa.feature.mfcc(y=audio, sr=sampling_rate, n_mfcc=100)
    return mfcc


def mfcc_to_audio(mfcc, sampling_rate):
    audio = librosa.feature.inverse.mfcc_to_audio(mfcc, sr=sampling_rate)
    return audio


def process_file(file: UploadFile):
    contents = file.file.read()

    with open(file.filename, 'wb') as f:
        f.write(contents)

    sound, sampling_rate = librosa.load(file.filename)
    sound = librosa.resample(y=sound, orig_sr=sampling_rate, target_sr=22050)

    l = []

    input_ = [
        sound[x:x + 22050 * 3]
        for x in range(0, len(sound), 22050 * 1)
        if x + 22050 * 3 < len(sound)
    ]

    input_ = np.array(input_)

    input_ = np.array([
        audio_to_mfcc(x, 22050)
        for x in input_
    ])

    # for x in input_:
    #     print(x.shape)

    input_ = np.expand_dims(input_, axis=-1)

    prediction = model.predict(input_)

    for pred in prediction:

        d = []

        for ins, pred in zip(instuments, pred):
            d.append((ins, float(pred)))

        l.append(d)

    return l


@app.post("/upload")
def upload(file: UploadFile = File(...), secret_token: str = None):
    try:
        if secret_token != SECRET:
            return {"message": "Invalid secret token"}

        # if file.content_type != "audio/wav":
        #     return {"message": "Invalid file type"}

        out = process_file(file)

        return {
            "message": f"Successfully processed {file.filename}",
            "data": out
        }

    except Exception as e:
        print(e)
        return {"message": "There was an error uploading the file"}

    finally:
        file.file.close()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
