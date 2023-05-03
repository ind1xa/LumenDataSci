import random
import tensorflow as tf
import numpy as np
import os
import keras


path = '../data_out'


def make_model():
    input_shape = (100, 130, 1)
    num_classes = 11

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(
            32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    return model


instuments = ['cel', 'cla', 'flu', 'gac', 'gel',
              'org', 'pia', 'sax', 'tru', 'vio', 'voi']

instuments_inv = {
    x: i for i, x in enumerate(instuments)
}


def generator():

    while True:

        x = random.choice(list(os.listdir(path)))

        if not x.endswith(".npz"):
            continue

        a = np.load(path + "/" + x, allow_pickle=True)

        out = []

        idx = np.random.choice(len(a["arr_0"]), 1000)

        for y in a["arr_0"][idx]:

            l = [0 for _ in range(11)]

            for z in y[0].split("_")[1:]:
                l[instuments_inv[z]] = 1

            out.append((y[1], l))

        yield np.array([x[0] for x in out]), np.array([x[1] for x in out])


def main():

    model = make_model()

    model.compile(optimizer="adam",
                  loss='categorical_crossentropy', metrics=["accuracy"])

    print("Training model...")

    # x, y = next(generator())

    # model.fit(x, y, epochs=10, batch_size=32)

    model.fit(generator(), epochs=1, steps_per_epoch=10)

    # test = model.evaluate(generator, steps=100)

    model.save("model.h5")

    # print(test)


if __name__ == '__main__':
    main()
