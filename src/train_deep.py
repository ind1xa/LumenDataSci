from wrangle_dataset import read_data
import tensorflow as tf


path = 'dataset/IRMAS_Training_Data'


def main():

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(
            32, (10, 10), activation='relu', input_shape=(100, 130, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(11, activation='softmax')
    ])

    gen = read_data(path)

    # print(next(gen)[1].shape)

    # exit()

    print(model.summary())

    model.compile(optimizer="adam",
                  loss='categorical_crossentropy', metrics=["accuracy"])

    print("Training model...")

    model.fit_generator(gen, epochs=10)

    test = model.evaluate(gen, steps=100)

    print(test)


if __name__ == '__main__':
    main()