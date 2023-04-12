from wrangle_dataset import read_data
import tensorflow as tf


path = 'dataset/IRMAS_Training_Data'


def main():
    input_shape = (130, 180, 1)
    num_classes = 10

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
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
