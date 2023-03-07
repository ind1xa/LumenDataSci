from wrangle_dataset import generate_all_data
import tensorflow as tf
import evaluation as ev

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

    """ model = tf.keras.applications.VGG19(
        include_top=True,
        weights="imagenet",
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000,
        classifier_activation="softmax",
    ) """

    gen = generate_all_data(path)


    # exit()

    print(model.summary())

    model.compile(optimizer="adam",
                  loss='categorical_crossentropy', metrics=["accuracy"])

    print("Training model...")

    model.fit_generator(gen, epochs=1)

    print("Evaluating model...")

    # get x_test and y_test from generate_all_data

    x_test = next(gen)[0]
    y_test = next(gen)[1]

    evaluate = ev.Evaluation(model, x_test, y_test)

    #test = model.evaluate(gen, steps=100)

    evaluate.print_evaluation()


if __name__ == '__main__':
    main()
