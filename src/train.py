import numpy as np
from tensorflow import keras
from .data.preprocess import preprocess_and_save
from .model.model import MyModel
from .utils.plotting import plot_training_history

SAVE_PATH = "../saved_data/"
PREPROCESS = True
n_epochs = 30
n_train = 50
n_test = 30

def main():
    if PREPROCESS:
        preprocess_and_save(SAVE_PATH, n_train, n_test)
    
    # Load pre-processed images
    q_train_images = np.load(SAVE_PATH + "q_train_images.npy")
    q_test_images = np.load(SAVE_PATH + "q_test_images.npy")
    (train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()
    train_images = train_images[:n_train] / 255.0
    test_images = test_images[:n_test] / 255.0
    train_images = train_images[..., keras.backend.newaxis]
    test_images = test_images[..., keras.backend.newaxis]

    q_model = MyModel()
    q_history = q_model.fit(
        q_train_images,
        train_labels,
        validation_data=(q_test_images, test_labels),
        batch_size=4,
        epochs=n_epochs,
        verbose=2,
    )

    c_model = MyModel()
    c_history = c_model.fit(
        train_images,
        train_labels,
        validation_data=(test_images, test_labels),
        batch_size=4,
        epochs=n_epochs,
        verbose=2,
    )

    plot_training_history(q_history, c_history)

if __name__ == "__main__":
    main()