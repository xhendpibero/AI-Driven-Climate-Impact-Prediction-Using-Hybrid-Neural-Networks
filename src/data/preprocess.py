import numpy as np
from .mnist_loader import load_mnist_data
from ..quantum.circuit import quanv

def preprocess_and_save(SAVE_PATH, n_train=50, n_test=30):
    """Quantum pre-processing of images and saving the results."""
    (train_images, train_labels), (test_images, test_labels) = load_mnist_data(n_train, n_test)
    
    q_train_images = []
    print("Quantum pre-processing of train images:")
    for idx, img in enumerate(train_images):
        print("{}/{}        ".format(idx + 1, n_train), end="\r")
        q_train_images.append(quanv(img))
    q_train_images = np.asarray(q_train_images)

    q_test_images = []
    print("\nQuantum pre-processing of test images:")
    for idx, img in enumerate(test_images):
        print("{}/{}        ".format(idx + 1, n_test), end="\r")
        q_test_images.append(quanv(img))
    q_test_images = np.asarray(q_test_images)

    # Save pre-processed images
    np.save(SAVE_PATH + "q_train_images.npy", q_train_images)
    np.save(SAVE_PATH + "q_test_images.npy", q_test_images)