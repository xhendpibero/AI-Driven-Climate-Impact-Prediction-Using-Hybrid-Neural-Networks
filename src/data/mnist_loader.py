import tensorflow as tf

def load_mnist_data(n_train=50, n_test=30):
    """Load and preprocess MNIST dataset."""
    mnist_dataset = tf.keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist_dataset.load_data()

    # Reduce dataset size
    train_images = train_images[:n_train]
    train_labels = train_labels[:n_train]
    test_images = test_images[:n_test]
    test_labels = test_labels[:n_test]

    # Normalize pixel values within 0 and 1
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # Add extra dimension for convolution channels
    train_images = train_images[..., tf.newaxis]
    test_images = test_images[..., tf.newaxis]

    return (train_images, train_labels), (test_images, test_labels)