import numpy as np
import os
import tensorflow as tf
from tqdm import tqdm

save_dir = "dataset/mnist_data/"
os.makedirs(save_dir, exist_ok=True)


def download_and_save_mnist(save_dir):
    (x_train_all, y_train_all), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Save training data with progress bar
    for array, name in zip(
        [x_train_all, y_train_all, x_test, y_test],
        ["x_train.npy", "y_train.npy", "x_test.npy", "y_test.npy"],
    ):
        with tqdm(total=len(array), desc=f"Saving {name}") as pbar:
            np.save(os.path.join(save_dir, name), array)
            pbar.update(len(array))

    print(f"Dataset downloaded and saved locally at {save_dir}")


def load_mnist_from_local(save_dir):
    x_train_all = np.load(os.path.join(save_dir, "x_train.npy"))
    y_train_all = np.load(os.path.join(save_dir, "y_train.npy"))
    x_test = np.load(os.path.join(save_dir, "x_test.npy"))
    y_test = np.load(os.path.join(save_dir, "y_test.npy"))
    print(f"Dataset loaded from local files at {save_dir}")
    return (x_train_all, y_train_all), (x_test, y_test)


def load_mnist():
    if not os.path.exists(os.path.join(save_dir, "x_train.npy")):
        download_and_save_mnist(save_dir)
    return load_mnist_from_local(save_dir)


if __name__ == "__main__":
    load_mnist()
