from src.models.MAML import MAML
import tensorflow as tf
import numpy as np
from Pyfhel import Pyfhel


class FMLEE:
    def __init__(self, no_clients, epochs):
        self.no_clients = no_clients
        self.epochs = epochs
        self.HE = self.CKKS()
        self.clients = []
        self.init_clients()

    def model_spec(self):
        model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Conv2D(
                    32, (3, 3), activation="relu", input_shape=(28, 28, 1)
                ),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dense(10),
            ]
        )
        return model

    def init_model(self):
        model = MAML(self.model_spec())
        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )
        return model

    def CKKS(self):
        HE = Pyfhel()
        ckks_params = {
            "scheme": "CKKS",
            "n": 2**14,  # Polynomial modulus degree. For CKKS, n/2 values can be
            "scale": 2**30,  # All the encodings will use it for float->fixed point
            "qi_sizes": [
                60,
                30,
                30,
                30,
                60,
            ],
        }
        HE.contextGen(**ckks_params)  # Generate context for ckks scheme
        HE.keyGen()  # Key Generation: generates a pair of public/secret keys
        HE.rotateKeyGen()
        HE.relinKeyGen()
        return HE

    def init_clients(self):
        for i in range(self.no_clients):
            self.clients.append(self.init_model())
            print(f"Client {i} initialized.")
