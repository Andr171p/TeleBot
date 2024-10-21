import keras
from keras import layers

from typing import Any


class BinaryNetworkClassifierModel:
    _model = keras.Sequential()

    def __init__(self, x_train: Any, y_train: Any) -> None:
        self._x_train = x_train
        self._y_train = y_train

    def create_model(self) -> None:
        self._model.add(layers.Dense(
            units=16,
            activation='relu',
            input_shape=(self._x_train.shape[1],)
        ))
        self._model.add(layers.Dense(
            units=8,
            activation='relu'
        ))
        self._model.add(layers.Dense(
            units=4,
            activation='relu'
        ))
        self._model.add(layers.Dense(
            units=1,
            activation='sigmoid'
        ))

    def compile_model(self) -> None:
        self._model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

    def train_model(self, epochs: int = 5, batch_size: int = 5) -> None:
        self._model.fit(
            x=self._x_train,
            y=self._y_train,
            epochs=epochs,
            batch_size=batch_size
        )

    def evaluate_model(self, x_test: Any, y_test: Any) -> None:
        self._model.evaluate(
            x=x_test,
            y=y_test
        )

    def model(self) -> Any:
        return self._model


from dataset import Dataset


x_train, y_train, x_test, y_test = Dataset().dataset()

model = BinaryNetworkClassifierModel(x_train=x_train, y_train=y_train)
model.create_model()
model.compile_model()
model.train_model()
model.evaluate_model(x_test=x_test, y_test=y_test)