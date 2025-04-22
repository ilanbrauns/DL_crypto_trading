from abc import ABC, abstractmethod

class Model(ABC):
    @abstractmethod
    def build(self, input_shape):
        pass

    @abstractmethod
    def train(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass
