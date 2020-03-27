from bearx.layers import Layer
from bearx.tensor import Tensor
from bearx.utils.data.DataLoader import DataLoader
from bearx.losses import Loss, MSE
from bearx.optimizers import Optimizer, SGD
from bearx.callbacks.callbacks import History

import numpy as np
from tqdm import tqdm

import os
import pickle


losses = {
    "mse": MSE,
}


class Sequential:
    """
    Sequential model. As initialization we
    create list of layers and flag of being
    compiled
    """

    def __init__(self) -> None:
        self.layers = []
        self.compiled = False
        #self.history = History()

    def add(self, layer: Layer):
        """
        Add new layer to the stack
        """
        self.layers.append(layer)

    def skeleton(self):
        """
        Prints out model architecture
        """
        print(24 * ' ', "Model Summary")
        print(63 * '=')

        layer_idx = 0
        if self.layers:
            for layer in self.layers:
                name = layer.__class__.__name__
                print(f"{name.upper()} (layer_{layer_idx})", end="\n")
                print(layer, end="\n")
                layer_idx += 1
        else:
            print(20 * " " + "Model has no layers yet!")
            print(17 * " " + "To add some use add() method")
        print(63 * '=')

    def feed_forward(self, inputs: Tensor) -> Tensor:
        """
        We iterate over all layers and forward
        propagate

        Parameters:
        -----------
        inputs: Tensor
            Input data

        Return:
        -------
        inputs: Tensor 
            Transformed input data
        """
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs

    def back_propagation(self, gradient: Tensor) -> Tensor:
        """
        We iterate over all layers and back
        propagate

        Parameters:
        -----------
        gradient: Tensor
            Gradient(Derivative) of the loss function

        Return:
        -------
        gradient: Tensor
        """
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)
        return gradient

    def get_params_and_gradients(self):
        """
        Helper function to get weights
        and gradients for all layers
        in the model
        """
        if self.optimizer.momentum != 0:
            for layer in self.layers:
                for params, w_update in zip(layer.params.items(), layer.w_update.values()):
                    name = params[0]
                    grad = layer.grads[name]
                    yield params[1], grad, w_update
        else:
            for layer in self.layers:
                for name, param in layer.params.items():
                    grad = layer.grads[name]
                    yield param, grad

    def compile(self,
                lr: float = 0.01,
                loss: str = 'mse',
                batch_size: int = 32,
                shuffle: bool = True,
                iterator: DataLoader = DataLoader(),
                optimizer: Optimizer = SGD()) -> None:
        self.loss = losses[loss]()
        self.iterator = iterator
        self.iterator.batch_size = batch_size
        self.iterator.shuffle = shuffle
        self.optimizer = optimizer
        self.optimizer.lr = lr
        self.compiled = True

    def train(self,
              inputs: Tensor,
              labels: Tensor,
              epochs: int = 5000,
              verbose: bool = False) -> History:
        assert self.compiled, ("Before Training You have "
                               "to compile the model!")
        print("The Training have begun!")
        history = History()
        for epoch in tqdm(range(epochs), desc=f"Training"):
            epoch_loss = 0.0
            for batch in self.iterator(inputs, labels):
                preds = self.feed_forward(batch.inputs)
                loss = self.loss.loss(preds, batch.targets)
                epoch_loss += loss
                gradient = self.loss.gradient(preds, batch.targets)
                self.back_propagation(gradient)
                self.optimizer.step(self)
            history.on_epoch_end(epoch, logs={"loss": loss})
            if verbose:
                print(f"Epoch: {epoch}, Loss: {epoch_loss:.5f}")
        return history

    def save_weights(self, name):
        i = 0
        dir_path = f"../model_data/weights/{name}_model_weights"
        print(os.listdir(".."))
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        for layer in self.layers:
            with open(os.path.join(dir_path, f"layer_{i}_weights.pkl"), "wb") as f:
                pickle.dump(layer.params, f, pickle.HIGHEST_PROTOCOL)
                i += 1
        print("Successfully saved model weights!")

    def load_weights(self, name):
        i = 0
        dir_path = f"../model_data/weights/{name}_model_weights"
        if os.path.exists(dir_path):
            for layer in self.layers:
                with open(os.path.join(dir_path, f"layer_{i}_weights.pkl"), "rb") as f:
                    params = pickle.load(f)
                    for name, param in params.items():
                        layer.params[name] = param
                i += 1
        else:
            raise NotADirectoryError(f"Cant find a directory: {dir_path}")
        print("Weight Loaded Successfully!")

    def predict(self, x: Tensor) -> Tensor:
        if len(x.shape) == 1:
            x = np.expand_dims(x, axis=0)
        preds = self.feed_forward(x)
        return preds
