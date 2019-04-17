"""Simple Model abstraction."""
import time
from tqdm import tqdm

import tensorflow as tf

from deep4rec.models.loss_functions import get_loss_fn
from deep4rec.models.optimizers import build_optimizer


class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()

    def train(self, ds, epochs, loss_function, optimizer="adam", verbose=True):
        if isinstance(loss_function, str):
            loss_function = get_loss_fn(loss_function)

        if isinstance(optimizer, str):
            optimizer = build_optimizer(optimizer)

        loss_plot = []
        for epoch in tqdm(range(epochs)):
            total_loss, count = 0, 0
            start = time.time()
            for (data, target) in ds:
                count += data.shape.as_list()[0]
                with tf.GradientTape() as tape:
                    pred_rating = self.call(data)
                    loss = loss_function(target, pred_rating)

                total_loss += loss.numpy()
                gradients = tape.gradient(loss, self.real_variables)
                optimizer.apply_gradients(
                    zip(gradients, self.real_variables),
                    tf.train.get_or_create_global_step(),
                )

            # storing the epoch end loss value to plot later
            loss_plot.append(total_loss / count)

            if verbose:
                print("Epoch {} Loss {:.6f}".format(epoch + 1, loss_plot[-1]))
                print("1 epoch = {} secs\n".format(time.time() - start))

    def call(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def real_variables(self):
        return self.variables
