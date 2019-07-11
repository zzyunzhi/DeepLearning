import tensorflow as tf
import numpy as np
import pickle
from SNGAN import SNGAN
from pyprind import ProgBar


class Trainer(object):
    def __init__(
            self,
            model,
            data_train,
            batch_size,
            n_dis,
            n_epochs,
            show_progress_per_epoch,
            sess=None,
    ):
        self.img_size = (32, 32, 3)
        self.latent_size = 128
        self.model = model
        self.data_train = data_train
        self.batch_size = batch_size
        self.n_dis = n_dis
        self.n_epochs = n_epochs
        self.show_progress_per_epoch = show_progress_per_epoch
        if sess is None:
            sess = tf.Session()
        self.sess = sess

    def train(self):
        with self.sess.as_default() as sess:
            sess.run(tf.initializers.global_variables())
            d_loss_train, g_loss_train, inception_scores = [], [], []
            n_batches = np.ceil(len(self.data_train) / self.batch_size)
            for epoch in range(self.n_epochs):
                print(f"\n--------- epoch {epoch} --------")
                pbar = ProgBar(n_batches)
                d_loss_train_batch, g_loss_train_batch, inception_scores_batch = [], [], []
                idx = np.arange(len(self.data_train))
                np.random.shuffle(idx)
                for batch_x_flatten in np.array_split(self.data_train[idx], n_batches):
                    pbar.update(1)
                    batch_x = np.reshape(batch_x_flatten, (-1,) + self.img_size)
                    batch_z = np.random.normal(0, 1, [self.batch_size, self.latent_size])
                    g_loss = self.model.train_g_step(batch_z)
                    g_loss_train_batch.append(g_loss)
                    for _ in range(self.n_dis):
                        d_loss = self.model.train_d_step(batch_x, batch_z)
                        d_loss_train_batch.append(d_loss)
                    # inception_scores_batch.append(self.model.test_step())

                d_loss_train.append(np.mean(d_loss_train_batch))
                g_loss_train.append(np.mean(g_loss_train_batch))
                if epoch % self.show_progress_per_epoch == 0:
                    print("at epoch", epoch, d_loss_train[-1], g_loss_train[-1])
                    self.model.show_progress()
                pbar.stop()


if __name__ == '__main__':
    tf.reset_default_graph()
    np.random.seed(0)
    tf.set_random_seed(0)

    data_train, data_test = tf.keras.datasets.cifar10.load_data()
    data_train, y_train = data_train
    data_test, y_test = data_test

    model = SNGAN(
        data_test,
    )
    trainer = Trainer(
        model,
        data_train,
        batch_size=128,
        n_dis=5,
        n_epochs=100,
        show_progress_per_epoch=1,
    )

    trainer.train()
