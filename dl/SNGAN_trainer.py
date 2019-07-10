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
            d_loss_train, g_loss_train = [], []
            n_batches = np.ceil(len(self.data_train) / self.batch_size)
            for epoch in range(self.n_epochs):
                print(f"\n--------- epoch {epoch} --------")
                pbar = ProgBar(n_batches)
                d_loss_train_batch, g_loss_train_batch = [], []
                idx = np.arange(len(self.data_train))
                np.random.shuffle(idx)
                for batch_x_flatten in np.array_split(self.data_train[idx], n_batches):
                    pbar.update(1)
                    batch_x = np.reshape(batch_x_flatten, (-1,) + self.img_size)
                    batch_z = np.random.normal(0, 1, [self.batch_size, self.latent_size])
                    d_loss = self.model.train_d_step(batch_x, batch_z)
                    g_loss = self.model.train_g_step(batch_z)
                    d_loss_train_batch.append(d_loss)
                    g_loss_train_batch.append(g_loss)
                d_loss_train.append(np.mean(d_loss_train_batch))
                g_loss_train.append(np.mean(g_loss_train_batch))
                if epoch % self.show_progress_per_epoch == 0:
                    print("at epoch", epoch, d_loss_train[-1], g_loss_train[-1])
                    self.model.show_progress()
                pbar.stop()
                self.model.display_meta_info()


if __name__ == '__main__':
    tf.reset_default_graph()
    np.random.seed(0)
    tf.set_random_seed(0)

    PARENT_DIR = './dataset/cifar-10-batches-py'

    with open(f'{PARENT_DIR}/data_batch_1', 'rb') as f:
        dict = pickle.load(f, encoding='bytes')
    print(dict.keys())
    data_train = dict[b'data']

    LEARNING_RATE = 1e-3
    BETA1=0.5
    BETA2=0.99

    model = SNGAN(
        lr=LEARNING_RATE,
        beta1=BETA1,
        beta2=BETA2,
    )
    trainer = Trainer(
        model,
        data_train,
        batch_size=128,
        n_dis=5,
        n_epochs=3,
        show_progress_per_epoch=1,
    )

    trainer.train()
