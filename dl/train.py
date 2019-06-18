import tensorflow as tf
import numpy as np
import pickle
from VAE import VAE
from pyprind import ProgBar


class Trainer(object):
    def __init__(
            self,
            model,
            data_train,
            data_test,
            batch_size=256,
            n_epochs=3,
            log_per_epoch=1,
            print_per_epoch=1,
            sess=None,
    ):
        self.model = model
        self.data_train = data_train
        self.data_test = data_test
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.log_per_epoch = log_per_epoch
        self.print_per_epoch = print_per_epoch
        if sess is None:
            sess = tf.Session()
        self.sess = sess

    def train(self):
        with self.sess.as_default() as _:
            loss_trn = []
            loss_val = []
            n_batches = np.ceil(len(self.data_train) / self.batch_size)

            for epoch in range(self.n_epochs):
                print("\n--------- epoch {} --------".format(epoch))
                pbar = ProgBar(n_batches)
                loss_trn_batch = []
                idx = np.arange(len(self.data_train))
                np.random.shuffle(idx)
                for batch in np.array_split(self.data_train[idx], n_batches):
                    loss = self.model.train_step(batch)
                    loss_trn_batch.append(loss)
                    pbar.update(1)
                pbar.stop()
                if epoch % self.log_per_epoch == 0:
                    loss_trn.append(np.mean(loss_trn_batch))
                    loss_val.append(self.model.test_step(self.data_test))
                if epoch % self.print_per_epoch == 0:
                    print("at epoch", epoch, loss_trn[-1], loss_val[-1])


if __name__ == '__main__':

    with open('./hw3-q2.pkl', 'rb') as f:
        data = pickle.load(f)
    data_train = data['train']
    data_test = data['test']

    model = VAE()
    trainer = Trainer(model, data_train, data_test)
    trainer.train()
