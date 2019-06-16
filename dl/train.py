import tensorflow as tf
import numpy as np
import pickle
from VAE import VAE


class Trainer(object):
    def __init__(
            self,
            model,
            data_train,
            data_test,
            batch_size=256,
            n_epochs=10,
            log_per_epoch=1,
            print_per_epoch=1,
            sess=None,
    ):
        if sess is None:
            sess = tf.Session()
        self.model = model
        self.data_train = data_train
        self.data_test = data_test
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.log_per_epoch = log_per_epoch
        self.print_per_epoch = print_per_epoch
        self.sess = sess

    def train(self):
        with self.sess.as_default() as sess:
            # initialize
            init_op = tf.initializers.global_variables()
            sess.run(init_op)

            loss_trn = []
            loss_val = []

            for epoch in range(self.n_epochs):
                print("\n--------- epoch {} --------".format(epoch))
                loss_trn_batch = []
                for batch in np.array_split(self.data_train, np.ceil(len(self.data_train) / self.batch_size)):
                    loss = self.model.train_step(batch)
                    loss_trn_batch.append(loss)

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
