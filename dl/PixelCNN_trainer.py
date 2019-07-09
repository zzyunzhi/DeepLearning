import tensorflow as tf
import numpy as np
import pickle
from PixelCNN import PixelCNN
import matplotlib.pyplot as plt
from pyprind import ProgBar


class Trainer(object):
    def __init__(
            self,
            model,
            data_train,
            data_test,
            batch_size,
            n_epochs,
            show_progress_per_epoch,
            sess=None,
    ):
        self.model = model
        self.data_train = data_train
        self.data_test = data_test
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.show_progress_per_epoch = show_progress_per_epoch
        if sess is None:
            sess = tf.Session()
        self.sess = sess

    def train(self):
        with self.sess.as_default() as sess:
            sess.run(tf.initializers.global_variables())
            loss_train = []
            loss_test = []
            n_batches = np.ceil(len(self.data_train) / self.batch_size)
            vis_batch = self.data_test[:5]
            pbar = ProgBar(self.n_epochs)
            for epoch in range(self.n_epochs):
                pbar.update(1)
                # print("\n--------- epoch {} --------".format(epoch))
                loss_train_batch = []
                idx = np.arange(len(self.data_train))
                np.random.shuffle(idx)
                for batch in np.array_split(self.data_train[idx], n_batches):
                    loss = self.model.train_step(batch)
                    loss_train_batch.append(loss)
                loss_train.append(np.mean(loss_train_batch))
                loss_test.append(self.model.test_step(self.data_test))
                if epoch % self.show_progress_per_epoch == 0:
                    print("at epoch", epoch, loss_train[-1], loss_test[-1])
                    self.model.show_progress(vis_batch, save_dir='./assets/', prefix='{}'.format(epoch))
            pbar.stop()


if __name__ == '__main__':
    tf.reset_default_graph()
    np.random.seed(0)
    tf.set_random_seed(0)

    with open('./dataset/mnist-hw1.pkl', 'rb') as f:
        data = pickle.load(f)
    data_train = data['train']
    data_test = data['test']

    IMG_SIZE = (28, 28, 3)
    COLOR_DIM = 4

    model = PixelCNN(
        img_size=IMG_SIZE,
        color_dim=COLOR_DIM,
    )
    trainer = Trainer(
        model,
        data_train,
        data_test,
        batch_size=32,
        n_epochs=50,
        show_progress_per_epoch=10,
    )

    trainer.train()

    with trainer.sess.as_default():
        images = model.reconstruct_images(data_test[:2])
        for image in images:
            plt.imshow(image/COLOR_DIM, interpolation='nearest')
        plt.show()
