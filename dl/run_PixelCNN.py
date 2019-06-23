import tensorflow as tf
import numpy as np
import pickle
from PixelCNN import PixelCNN
from train import Trainer
import matplotlib.pyplot as plt


if __name__ == '__main__':
    tf.reset_default_graph()
    np.random.seed(0)
    tf.set_random_seed(0)

    with open('dataset/mnist-hw1.pkl', 'rb') as f:
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
    )

    trainer.train()

    with trainer.sess.as_default():
        images = model.reconstruct_images(data_test[:2])
        for image in images:
            plt.imshow(image/COLOR_DIM, interpolation='nearest')
        plt.show()
