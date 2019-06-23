import tensorflow as tf
import numpy as np
import pickle
from VAE import VAE
from train import Trainer


if __name__ == '__main__':
    tf.reset_default_graph()
    np.random.seed(0)
    tf.set_random_seed(0)

    with open('./hw3-q2.pkl', 'rb') as f:
        data = pickle.load(f)
    data_train = data['train']
    data_test = data['test']
    print(data_test[0])

    model = VAE()
    trainer = Trainer(model, data_train, data_test)
    trainer.train()
