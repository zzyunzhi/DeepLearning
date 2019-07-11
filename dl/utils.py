import tensorflow as tf
from datetime import datetime
import sys
import scipy
import os
import matplotlib.pyplot as plt


def pprint(tensor, msg=""):
    #     tensor_to_print = tf.reshape(tensor, [-1, np.prod(tensor.get_shape()[1:])]) if reshape else tensor
    #mean, var = tf.nn.moments(tensor, axes=None)

    with tf.control_dependencies([tf.print(
            msg + "\n",
            tensor.name, " has shape ",
            tf.shape(tensor), "\n",
            #"mean: ", mean, "\n",
            #"var: ", var, "\n",
            #"max: ", tf.reduce_max(tensor), "\n",
            #"min: ", tf.reduce_min(tensor), "\n",
            #                                            tensor_to_print,
            output_stream=sys.stdout,
    )]):
        return tf.check_numerics(tensor, tensor.name)


def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")


def save_images(images, prefix='', save_dir='./', append_timestamp=False):
    os.makedirs(save_dir, exist_ok=True)
    for idx, image in enumerate(images):
        filename = '{}_img_{}.png'.format(prefix, idx)
        scipy.misc.imsave(os.path.join(save_dir, filename), image)


def display_images(images, n_rows, n_cols, width=10, height=10):
    assert len(images) <= n_rows * n_cols
    fig = plt.figure(figsize=(width, height))
    for idx, image in enumerate(images):
        fig.add_subplot(n_rows, n_cols, idx+1)
        plt.axis('off')
        plt.imshow(images[idx])
