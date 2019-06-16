import tensorflow as tf
import sys


def pprint(tensor, msg=""):
    #     tensor_to_print = tf.reshape(tensor, [-1, np.prod(tensor.get_shape()[1:])]) if reshape else tensor
    mean, var = tf.nn.moments(tensor, axes=None)

    with tf.control_dependencies([tf.print(msg + "\n",
                                           tensor.name, " has shape ",
                                           tf.shape(tensor), "\n",
                                           "mean: ", mean, "\n",
                                           "var: ", var, "\n",
                                           "max: ", tf.reduce_max(tensor), "\n",
                                           "min: ", tf.reduce_min(tensor), "\n",
                                           #                                            tensor_to_print,
                                           output_stream=sys.stdout)]):
        return tf.check_numerics(tensor, tensor.name)

