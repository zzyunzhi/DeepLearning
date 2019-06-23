import tensorflow as tf
import sys


def minimize_and_clip(optimizer, objective, var_list, clip_val=10):
    """
    minimized `objective` using `optimizer` w.r.t. variables in
    `var_list` while ensure the norm of the gradients for each
    variable is clipped to `clip_val`
    """
    gradients = optimizer.compute_gradients(objective, var_list=var_list)
    for i, (grad, var) in enumerate(gradients):
        if grad is not None:
            gradients[i] = (tf.clip_by_norm(grad, clip_val), var)
    return optimizer.apply_gradients(gradients)


def pprint(tensor, msg=""):
    #     tensor_to_print = tf.reshape(tensor, [-1, np.prod(tensor.get_shape()[1:])]) if reshape else tensor
    mean, var = tf.nn.moments(tensor, axes=None)

    with tf.control_dependencies([tf.print(msg + "\n",
                                           tensor.name, " has shape ",
                                           tf.shape(tensor), "\n",
                                           "mean: ", mean, "\n",
                                           #"var: ", var, "\n",
                                           "max: ", tf.reduce_max(tensor), "\n",
                                           "min: ", tf.reduce_min(tensor), "\n",
                                           #                                            tensor_to_print,
                                           output_stream=sys.stdout)]):
        return tf.check_numerics(tensor, tensor.name)

