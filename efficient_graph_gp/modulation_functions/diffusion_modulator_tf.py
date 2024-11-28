import tensorflow as tf

def diffusion_modulator_tf(length: tf.Tensor, beta: tf.Tensor) -> tf.Tensor:
    length = tf.cast(length, tf.float64)
    beta = tf.cast(beta, tf.float64)
    two = tf.constant(2.0, dtype=tf.float64)
    numerator = tf.pow(-beta, length)
    denominator = tf.pow(two, length) * tf.exp(tf.math.lgamma(length + 1.0))
    return numerator / denominator