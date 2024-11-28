import tensorflow as tf

@tf.function
def diffusion_modulator_tf(length, beta):
    numerator = tf.pow(-beta, length)
    denominator = tf.pow(2.0, tf.cast(length, tf.float32)) * tf.math.gamma(tf.cast(length + 1, tf.float32))
    return numerator / denominator