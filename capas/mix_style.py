import tensorflow as tf
from tensorflow import keras
from keras import layers

class MixStyle(layers.Layer):
    def __init__(self, p=0.5, alpha=0.1, eps=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.p = p
        self.alpha = alpha
        self.eps = eps

    def call(self, x, training=None):
        # Si no está entrenando, devolver x directamente
        if training is False:
            return x

        def apply_mixstyle():
            # Calcular mean y std por muestra
            mean, var = tf.nn.moments(x, axes=[1, 2], keepdims=True)
            std = tf.sqrt(var + self.eps)
            x_normed = (x - mean) / std

            # Parámetros aleatorios
            lmda = tf.random.gamma(shape=[tf.shape(x)[0], 1, 1, 1],
                                   alpha=self.alpha, beta=1.0)
            lmda = tf.clip_by_value(lmda, 0.0, 1.0)

            # Mezclar por permutación
            perm = tf.random.shuffle(tf.range(tf.shape(x)[0]))
            mean2 = tf.gather(mean, perm)
            std2 = tf.gather(std, perm)

            mean_mix = lmda * mean + (1 - lmda) * mean2
            std_mix = lmda * std + (1 - lmda) * std2

            return x_normed * std_mix + mean_mix

        # Aplicar MixStyle con probabilidad p
        return tf.cond(
            tf.less(tf.random.uniform([], 0, 1), self.p),
            apply_mixstyle,
            lambda: x
        )

    def get_config(self):
        config = super().get_config()
        config.update({"p": self.p, "alpha": self.alpha, "eps": self.eps})
        return config
