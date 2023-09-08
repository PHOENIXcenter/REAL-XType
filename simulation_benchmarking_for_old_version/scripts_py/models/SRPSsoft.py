import tensorflow as tf
import numpy as np
import time
from tensorflow.keras import layers, losses, regularizers
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.regularizers import l2

class SRPSsoft(Model):
    def __init__(self, layer_params):
        super(SRPSsoft, self).__init__()
        self.encoder_layers = [layers.Dense(
                layer_params['encoder_hdim'], 
                use_bias=layer_params['use_bias'], 
                activation=layer_params['activation'], 
                kernel_regularizer=regularizers.l1(l=layer_params['regularization'])
            ) 
            for _ in range(layer_params['encoder_layer_num'])
        ]
        self.class_layers = [layers.Dense(
                layer_params['class_hdim'], 
                use_bias=layer_params['use_bias'], 
                activation=layer_params['activation'], 
                kernel_regularizer=regularizers.l1(l=layer_params['regularization'])
            ) 
            for _ in range(layer_params['class_layer_num'] - 1)
        ]
        self.class_layers.append(layers.Dense(
                layer_params['subtype_num'], 
                use_bias=layer_params['use_bias'], 
                activation=None, 
                kernel_regularizer=regularizers.l1(l=layer_params['regularization'])
            )
        )
    
    def call(self, x, drop_rate=0., is_training=False):
        for layer in self.encoder_layers:
            x = layer(tf.nn.dropout(x, rate=drop_rate))
        for layer in self.class_layers:
            x = layer(tf.nn.dropout(x, rate=drop_rate))
        return x

    def predict(self, x):
        x = self.call(x)
        a = tf.math.argmax(x, axis=1)
        return tf.squeeze(a).numpy()

    def predict_proba(self, x):
        x = self.call(x)
        p = tf.nn.softmax(x, axis=1)
        return tf.squeeze(p).numpy()

def make_models(layer_params, learning_rate):
    classifier = SRPSsoft(layer_params)
    # optimizer = tf.keras.optimizers.Adam(learning_rate)
    optimizer = tf.keras.optimizers.SGD(learning_rate, momentum=0.9)
    return classifier, optimizer

def rmst_loss(logits, lifetime, dead):
    outs = tf.nn.softmax(logits)
    k = outs.shape[1]
    RMST = [findRMST(lifetime, dead, outs[:, ci]) for ci in range(k)]
    loss = -tf.reduce_mean(
        tf.stack(
            [tf.reduce_mean(RMST[i]) - tf.reduce_mean(RMST[i + 1]) for i in range(k - 1)]
        )
    )
    return loss

def findRMST(lifetimes, deads, weights=None, max_life_time=601, resolution=0.1):
    if weights is None:
        weights = tf.ones_like(lifetimes, dtype=tf.float32)
    freq_lifetimes = tf.math.bincount(lifetimes, weights, minlength=max_life_time)
    freq_lifetimesDead = tf.math.bincount(lifetimes, weights * tf.cast(deads, tf.float32), minlength=max_life_time)
    nAlive = tf.reverse(tf.math.cumsum(tf.reverse(freq_lifetimes, [0])), [0])
    KMLambda = freq_lifetimesDead / nAlive
    KMProd = tf.math.cumprod(1 - KMLambda, 0)
    RMST = tf.reduce_sum(KMProd) * resolution
    return RMST

@tf.function
def train_step(x1, y1, x2, OS, status, DFS, recurrence, classifier, optimizer, loss_rates, dropout, lr):

    with tf.GradientTape(persistent=True) as tape:
        # supervised loss
        logits = classifier(x1, dropout)
        loss_su = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y1, logits=logits))

        # survival loss
        prob = tf.nn.softmax(classifier(x2, dropout))
        loss_surv_OS = rmst_loss(prob, tf.cast(OS * 10, dtype=tf.int32), status)
        loss_surv_DFS = rmst_loss(prob, tf.cast(DFS * 10, dtype=tf.int32), recurrence)
        loss_surv = (loss_surv_OS + loss_surv_DFS) / 2

        # l1 normalization loss
        loss_l1 = tf.add_n(classifier.losses)

        loss = loss_su * loss_rates[0] + loss_surv * loss_rates[1] + loss_l1

    gradients = tape.gradient(loss, classifier.trainable_variables)
    optimizer.learning_rate = lr
    optimizer.apply_gradients(zip(gradients, classifier.trainable_variables))

    return loss_su, loss_surv, loss_l1
