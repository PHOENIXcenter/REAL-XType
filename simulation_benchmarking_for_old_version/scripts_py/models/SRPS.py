import tensorflow as tf
import numpy as np
import time
from tensorflow.keras import layers, losses, regularizers
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.regularizers import l2

class SRPSNet(Model):
    def __init__(self, layer_params):
        super(SRPSNet, self).__init__()
        self.layer_list = [layers.Dense(
            hdim, use_bias=bias, activation=activation, kernel_regularizer=regularizers.l1(l=regularization)
        ) for hdim, bias, activation, regularization in layer_params]
    
    def call(self, x, drop_rate):
        for layer in self.layer_list:
            x = layer(tf.nn.dropout(x, rate=drop_rate))
        return x

    def predict(self, x):
        for layer in self.layer_list:
            x = layer(x)
        a = tf.math.argmax(x, axis=1)
        return tf.squeeze(a).numpy()

    def sample(self, x):
        for layer in self.layer_list:
            x = layer(x)
        a_sample = tf.random.categorical(x, 1)
        a_argmax = tf.math.argmax(x, axis=1)
        return tf.squeeze(a_sample).numpy(), tf.squeeze(a_argmax).numpy()

    def predict_proba(self, x):
        for layer in self.layer_list:
            x = layer(x)
        p = tf.nn.softmax(x, axis=1)
        return tf.squeeze(p).numpy()

class Baseline(Model):
    def __init__(self, layer_params, use_bias=False):
        super(Baseline, self).__init__()
        self.layer_list = [layers.Dense(
            hdim, activation=activation
        ) for hdim, bias, activation in layer_params]
        self.c = self.add_weight(
            shape=[], 
            dtype=tf.float32,
            initializer=tf.zeros_initializer(),
            trainable=True,
            name='baseline_c'
        )

    def call(self, x, r):
        x = self.layer_list[0](tf.stop_gradient(x))
        for layer in self.layer_list[1:]:
            x = layer(x)
        return tf.stop_gradient(r) - x - self.c

def make_models(layer_params_clf, layer_params_baseline, learning_rates):
    classifier = SRPSNet(layer_params_clf)
    baseline = Baseline(layer_params_baseline)
    classifier_optimizer = tf.keras.optimizers.Adam(learning_rates[0])
    baseline_optimizer = tf.keras.optimizers.Adam(learning_rates[1])
    return [classifier, baseline], [classifier_optimizer, baseline_optimizer]
    
def supervise_loss(logits, y):
    return tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)

def reinforce_and_baseline_loss(x, a, logits, reward):
    reinforce_loss = reward * tf.nn.sparse_softmax_cross_entropy_with_logits(labels=a, logits=logits)
    baseline_loss = tf.math.square(reward)
    return reinforce_loss, baseline_loss

@tf.function
def train_step_rl_no_baseline(x1, y1, x2, a, r, models, optimizers, loss_rates, dropout, dropout_rl):
    classifier, baseline = models
    classifier_optimizer, baseline_optimizer = optimizers

    with tf.GradientTape(persistent=True) as tape:
        # supervised loss
        logits1 = classifier(x1, dropout)
        loss_su = supervise_loss(logits1, y1)

        # reinforce loss
        logits2 = classifier(x2, dropout_rl)
        loss_rl, _ = reinforce_and_baseline_loss(x2, a, logits2, r)

        # l1 normalization loss
        loss_l1 = tf.add_n(classifier.losses)

        loss = tf.reduce_mean(loss_su) * loss_rates[0] + \
            tf.reduce_mean(loss_rl) * loss_rates[1] + \
            loss_l1

    classifier_gradients = tape.gradient(loss, classifier.trainable_variables)

    classifier_optimizer.apply_gradients(zip(classifier_gradients, classifier.trainable_variables))

    return loss_su, loss_rl, 0., loss_l1, [logits1, logits2]

@tf.function
def train_step_rl(x1, y1, x2, a, r, models, optimizers, loss_rates, dropout, dropout_rl):
    classifier, baseline = models
    classifier_optimizer, baseline_optimizer = optimizers

    with tf.GradientTape(persistent=True) as tape:
        # supervised loss
        logits1 = classifier(x1, dropout)
        loss_su = supervise_loss(logits1, y1)

        # reinforce loss
        logits2 = classifier(x2, dropout_rl)
        reward = baseline(x2, r)
        loss_rl, loss_bl = reinforce_and_baseline_loss(x2, a, logits2, reward)

        # l1 normalization loss
        loss_l1 = tf.add_n(classifier.losses)

        loss = tf.reduce_mean(loss_su) * loss_rates[0] + \
            tf.reduce_mean(loss_rl) * loss_rates[1] + \
            loss_l1

    classifier_gradients = tape.gradient(loss, classifier.trainable_variables)
    baseline_gradients = tape.gradient(loss_bl, baseline.trainable_variables)

    classifier_optimizer.apply_gradients(zip(classifier_gradients, classifier.trainable_variables))
    baseline_optimizer.apply_gradients(zip(baseline_gradients, baseline.trainable_variables))

    return loss_su, loss_rl, loss_bl, loss_l1, [logits1, logits2]