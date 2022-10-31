import tensorflow as tf
import numpy as np
import time
from data_utils import get_rmst
from tensorflow.keras import layers, losses, regularizers
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.regularizers import l2

class Classifier(Model):
    def __init__(self, h_dim, subtype_num, regularization, use_bias=False):
        super(Classifier, self).__init__()
        self.dense1 = layers.Dense(h_dim, activation='relu', name='clf_dense1', 
            kernel_regularizer=regularizers.l1(l=regularization), use_bias=use_bias
        )
        self.dense2 = layers.Dense(subtype_num, activation=None, name='clf_dense2')
    
    def call(self, x, drop_rate1, drop_rate2):
        h = self.dense1(tf.nn.dropout(x, rate=drop_rate1))
        out = self.dense2(tf.nn.dropout(h, rate=drop_rate2))
        return out

    def sample(self, x):
        h = self.dense1(x)
        out = self.dense2(h)
        a_sample = tf.random.categorical(out, 1)
        a_argmax = tf.math.argmax(out, axis=1)
        return tf.squeeze(a_sample).numpy(), tf.squeeze(a_argmax).numpy()

    def predict(self, x):
        h = self.dense1(x)
        out = self.dense2(h)
        a = tf.math.argmax(out, axis=1)
        return tf.squeeze(a).numpy()

    def encode(self, x):
        h = self.dense1(x)
        return h.numpy()

    def prob(self, x):
        h = self.dense1(x)
        p = tf.nn.softmax(self.dense2(h))
        return p.numpy()

class Masked_linear_layer(tf.keras.layers.Layer):
    def __init__(self, output_dim, regularization, use_bias, mask, reweight, **kwargs):
        super(Masked_linear_layer, self).__init__()
        self.output_dim = output_dim
        self.regularization = regularization
        self.use_bias = use_bias
        self.mask_init = mask
        self.reweight = reweight

    def build(self, input_shape):
        self.kernel = self.add_weight(
            'kernel',
            shape=[int(input_shape[-1]), self.output_dim],
            initializer=tf.keras.initializers.GlorotUniform(),
            trainable=True,
            dtype=tf.float32,
            regularizer=regularizers.l1(l=self.regularization)
        )
        if self.use_bias:
            self.bias = self.add_weight(
            'bias',
            shape=[self.output_dim],
            initializer=tf.keras.initializers.Zeros(),
            trainable=True,
            dtype=tf.float32,
            regularizer=regularizers.l1(l=self.regularization)
        )
        self.mask = tf.constant(
            np.ones([int(input_shape[-1]), self.output_dim]), 
            name='mask', 
            dtype=tf.float32
        ) if self.mask_init is None else tf.constant(
            self.mask_init, 
            name='mask', 
            dtype=tf.float32
        )

    def call(self, x):
        out = tf.matmul(x, self.kernel * self.mask)
        if self.reweight:
            input_num = tf.reduce_sum(self.mask, axis=0)
            out = out / input_num * tf.reduce_mean(input_num)
        if self.use_bias:
            out += self.bias
        return out

    def update_mask(self, thresh):
        self.mask = tf.where(
            tf.math.less(tf.math.abs(self.kernel), thresh), 
            tf.zeros_like(self.mask), 
            self.mask
        )

    def set_mask(self, mask):
        self.mask.set_weights(mask)
        
    def get_mask(self):
        return self.mask.numpy()

    def get_masked_weight(self):
        return (self.kernel * self.mask).numpy()

class Classifier_simple(Model):
    def __init__(self, subtype_num, regularization, use_bias=False):
        super(Classifier_simple, self).__init__()
        self.masked_linear = Masked_linear_layer(subtype_num, regularization, use_bias, None, False, name='masked_linear')
    
    def call(self, x, drop_rate):
        out = self.masked_linear(tf.nn.dropout(x, rate=drop_rate))
        return out

    def sample(self, x):
        out = self.masked_linear(x)
        a_sample = tf.random.categorical(out, 1)
        a_argmax = tf.math.argmax(out, axis=1)
        return tf.squeeze(a_sample).numpy(), tf.squeeze(a_argmax).numpy()

    def predict(self, x):
        out = self.masked_linear(x)
        a = tf.math.argmax(out, axis=1)
        return tf.squeeze(a).numpy()

    def encode(self, x):
        out = self.masked_linear(x)
        return out.numpy()

    def prob(self, x):
        p = tf.nn.softmax(self.masked_linear(x))
        return p.numpy()

class Classifier_grouped(Model):
    def __init__(self, subtype_num, group_num, mask, regularization, use_bias=False):
        super(Classifier_grouped, self).__init__()
        self.masked_linear = Masked_linear_layer(group_num, regularization, use_bias, mask, True, name='masked_linear')
        # self.dense = layers.Dense(subtype_num, activation='relu', name='clf_dense', 
        #     kernel_regularizer=regularizers.l1(l=regularization), use_bias=use_bias
        # )
        self.dense = layers.Dense(subtype_num, activation='relu', name='clf_dense', use_bias=use_bias)
    
    def call(self, x, drop_rate):
        group_score = self.masked_linear(tf.nn.dropout(x, rate=drop_rate))
        out = self.dense(tf.nn.dropout(group_score, rate=drop_rate))
        return out

    def sample(self, x):
        out = self.dense(self.masked_linear(x))
        a_sample = tf.random.categorical(out, 1)
        a_argmax = tf.math.argmax(out, axis=1)
        return tf.squeeze(a_sample).numpy(), tf.squeeze(a_argmax).numpy()

    def predict(self, x):
        out = self.dense(self.masked_linear(x))
        a = tf.math.argmax(out, axis=1)
        return tf.squeeze(a).numpy()

    def encode(self, x):
        out = self.masked_linear(x)
        return out.numpy()

    def group_score(self, x):
        out = self.masked_linear(x)
        return out.numpy()

    def prob(self, x):
        p = tf.nn.softmax(self.dense(self.masked_linear(x)))
        return p.numpy()

class Baseline(Model):
    def __init__(self, h_dim=128):
        super(Baseline, self).__init__()
        self.baseline_x1 = layers.Dense(h_dim, activation='sigmoid', name='baseline_x1')
        self.baseline_x2 = layers.Dense(h_dim, activation='sigmoid', name='baseline_x2')
        self.baseline_x3 = layers.Dense(1, activation=None, name='baseline_x3')
        self.baseline_c = self.add_weight(
            shape=[], 
            dtype=tf.float32,
            initializer=tf.zeros_initializer(),
            trainable=True,
            name='baseline_c'
        )

    def call(self, x, r):
        h = self.baseline_x1(tf.stop_gradient(x))
        h = self.baseline_x2(h)
        baseline_x = self.baseline_x3(h)
        return tf.stop_gradient(r) - baseline_x - self.baseline_c

def make_models(h_dim, subtype_num, learning_rates, regularization, use_bias):
    classifier = Classifier(h_dim, subtype_num, regularization, use_bias)
    baseline = Baseline(h_dim)
    classifier_optimizer = tf.keras.optimizers.Adam(learning_rates[0])
    baseline_optimizer = tf.keras.optimizers.Adam(learning_rates[1])
    return [classifier, baseline], [classifier_optimizer, baseline_optimizer]

def make_simple_models(h_dim, subtype_num, learning_rates, regularization, use_bias=False):
    classifier = Classifier_simple(subtype_num, regularization, use_bias)
    baseline = Baseline(h_dim)
    classifier_optimizer = tf.keras.optimizers.Adam(learning_rates[0])
    baseline_optimizer = tf.keras.optimizers.Adam(learning_rates[1])
    return [classifier, baseline], [classifier_optimizer, baseline_optimizer]

def make_grouped_models(h_dim, subtype_num, group_num, mask, learning_rates, regularization, use_bias=False):
    classifier = Classifier_grouped(subtype_num, group_num, mask, regularization, use_bias)
    baseline = Baseline(h_dim)
    classifier_optimizer = tf.keras.optimizers.Adam(learning_rates[0])
    baseline_optimizer = tf.keras.optimizers.Adam(learning_rates[1])
    return [classifier, baseline], [classifier_optimizer, baseline_optimizer]

def supervise_loss(logits, y):
    return tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)

def reinforce_and_baseline_loss(x, a, logits, reward):
    reinforce_loss = reward * tf.nn.sparse_softmax_cross_entropy_with_logits(labels=a, logits=logits)
    baseline_loss = tf.math.square(reward)
    return reinforce_loss, baseline_loss

def mvi_loss(logits, mvi):
    p = tf.nn.softmax(logits)
    [s0, s1, s2] = tf.split(p, 3, axis=-1)
    # mvi0 = tf.reduce_mean(tf.squeeze(s0) * mvi)
    mvi0 = tf.reduce_sum(
        tf.squeeze(s0) * mvi) / tf.reduce_sum(
        tf.squeeze(s0)
    )
    # mvi2 = tf.reduce_sum(
    #     tf.squeeze(s2) * mvi) / tf.reduce_sum(
    #     tf.squeeze(s2)
    # )
    return mvi0

@tf.function
def train_step_rl(x1, y1, x2, a, r, mvi, models, optimizers, loss_rates, dropouts=[0.8, 0.5], su_loss_w=1., simple=False, group=False):
    classifier, baseline = models
    classifier_optimizer, baseline_optimizer = optimizers

    with tf.GradientTape(persistent=True) as tape:
        logits1 = classifier(x1, dropouts[0])  if simple or group else classifier(x1, dropouts[0], dropouts[1]) 
        logits2 = classifier(x2, 0.0) if simple or group  else classifier(x2, 0.0, 0.0)
        reward = baseline(x2, r)
        loss_su = supervise_loss(logits1, y1)
        loss_rl, loss_bl = reinforce_and_baseline_loss(x2, a, logits2, reward)
        loss_mvi = mvi_loss(logits2, mvi)
        loss_l1 = tf.add_n(classifier.losses)
        loss = tf.reduce_mean(loss_su * su_loss_w) * loss_rates[0] + \
            tf.reduce_mean(loss_rl) * loss_rates[1] + \
            tf.reduce_mean(loss_mvi) * loss_rates[2] + \
            loss_l1

    classifier_gradients = tape.gradient(loss, classifier.trainable_variables)
    baseline_gradients = tape.gradient(loss_bl, baseline.trainable_variables)

    classifier_optimizer.apply_gradients(zip(classifier_gradients, classifier.trainable_variables))
    baseline_optimizer.apply_gradients(zip(baseline_gradients, baseline.trainable_variables))

    return loss_su, loss_rl, loss_bl, loss_mvi, loss_l1, [logits1, logits2]