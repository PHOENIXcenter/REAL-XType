import tensorflow as tf
import numpy as np
import time
from data_utils import get_rmst
from tensorflow.keras import layers, losses, regularizers
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.regularizers import l2

@tf.custom_gradient
def gradient_reversal(x, alpha=1.0):
    def grad(dy):
        return -dy * alpha, None
    return x, grad

class GradientReversalLayer(layers.Layer):
    def __init__(self,**kwargs):
        super(GradientReversalLayer,self).__init__(**kwargs)

    def call(self, x, alpha=1.0):
        return gradient_reversal(x, alpha)

class Classifier_simple(Model):
    def __init__(self, h_dim, subtype_num, domain_num, regularization, activ, use_bias=False):
        super(Classifier_simple, self).__init__()
        self.linear1 = layers.Dense(h_dim, activation=activ, name='linear', 
            kernel_regularizer=regularizers.l2(l=regularization), use_bias=use_bias
        )
        self.linear2 = layers.Dense(subtype_num, activation=None, name='linear', 
            kernel_regularizer=regularizers.l2(l=regularization), use_bias=use_bias
        )
        self.linear3 = layers.Dense(domain_num, activation=None, name='linear', use_bias=use_bias
        )
        self.grad_reversal_layer = GradientReversalLayer()
        self.domain_num = domain_num
        
    def call(self, x, drop_rate):
        x = self.linear1(tf.nn.dropout(x, rate=drop_rate))
        x = self.linear2(x)
        return x

    def encode(self, x, drop_rate):
        x = self.linear1(tf.nn.dropout(x, rate=drop_rate))
        return x 

    def subtype(self, x):
        x = self.linear2(x)
        return x

    def risk(self, x):
        x = self.linear2(x)
        p = tf.nn.softmax(x)
        risk = p[:, -1] - p[:, 0]
        # risk = p[:, 2]*3 + p[:, 1]*2 + p[:, 0]
        return risk

    def domain_predict(self, x, alpha=1.):
        x = self.grad_reversal_layer(x, alpha)
        x = self.linear3(x)
        return x

    def predict(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        a = tf.math.argmax(x, axis=1)
        return tf.squeeze(a).numpy()

    def prob(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        p = tf.nn.softmax(x)
        return p.numpy()

    def domain_test(self, x):
        feat = self.linear1(x)
        softmax_class = tf.nn.softmax(self.linear2(feat), axis=-1)
        x = tf.linalg.matmul(tf.expand_dims(feat, axis=2), tf.expand_dims(softmax_class, axis=1))
        x = tf.reshape(x, (-1, softmax_class.shape[1] * feat.shape[1]))
        x = self.linear3(x)
        return x

def make_simple_models(h_dim, subtype_num, domain_num, learning_rate, regularization, activ, use_bias=False):
    classifier = Classifier_simple(h_dim, subtype_num, domain_num, regularization, activ, use_bias)
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    return classifier, optimizer

def negative_log_likelihood(risk, t, e):
    # sort the data in a batch with descending observed event time t
    risk = tf.squeeze(risk)
    t = tf.squeeze(t)
    e = tf.squeeze(e)
    sorted_ids = tf.argsort(t, direction='DESCENDING')
    sorted_risk = tf.gather(risk, sorted_ids)
    sorted_e = tf.gather(e, sorted_ids)

    # calculate negative likelihood
    hazard_ratio = tf.math.exp(sorted_risk)
    log_hr = tf.math.log(tf.math.cumsum(hazard_ratio))
    uncensored_likelihood = sorted_risk - log_hr
    censored_likelihood = uncensored_likelihood * sorted_e
    num_observed_events = tf.reduce_sum(sorted_e) + 1e-10
    neg_likelihood = -tf.reduce_sum(censored_likelihood) / num_observed_events
    return neg_likelihood

def cal_entropy(x):
    entropy = -x * tf.math.log(x + 1e-5)
    entropy = tf.reduce_sum(entropy, axis=1)
    return entropy

@tf.function
def train_step(data_subtype, data_cox, data_da, classifier, optimizer, loss_rates, droprate, alpha, use_entropy=True):

    x_cls, y_cls = data_subtype
    x_da, y_da = data_da

    with tf.GradientTape(persistent=True) as tape:
        # supervised training
        logits = classifier(x_cls, droprate)
        loss_su = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_cls, logits=logits))

        # cox training
        # all population
        x_cox, t1, e1, t2, e2 = data_cox[0]
        risk = classifier.risk(classifier.encode(x_cox, droprate))
        loss_nll = (negative_log_likelihood(risk, t1, e1) + negative_log_likelihood(risk, t2, e2)) / 2
        # low risk
        if len(data_cox) > 1:
            x_cox, t1, e1, t2, e2 = data_cox[1]
            risk = classifier.risk(classifier.encode(x_cox, droprate))
            loss_nll_low_risk = negative_log_likelihood(risk, t2, e2) * 0.5
            loss_nll += loss_nll_low_risk

        # CDAN loss
        feat = classifier.encode(x_da, droprate)
        logits_class = classifier.subtype(feat)
        softmax_class = tf.nn.softmax(logits_class, axis=-1)
        op_out = tf.linalg.matmul(tf.expand_dims(feat, axis=2), tf.expand_dims(softmax_class, axis=1))
        domain_in = tf.reshape(op_out, (-1, softmax_class.shape[1] * feat.shape[1]))
        logits_domain = classifier.domain_predict(domain_in, alpha)
        if use_entropy:
            entropy = cal_entropy(softmax_class)
            entropy = classifier.grad_reversal_layer(entropy, alpha)
            entropy = 1.0 + tf.math.exp(-entropy)
            weight = tf.zeros_like(y_da, dtype=tf.float32)
            for d_id in range(classifier.domain_num):
                d_weight = tf.where(y_da == d_id, entropy, 0.)
                d_weight = d_weight / tf.stop_gradient(tf.reduce_sum(d_weight) + 1e-10)
                weight += d_weight
            loss_da = tf.reduce_sum(weight * tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_da, logits=logits_domain)
            ) / tf.stop_gradient(tf.reduce_sum(weight))
        else:
            loss_da = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_da, logits=logits_domain))
        loss_l1 = tf.add_n(classifier.losses)
        loss_var = tf.math.reduce_variance(tf.reduce_sum(softmax_class, axis=0))
        loss = loss_su * loss_rates[0] + \
                loss_nll * loss_rates[1] + \
                loss_da * loss_rates[2] + \
                loss_l1 + \
                loss_var * loss_rates[3]
    classifier_gradients = tape.gradient(loss, classifier.trainable_variables)
    optimizer.apply_gradients(zip(classifier_gradients, classifier.trainable_variables))

    return loss_su, loss_nll, loss_da, loss_l1, logits