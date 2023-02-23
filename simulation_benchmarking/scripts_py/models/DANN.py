import tensorflow as tf
import numpy as np
import time
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
        super(GradientReversalLayer,self).__init__(kwargs)

    def call(self, x, alpha=1.0):
        return gradient_reversal(x, alpha)

class DANN(Model):
    def __init__(self, layer_params=None, simple_encoder=True):
        super(DANN, self).__init__()
        if layer_params is None:
            if simple_encoder:
                self.encoder_layers = [
                    layers.Conv2D(filters=32, kernel_size=5,strides=1),
                    layers.Activation('relu'),
                    layers.MaxPool2D(pool_size=(2, 2), strides=2),
                    layers.Conv2D(filters=48, kernel_size=5,strides=1),
                    layers.Activation('relu'),
                    layers.MaxPool2D(pool_size=(2, 2), strides=2),
                    layers.Flatten()
                ]
            else:
                self.encoder_layers = [
                    layers.Conv2D(filters=32, kernel_size=5,strides=1),
                    layers.BatchNormalization(),
                    layers.Activation('relu'),
                    layers.MaxPool2D(pool_size=(2, 2), strides=2),
                    layers.Conv2D(filters=64, kernel_size=5,strides=1),
                    layers.BatchNormalization(),
                    layers.Activation('relu'),
                    layers.MaxPool2D(pool_size=(2, 2), strides=2),
                    layers.Conv2D(filters=128, kernel_size=3,strides=1),
                    layers.BatchNormalization(),
                    layers.Activation('relu'),
                    layers.Flatten()
                ]
            self.class_layers = [
                layers.Dense(100),
                layers.Activation('relu'),
                layers.Dense(100,activation='relu'),
                layers.Dense(10,activation=None,name="image_cls_pred")
            ]
            self.domain_layers = [
                layers.Dense(100),
                layers.Activation('relu'),
                layers.Dense(2, activation=None, name="domain_cls_pred")
            ]
            self.use_regu = False
        else:
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
            self.domain_layers = [layers.Dense(
                    layer_params['class_hdim'], 
                    use_bias=layer_params['use_bias'], 
                    activation=layer_params['activation'], 
                    kernel_regularizer=regularizers.l1(l=layer_params['regularization'])
                ) 
                for _ in range(layer_params['class_layer_num'] - 1)
            ]
            self.domain_layers.append(layers.Dense(
                    2, 
                    use_bias=layer_params['use_bias'], 
                    activation=None, 
                    kernel_regularizer=regularizers.l1(l=layer_params['regularization'])
                )
            )
            self.use_regu = True

        self.grad_reversal_layer = GradientReversalLayer()

    def call(self, x, drop_rate=0., is_training=False):
        for layer in self.encoder_layers:
            x = layer(tf.nn.dropout(x, rate=drop_rate), training=is_training)
        for layer in self.class_layers:
            x = layer(tf.nn.dropout(x, rate=drop_rate))
        return x

    def predict(self, x, drop_rate=0., is_training=False):
        for layer in self.encoder_layers:
            x = layer(tf.nn.dropout(x, rate=drop_rate), training=is_training)
        for layer in self.class_layers:
            x = layer(tf.nn.dropout(x, rate=drop_rate))
        return tf.argmax(x, axis=-1).numpy()

    def predict_proba(self, x, drop_rate=0., is_training=False):
        for layer in self.encoder_layers:
            x = layer(tf.nn.dropout(x, rate=drop_rate), training=is_training)
        for layer in self.class_layers:
            x = layer(tf.nn.dropout(x, rate=drop_rate))
        p = tf.nn.softmax(x, axis=1)
        return tf.squeeze(p).numpy()

    def domain_predict(self, x, drop_rate=0.1, alpha=1., is_training=False):
        for layer in self.encoder_layers:
            x = layer(tf.nn.dropout(x, rate=drop_rate), training=is_training)
        x = self.grad_reversal_layer(x, alpha)
        for layer in self.domain_layers:
            x = layer(tf.nn.dropout(x, rate=drop_rate))
        return x

    def domain_pred_test(self, x, drop_rate=0., alpha=1.):
        return self.domain_predict(x, drop_rate, alpha)

def make_models(learning_rate, layer_params=None, simple_encoder=True):
    classifier = DANN(layer_params, simple_encoder=simple_encoder)
    optimizer = tf.keras.optimizers.SGD(learning_rate, momentum=0.9)
    return classifier, optimizer

@tf.function
def train_step(x_source, y_source, x_target, classifier, optimizer, drop_rate, alpha, lr):
    y_domain = tf.concat([tf.zeros(len(x_source), dtype=tf.int32), tf.ones(len(x_target), dtype=tf.int32)], axis=0)

    with tf.GradientTape(persistent=True) as tape:
        logits_class = classifier(x_source, drop_rate, is_training=True)
        loss_su = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_source, logits=logits_class))

        logits_domain = classifier.domain_predict(tf.concat([x_source, x_target], axis=0), drop_rate, alpha, is_training=True)
        loss_do = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_domain, logits=logits_domain))

        loss_l1 = tf.add_n(classifier.losses) if classifier.use_regu else 0.

        loss = loss_su + loss_do + loss_l1

    gradients = tape.gradient(loss, classifier.trainable_variables)
    optimizer.learning_rate = lr
    optimizer.apply_gradients(zip(gradients, classifier.trainable_variables))
    return loss_su, loss_do

