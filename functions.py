import tensorflow as tf
from scipy import stats
from scipy.special import gamma

import keras
import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.models import Sequential
from art.attacks.evasion import FastGradientMethod, CarliniL2Method, CarliniLInfMethod, ProjectedGradientDescentNumpy
from art.estimators.classification import TensorFlowV2Classifier

from datetime import datetime
from pytz import timezone
import numpy as np


'''
ACTIVATION FUNCTIONS
'''
# https://stackoverflow.com/questions/43915482/how-do-you-create-a-custom-activation-function-with-keras
# https://www.tensorflow.org/guide/autodiff

# def relu_test(x):
#     return tf.maximum(0.0, x)
#
# def sigmoid_test(x):
#     return 1 / (1 + tf.math.exp(-x))

tent_delta = 1

def tent_activation(x):
    return tf.math.maximum(0, tent_delta-tf.math.abs(x))

def tent_derivative(x):
    x = tf.Variable(x, name='x')
    with tf.GradientTape(persistent=True) as tape: 
        y = tf.constant(tent_activation(x), dtype='float32')
    dy_dx = tape.gradient(y, x)
    return dy_dx

    
'''
GENERALIZED GAMMA HYPERPARAMETERS
'''

a = 1 # alpha
c = 3 # gamma
mu = -2.6  # same as mu location parameter in Mathematica parametrization of generalized gamma distribution
b = 3 # have to double check what this means
sf = 1.17 # scale factor


'''
FUNCTIONS
'''
def generalized_gamma(x):
    x = tf.math.divide(x-mu, b)
    func = tf.math.divide(tf.math.exp(-x**c)*c*x**((c*a)-1), gamma(a))    
    return tf.where(x>0, tf.math.divide(func, sf), 0)

# automatically approximate the derivative (nice and convenient)
def gamma_derivative(x):
    x = tf.Variable(x, name='x')
    with tf.GradientTape(persistent=True) as tape: 
        y = tf.constant(generalized_gamma(x), dtype='float32')
    dy_dx = tape.gradient(y, x)
    return dy_dx


'''
LEARNING MODELS 
'''
def define_model(af='tanh'): 
    model = Sequential()

    # C1 convolutional layer 
    model.add(layers.Conv2D(6, kernel_size=(5,5), strides=(1,1), activation=af, padding='same', dynamic=True))

    # S2 pooling layer
    model.add(layers.AveragePooling2D(pool_size=(2,2), strides=(1,1), padding='valid'))

    # C3 convolutional layer
    model.add(layers.Conv2D(16, kernel_size=(5,5), strides=(1,1), activation=af, padding='valid', dynamic=True))

    # S4 pooling layer
    model.add(layers.AveragePooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

    # C5 fully connected convolutional layer
    model.add(layers.Conv2D(120, kernel_size=(5,5), strides=(1,1), activation=af, padding='valid', dynamic=True))
    model.add(layers.Flatten())

    # FC6 fully connected layer
    model.add(layers.Dense(84, activation=af, dynamic=True))

    # Output layer with softmax activation
    model.add(layers.Dense(10, activation='softmax'))
    return model

def train_step(model, images, labels):
    loss_object = tf.keras.losses.CategoricalCrossentropy()
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
    
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_object(labels, predictions)
        
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
def train_model(model, x_train, y_train, x_test, y_test, eps=10, batch=128, lr=0.01, clip_values=(0.0, 1.0)):
    loss_object = tf.keras.losses.CategoricalCrossentropy()
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

    classifier = TensorFlowV2Classifier(model=model,
                                        clip_values=clip_values, 
                                        input_shape=x_train.shape[1:], 
                                        nb_classes=10,  
                                        train_step=train_step,
                                        loss_object=loss_object)
    print('...created classifier')
    
    tz = timezone('EST')
    print('Before training time (EST): ', datetime.now(tz))
    
    hist = classifier.fit(x_train, y_train, nb_epochs=eps, batch_size=batch)
    print('...finished training')
    
    print('Time after training finished: ', datetime.now(tz))
    
    # Evaluate the classifier on the test set
    preds = np.argmax(classifier.predict(x_test), axis=1)
    acc = np.sum(preds == np.argmax(y_test, axis=1)) / y_test.shape[0]
    print("Test accuracy: %.2f%%\n" % (acc * 100))

    return classifier


'''
SUPPORT EVALUATION FUNCTIONS
'''
def get_successful_test(classifier, x_test, y_test):
    preds = np.argmax(classifier.predict(x_test), axis=1)
    acc = np.sum(preds == np.argmax(y_test, axis=1)) / y_test.shape[0]
    print("Original test accuracy: %.2f%%" % (acc * 100))
    
    preds = np.argmax(classifier.predict(x_test), axis=1)
    correct = np.nonzero(preds == np.argmax(y_test, axis=1))

    eval_x_test = x_test[correct]
    eval_y_test = y_test[correct]

    eval_x_test_final = eval_x_test[:1000]
    print(eval_x_test_final.shape)
    eval_y_test_final = eval_y_test[:1000]
    print(eval_y_test_final.shape)
    
    preds = np.argmax(classifier.predict(eval_x_test_final), axis=1)
    acc = np.sum(preds == np.argmax(eval_y_test_final, axis=1)) / eval_y_test_final.shape[0]
    print("Test set of correctly predicted (benign): %.2f%%" % (acc * 100))
    
    return eval_x_test_final, eval_y_test_final

def attack_success(x):
    return (100 - np.array(x))

'''
SUPPORT ATTACK FUNCTIONS
'''
def fgsm_attack(classifier, x_test, y_test, eps=0.2):
    epsilon = eps  # Maximum perturbation
    adv_crafter = FastGradientMethod(classifier, eps=epsilon)
    x_test_adv = adv_crafter.generate(x=x_test)

    # Evaluate the classifier on the adversarial examples
    preds = np.argmax(classifier.predict(x_test_adv), axis=1)
    acc = np.sum(preds == np.argmax(y_test, axis=1)) / y_test.shape[0]
    return acc * 100

def pgd_attack(classifier, x_test, y_test, eps=0.2):
    epsilon = eps  # Maximum perturbation
    adv_crafter = ProjectedGradientDescentNumpy(classifier, eps=epsilon, verbose=False)
    x_test_adv = adv_crafter.generate(x=x_test)

    # Evaluate the classifier on the adversarial examples
    preds = np.argmax(classifier.predict(x_test_adv), axis=1)
    acc = np.sum(preds == np.argmax(y_test, axis=1)) / y_test.shape[0]
    return acc * 100

def cwl2_attack(classifier, x_test, y_test, eps=0.2):
    adv_crafter = CarliniL2Method(classifier, confidence=eps)
    print('...creating adversarial examples')
    x_test_adv = adv_crafter.generate(x=x_test)

    # Evaluate the classifier on the adversarial examples
    preds = np.argmax(classifier.predict(x_test_adv), axis=1)
    acc = np.sum(preds == np.argmax(y_test, axis=1)) / y_test.shape[0]
    print("Test accuracy on adversarial sample: %.2f%%" % (acc * 100))
    return acc * 100    