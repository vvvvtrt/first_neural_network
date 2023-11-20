from numpy import exp, array, random, dot
from perceptron import *
from data_training import *




if __name__ == '__main__':
    create_new_ii = Perceptron(training_set_inputs, training_set_outputs)
    create_new_ii.training()
    print(create_new_ii.recognition_answer(array([0, 0, 0, 0])))
