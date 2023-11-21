from numpy import exp, array, random, dot
from perceptron import *
from data_training import *

def tests(training_set_inputs, training_set_outputs, test_):
    arr_per = []

    for i in range(len(test_)):
        arr_per.append(Perceptron(array(training_set_inputs[i]), training_set_outputs))
        arr_per[-1].training()

    sum_ = 0
    for i in range(len(test_)):
        sum_ += arr_per[i].recognition_answer(array(test_[i]))

    return sum_ / len(test_)


if __name__ == '__main__':
    inputs = list(num.values())
    training_set_inputs = [[] for i in range(len(inputs[0][0]))]
    training_set_outputs = [[]]

    for i in range(len(inputs)):
        for j in range(len(inputs[i])):
            for k in range(len(inputs[i][j])):
                training_set_inputs[k].append(inputs[i][j][k])

            training_set_outputs[0].append(1 if i == 0 else 0)

    training_set_outputs = array(training_set_outputs)

    print(tests(training_set_inputs, training_set_outputs, test1))
    print(tests(training_set_inputs, training_set_outputs, test2))

