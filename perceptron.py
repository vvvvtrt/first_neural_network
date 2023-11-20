from numpy import exp, array, random, dot

class Perceptron:
    def __init__(self, training_set_inputs, training_set_outputs):
        self.training_set_inputs = array(training_set_inputs)
        self.training_set_outputs = array(training_set_outputs).T

        random.seed(1)
        self.synaptic_weights = 2 * random.random((len(self.training_set_inputs[0]), 1)) - 1

    # Процесс обучения сети
    def training(self):
        for iteration in range(10000):
            # Рассчитываем выходные данные с использованием сигмоидной функции активации
            output = 1 / (1 + exp(-(dot(self.training_set_inputs, self.synaptic_weights))))
            # Корректируем веса синапсов на основе ошибки предсказания
            self.synaptic_weights += dot(self.training_set_inputs.T, (self.training_set_outputs - output) * output * (1 - output))

    def recognition_answer(self, data):
        return 1 / (1 + exp(-(dot(data, self.synaptic_weights))))