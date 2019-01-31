import numpy as np
import matplotlib.pyplot as plt


class LinearEquationNeuralNetwork:
    """
    the purpose of this NeuralNetwork is to solve linear equations for which a few points are known
    and more importantly, to do linear regression.
    """
    def __init__(self, dimensions):
        self.weights = np.random.random(dimensions)
        self.dimensions = dimensions

    def get_value(self, positions, hstack=False):
        """
        return what the ai currently expects the values at 'positions' to be

        :param np.ndarray positions: the positions for which values are sought
        :param bool hstack: if this is set to True, the ai adds a constant one to
                            all 'positions' so that the value at (0, 0, ..., 0) can differ from 0
        :return: np.array
        """
        if hstack:
            positions = np.hstack((positions, [[1]] * len(positions)))
            # add a constant one which will allow us to change
            # the intersection with the y axis.
        return self.weights @ positions.T

    def current_guess(self):
        """
        returns what the current guess/function of the ai looks like

        :return: str
        """
        return ( 
            f"{' + '.join([str(round(self.weights[i-1], 4))+'X'+str(i) for i in range(1, self.dimensions)])}"
            f" + {round(self.weights[-1], 4)}")

    def train(self, positions, values, generations):
        """
        update the ai's weights so that it will create a function that has the best fit with the input
        so that means that self.get_value(positions) will as close as possible resemble values

        :param positions: the positions for which we know the values
        :param values: the values that correspond to 'positions'
        :param generations: the number of generations over which the ai will be trained
        :return: None
        """
        print("\nguess before training " + self.current_guess())
        print("with an average error of "
              f"{round(abs(np.average(self.get_value(positions)-values)), 7)}\n")

        for generation in range(generations):
            if generation % 100000 == 0:
                print(f"guess during training " + self.current_guess())
                print(f"with an average error of "
                      f"{round(abs(np.average(self.get_value(positions)-values)), 7)}\n")
            guess = self.get_value(positions)
            error = values - guess
            # find how far the ai is off from the real value
            adjustment = 1e-4 * error @ positions
            # when we find a new value, we do not want to completely get away with
            # the old value, which is why we multiply by 1e-4
            self.weights += adjustment.squeeze()

        print("guess after training " + self.current_guess())
        print("with an average error of "
              f"{round(abs(np.average(self.get_value(positions)-values)), 7)}")


if __name__ == '__main__':
    inputs_ = np.array([[float(f) for f in input("inputs: ").split()]]).T

    outputs = np.array([float(f) for f in input("outputs: ").split()]).T

    through_origin = 'y' in input("do you want the line to go through the origin? ")

    if through_origin:
        inputs = np.hstack((inputs_, [[1]] * len(inputs_)))
        # add a constant one which will allow us to change
        # the intersection with the y axis.
    else:
        inputs = inputs_
    
    ai = LinearEquationNeuralNetwork(2 if through_origin else 1)

    ai.train(inputs, outputs, 1000000)

    plt.plot(inputs_, outputs, 'r--', inputs_, ai.get_value(inputs), 'b:')
    plt.legend(['the data', 'the linearly regressed line'])
    plt.show()
