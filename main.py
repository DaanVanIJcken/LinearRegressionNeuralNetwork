import numpy as np
import matplotlib.pyplot as plt

# disclaimer
# this neural network was build with the purpose of solving linear equations
# it can, however, also be used for linear regression
# do not use this ai for industrial purposes.
# There are far faster ai's out there. Take a look at matplotlib. 
class LinearEquationNeuralNetwork:
  def __init__(self, dimensions):
    self.weights = np.random.random((dimensions))
    self.dimensions = dimensions

  def get_value(self, positions, hstack=False):
    if hstack:
      inputs = np.hstack((inputs, [[1]]*len(inputs)))
      # add a constant one which will allow us to change
      # the intersection with the y axis.
    return self.weights@positions.T

  def current_guess(self):
    return (
      f"{' + '.join([str(round(self.weights[i-1], 4))+'X'+str(i) for i in range(1, self.dimensions)])}"
      f" + {round(self.weights[-1], 4)}")
  def train(self, inputs, outputs, generations):
    print("\nguess before training " + self.current_guess())
    print("with an average error of "
          f"{round(abs(np.average(self.get_value(inputs)-outputs)), 7)}\n")

    for generation in range(generations):
      if generation % 100000 == 0:
        print(f"guess during training " + self.current_guess())
        print(f"with an average error of "
              f"{round(abs(np.average(self.get_value(inputs)-outputs)), 7)}\n")
      guess = self.get_value(inputs)
      error = outputs - guess
      # find how far the ai is off from the real value
      adjustment = 1e-4 * error @ inputs
      # when we find a new value, we do not want to completely get away with
      # the old value, which is why we multiply by 1e-4
      self.weights += adjustment
      
    print("guess after training " + self.current_guess())
    print("with an average error of "
          f"{round(abs(np.average(self.get_value(inputs)-outputs)), 7)}")


if __name__ == '__main__':
  if input('function or linear regression?\n') == 'function':
    def f(inputs):
      return np.array([3,3])@inputs.T
      # change the values in the array to change the function
      # make the array longer to make the function deal with more dimensions

    inputs = np.random.random((10,1))
    # the second number relates to the number of dimensions

    inputs = np.hstack((inputs, [[1]]*len(inputs)))
    # add a constant one which will allow us to change
    # the intersection with the y axis.
    
    outputs = f(inputs)
    # if you already have inputs and outputs, but don't know the function,
    # directly input the inputs and outputs here.
    # keep in mind, this ai only works with linear equations
    

    ai = LinearEquationNeuralNetwork(2)
    # input dimensions + 1 (output dimension)

    ai.train(inputs, outputs, 300000)
    # if the ai doesn't find the exact the function,
    # increase the number of generations
    if ai.dimensions == 2:
      test_inputs_ = np.array([np.arange(-5, 5, 0.5)]).T

      test_inputs = np.hstack((test_inputs_, [[1]]*len(test_inputs_)))
      
      test_outputs = f(test_inputs)
      plt.plot(test_inputs_, test_outputs, 'r--',
              test_inputs_, ai.get_value(test_inputs), 'b:')
      plt.legend(['the correct line', 'the AI\'s guess'])
      plt.show()
  else: 
    from random import randint
    def rand_f(inputs):
      return np.array([ii@np.array([3,3]) + randint(-3, 2) for ii in inputs])

    inputs_ = np.array([np.arange(-5, 5, 0.5)]).T

    inputs = np.hstack((inputs_, [[1]]*len(inputs_)))
    # add a constant one which will allow us to change
    # the intersection with the y axis.

    outputs = rand_f(inputs)

    ai = LinearEquationNeuralNetwork(2)

    ai.train(inputs, outputs, 10000)

    plt.plot(inputs_, outputs, 'r--', inputs_, ai.get_value(inputs), 'b:')
    plt.legend(['the data', 'the linearly regressed line'])
    plt.show()
