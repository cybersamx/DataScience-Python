# Credits:
# https://github.com/mattnedrich/GradientDescentExample
# https://eli.thegreenplace.net/2016/drawing-animated-gifs-with-matplotlib/

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Class to encapsulate all attributes for the data and graphs.

class Graph(object):
  def __init__(self, points):
    # Datasets
    self.points = points
    self.loss_x = []
    self.loss_y = []

    # Regression initial guess
    self.learning_rate = 0.00025
    self.b = 0
    self.m = 0
    self.num_iterations = 50

    # Matplot settings
    self.figure = plt.figure(figsize=(9, 6))
    self.main_axes = self.figure.add_subplot(121)
    self.loss_axes = self.figure.add_subplot(122)

# Animation function

def animate(i, graph):
  if i >= graph.num_iterations:
    print('Please wait. Finishing up...')
    exit(0)

  # Scatter chart of the data points
  scatter_x = [sublist[0] for sublist in graph.points]
  scatter_y = [sublist[1] for sublist in graph.points]
  graph.main_axes.clear()
  graph.main_axes.scatter(scatter_x, scatter_y, color='red', marker='o')
  graph.main_axes.set_title('Dataset')
  graph.main_axes.set_autoscalex_on(False)
  graph.main_axes.set_xlim([0, 16])
  graph.main_axes.set_autoscaley_on(False)
  graph.main_axes.set_ylim([0, 16])

  # Regression line
  formula = lambda m, b, x: m * x + b
  regression_x = np.array(range(0, 16))
  regression_y = formula(graph.m, graph.b, regression_x)
  graph.main_axes.plot(regression_x, regression_y, color='blue')

  # Loss function (error vs iteration)
  graph.m, graph.b = increment(graph.m, graph.b, np.array(graph.points), graph.learning_rate)
  loss = compute_loss(graph.m, graph.b, graph.points)
  print("i={0} m={1} b={2} loss={3}".format(i, graph.m, graph.b, loss))
  graph.loss_x.append(i)
  graph.loss_y.append(loss)

  graph.loss_axes.clear()
  graph.loss_axes.set_title('Loss over time')
  graph.loss_axes.scatter(graph.loss_x, graph.loss_y, color='green', marker='.')


def compute_loss(m, b, points):
  total_loss = 0
  for i in range(0, len(points)):
    x = points[i, 0]
    y = points[i, 1]
    total_loss += (y - (m * x + b)) ** 2
  return total_loss / float(len(points))


def increment(current_m, current_b, points, learning_rate):
  b = 0
  m = 0
  n = float(len(points))
  for i in range(0, len(points)):
    x = points[i, 0]
    y = points[i, 1]
    b += -(2/n) * (y - ((current_m * x) + current_b))
    m += -(2/n) * x * (y - ((current_m * x) + current_b))
  new_b = current_b - (learning_rate * b)
  new_m = current_m - (learning_rate * m)
  return [new_m, new_b]


if __name__ == '__main__':
  graph = Graph(np.genfromtxt('data.csv', delimiter=','))
  anime = animation.FuncAnimation(graph.figure, animate, fargs=(graph,), interval=100)

  if len(sys.argv) > 1 and sys.argv[1] == 'save':
    anime.save('regression.gif', dpi=80, writer='imagemagick')
  else:
    plt.show()
