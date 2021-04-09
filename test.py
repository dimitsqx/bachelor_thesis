from matplotlib import pyplot as plt
import numpy as np

# x = np.arange(-6, 6, 0.05)
# sig = 1/(1+np.exp(-x))
# tanh = (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
# relu = np.array([max(0, i) for i in x])
# sw = x/(1+np.exp(-x))

# fig, axs = plt.subplots(2, 2)

# axs[0, 0].plot(x, sig)
# axs[0, 0].set_title('Sigmoid')
# axs[0, 1].plot(x, tanh, 'tab:orange')
# axs[0, 1].set_title('Tanh')
# axs[1, 0].plot(x, relu, 'tab:green')
# axs[1, 0].set_title('ReLU')
# axs[1, 1].plot(x, sw, 'tab:red')
# axs[1, 1].set_title('Swish')
# fig.tight_layout()
# plt.show()


# def f_x(x): return (x**2)-4*(x)+6

def f_x(x):
    print(x)
    return 3*(x**4) - 16*(x**3) + 18*(x**2)


x = np.linspace(-1.1, 4, 1000)
# def f_x_derivative(x): return 2*(x)-4
def f_x_derivative(x): return 12*(x**3)-48*(x**2)+26*(x)


def plot_gradient(x, y, x_vis, y_vis):
    plt.scatter(x_vis, y_vis, c="b")
    plt.plot(x, f_x(x), c="r")
    plt.title("Gradient Descent")
    plt.xlabel("\u03B8")
    plt.ylabel("Loss")
    plt.show()


def gradient_iterations(x_start, iterations, learning_rate):

    # These x and y value lists will be used later for visualization.
    x_grad = [x_start]
    y_grad = [f_x(x_start)]
    # Keep looping until number of iterations
    for i in range(iterations):

        # Get the Slope value from the derivative function for x_start
        # Since we need negative descent (towards minimum), we use '-' of derivative
        x_start_derivative = - f_x_derivative(x_start)

        # calculate x_start by adding the previous value to
        # the product of the derivative and the learning rate calculated above.
        x_start += (learning_rate * x_start_derivative)

        x_grad.append(x_start)
        y_grad.append(f_x(x_start))

    print("Local minimum occurs at: {:.2f}".format(x_start))
    print("Number of steps: ", len(x_grad)-1)
    plot_gradient(x, f_x(x), x_grad, y_grad)


gradient_iterations(-1, 1000, 0.001)
