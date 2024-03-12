import matplotlib.pyplot as plt

def plot_2d_data(data, labels):
    plt.scatter(data[:, 0], data[:, 1], c=labels)