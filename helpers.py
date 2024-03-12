import matplotlib.pyplot as plt

def plot_2d_data(data, labels):
    plt.scatter(data[:, 0], data[:, 1], c=labels)
    
def plot_classified_data_on_bloch(op_state,train_y):
    plt.rcParams["figure.figsize"] = [7.00, 7.00]
    plt.rcParams["figure.autolayout"] = True
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    plt.scatter(op_state[:, 0], op_state[:, 1], c=train_y)
    