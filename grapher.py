from matplotlib import pyplot as plt
import numpy as np

def plot(data, pattern, title="", save_fig=False):
    plt.plot(data)
    q_size = len(data)/len(pattern)
    for i in range(len(pattern)):
        plt.axvspan(i*q_size, (i+1)*q_size, facecolor="orange", alpha=pattern[i]/(max(pattern)+1))
    plt.title(title)
    if save_fig:
        plt.savefig(f"{title}.png")
    else:
        plt.show()

plot(np.random.random(100), [1, 1, 2, 1, 2, 3, 1, 2, 3, 4, 1, 2, 3, 4, 5], "test", True)
