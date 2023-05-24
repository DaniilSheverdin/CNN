from CNN.network import *
from CNN.helper import *

from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import pickle

parser = argparse.ArgumentParser(description='Predict the network accuracy.')
parser.add_argument('parameters', metavar='parameters',
                    help='name of file parameters were saved in. These parameters will be used to measure the accuracy.')

if __name__ == '__main__':

    args = parser.parse_args()
    save_path = args.parameters

    cost = train(save_path=save_path)

    params = pickle.load(open(save_path, 'rb'))
    [f1, f2, w3, w4, b1, b2, b3, b4] = params

    # Plot cost
    plt.plot(cost, 'r')
    plt.xlabel('# Iterations')
    plt.ylabel('Cost')
    plt.legend('Loss', loc='upper right')
    plt.show()
