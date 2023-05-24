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

    params = pickle.load(open(save_path, 'rb'))
    [f1, f2, w3, w4, b1, b2, b3, b4] = params

    # Get test data
    m = 4000
    X = extract_data('valid-images.idx3-ubyte.gz', m, 28)
    y_dash = extract_labels('valid-labels.idx1-ubyte.gz', m).reshape(m, 1)
    # Normalize the data
    X -= int(np.mean(X))  # subtract mean
    X /= int(np.std(X))  # divide by standard deviation
    test_data = np.hstack((X, y_dash))

    X = test_data[:, 0:-1]
    X = X.reshape(len(test_data), 1, 28, 28)
    y = test_data[:, -1]

    corr = 0
    digit_count = [0 for i in range(4)]
    digit_correct = [0 for i in range(4)]

    print()
    print("Вычисление точности на тестовом наборе:")

    t = tqdm(range(len(X)), leave=True)

    for i in t:
        x = X[i]
        pred, prob = predict(x, f1, f2, w3, w4, b1, b2, b3, b4)
        digit_count[int(y[i])] += 1
        if pred == y[i]:
            corr += 1
            digit_correct[pred] += 1

        t.set_description("Точность:%0.2f%%" % (float(corr / (i + 1)) * 100))

    print("Общая точность: %.2f" % (float(corr / len(test_data) * 100)))
    x = np.arange(4)
    digit_recall = [(x+1) / (y+1) for x, y in zip(digit_correct, digit_count)]
    plt.xlabel('Классы фигур (0-круг; 1-квадрат; 2-звезда; 3-треугольник')
    plt.ylabel('Точность')
    plt.title("Точность на тестовом наборе")
    plt.bar(x, digit_recall)
    plt.show()