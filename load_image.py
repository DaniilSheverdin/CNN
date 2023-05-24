import numpy

from CNN.helper import *

from PIL import Image
import argparse
import pickle

parser = argparse.ArgumentParser(description='Predict the network accuracy.')
parser.add_argument('image', metavar='image',
                    help='name of image')
parser.add_argument('parameters', metavar='parameters',
                    help='name of file parameters were saved in. These parameters will be used to measure the accuracy.')


def imageConvert(path):
    image = Image.open(path)
    # Преобразуем изображение в черно-белый формат
    image = image.convert("L")
    data = np.array([])
    # Изменяем размер изображения до 28x28 пикселей
    image = image.resize((28, 28))
    # Сохраняем преобразованное изображение
    for i in range(28):
        for j in range(28):
            data = numpy.append(data, image.getpixel((j,i)))

    data = data.reshape(1, 28*28)
    return data

if __name__ == '__main__':
    args = parser.parse_args()
    save_path = args.parameters
    img_path = args.image
    # img_path="test.jpg"
    # save_path="params.pkl"
    X = imageConvert(img_path)
    params = pickle.load(open(save_path, 'rb'))
    [f1, f2, w3, w4, b1, b2, b3, b4] = params

    label = int(input("[0-круг, 1-квадрат, 2-звезда, 3-треугольник] label = "))
    y=np.array([[label]])
    # # Normalize the data
    X -= int(np.mean(X))  # subtract mean
    X /= int(np.std(X))  # divide by standard deviation
    test_data = np.hstack((X, y))
    #
    X = test_data[:, 0:-1]
    X = X.reshape(len(test_data), 1, 28, 28)
    y = test_data[:, -1]



    x = X[0]
    pred, prob = predict(x, f1, f2, w3, w4, b1, b2, b3, b4)
    print("predict: "+str(pred))