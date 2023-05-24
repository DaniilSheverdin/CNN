import os
import numpy as np
import idx2numpy
from PIL import Image

def getUByteDataSet(appendix):
    img_path = 'C:/Users/Daniil/PycharmProjects/GeometryAI/images/' + appendix
    classes = ['circle', 'square', 'star', 'triangle']

    data_list = []
    labels_list = []
    for i in range(len(classes)):
        images = os.listdir(img_path + classes[i])
        data = []

        for f in images:
            image = Image.open(img_path+classes[i]+'/'+f).convert('L').resize((28, 28))
            data.append(np.array(image))

        data_list.append(np.array(data))
        labels_list.append(np.full(len(data), i, dtype=np.uint8))

    # Объединяем массивы для изображений и меток классов
    images = np.concatenate(data_list)
    labels = np.concatenate(labels_list)

    # Сохраняем массивы в формате idx3-ubyte
    if appendix == "train/":
        idx2numpy.convert_to_file('train-labels.idx1-ubyte', labels)
        idx2numpy.convert_to_file('train-images.idx3-ubyte', images)
    else:
        idx2numpy.convert_to_file('valid-images.idx3-ubyte', images)
        idx2numpy.convert_to_file('valid-labels.idx1-ubyte', labels)

getUByteDataSet("train/")
getUByteDataSet("valid/")