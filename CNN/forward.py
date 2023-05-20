import numpy as np

def convolution(image, filt, bias, s=1):
    '''
    Свертка `filt` поверх `image`, используя шаг`s`
    '''
    (n_f, n_c_f, f, _) = filt.shape  #количество фильтров, кол-во каналов, размер фильтра
    n_c, in_dim, _ = image.shape  #кол-во каналов, размер изображения

    out_dim = int((in_dim - f) / s) + 1  # размер выхода

    #  размеры канлов фильтра соответствуют размерам каналов входного изображения?
    assert n_c == n_c_f, "Размер каналов фильтра должен соответствовать размеру каналов входного изображения"

    out = np.zeros((n_f, out_dim, out_dim))  # для хранения значений операции свертки

    for curr_f in range(n_f):
        curr_y = out_y = 0
        # перемещение фильтра вертикально
        while curr_y + f <= in_dim:
            curr_x = out_x = 0
            # горизонтально
            while curr_x + f <= in_dim:
                # свертка + смещение
                out[curr_f, out_y, out_x] = np.sum(filt[curr_f] * image[:, curr_y:curr_y + f, curr_x:curr_x + f]) + \
                                            bias[curr_f]
                curr_x += s
                out_x += 1
            curr_y += s
            out_y += 1

    return out


def maxpool(image, f=2, s=2):
    '''
    Уменьшение размерности `image` используя ядро размером `f` и шаг `s`
    '''
    n_c, h_prev, w_prev = image.shape

    h = int((h_prev - f) / s) + 1
    w = int((w_prev - f) / s) + 1

    downsampled = np.zeros((n_c, h, w))
    for i in range(n_c):
        # окно по каждой части изображения, используя шаг s
        curr_y = out_y = 0
        while curr_y + f <= h_prev:
            curr_x = out_x = 0
            while curr_x + f <= w_prev:
                downsampled[i, out_y, out_x] = np.max(image[i, curr_y:curr_y + f, curr_x:curr_x + f])
                curr_x += s
                out_x += 1
            curr_y += s
            out_y += 1
    return downsampled


def softmax(X):
    out = np.exp(X)
    return out / np.sum(out)


def categoricalCrossEntropy(probs, label):
    return -np.sum(label * np.log(probs))