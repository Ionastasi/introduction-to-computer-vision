from numpy import array, sqrt, power, arctan2, zeros, hstack
from scipy.ndimage.filters import convolve
from skimage.transform import resize
from math import pi

def extract_hog(img, roi):
    # grayscale, resize, roi
    #img = (img[:, :, 0] + img[:, :, 1] + img[:, :, 2]) / 3
    img = 0.299*img[:, :, 0] + 0.587*img[:, :, 0] + 0.114*img[:, :, 0]
    img = resize(img[roi[1]:roi[3], roi[0]:roi[2]], (48, 48))  # x1, y1, x2, y2

    # свертка
    sobel_x = array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    ix = convolve(img, sobel_x)
    iy = convolve(img, sobel_y)

    # градиент
    abs_grad = sqrt(power(ix, 2) + power(iy, 2))
    dir_grad = arctan2(ix, iy) + pi  # 0..2pi

    # параметры ячеек
    cellRows = 8
    cellCols = 8
    binCount = 9

    part = 2 * pi / binCount

    # считаем гистограммы
    h, w = dir_grad.shape[:2]
    cells = zeros(shape=(h // cellRows, w // cellCols, binCount))
    for iCell in range(h // cellRows):
        for jCell in range(w // cellCols):
            for i in range(iCell * cellRows, (iCell+1) * cellRows):
                for j in range(jCell * cellCols, (jCell+1) * cellCols):
                    x = (dir_grad[i][j] // part) % binCount
                    cells[iCell][jCell][x] = abs_grad[i][j]

    # блоки (с пересечением)
    blockRowCell = 3
    blockColCell = 3
    eps = 10 ** -4
    h, w = cells.shape[:2]
    new_h = h - blockRowCell + 1
    new_w = w - blockColCell + 1
    new_bin = binCount * blockColCell * blockRowCell
    blocks = zeros(shape=(new_h, new_w, new_bin))
    for iBlock in range(new_h):
        for jBlock in range(new_w):
            to_concate = list()
            for i in range(iBlock, iBlock+blockRowCell):
                for j in range(jBlock, jBlock+blockColCell):
                    to_concate.append(cells[i, j])
            concated = hstack(to_concate)
            blocks[iBlock, jBlock] = concated / sqrt(sum(concated ** 2) + eps)

    ans = blocks.reshape(new_bin * new_h * new_w)  # 1296

    return ans
