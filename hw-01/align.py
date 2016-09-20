from numpy import roll, dstack, sum, power
from skimage.transform import rescale


def mse(green, other, dx, dy):
    h, w = green.shape
    i, j = abs(dx), abs(dy)
    if dx >= 0 and dy >= 0:
        green = green[i:, j:]
        other = other[:h-i, :w-j]
    elif dx <= 0 and dy <= 0:
        green = green[:h-i, :w-j]
        other = other[i:, j:]
    elif dx > 0 and dy < 0:
        green = green[i:, :w-j]
        other = other[:h-i, j:]
    else:
        green = green[:h-i, j:]
        other = other[i:, :w-j]
    h, w = green.shape
    return sum(power(green - other, 2)) / (h * w)


def align(bgr_image):
    h, w = bgr_image.shape
    init_blue = bgr_image[:h//3, :]
    init_green = bgr_image[h//3:2*h//3, :]
    init_red = bgr_image[2*h//3:, :]
    h = h // 3
    del_x = h * 7 // 100
    del_y = w * 7 // 100
    init_red = init_red[del_x:h-del_x,     del_y:w-del_y]
    init_green = init_green[del_x:h-del_x, del_y:w-del_y]
    init_blue = init_blue[del_x:h-del_x,   del_y:w-del_y]

    red = init_red.copy()
    green = init_green.copy()
    blue = init_blue.copy()

    scale = 1
    while max(green.shape) > 500:
        red = rescale(red, 0.5)
        green = rescale(green, 0.5)
        blue = rescale(blue, 0.5)
        scale *= 2

    delta = 15
    min_red = min_blue = float("+inf")
    for i in range(-delta, delta+1):
        for j in range(-delta, delta+1):
            cur = mse(green.copy(), red.copy(), i, j)
            if cur < min_red:
                min_red, dx_red, dy_red = cur, i, j
            cur = mse(green.copy(), blue.copy(), i, j)
            if cur < min_blue:
                min_blue, dx_blue, dy_blue = cur, i, j
    out_im = dstack((roll(roll(init_red, dx_red*scale, axis=0), dy_red*scale, axis=1),
                    init_green,
                    roll(roll(init_blue, dx_blue*scale, axis=0), dy_blue*scale, axis=1)))
    return out_im
