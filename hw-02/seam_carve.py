from numpy import zeros, argmin, ndarray, transpose

def calculate_energy(img, mask):
    h, w = img.shape[:2]
    luma = 0.299 * img[:,:, 0] + 0.587 * img[:,:, 1] + 0.114 * img[:,:, 2]
    energy = ndarray(shape=(h, w))
    for y in range(h):
        for x in range(w):
            dx = dy = 0
            if x != 0:
                dx += luma[y][x-1] - luma[y][x]
            if x != w-1:
                dx += luma[y][x] - luma[y][x+1]
            if y != 0:
                dy += luma[y-1][x] - luma[y][x]
            if y != h-1:
                dy += luma[y][x] - luma[y+1][x]
            energy[y][x] = (dx ** 2 + dy ** 2) ** 0.5
    if not mask is None:
        delta = w * h * 256
        energy += mask * delta
    return energy


def calculate_dynamic_table(energy):
    h, w = energy.shape
    for y in range(1, h):
        for x in range(w):
            min_neighbour = energy[y-1][x]
            if x != 0:
                min_neighbour = min(min_neighbour, energy[y-1][x-1])
            if x != w-1:
                min_neighbour = min(min_neighbour, energy[y-1][x+1])
            energy[y][x] += min_neighbour

    return energy


def find_min_carve(dynamic_table):
    h, w = dynamic_table.shape
    carve_mask = zeros(shape=(h, w))
    x = argmin(dynamic_table[-1])
    carve_mask[h-1][x] = 1
    for y in range(h-2, -1, -1):
        coord_min = x
        if x != 0 and dynamic_table[y][x-1] <= dynamic_table[y][x]:
            coord_min = x-1
        if x != w-1 and dynamic_table[y][coord_min] > dynamic_table[y][x+1]:
            coord_min = x+1
        x = coord_min
        carve_mask[y][x] = 1

    return carve_mask


def make_a_shrink(img, mask, carve_mask):
    h, w = img.shape[:2]
    resized_img = ndarray(shape=(h, w-1, 3))
    resized_mask = ndarray(shape=(h, w-1))
    for y in range(h):
        new_x = 0
        for x in range(w):
            if carve_mask[y][x] == 1:
                continue
            resized_img[y][new_x] = img[y][x]
            resized_mask[y][new_x] = mask[y][x]
            new_x += 1
    return (resized_img, resized_mask)


def make_an_expand(img, mask, carve_mask):
    h, w = img.shape[:2]
    resized_img = ndarray(shape=(h, w+1, 3))
    resized_mask = ndarray(shape=(h, w+1))
    for y in range(h):
        new_x = 0
        for x in range(w):
            resized_img[y][new_x] = img[y][x]
            resized_mask[y][new_x] = mask[y][x]
            new_x += 1
            if carve_mask[y][x] == 1:
                resized_mask[y][new_x] = 1
                next_cell = img[y][x]
                if x != w-1:
                    next_cell = img[y][x+1]
                resized_img[y][new_x] = (img[y][x] + next_cell) // 2
                new_x += 1
    return (resized_img, resized_mask)


def seam_carve(img, mode, mask=None):
    if 'vertical' in mode:
        img = img.transpose(1, 0, 2)
        if not mask is None:
            mask = transpose(mask)

    energy = calculate_energy(img, mask)
    dynamic_table = calculate_dynamic_table(energy)
    carve_mask = find_min_carve(dynamic_table)

    mask_is_none = mask is None
    if mask is None:
        mask = zeros(shape=img.shape[:2])

    if 'shrink' in mode:
        resized_img, resized_mask = make_a_shrink(img, mask, carve_mask)
    else:
        resized_img, resized_mask = make_an_expand(img, mask, carve_mask)

    if 'vertical' in mode:
        resized_img = resized_img.transpose(1, 0, 2)
        resized_mask = transpose(resized_mask)
        carve_mask = transpose(carve_mask)

    if mask_is_none:
        resized_mask = None

    return (resized_img, resized_mask, carve_mask)
