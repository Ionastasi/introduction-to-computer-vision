import numpy as np
from glob import iglob
from skimage.io import imread
from skimage import exposure, filters, transform, feature, measure


class Elem:
    def __init__(self):
        self.top = float('+inf')
        self.down = -1
        self.right = -1
        self.left = float('+inf')
        self.height = 0
        self.width = 0


def generate_template(digit_dir_path):
    template = np.zeros(shape=(42, 42))
    count = 0
    for filename in iglob(digit_dir_path + '/*.bmp'):
        count += 1
        img = transform.resize(imread(filename, plugin='matplotlib'), (42, 42))
        template += img
    template /= count
    return template


def get_objects(labels):
    h, w = labels.shape[:2]
    col_num = len(np.unique(labels))
    objects = [Elem() for i in range(col_num)]
    for y in range(h):
        for x in range(w):
            obj = objects[labels[y, x]]
            obj.down = max(obj.down, y)
            obj.top = min(obj.top, y)
            obj.left = min(obj.left, x)
            obj.right = max(obj.right, x)
    for obj in objects:
        obj.height = obj.right - obj.left
        obj.width = obj.down - obj.top
    return objects


def recognize(root, digit_templates):
    h, w = root.shape[:2]

    # выравнивание контраста
    img = exposure.adjust_gamma(root, gamma=0.9)

    # шумоподавление
    img = filters.gaussian(img, sigma=0.1)

    #бинаризация

    img = filters.threshold_adaptive(img, block_size=43)
    thresh = filters.threshold_otsu(img)
    img = (img <= thresh).astype(int)


    # выделение связных компонент
    labels = measure.label(img)

    # objects
    objects = get_objects(labels)

    #metric
    numbers = [0, 0, 0]  # colors
    optimum_height = 0
    optimum_total = float('+inf')

    for i in range(len(objects)):
        for j in range(i+1, len(objects)):
            for k in range(j+1, len(objects)):
                total = 0
                objs = [objects[i], objects[j], objects[k]]
                # проверяем ширину, высоту и уровень
                delta_h = float('-inf')
                delta_w = float('-inf')
                delta_level = float('-inf')
                too_small = False
                for q in range(3):
                    if objs[q].height < 5 or objs[q].width < 5:
                        too_small = True
                        break
                    for r in range(q+1, 3):
                        delta_h = max(delta_h, abs(objs[q].height - objs[r].height))
                        delta_w = max(delta_w, abs(objs[q].width - objs[r].width))
                        delta_level = max(delta_level, abs(objs[q].top - objs[r].top),
                                               abs(objs[q].down - objs[r].down))
                if too_small or delta_h > 5 or delta_w > 4 or delta_level > 3:  # yeap, it's magic numbers
                    continue
                total += delta_h + delta_w + delta_level

                # проверяем непересечение областей и расстояние между
                left_obj, right_obj = objs[0], objs[2]
                first, second, third = 0, 1, 2
                for q in range(3):
                    obj = objs[q]
                    if obj.left < left_obj.left:
                        left_obj = obj
                        first = q
                    if obj.right > right_obj.right:
                        right_obj = obj
                        third = q
                for q in range(3):
                    obj = objs[q]
                    if obj != left_obj and obj != right_obj:
                        mid_obj = obj
                        second = q
                if left_obj.right > mid_obj.left or mid_obj.right > right_obj.left:
                    continue
                distance = max(mid_obj.left - left_obj.right, right_obj.left - mid_obj.right)
                if distance > 7:
                    continue
                total += distance

                # so, it's look like numbers...
                max_height = max([obj.height for obj in objs])
                if optimum_total - total > 6 or abs(optimum_total - total) <= 6 and optimum_height <= max_height:
                    optimum_height = max_height
                    optimum_total = total
                    numbers = list()
                    for num in [first, second, third]:
                        if num == 0:
                            numbers.append(i)
                        elif num == 1:
                            numbers.append(j)
                        else:
                            numbers.append(k)


    # опознаем цифры....
    answer = [0, 0, 0]
    for i in range(3):
        obj = objects[numbers[i]]
        max_peak = float('-inf')
        my_digit = root[obj.top:obj.down, obj.left:obj.right]
        my_digit = transform.resize(my_digit, digit_templates[0].shape)
        for j in range(len(digit_templates)):
            digit = digit_templates[j]
            result = feature.match_template(my_digit, digit)
            if np.amax(result) > max_peak:
                max_peak = np.amax(result)
                answer[i] = j
    return tuple(answer)
