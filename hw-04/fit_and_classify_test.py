#!/usr/bin/python3

from sys import argv, stdout, exit
from numpy import array, loadtxt, zeros
from fit_and_classify import fit_and_classify, test
from extract_hog import extract_hog
from skimage.io import imread
from time import time

def read_gt(path):
    filename = path + '/gt.txt'
    data = loadtxt(filename, delimiter=',', skiprows=1, usecols=range(1,6))
    rois = data[:, 0:4]
    labels = data[:, 4]

    filenames = []
    for line in open(filename).readlines()[1:]:
        filenames.append(line[0:9])

    return (filenames, rois, labels)


def extract_features(path, filenames, rois):
    hog_length = len(extract_hog(imread(path + '/' + filenames[0], plugin='matplotlib'), rois[0, :]))
    print(rois[0, :])
    data = zeros((len(filenames), hog_length))
    for i in range(0, len(filenames)):
        filename = path + '/' + filenames[i]
        data[i, :] = extract_hog(imread(filename, plugin='matplotlib'), rois[i, :])
    return data



train_data_path = './data/train/'

(train_filenames, train_rois, train_labels) = read_gt(train_data_path)

print('extract features')
t = time()
train_features = extract_features(train_data_path, train_filenames, train_rois)
t = time() - t
print('{} min, {} sec'.format(t // 60, round(t%60)))

print('cross validation')
t = time()
test(train_features, train_labels)
t = time() - t
print('{} min, {} sec'.format(t // 60, round(t%60)))

