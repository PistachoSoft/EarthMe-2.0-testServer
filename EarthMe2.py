# coding=utf-8
import os
import time
import psutil
from sys import getsizeof
from random import randint, shuffle
from concurrent.futures import ThreadPoolExecutor

from threading import Lock
import cv2
from scipy.spatial.ckdtree import cKDTree
from flask import Flask, request, send_from_directory, abort, render_template
from werkzeug.utils import secure_filename
import numpy as np

"""
    Main cKDTree, will contain all the images¿? or just the names of the dirs
    to be considered ... whether this uses too much memory, the disk will be used
"""
__tree = cKDTree([(0, 0, 0)])
__MAX_SIZE = 150
__EARTH_IMG_EDGE = 50
__MIN_CLASS_DATA = int(150)
__PY_CLASS_DIR = "../em-data/py_class_50"
__DATA_DIR = "../em-data/data_50"
__COLOR_MASK = 0xF8

__current_milli_time = lambda: int(round(time.time() * 1000))


def __get_index_array(array):
    """
    Given an array, which must be formed by three ints [b,g,r], generates
    the str with the COLOR_MASK
    """
    return __get_index_bgr(array[0], array[1], array[2])


def __get_index_img(img):
    """
    Given an image (numpy.array) this method generates the str that matches
    the correspondent image mean color
    """
    b, g, r, _ = cv2.mean(img)
    return __get_index_bgr(b, g, r)


def __get_index_bgr(b, g, r):
    """
    Given a BGR color this method returns a string applying COLOR_MASK
    to the components
    """
    return str(int(round(b)) & __COLOR_MASK).zfill(3) + \
           str(int(round(g)) & __COLOR_MASK).zfill(3) + \
           str(int(round(r)) & __COLOR_MASK).zfill(3)


def __max(x, y):
    """
    returns a tuple where the left component is always the greater
    """
    if (y > x):
        return (y, x)
    else:
        return (x, y)


# def allowed_file(filename):
# """
# Verifies whether the filename is allowed in ALLOWED_EXTENSIONS or not
#     """
#     return '.' in filename and \
#            filename.rsplit('.', 1)[1] in _ALLOWED_EXTENSIONS


def __gen_colors(dir):
    """
    Generates the color dir, which is named after its BGR
    """
    global __tree
    print "generating color", dir
    # some timestamp to avoid collisions
    timestamp = __current_milli_time()
    color = dir.rsplit("/", 1)[1]
    b = int(color[0:3])
    g = int(color[3:6])
    r = int(color[6:9])
    color = np.array((b, g, r))

    if not os.path.isdir(dir):
        os.mkdir(dir)

    # get the k neighbours of the color
    indexes = __tree.query(np.array((b, g, r)), k=__MIN_CLASS_DATA)[1]

    # get random colors from the neighbours
    for x in range(__MIN_CLASS_DATA):
        rand = randint(0, len(indexes) - 1)
        some_color = __tree.data[indexes[rand]]
        some_color_dir = __get_index_array(some_color)
        dir_list = os.listdir(os.path.join(__PY_CLASS_DIR,
                                           some_color_dir))
        rand = randint(0, len(dir_list) - 1)
        img = cv2.imread(os.path.join(__PY_CLASS_DIR, some_color_dir, dir_list[rand])).astype("int")
        cv2.imwrite(os.path.join(dir, str(timestamp) + str(x) + ".jpg"), (img + color) / 2)

    # get new cKDTree with the new color
    _dummy = __tree.data.tolist()
    _dummy.append([b, g, r])
    __tree = cKDTree(_dummy)
    pass


lock = Lock()


def get_pix_img(pixel):
    color_dir = os.path.join(__PY_CLASS_DIR, __get_index_array(pixel))

    # verify the color's existence and its variety
    lock.acquire()
    if not os.path.isdir(color_dir) or len(os.listdir(color_dir)) < __MIN_CLASS_DATA:
        __gen_colors(color_dir)
    lock.release()

    color_images = os.listdir(color_dir)
    index = randint(0, len(color_images) - 1)
    return cv2.imread(os.path.join(color_dir, color_images[index]))


def process_img(img_name, img_dir="./upload", output_dir="./processed"):
    def process_reduced_img(img):
        #
        processed = np.array(()).astype("int")
        height, width, _ = img.shape
        # for each row
        for i in range(height):
            if i % 10 == 0:
                print "process", (float(i) / height * 100), "%"

            # get an image for each pixel
            row = np.hstack(map(get_pix_img, img[i, :, :]))
            if processed.shape == (0,):
                processed = row
            else:
                processed = np.vstack((processed, row))

        return processed

    # if the image exist on the output dir smth went wrong, shouldn't happen
    if os.path.exists(os.path.join(output_dir, img_name)):
        return False

    img = cv2.imread(os.path.join(img_dir, img_name))

    # reduce the image's size, ued area interpolation to blend the colors
    # this is used to easily get a relation pixel <-> image when processing
    height, width, _ = img.shape
    max_edge, _ = __max(height, width)
    if max_edge > __MAX_SIZE:
        ratio = float(__MAX_SIZE) / max_edge
        img = cv2.resize(img, (0, 0), fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA)
    height, width, _ = img.shape

    # split the image to use a bit of concurrency
    m_height = height / 2
    m_width = width / 2

    executor = ThreadPoolExecutor(max_workers=4)
    m1 = executor.submit(process_reduced_img, img[:m_height, :m_width, :])
    m2 = executor.submit(process_reduced_img, img[:m_height, m_width:, :])
    m3 = executor.submit(process_reduced_img, img[m_height:, :m_width, :])
    m4 = executor.submit(process_reduced_img, img[m_height:, m_width:, :])
    d1 = np.hstack((m1.result(), m2.result()))
    d2 = np.hstack((m3.result(), m4.result()))

    cv2.imwrite(os.path.join(output_dir, img_name), np.vstack((d1, d2)))

    return True


def setup(gen_palette = False):
    """
    setups the environment cKDTree and stuff, must be used before processing
    :return:
    """
    global __tree

    # if no data is classified, do it
    if not os.path.isdir(__PY_CLASS_DIR) or len(os.listdir(__PY_CLASS_DIR)) == 0:
        os.mkdir(__PY_CLASS_DIR)

        for file in os.listdir(__DATA_DIR):
            img = cv2.imread(os.path.join(__DATA_DIR, file))
            __dir_name = os.path.join(__PY_CLASS_DIR, __get_index_img(img))
            if not os.path.isdir(__dir_name):
                os.mkdir(__dir_name)
            cv2.imwrite(os.path.join(__dir_name, file + ".jpg"), img)

    # gen a cKDTree with all the colors we start with
    __dummy = []
    for file in os.listdir(__PY_CLASS_DIR):
        __dummy.append((float(file[:3]), float(file[3:6]), float(file[6:])))

    __tree = cKDTree(__dummy)

    # TODO: should be done randomly instead of sequentially
    if gen_palette:
        bits_one = bin(__COLOR_MASK).count("1")
        shift = 8 - bits_one
        counter = 0
        dummy = 2 ** bits_one
        total = dummy ** 3
        p1,p2,p3 = np.mgrid[:dummy, :dummy, :dummy]
        dummy = (np.array(zip(p1.ravel(), p2.ravel(), p3.ravel())) << shift).tolist()
        shuffle(dummy)
        for pixel in dummy:
            counter += 1
            if counter % 20 == 0:
                print "filling up palette:",float(counter)/total*100,"%"
            __gen_colors(os.path.join(__PY_CLASS_DIR,__get_index_array(pixel)))
    # print "TREE:",len(__tree.data)
    # total = (2**bits_one)**3
    # for i in range(2**bits_one):
    #     for j in range(2**bits_one):
    #         for k in range(2**bits_one):
    #             counter += 1
    #             if counter % 50 == 0:
    #                 print "filling up palette:",float(counter)/total*100,"%"
    #             __gen_colors(os.path.join(__PY_CLASS_DIR,__get_index_bgr(i << shift, j << shift, k << shift)))
    # __TREEOP = __TREEOP.rebalance()
    # app.run()
