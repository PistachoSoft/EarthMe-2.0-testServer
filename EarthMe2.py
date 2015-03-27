# coding=utf-8
import os
import time
import psutil
import kdtree
from sys import getsizeof
from random import randint
from concurrent.futures import ThreadPoolExecutor

from threading import Lock
import cv2
from scipy.spatial import KDTree
from flask import Flask, request, send_from_directory, abort, render_template
from werkzeug.utils import secure_filename
import numpy as np

"""
    Main kdtree, will contain all the images¿? or just the names of the dirs
    to be considered ... whether this uses too much memory, the disk will be used
"""
__tree = KDTree([(0,0,0)])
__VARS = {}
__MAX_SIZE = 150
__EARTH_IMG_EDGE = 50
__MIN_CLASS_DATA = int(150)
__PY_CLASS_DIR = "./py_class_50"
__DATA_DIR = "./data_50"
__UPLOAD_FOLDER = './uploads'
__PROCESSED_FOLDER = './processed'
__ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
__COLOR_MASK = 0xF8

__app = Flask(__name__)
__app.config['UPLOAD_FOLDER'] = __UPLOAD_FOLDER
__app.config['PROCESSED_FOLDER'] = __PROCESSED_FOLDER

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
#     Verifies whether the filename is allowed in ALLOWED_EXTENSIONS or not
#     """
#     return '.' in filename and \
#            filename.rsplit('.', 1)[1] in _ALLOWED_EXTENSIONS


def __gen_colors(dir):
    """
    Generates the color dir, which is named after its BGR
    """
    global __tree
    # print "generating color", dir
    timestamp = __current_milli_time()
    color = dir.rsplit("/", 1)[1]
    b = int(color[0:3])
    g = int(color[3:6])
    r = int(color[6:9])
    color = np.array((b, g, r))

    if not os.path.isdir(dir):
        os.mkdir(dir)

    # __neighbours = __tree.search_knn((b,g,r),__MIN_CLASS_DATA)
    indexes = __tree.query(np.array((b,g,r)),k=__MIN_CLASS_DATA)[1]
    for x in range(__MIN_CLASS_DATA):
        rand = randint(0, len(indexes) - 1)
        some_color = __tree.data[indexes[rand]]
        some_color_dir = __get_index_array(some_color)
        dir_list = os.listdir(os.path.join(__PY_CLASS_DIR,
                                           some_color_dir))
        rand = randint(0, len(dir_list) - 1)
        img = cv2.imread(os.path.join(__PY_CLASS_DIR,some_color_dir, dir_list[rand])).astype("int")
        cv2.imwrite(os.path.join(dir, str(timestamp) + str(x) + ".jpg"), (img + color) / 2)

    _dummy = __tree.data.tolist()
    _dummy.append([b,g,r])
    __tree = KDTree(_dummy)
    # __tree.add((b,g,r))
    # __tree = __tree.rebalance()
    pass


def process_img(img_name, img_dir="./upload", output_dir="./processed"):
    lock = Lock()
    def process_reduced_img(img):
        #
        processed = np.array(()).astype("int")
        height, width, _ = img.shape
        # for each row
        for i in range(height):
            if i % 10 == 0:
                print "process",(float(i) / height * 100),"%"

            # dir = os.path.join(PY_CLASS_DIR, get_index_array(img[i, 0, :]))
            # if not os.path.isdir(dir) or len(os.listdir(dir)) < MIN_CLASS_DATA:
            # gen_colors(dir)
            # images = os.listdir(dir)
            # rand = randint(0, len(images) - 1)
            # row = cv2.imread(os.path.join(dir, images[rand]))
            row = np.array(())

            # for each pixel in this row
            for j in range(width):
                dir = os.path.join(__PY_CLASS_DIR, __get_index_array(img[i, j, :]))
                lock.acquire()
                if not os.path.isdir(dir) or len(os.listdir(dir)) < __MIN_CLASS_DATA:
                    __gen_colors(dir)
                lock.release()
                images = os.listdir(dir)
                rand = randint(0, len(images) - 1)

                if row.shape == (0,):
                    row = cv2.imread(os.path.join(dir, images[rand]))
                else:
                    row = np.hstack((row, cv2.imread(os.path.join(dir, images[rand]))))

            if processed.shape == (0,):
                processed = row
            else:
                processed = np.vstack((processed, row))

        return processed

    # if the image exist on the output dir smth went wrong
    if os.path.exists(os.path.join(output_dir, img_name)):
        return False
    # filename = os.path.basename(img_path)
    img = cv2.imread(os.path.join(img_dir, img_name))

    # reduce the image's size, area interpolation to keep the image aspect
    # this is used to easily get a relation pixel <-> image when processing
    height, width, _ = img.shape
    max_edge, _ = __max(height, width)
    if max_edge > __MAX_SIZE:
        ratio = float(__MAX_SIZE) / max_edge
        img = cv2.resize(img, (0, 0), fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA)
    height, width, _ = img.shape

    # cv2.imwrite(os.path.join(output_dir, img_name),
    #             process_reduced_img(img))

    m_height = height / 2
    m_width = width / 2

    executor = ThreadPoolExecutor(max_workers=4)
    m1 = executor.submit(process_reduced_img,img[:m_height,:m_width,:])
    m2 = executor.submit(process_reduced_img,img[:m_height,m_width:,:])
    m3 = executor.submit(process_reduced_img,img[m_height:,:m_width,:])
    m4 = executor.submit(process_reduced_img,img[m_height:,m_width:,:])
    d1 = np.hstack((m1.result(),m2.result()))
    d2 = np.hstack((m3.result(),m4.result()))

    cv2.imwrite(os.path.join(output_dir, img_name), np.vstack((d1,d2)))

    return True

# @app.route('/processed/<filename>')
# def uploaded_file(filename):
# """
#     Serves all the files stored in PROCESSED_FOLDER
#     """
#     return send_from_directory(app.config['PROCESSED_FOLDER'],
#                                filename)


# @app.route('/usage', methods=['GET'])
# def usage_stats():
#     return render_template("usage.html",
#                            memory=zip(psutil.virtual_memory()._fields, psutil.virtual_memory()),
#                            cpu=psutil.cpu_percent(),
#                            kdtree=getsizeof(tree),
#                            vars=get_vars_data(VARS))


# def get_vars_data(dict):
#     r = {}
#     for k, v in dict.iteritems():
#         r[k] = getsizeof(v)
#     return r


# __VARS['kdtree'] = __tree
# if there's no classified dir created, make a new one
def setup():
    global __tree
    if not os.path.isdir(__PY_CLASS_DIR) or len(os.listdir(__PY_CLASS_DIR)) == 0:
        os.mkdir(__PY_CLASS_DIR)

        for file in os.listdir(__DATA_DIR):
            img = cv2.imread(os.path.join(__DATA_DIR, file))
            __dir_name = os.path.join(__PY_CLASS_DIR, __get_index_img(img))
            if not os.path.isdir(__dir_name):
                os.mkdir(__dir_name)
            cv2.imwrite(os.path.join(__dir_name, file + ".jpg"), img)

    # add all the colors into the kdtree and images collections
    __dummy = []
    for file in os.listdir(__PY_CLASS_DIR):
        __dummy.append((float(file[:3]), float(file[3:6]), float(file[6:])))
    #     __TREEOP.add((int(file[:3]), int(file[3:6]), int(file[6:])))

    __tree = KDTree(__dummy)
    print "TREE:",len(__tree.data)
    bits_one = bin(__COLOR_MASK).count("1")
    shift = 8 - bits_one
    counter = 0
    total = (2**bits_one)**3
    for i in range(2**bits_one):
        for j in range(2**bits_one):
            for k in range(2**bits_one):
                counter += 1
                if counter % 50 == 0:
                    print "filling up palette:",float(counter)/total*100,"%"
                __gen_colors(os.path.join(__PY_CLASS_DIR,__get_index_bgr(i << shift, j << shift, k << shift)))
    # __TREEOP = __TREEOP.rebalance()
    # app.run()
