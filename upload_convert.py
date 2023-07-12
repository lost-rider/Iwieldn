from tqdm import tqdm
import cv2
from PIL import Image, ImageTk
import datetime
from datetime import date
import os
import numpy as np
import sys
from matplotlib import pyplot as plt
import shutil
from pathlib import Path
import glob
from pydicom import dcmread
from pydicom.data import get_testdata_file
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
from scipy.ndimage.filters import median_filter
import sys
from flask import Flask, request
import flask
import json
from flask_cors import CORS

# @app.route('/users', methods=["GET", "POST"])
# def users():
#     print("users endpoint reached...")
#     if request.method == "GET":
#         with open("users.json", "r") as f:
#             data = json.load(f)
#             data.append({
#                 "username": "user4",
#                 "pets": ["hamster"]
#             })

#             return flask.jsonify(data)
#     if request.method == "POST":
#         received_data = request.get_json()
#         print(f"received data: {received_data}")
#         message = received_data['data']
#         return_data = {
#             "status": "success",
#             "message": f"received: {message}"
#         }
#         return flask.Response(response=json.dumps(return_data), status=201)


# f1=sys.argv[1]
f1 = "C:\\registration-main3\\img_result\\"

# img_path=sys.argv[2]
# img_path=img_path.replace("||"," ")
# img_path="E:\\Defects 03.01.2022\\Cluster of porosities\\3022 - A1KFC + A2KFPS - S20 - DK30 - S - 23.08.2020@18MM.dcm"


def stretch1(a, lower_thresh, upper_thresh):
    r = 65535.0 / (upper_thresh - lower_thresh + 2)  # unit of stretching
    out = np.round(r * np.where(a >= lower_thresh, a -
                   lower_thresh + 1, 0)).clip(max=65535)
    return out.astype(a.dtype)


def stretch2(a, lower_thresh, upper_thresh):
    r = 255.0 / (upper_thresh - lower_thresh + 2)  # unit of stretching
    out = np.round(r * np.where(a >= lower_thresh, a -
                   lower_thresh + 1, 0)).clip(max=255)
    return out.astype(a.dtype)


files = glob.glob("C:\\registration-main3\\test_images\\*")

for img_path in tqdm(files):
    try:
        ds = dcmread(img_path)
    except:
        print("fail to read")
    M = ds.pixel_array
    array = np.full((M.shape[0], M.shape[1]), 65535)
    array = array.astype(np.uint16)
    DAPI = array - M
    histogram, bin_edges = np.histogram(DAPI.ravel(), bins=range(65536))

    x = []
    d = 0
    for i in range(65534, 0, -1):
        if histogram[i] > 25000 and d < 5:
            x.append(i)
            d = d + 1

    y = []
    c = 0
    for i in range(0, 65535):
        if histogram[i] > 50 and c < 5:
            y.append(i)
            c = c + 1

    z = []
    d = 0
    for i in range(65534, 0, -1):
        if histogram[i] > 250 and d < 40:
            z.append(i)
            d = d + 1

    if len(x) == 0:
        m = stretch2(DAPI, lower_thresh=y[len(
            y) - 1] - 5, upper_thresh=z[len(z) - 1] + 800)
    else:
        m = stretch2(DAPI, lower_thresh=y[len(
            y) - 1] - 5, upper_thresh=x[len(x) - 1] + 800)
    pil1 = Image.fromarray(m.astype(np.uint8))
    pil1.save(f1 + os.path.basename(img_path) + ".png")
    image_bw = cv2.imread(f1 + os.path.basename(img_path) + ".png", 0)
    image_bw = cv2.medianBlur(image_bw, 5)
    clahe = cv2.createCLAHE(clipLimit=5, tileGridSize=(32, 32))
    final_img = clahe.apply(image_bw)
    cv2.imwrite(f1 + os.path.basename(img_path) + ".png", final_img)
