from app import app
from flask import Flask, jsonify, request, make_response
from matplotlib import pyplot as plt
from matplotlib.image import imread
import os
import pandas as pd
import time
from matplotlib import pyplot as plt
from matplotlib.image import imread
import numpy as np
import cv2
import json
import MySQLdb
from time import sleep
import MySQLdb
from PIL import Image
import requests




@app.route("/")
def home():
    message = {
        'status': 200,
        'message': 'Success access Api',
    }
    resp = jsonify(message)
    resp.status_code = 200
    return resp


@app.errorhandler(404)
def not_found(error=None):
    message = {
        'status': 404,
        'message': 'Not Found'+request.url,
    }
    resp = jsonify(message)
    resp.status_code = 404
    return resp

@app.get("/cron")
def cron():
    try:
        # folder_path = 'c:/xampp/htdocs/klasifikasi/modal/dataset/uploads'
        # new_folder_path = 'c:/xampp/htdocs/klasifikasi/data'
        # image_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]
        # hsv_data = []
        # for filename in os.listdir(folder_path):
        #     img = cv2.imread(os.path.join(folder_path, filename))
        #     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        #     hsv_mean = np.mean(hsv, axis=(0, 1))
        #     hsv_data.append(hsv_mean)
        # df = pd.DataFrame(hsv_data, columns=['H', 'S', 'V'])
        width = 128
        height = 128


        def setNorm(img, b, c):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = myresize(img, b, c)
            return img


        def myresize(img, b, c):
            dim = (b, c)
            resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
            # print(resized)
            return resized


        def cekHasil(img, train_image_names, proj_data, w):

            unknown_manggo = plt.imread(img)
            ad = setNorm(unknown_manggo, height, width)
            unknown_manggo_vector = np.array(ad, dtype='float64').flatten()
            normalised_umanggo_vector = np.subtract(unknown_manggo_vector, mean_manggo)

            w_unknown = np.dot(proj_data, normalised_umanggo_vector)
            diff = w - w_unknown
            norms = np.linalg.norm(diff, axis=1)
            index = np.argmin(norms)
            hasil = ''
        #     if norms[index] < t0:
        #         hasil=train_image_names[index].split('_')[0]
        #     else:
        #         hasil='tidak ada yang matching'
            Value = norms[index]
            Nmax = t0
            Nmin = 0  # interval
            fuzzy = ((Value-Nmin)/(Nmax-Nmin))
            print(fuzzy)
            print(Value-Nmin)
            print(Value-Nmin)

            if fuzzy < 0.5:
                hasil = train_image_names[index].split('_')[0]
            else:
                hasil = 'tidak ada yang matching'

            return hasil


        PATH_UJI = "C:/xampp/htdocs/klasifikasi_mangga/ypathfile/"


        ada = 0
        db = MySQLdb.connect("localhost", "root", "", "db_mangga1")
        cursor = db.cursor()

        cursor.execute(
            "SELECT `id_pengujian`, `nama_pengujian`, `gambar`,`status`,jenis_mangga FROM `tb_pengujian` where status='0' order by id_pengujian desc")
        for row in cursor.fetchall():
            ada = 1
            idx = row[0]
            nama_pengujian = row[1]
            gambar = row[2]
            status = row[3]
            jenis_mangga = row[4]
            if jenis_mangga == 'Mangga Gedong':
                mean_manggo = np.load(
                    'hasil traning/gedong/mean_manggo_gedong.npy')
                proj_data = np.load('hasil traning/gedong/proj_data_gedong.npy')
                train_image_names = np.load(
                    'hasil traning/gedong/train_image_names_gedong.npy')
                TEST_IMG_FOLDER = 'dataset/Gedong/testing/'
                t0 = np.load('hasil traning/gedong/t0_gedong.npy')
                w = np.load('hasil traning/gedong/w_gedong.npy')
            elif jenis_mangga == 'Mangga Indramayu':
                mean_manggo = np.load(
                    'hasil traning/indramayu/mean_manggo_indramayu.npy')
                proj_data = np.load(
                    'hasil traning/indramayu/proj_data_indramayu.npy')
                train_image_names = np.load(
                    'hasil traning/indramayu/train_image_names_indramayu.npy')
                TEST_IMG_FOLDER = 'dataset/Indramayu/testing/'
                t0 = np.load('hasil traning/indramayu/t0_indramayu.npy')
                w = np.load('hasil traning/indramayu/w_indramayu.npy')
            elif jenis_mangga == 'Mangga Manalagi':
                mean_manggo = np.load(
                    'hasil traning/manalagi/mean_manggo_manalagi.npy')
                proj_data = np.load(
                    'hasil traning/manalagi/proj_data_manalagi.npy')
                train_image_names = np.load(
                    'hasil traning/manalagi/train_image_names_manalagi.npy')
                TEST_IMG_FOLDER = 'dataset/Manalagi/testing/'
                t0 = np.load('hasil traning/manalagi/t0_manalagi.npy')
                w = np.load('hasil traning/manalagi/w_manalagi.npy')
            elif jenis_mangga == 'Mangga Harumanis':
                mean_manggo = np.load(
                    'hasil traning/harumanis/mean_manggo_harumanis.npy')
                proj_data = np.load(
                    'hasil traning/harumanis/proj_data_harumanis.npy')
                train_image_names = np.load(
                    'hasil traning/harumanis/train_image_names_harumanis.npy')
                TEST_IMG_FOLDER = 'dataset/Harumanis/testing/'
                t0 = np.load('hasil traning/harumanis/t0_harumanis.npy')
                w = np.load('hasil traning/harumanis/w_harumanis.npy')
            else:
                mean_manggo = np.load('mean_manggo.npy')
                proj_data = np.load('proj_data.npy')
                train_image_names = np.load('train_image_names.npy')
                TEST_IMG_FOLDER = 'dataset/Gedong/testing/'
                t0 = np.load('t0.npy')
                w = np.load('w.npy')
            lokasi = PATH_UJI+gambar
            hasil = cekHasil(lokasi, train_image_names, proj_data, w)

            katagori = hasil

            cursor2 = db.cursor()
            sql2 = "UPDATE `tb_pengujian`  set `status`='1',`katagori`='%s'where id_pengujian='%s'" % (
                katagori, idx)

            try:
                cursor.execute(sql2)
                db.commit()
                print(sql2)
                print("Berhasil Update")

            except:
                print("Failed Update")
                print(sql2)

    except Exception as e:
        return make_response(jsonify({"error":str(e)}),400)
    return make_response(jsonify({"status": "success",
                                    "message": "success update data",
                                    }))  
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
