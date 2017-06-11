#-*- coding:utf-8 -*-
#引数1：ランク(ファイル名)
#引数2：サイズ(ピクセル)
#引数3：画像数
#インプットデータを12倍(反転、回転、コントラスト変更)にするコード追加

import os
import numpy as np
import cv2
import sys
import csv

#引数のimageをreshapeしてcsvに出力する
def numpy_2_csv(image):
    image_reshape = np.reshape(image,(int(args[2])*int(args[2])))
    if len(image_reshape) > 0:
        csvWriter.writerow(image_reshape)
    else:
        pass

#引数のpath、nameでimageを保存する
def pic_write(path, name, image):
    cv2.imwrite(path + name, image)

#反転
def flip(image):
    image = np.fliplr(image)
    return image

#回転
def rotate(image,theta):
    center = tuple(np.array(image.shape[0:2])/2)
    rotMat = cv2.getRotationMatrix2D(center, theta, 1.0)
    image = cv2.warpAffine(image, rotMat, image.shape[0:2], flags=cv2.INTER_LINEAR)
    return image

#塩
def salt(image, percentage):
    num_salt = np.ceil(percentage * image.size * 0.5)
    coords = [np.random.randint(0, i-1 , int(num_salt)) for i in image.shape]
    image[coords[:-1]] = (255,255,255)
    return image

#ハイコントラスト
def high_contrast(image):
    min_table = 50
    max_table = 205
    diff_table = max_table - min_table

    LUT_HC = np.arange(256, dtype = 'uint8' )

    for i in range(0, min_table):
        LUT_HC[i] = 0
    for i in range(min_table, max_table):
        LUT_HC[i] = 255 * (i - min_table) / diff_table
    for i in range(max_table, 255):
        LUT_HC[i] = 255

    image = cv2.LUT(image, LUT_HC)
    return image

#ローコントラスト
def low_contrast(image):
    min_table = 50
    max_table = 205
    diff_table = max_table - min_table

    LUT_LC = np.arange(256, dtype = 'uint8' )  
    
    for i in range(256):
        LUT_LC[i] = min_table + i * (diff_table) / 255

    image = cv2.LUT(image, LUT_LC)
    return image

def pic_2_csv(path):

    count = 0

    for i in path:
        if i.find('.png') > 0 or i.find('.jpg') > 0 or i.find('.jpeg') > 0:
            print(i)
            I = args[1] + "/" + i
            image = cv2.imread(I)
#            if len(image.shape) == 3:
#                height, width, channels = image.shape[:3]
#            else:
#                height, width = image.shape[:2]
#                channels = 1
#            image_gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
            height,width = image.shape[:2]
            image_gray = image
            cascade = cv2.CascadeClassifier(cascade_path)
            faces = face_cascade.detectMultiScale(image_gray, 1.3, 5)
            if len(faces) > 0:
                for (x, y, w, h) in faces:
                    buf = max(w,h)
                    center_x = x + w / 2
                    center_y = y + h / 2
                out_len = min(buf,center_x,center_y,width-center_x,height-center_y)
                cutted = image[center_y-out_len:center_y+out_len, center_x-out_len:center_x+out_len]
                cutted_gray = cv2.cvtColor(cutted,cv2.COLOR_RGB2GRAY)
                cutted_gray_resize = cv2.resize(cutted_gray,(int(args[2]),int(args[2])))
                #---切り取り後灰色画像生成完了----                
                folder_pic = "./pic_changed/"
                original = cutted_gray_resize.copy()
                flipped = flip(original)

                #bruto force!!
                #誰かnp.arrayにappendしてうまいことfor文で回せるように変換してください！
                pic_write(folder_pic, args[1] + "/" + i + "_original.jpg", original)
                #numpy_2_csv(original)
                
                pic_write(folder_pic, args[1] + "/" + i + "_flip.jpg", flipped)
                #numpy_2_csv(flipped)

                pic_write(folder_pic, args[1] + "/" + i + "_rotate(5).jpg", rotate(original,5))
                #numpy_2_csv(rotate(original,5))

                pic_write(folder_pic, args[1] + "/" + i + "_rotate(-5).jpg", rotate(original,-5))
                #numpy_2_csv(rotate(original,-5))

                pic_write(folder_pic, args[1] + "/" + i + "_rotate(10).jpg", rotate(original,10))
                #numpy_2_csv(rotate(original,10))

                pic_write(folder_pic, args[1] + "/" + i + "_rotate(-10).jpg", rotate(original,-10))
                #numpy_2_csv(rotate(original,-10))

                pic_write(folder_pic, args[1] + "/" + i + "_flip_rotate(5).jpg", rotate(flipped,5))
                #numpy_2_csv(rotate(flipped,5))

                pic_write(folder_pic, args[1] + "/" + i + "_flip_rotate(-5).jpg", rotate(flipped,-5))
                #numpy_2_csv(rotate(flipped,-5))

                pic_write(folder_pic, args[1] + "/" + i + "_flip_rotate(10).jpg", rotate(flipped,10))
                #numpy_2_csv(rotate(flipped,10))

                pic_write(folder_pic, args[1] + "/" + i + "_flip_rotate(-10).jpg", rotate(flipped,-10))
                #numpy_2_csv(rotate(flipped,-10))

                pic_write(folder_pic, args[1] + "/" + i + "_original_hicon.jpg", high_contrast(original))
                #numpy_2_csv(high_contrast(original))
                
                pic_write(folder_pic, args[1] + "/" + i + "_flip_hicon.jpg", high_contrast(flipped))
                #numpy_2_csv(high_contrast(flipped))

            else:
                pass

#prepare face_detection
cascade_path = "C:/opencv/build/etc\haarcascades/haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

#get face directory pass 
args = sys.argv
path = os.listdir(args[1])

#file prepare
filename = args[1] + ".csv"
f = open(filename,'w')
csvWriter = csv.writer(f,delimiter=',',lineterminator="\n")

print("folder name = " + args[1])
print("size = " + args[2])
pic_2_csv(path)





