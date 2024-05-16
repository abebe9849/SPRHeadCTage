import os
import numpy as np
import glob,cv2,multiprocessing
from PIL import Image
import matplotlib.pyplot as plt

"""
3dで保存したwindow済みの画像を2.5dに変換する


"""
DATA_DIR = "../data"

output_base_dir = "f{DATA_DIR}/window_png/"  # Update this path as necessary
output_z_pos = output_base_dir+"z_pos"
output_3d = output_base_dir+"3d"
imgs = glob.glob(f"{output_3d}/*")

def func(img):
    x = 255-cv2.morphologyEx(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:,:,1], cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))
    retval, im_bw = cv2.threshold(x, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(255-im_bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_rect = 0
    xywh = [0,0,1,1]
    for i in range(len(contours)):
        
        x, y, w, h = cv2.boundingRect(contours[i])
        rect = w*h
        if max_rect < rect:
            max_rect=rect
            xywh = [x, y, w, h]
    size_ = xywh[2]*xywh[3]/(img.shape[0]*img.shape[1])

        
    return xywh,size_

output_25d = output_base_dir+"25d"
os.makedirs(output_25d,exist_ok=True)

def save_3stack(path):
    imgs = np.load(path)
    p = path.split("/")[-1].replace(".npy","")#studyid____seriesid
    new_img = []
    
    os.makedirs(f"{output_25d}/{p}",exist_ok=True)
    for i in range(imgs.shape[-1]-2):
        img = (imgs[:,:,i:i+3]*255.).astype(np.uint8)
        cv2.imwrite(f"{output_25d}/{p}/{p}_{i}.png",img)
    return 
#same as 25d
output_25d_2 = output_base_dir+"25d_2"
os.makedirs(output_25d_2,exist_ok=True)
def save_3stack_2(path):
    imgs = np.load(path)
    p = path.split("/")[-1].replace(".npy","")#studyid____seriesid
    new_img = []
    
    os.makedirs(f"{output_25d_2}/{p}",exist_ok=True)
    for i in range(imgs.shape[-1]-4):
        img = (imgs[:,:,[i,i+2,i+4]]*255.).astype(np.uint8)
        cv2.imwrite(f"{output_25d_2}/{p}/{p}_{i}.png",img)
    return 
#not work
output_25d_3 = output_base_dir+"25d_3"
os.makedirs(output_25d_3,exist_ok=True)
def save_5stack_3(path):
    imgs = np.load(path)
    p = path.split("/")[-1].replace(".npy","")#studyid____seriesid
    new_img = []
    
    os.makedirs(f"{output_25d_3}/{p}",exist_ok=True)
    for i in range(imgs.shape[-1]-10):
        img = (imgs[:,:,[i,i+2,i+4,i+6,i+10]]*255.).astype(np.uint8)
        np.save(f"{output_25d_3}/{p}/{p}_{i}.npy",img)
    return 

output_25d_4 = output_base_dir+"25d_4"
os.makedirs(output_25d_4,exist_ok=True)
#not work
def save_7stack_int2(path):
    imgs = np.load(path)
    p = path.split("/")[-1].replace(".npy","")#studyid____seriesid
    new_img = []
    
    os.makedirs(f"{output_25d_4}/{p}",exist_ok=True)
    os.makedirs(f"{output_25d_4}/{p}".replace("25d_4","25d_4_xywh"),exist_ok=True)
    for i in range(imgs.shape[-1]-14):
        img = (imgs[:,:,[i,i+2,i+4,i+6,i+10,i+12,i+14]]*255.).astype(np.uint8)
        np.save(f"{output_25d_4}/{p}/{p}_{i}.npy",img)
        
        xywh,size_ = func(img[:,:,[3,4,5]])
        xywh.append(size_)
        path_npy = f"{output_25d_4}/{p}/{p}_{i}.npy".replace("25d_4","25d_4_xywh")
        #print(path_npy,xywh)
        np.save(path_npy,np.array(xywh))

        
    return 
    
import time
with multiprocessing.Pool(10) as p:
    p.map(save_3stack,imgs)