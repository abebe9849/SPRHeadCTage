#%%
import numpy as np
import cv2,glob,os,multiprocessing

DATA_DIR = "../data"
img_paths = glob.glob("f{DATA_DIR}/25d/*")

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


def crop(paths_):
    os.makedirs(paths_.replace("25d_2","25d_2_xywh"),exist_ok=True)
    
    paths_ =  glob.glob(f"{paths_}/*")
    for p in paths_:

        img  = cv2.imread(p)

        xywh,size_ = func(img)
        xywh.append(size_)
        path_npy = p.replace("25d_2","25d_2_xywh").replace(".png",".npy")
        #print(path_npy,xywh)
        np.save(path_npy,np.array(xywh))
        
def crop_5(paths_):
    # 5ch stack
    os.makedirs(paths_.replace("25d_3","25d_3_xywh"),exist_ok=True)
    
    paths_ =  glob.glob(f"{paths_}/*")
    print(paths_)
    for p in paths_:

        img  = np.load(p)[:,:,[1,2,3]]

        xywh,size_ = func(img)
        xywh.append(size_)
        path_npy = p.replace("25d_3","25d_3_xywh").replace(".npy",".npy")
        #print(path_npy,xywh)
        np.save(path_npy,np.array(xywh))
        
def crop_7(paths_):
    # 5ch stack
    os.makedirs(paths_.replace("25d_4","25d_4_xywh"),exist_ok=True)
    
    paths_ =  glob.glob(f"{paths_}/*")
    for p in paths_:

        img  = np.load(p)[:,:,[3,4,5]]

        xywh,size_ = func(img)
        xywh.append(size_)
        path_npy = p.replace("25d_4","25d_4_xywh").replace(".npy",".npy")
        #print(path_npy,xywh)
        np.save(path_npy,np.array(xywh))
        



with multiprocessing.Pool(4) as p:
    p.map(crop,img_paths[:])


# %%
