#%%
import pandas as pd
import seaborn as sns
import glob,pydicom,os,sys
import numpy as np

DATA_DIR = "../data"
tra_df = pd.read_csv("f{DATA_DIR}/train.csv")
os.makedirs("f{DATA_DIR}/window_png",exist_ok=True)
print(tra_df.columns)
# %%


print(tra_df.shape)

sns.histplot(tra_df["Age"],bins=100)

# %%
ex_tra = pd.read_csv("f{DATA_DIR}/test.csv")
print(ex_tra.shape)

tra_id = tra_df["StudyID"].unique()
ex_id = ex_tra["StudyID"].unique()
ex_id = [str(i).zfill(6) for i in ex_id]

test = pd.read_csv("f{DATA_DIR}/submission2.csv")
test_id = test["StudyID"].unique()
test_id = [str(i).zfill(6) for i in test_id]


print(len(tra_id),len(ex_id),len(test_id))

print(set(tra_id)&set(ex_id))
print(set(tra_id)&set(test_id))
print(tra_id[0],test_id[0])


sns.histplot(ex_tra["Age"],bins=100)


# %%
tra_df.head()
# %%

# %%
image_size_seg = (128, 128, 128)

# %%
def window_image(img, window_center,window_width, intercept, slope, rescale=True):
    '''
    This fucntion came from this notebook https://www.kaggle.com/code/redwankarimsony/ct-scans-dicom-files-windowing-explained
    If you want to understand more about windowing the referenced notebook is a good read.
    '''
    img = (img*slope +intercept) #for translation adjustments given in the dicom file. 
    img_min = window_center - window_width//2 #minimum HU level
    img_max = window_center + window_width//2 #maximum HU level
    img[img<img_min] = img_min #set img_min for all HU levels less than minimum HU level
    img[img>img_max] = img_max #set img_max for all HU levels higher than maximum HU level
    if rescale: 
        img = (img - img_min) / (img_max - img_min)*255.0 
    return img

def normalize_image(image):
    """
    Normalize image to the range [0, 1].
    """
    image = image - np.min(image)
    return image / np.max(image)
def load_dicom(dcm, window_center=None, window_width=None):
    """
    Process a DICOM file and save it as a PNG file.
    """    
    try:
        image = dcm.pixel_array
    except:
        return
    if window_center is not None and window_width is not None:
        
        image = window_image(image, window_center, window_width, dcm.RescaleIntercept, dcm.RescaleSlope)

    if dcm.PhotometricInterpretation == "MONOCHROME1":
        image = np.invert(image)
    
    normalized_image = normalize_image(image)
    return normalized_image
def load_dicom_line_par(path):

    t_paths = sorted(glob.glob(os.path.join(path, "*"))[1:], key=lambda x: int(x.split('/')[-1].split(".")[0]))

    #indices = np.quantile(list(range(n_scans)), np.linspace(0., 1., image_size_seg[2])).round().astype(int)
    #t_paths = [t_paths[i] for i in indices]
    dicoms = [pydicom.dcmread(d, force= True) for d in t_paths]
    sereis_instance_uid = [d.SeriesInstanceUID for d in dicoms]
    print(set(sereis_instance_uid))
        
    images = {i:[] for i in set(sereis_instance_uid)}
    z_pos = {i:[] for i in set(sereis_instance_uid)}
    for dcm,s_id in zip(dicoms,sereis_instance_uid):
        d = load_dicom(dcm,40,80)
        #d = load_dicom(dcm,500,2000)
        if d is None:
            continue
        images[s_id].append(d)
        z_pos[s_id].append(float(dcm.ImagePositionPatient[-1]))
        
    #z_pos = [float(d.ImagePositionPatient[-1]) for d in dicoms]
    for s_id in set(sereis_instance_uid):
        images[s_id] = np.stack(images[s_id], -1)[:,:,np.argsort(z_pos[s_id])]
        z_pos[s_id] = np.sort(z_pos[s_id])
    return images,z_pos


def save_3stack(imgs):
    new_img = []
    for i in range(imgs.shape[-1]-2):
        img = imgs[:,:,i:i+3]
        print(img.shape)
       

base_dir = "f{DATA_DIR}/dataset_jpr_train/dataset_jpr_train"
output_base_dir = "f{DATA_DIR}/window_png/"  # Update this path as necessary
os.makedirs(output_base_dir,exist_ok=True)
output_z_pos = output_base_dir+"z_pos"
output_3d = output_base_dir+"3d"
os.makedirs(output_3d,exist_ok=True)

os.makedirs(output_z_pos,exist_ok=True)

for root_dir in ["1","2","3"]:
    root_path = os.path.join(base_dir, root_dir)
    for accession_number in os.listdir(root_path)[::-1]: 
        if len(glob.glob(f"{output_3d}/{accession_number}*"))>0:
            continue
        accession_path = os.path.join(root_path, accession_number)
        img_3d,z_pos = load_dicom_line_par(accession_path)
        for i in img_3d.keys():
            #np.save(f"{output_z_pos}/{accession_number}____{i}.npy",z_pos[i])
            np.save(f"{output_3d}/{accession_number}____{i}.npy",img_3d[i])

        
base_dir = "f{DATA_DIR}/dataset_jpr_test_notarget/dataset_jpr_test/4"
output_base_dir = "f{DATA_DIR}/window_png/"  # Update this path as necessary
os.makedirs(output_base_dir,exist_ok=True)
output_z_pos = output_base_dir+"z_pos"
output_3d = output_base_dir+"3d"
os.makedirs(output_3d,exist_ok=True)

os.makedirs(output_z_pos,exist_ok=True)
        


for accession_number in os.listdir(base_dir)[::-1]:    
    accession_path = os.path.join(base_dir, accession_number)
    img_3d,z_pos = load_dicom_line_par(accession_path)
    for i in img_3d.keys():
        np.save(f"{output_z_pos}/{accession_number}____{i}.npy",z_pos[i])
        np.save(f"{output_3d}/{accession_number}____{i}.npy",img_3d[i])
        

base_dir = "f{DATA_DIR}/dataset_jpr_test2/dataset_jpr_test2"
output_base_dir = "f{DATA_DIR}/window_png/"  # Update this path as necessary
os.makedirs(output_base_dir,exist_ok=True)
output_z_pos = output_base_dir+"z_pos"
output_3d = output_base_dir+"3d"
os.makedirs(output_3d,exist_ok=True)

os.makedirs(output_z_pos,exist_ok=True)
        


for accession_number in os.listdir(base_dir)[::-1]:    
    accession_path = os.path.join(base_dir, accession_number)
    img_3d,z_pos = load_dicom_line_par(accession_path)
    for i in img_3d.keys():
        np.save(f"{output_z_pos}/{accession_number}____{i}.npy",z_pos[i])
        np.save(f"{output_3d}/{accession_number}____{i}.npy",img_3d[i])