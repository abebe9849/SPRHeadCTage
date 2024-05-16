import pandas as pd
import glob
DATA_DIR = "../data"
tr_df = pd.read_csv("f{DATA_DIR}/train.csv")
ex_tra = pd.read_csv("f{DATA_DIR}/test.csv")
ex_id = ex_tra["StudyID"].unique()
ex_id = [str(i).zfill(6) for i in ex_id]
ex_tra["StudyID"]=ex_id
tr_df = pd.concat([tr_df,ex_tra]).reset_index(drop=True)

tra_df = pd.read_csv("f{DATA_DIR}/train.csv")
test = pd.read_csv("f{DATA_DIR}/submission2.csv")
df = pd.DataFrame()
paths = glob.glob("f{DATA_DIR}/window_png/25d/*")

from multiprocessing import Pool

# パスからStudyIDを抽出する関数
def extract_study_id(path):
    return path.split('/')[-1].split('____')[0]

def extract_study_id_(path):
    return path.split('/')[-1]


with Pool(processes=10) as pool:  # 利用するプロセス数を指定
    study_ids = pool.map(extract_study_id, paths)
    study_ids_ = pool.map(extract_study_id_, paths)
#df["StudyID"]= [i.split("/")[-2].split("____")[0] for i in paths]
df["StudyID"]= study_ids

df["StudyID___SeriesID"]= study_ids_#[i.split("/")[-2] for i in paths]


df = df.merge(tr_df, on="StudyID", how='left')
df_concat = pd.concat([
    df,
    tr_df.set_index('StudyID')
                .reindex(df['StudyID'].values)
                .reset_index(drop=True)
], axis=1)

print(df.head())


# %%
print(ex_tra.shape)

tra_id = tra_df["StudyID"].unique()



test_id = test["StudyID"].unique()
test_id = [str(i).zfill(6) for i in test_id]
test_df = df[df["StudyID"].isin(test_id)].reset_index(drop=True)
test_df.to_csv("f{DATA_DIR}/25d_test.csv",index=False) 
test_study_id = [int(i[2:]) for i in test_df["StudyID"].unique()]
print(len(test_study_id))


df = df[~df["StudyID"].isin(test_id)].reset_index(drop=True)
print(df["StudyID"].nunique())
df["bin_targets"]=pd.cut(df["Age"], 8, labels=False)
print(df["bin_targets"].value_counts())

from sklearn.model_selection import StratifiedKFold,StratifiedGroupKFold
import glob
from tqdm import tqdm
import numpy as np

sgkf = StratifiedGroupKFold(n_splits=10,random_state=2025,shuffle=True)
for fold, ( _, val_) in enumerate(sgkf.split(X=df, y=df.bin_targets.to_numpy(),groups=df.StudyID)):
    df.loc[val_ , "fold"] = fold
    
    val_df = df[df["fold"]==fold]
    print(val_df["bin_targets"].value_counts())

df.to_csv("f{DATA_DIR}/25d_10folds.csv",index=False)