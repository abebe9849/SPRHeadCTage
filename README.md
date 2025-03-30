

### install
conda env create -n SCC -file env.yaml

### download data
cd /data
kaggle competitions download -c spr-head-ct-age-prediction-challenge

### preprocess&train
bash preprocess.sh  

### trained weights
M004GRU:https://www.kaggle.com/datasets/abebe9849/SPRheadCTM004GRU
