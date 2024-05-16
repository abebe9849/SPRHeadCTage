

### install
conda create -n SCC -file env.yaml

### download data
cd /data
kaggle competitions download -c spr-head-ct-age-prediction-challenge

### preprocess&train
bash preprocess.sh  
