

### install
conda create -n SCC -file env.yaml

### download data
cd /data
kaggle competitions download -c medical-ai-contest2024

### preprocess
bash preprocess.sh  
