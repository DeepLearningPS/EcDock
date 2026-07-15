import shutil
import os
import csv
from tqdm import tqdm

from pathlib import Path

# 
CURRENT_DIR = os.getcwd()

# 
#os.chdir(CURRENT_DIR)
# 
print(":", CURRENT_DIR)



s_path = 'Distance_model/interface/tmpdata_predict_sdf_boxsize10'
t_path = f'{CURRENT_DIR}/tmpdata/tmpdata'
count = 0
for i in os.listdir(s_path):
    path = os.path.join(s_path, i)
    if os.path.exists(path) and os.path.isdir(path) and os.listdir(path): #
        #print('i:', i)
        s_file = os.path.join(s_path, i, f'interaction_{i}.pkl')
        t_file = os.path.join(t_path, i, f'interaction_{i}_v2.pkl')
        shutil.copy2(s_file, t_file)
        count += 1  


print('success num:', count) #428

path = Path(s_path)
if path.is_dir():  # 
        shutil.rmtree(path)
        print(f" {s_path} ")



s_path = 'Distance_model/interface/tmpdata_predict_sdf_random_protein_cutoff'

path = Path(s_path)
if path.is_dir():  # 
        shutil.rmtree(path)
        print(f" {s_path} ")








