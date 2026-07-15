import os
import shutil
from tqdm import tqdm 
import dill

#‘/data2/fzg/MDocking/VSDS_DTEBV-D/exist_data_name_dict.pkl’
#/data2/fzg/MDocking/VSDS_DTEBV-D/data_name.txt

with open('/data2/fzg/MDocking/VSDS_DTEBV-D/exist_data_name_dict.pkl', 'rb') as f:
    valid_name_dict = dill.load(f)


data_name_list = []
with open('/data2/fzg/MDocking/VSDS_DTEBV-D/data_name.txt') as f:
    for name in f:
        data_name_list.append(name.strip())

#print('data_name_list num:', len(data_name_list))
#print('valid_name_dict num:', len(valid_name_dict))

loss_num = {}
for data_name in tqdm(valid_name_dict):
    num = 0
    for name in valid_name_dict[data_name]:
        s_dir = f'/data/fzg/MDocking/unimol_docking_v2/interface/vsds2/{data_name}_sdf/{name}' #O00329-6PYR_sdf
        if not os.path.exists(s_dir):
            s_dir = f'/data2/fzg/MDocking/unimol_docking_v2/interface/vsds/{data_name}_sdf/{name}' #O00329-6PYR_sdf
            if not os.path.exists(s_dir):
                num += 1
                continue
    
        t_dir = f'/data2/fzg/MDocking/Docking_baseline/unimol_docking_v2/interface/valid_vsds/{data_name}/{name}'
        os.makedirs(t_dir, exist_ok= True)
        shutil.copytree(s_dir, t_dir, dirs_exist_ok = True)
    
    loss_num[data_name] = num

#print('loss_num:', loss_num)
'''
loss_num: {'P00533-8A27': 0, 'P06276-6ZWI': 0, 'P20309-8EA0': 0, 'P22303-4M0E': 0, 'P11511-5JKV': 0, 'P01116-4TQA': 0, 'Q15078-1UNL': 0, 
'P00746-5NAT': 0, 'Q16790-6G9U': 0, 'P04626-7PCD': 0, 'Q02127-4OQV': 0, 'Q00987-6Q9L': 0, 'P00918-1LUG': 0, 'P36544-5AFN': 0, 'P28223-6WHA': 0, 
'P18031-2F71': 0, 'P00374-1KMV': 0, 'Q8IXJ6-4RMH': 0, 'P08172-6U1N': 0, 'P07858-6AY2': 0, 'P28335-8DPF': 0, 'P34969-7XTC': 0, 'P10415-6GL8': 0, 
'P42574-2XYG': 0, 'P00749-1GJ7': 0, 'Q9NWZ3-6EGE': 0, 'P35462-8IRT': 0, 'P03372-7NFB': 0, 'P21397-2Z5Y': 0, 'P07711-2XU3': 0, 'O14965-5L8L': 0, 
'P07900-5XRD': 0, 'P53779-7ORF': 0, 'P06493-6GU2': 0, 'P00519-2HZI': 0, 'P24666-7KH8': 0, 'Q9Y337-6QFE': 0, 'P27338-6FW0': 0, 'P04150-4UDD': 0, 
'O60341-2Z5U': 0, 'Q06124-5EHR': 0, 'P33527-2CBZ': 0, 'P11388-1ZXM': 0, 'P21453-7EO4': 0, 'Q07820-8G3S': 0, 'P01375-7JRA': 0, 'P29274-5NM4': 0, 
'P08238-6N8Y': 0, 'P10275-8E1A': 0, 'P31645-5I6X': 0, 'Q9UBN7-8G44': 0, 'P07550-6PS2': 0, 'P56524-6FYZ': 0, 'P00747-5UGD': 0, 'P08908-7E2Z': 0, 
'P98170-6GJW': 0, 'P03952-5TJX': 0, 'P11362-5EW8': 0, 'P36888-6JQR': 0, 'Q16539-5WJJ': 0, 'P14416-7JVR': 0, 'P28482-8AOJ': 0, 'Q92769-7KBG': 0, 
'P41143-4N6H': 0, 'P06239-1QPC': 0, 'P00742-2JKH': 0, 'P07339-4OD9': 0, 'P08246-5ABW': 0, 'P24941-6Q4G': 0, 'P53350-2RKU': 0, 'Q99500-7C4S': 0, 
'P49841-1O6L': 0, 'P09917-3V99': 0, 'P08581-4R1V': 0, 'P11802-7SJ3': 0, 'P55055-6S5K': 0, 'P29275-7XY6': 0, 'P29474-4D1P': 0, 'P40763-5AX3': 0, 
'P12931-7NG7': 0, 'P09874-6NRH': 0, 'Q05397-6YOJ': 0, 'P35354-5F19': 0, 'Q14432-7LRC': 0, 'P00734-1BA8': 0, 'Q9BY41-5BWZ': 0, 'P30542-5UEN': 0, 
'O15379-4A69': 0, 'Q9Y233-2OUR': 0, 'P41594-4OO9': 0, 'P35968-2XIR': 0, 'P00338-5W8J': 0, 'Q92731-3OLL': 0, 'Q07817-6VWC': 0, 'P27986-4JPS': 0, 
'Q96GD4-4AF3': 0, 'P08684-3NXU': 0, 'P08253-7XJO': 0, 'P00748-6X0S': 0, 'P14780-6ESM': 0, 'P33261-4GQS': 0, 'Q96RI1-6HL1': 0, 'O14757-2YEX': 0, 
'P78536-2DDF': 0, 'Q03181-5U3Q': 0, 'P10721-1T46': 0, 'P05177-2HI4': 0, 'P10635-3TBG': 0, 'Q13255-3KS9': 0, 'P51449-7NPC': 0, 'Q08499-1Y2K': 0, 
'Q9GZT9-4BQY': 0, 'P15056-8C7X': 0, 'P27487-4A5S': 0, 'P34972-6KPF': 0, 'P37231-6MS7': 0, 'P11712-5X23': 0, 'P03956-1HFC': 0, 'Q14416-4XAQ': 0, 
'P21554-6KPG': 0, 'P35372-8EFB': 0, 'Q9H7B4-6P7Z': 0, 'P49840-7SXF': 0, 'Q07869-6LXA': 0, 'O75874-6BKX': 0, 'O43614-7XRR': 0, 'P45452-5B5O': 0, 
'O75116-7JNT': 0, 'Q6V1X1-7AYQ': 0, 'P23458-6N7A': 0, 'Q9UHL4-3N0T': 0, 'Q86TI2-7ZXS': 0, 'P08254-1CAQ': 0, 'O60885-4QB3': 0, 'Q06187-5P9J': 0, 
'P42336-8EXL': 0, 'Q13464-7S25': 0, 'P48736-6AUD': 0, 'Q92793-5J0D': 0, 'O43613-6TOD': 0, 'O60674-7LL4': 0, 'P37173-5QIN': 0, 'Q13546-6NW2': 0, 
'O75469-6TFI': 0, 'P29597-3LXP': 0, 'O00329-6PYR': 0, 'P43405-4YJR': 0}
'''