from torch_geometric.data import Data, Batch
#from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader
import torch

# 
data_list = [
    Data(x=torch.tensor([[1], [2], [3]]), y=torch.tensor([1]), z=set(torch.tensor([[1, 2, 3], [1, 2, 3]]))),
    Data(x=torch.tensor([[4], [5], [6]]), y=torch.tensor([0]), z=set(torch.tensor([[4, 5], [4, 5], [4, 5]]))),
    Data(x=torch.tensor([[4], [5], [6]]), y=torch.tensor([0]), z=set(torch.tensor([[4, 5], [4, 5], [4, 5]]))),
    Data(x=torch.tensor([[1], [2], [3]]), y=torch.tensor([1]), z=set(torch.tensor([[1, 2, 3], [1, 2, 3]]))),
]


'''
data_list = [
    Data(x=torch.tensor([[1], [2], [3]]), y=torch.tensor([1]), z='a'),
    Data(x=torch.tensor([[4], [5], [6]]), y=torch.tensor([0]), z='b'),
    Data(x=torch.tensor([[4], [5], [6]]), y=torch.tensor([0]), z='c'),
    Data(x=torch.tensor([[1], [2], [3]]), y=torch.tensor([1]), z='d'),
]
'''

# collate
def custom_collate(batch):
    # （ 'z'）
    exclude_keys = ['z']

    # 
    batch_data = {}
    
    keys = batch[0].keys #pyg2.1.0，，
    # 
    for key in keys:
        if key in exclude_keys:
            # ，
            batch_data[key] = [getattr(data, key) for data in batch]
        else:
            # ，
            batch_data[key] = torch.cat([getattr(data, key) for data in batch], dim=0)

    return batch_data



# collate
def custom_collate2(batch):
    # （ 'z'）
    exclude_keys = ['z']
    

    # 
    batch_data = {}
    
    # 
    for key in batch[0].keys:
        if key in exclude_keys:
            # ，
            batch_data[key] = [getattr(data, key) for data in batch]
        else:
            # ，
            batch_data[key] = torch.cat([getattr(data, key) for data in batch], dim=0)

    return batch_data

exclude_keys = []

# DataLoadercollate
loader = DataLoader(data_list, batch_size=2, collate_fn=custom_collate2, exclude_keys = exclude_keys) 
#PYG dataloadercollate_fn，，exclude_keys，，exclude_keys

# DataLoader
for batch_data_ in loader:
    batch_data = batch_data_.cuda()
    print(batch_data)
    zz = batch_data.z
    print('batch_data.x:', batch_data.x)
    print('zz:', zz)
    zzz = []
    #[{tensor([1, 2, 3]), tensor([1, 2, 3])}, {tensor([4, 5]), tensor([4, 5]), tensor([4, 5])}]
    for i in zz:
        print('i:', i)
        ii = torch.stack(list(i), dim = 0) #list
        zzz.append(ii.cuda())
    
    print('zzz:', zzz)

    '''
    zzz: [tensor([[1, 2, 3],
        [1, 2, 3]]), tensor([[4, 5],
        [4, 5],
        [4, 5]])]
    '''
