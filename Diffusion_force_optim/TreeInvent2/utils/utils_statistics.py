from tqdm import tqdm 
import math , os
from .utils_os import Find_Files
#from rdkit.Chem import Draw
import pickle 
from rdkit import Chem
import numpy as np 

def Statistic_Group_Systems(LFiles,SavePath='./datasets/STAT',Fname='PDBBind_Ligands'):
    ring_type_statistics={}
    ring_smis_statistics={} 
    ring_coming_statistics={}
    group_type_statistics={}
    group_smis_statistics={}
    group_coming_statistics={}
    
    for LigandF in tqdm(LFiles):
        try:
            with open(LigandF,'rb') as f:
                mg=pickle.load(f)
            max_single_ring_size=np.max(mg.l_groups_max_ring_size)
            flag=True
            if max_single_ring_size>8:
                flag=False
            if flag:
                for gid in range(mg.l_ngroups):
                    group_natoms=len(mg.l_groups[gid])
                    f=mg.l_groups_feats[gid]
         
                    fragmol=mg.Trans_ring_to_Mol(gid)
                    fragsmi=Chem.MolToSmiles(fragmol)
                    descriptor=mg.get_l_group_descriptor(gid)
                    if f[0]>=1:
                        if descriptor not in ring_type_statistics.keys():
                            ring_type_statistics[descriptor]=0
                            ring_smis_statistics[descriptor]={}
                            ring_coming_statistics[descriptor]={}
                            
                        ring_type_statistics[descriptor]+=1
                        if fragsmi not in ring_smis_statistics[descriptor].keys():
                            ring_smis_statistics[descriptor][fragsmi]=0
                            ring_coming_statistics[descriptor][fragsmi]=[]
                            
                        ring_smis_statistics[descriptor][fragsmi]+=1
                        ring_coming_statistics[descriptor][fragsmi].append(mg.savepath)
                    else:   
                        if descriptor not in group_type_statistics.keys():
                            group_type_statistics[descriptor]=0
                            group_smis_statistics[descriptor]={}                    
                            group_coming_statistics[descriptor]={}
                            
                        group_type_statistics[descriptor]+=1
                        if fragsmi not in group_smis_statistics[descriptor].keys():
                            group_smis_statistics[descriptor][fragsmi]=0
                            group_coming_statistics[descriptor][fragsmi]=[]
                            
                        group_smis_statistics[descriptor][fragsmi]+=1
                        group_coming_statistics[descriptor][fragsmi].append(mg.savepath)
                    
        except Exception as e:
            print (f'Failed to process {LigandF} due to {e}')
            continue

    with open (f'{SavePath}/{Fname}_ring.statis_pkl','wb') as f:
        pickle.dump((ring_type_statistics,ring_smis_statistics),f)
    with open(f'{SavePath}/{Fname}_group.statis_pkl','wb') as f:
        pickle.dump((group_type_statistics,group_smis_statistics),f)
    with open(f'{SavePath}/{Fname}_ring_coming.statis_pkl','wb') as f:
        pickle.dump(ring_coming_statistics,f)
    with open(f'{SavePath}/{Fname}_group_coming.statis_pkl','wb') as f:
        pickle.dump(group_coming_statistics,f)
    return 

def Statistic_Group_Systems_Parallel(LFiles,SavePath,NProcs=28,NFiles_Per_Job=1000):
    from multiprocessing import Pool,Queue,Manager,Process
    manager=Manager()
    DQueue=manager.Queue()
    p=Pool(NProcs)
    resultlist=[]
    
    if not os.path.exists(SavePath):
        os.system(f'mkdir -p {SavePath}')
    
    njobs=math.ceil(len(LFiles)/NFiles_Per_Job)
    
    for i in range(njobs):
        Files=LFiles[i*NFiles_Per_Job:(i+1)*NFiles_Per_Job]
        result=p.apply_async(Statistic_Group_Systems,(Files,SavePath,f'Jobs-{i}'))
        resultlist.append(result)
    
    for i in range(njobs):
        tmp=resultlist[i].get()
        if tmp is not None:
            print (tmp)
    
    p.terminate()
    p.join()
    
def Merge_Statistics(STATIS_FNAMES=None,SavePath=None,mode='group'):
    if STATIS_FNAMES is None:
        assert SavePath is not None, "SavePath should be provided when STATIS_FNAMES is None"
        STATIS_FNAMES=Find_Files(SavePath,f'_{mode}.statis_pkl')
    ftype_statistics={}
    fsmis_statistics={}
    
    for i in tqdm(range(len(STATIS_FNAMES))):
        with open(STATIS_FNAMES[i],'rb') as f:
            ftypes_dict,fsmis_dict=pickle.load(f)
        #print (fsmis_dict)
        for key in ftypes_dict.keys():
            if key not in ftype_statistics.keys():
                ftype_statistics[key]=0
            ftype_statistics[key]+=ftypes_dict[key]
        
        for key in fsmis_dict.keys():
            if key not in fsmis_statistics.keys():
                fsmis_statistics[key]={}
            for smi in fsmis_dict[key].keys():
                if smi not in fsmis_statistics[key].keys():
                    fsmis_statistics[key][smi]=0
                fsmis_statistics[key][smi]+=fsmis_dict[key][smi]
            
    with open(f'{SavePath}/{mode}_type_statistics.pkl','wb') as f:
        pickle.dump(ftype_statistics,f)
        
    with open(f'{SavePath}/{mode}_smis_statistics.pkl','wb') as f:
        pickle.dump(fsmis_statistics,f)

def Merge_Common_Statistics(type_dicts,smis_dicts,SavePath='./datasets',mode='group'):

    type_statistics={}
    smis_statistics={}
    
    for i in tqdm(range(len(type_dicts))):
        #print (fsmis_dict)
        for key in type_dicts[i].keys():
            if key not in type_statistics.keys():
                type_statistics[key]=0
            type_statistics[key]+=type_dicts[i][key]
        
        for key in smis_dicts[i].keys():
            if key not in smis_statistics.keys():
                smis_statistics[key]={}
            for smi in smis_dicts[i][key].keys():
                if smi not in smis_statistics[key].keys():
                    smis_statistics[key][smi]=0
                smis_statistics[key][smi]+=smis_dicts[i][key][smi]
            
    with open(f'{SavePath}/{mode}_type_statistics.pkl','wb') as f:
        pickle.dump(type_statistics,f)
        
    with open(f'{SavePath}/{mode}_smis_statistics.pkl','wb') as f:
        pickle.dump(smis_statistics,f)
    return 

 
def Select_Group_Systems(SavePaths='./datasets/STAT',cover=0.99):    
    with open (f'{SavePaths}/ring_type_statistics.pkl','rb') as f:
        ring_type_statistics=pickle.load(f)
    with open (f'{SavePaths}/ring_smis_statistics.pkl','rb') as f:
        ring_smis_statistics=pickle.load(f) 
    
    ring_type_num=np.array([v for k,v in ring_type_statistics.items()])
    ring_type_key=[k for k,v in ring_type_statistics.items()]    
    
    order=np.argsort(-ring_type_num)
    total_times=np.sum(ring_type_num)
    
    cum_ring_num=np.cumsum(ring_type_num[order])
    ring_type_label_dict={}
    ring_smis_label_dict={}
    
    for i in range(len(order)):
        if cum_ring_num[i]<cover*total_times and ring_type_statistics[ring_type_key[order[i]]]>10:
            ring_type_label_dict[ring_type_key[order[i]]]=ring_type_statistics[ring_type_key[order[i]]]
            ring_smis_label_dict[ring_type_key[order[i]]]=ring_smis_statistics[ring_type_key[order[i]]]
    
    with open(f'{SavePaths}/ring_label_num_{cover}.pkl','wb') as f:
        pickle.dump(ring_type_label_dict,f)
    
    with open(f'{SavePaths}/ring_label_smis_{cover}.pkl','wb') as f:
        pickle.dump(ring_smis_label_dict,f) 
    
    os.system(f"mkdir -p {SavePaths}/PNG_{cover:.3f}/ring")
    os.system(f"mkdir -p {SavePaths}/PNG_{cover:.3f}/group")
    
    with open (f'{SavePaths}/group_type_statistics.pkl','rb') as f:
        group_type_statistics=pickle.load(f)
    
    with open (f'{SavePaths}/group_smis_statistics.pkl','rb') as f:
        group_smis_statistics=pickle.load(f)
    
    single_atom_descriptors=[]
    single_atom_num=[]
    pharm_group_descriptors=[]
    pharm_group_num=[]
    
    group_type_label_dict={}
    group_smis_label_dict={}
    for key in group_type_statistics.keys():
        vars=key.split('-')
        Besnum=int(vars[10][3:])
        if Besnum!=1:
            if group_type_statistics[key]>5:
                group_type_label_dict[key]=group_type_statistics[key]
                group_smis_label_dict[key]=group_smis_statistics[key]
                pharm_group_descriptors.append(key)
        else:
            group_type_label_dict[key]=group_type_statistics[key]
            group_smis_label_dict[key]=group_smis_statistics[key]
            single_atom_descriptors.append(key)
            
    single_atom_descriptors.sort(reverse=True)
    pharm_group_descriptors.sort(reverse=True)
    
    with open(f'{SavePaths}/group_label_num_{cover}.pkl','wb') as f:
        pickle.dump(group_type_label_dict,f)
    
    with open(f'{SavePaths}/group_label_smis_{cover}.pkl','wb') as f:
        pickle.dump(group_smis_label_dict,f) 
    
    with open(f'{SavePaths}/descriptor_{cover:.3f}.csv','w') as f:
        for key in single_atom_descriptors:
            f.write(f'{key} {group_type_statistics[key]}\n')
            
        for key in pharm_group_descriptors:
            f.write(f'{key} {group_type_label_dict[key]}\n')
            try:
                groupmols=[Chem.MolFromSmiles(smi) for smi in group_smis_label_dict[key].keys()]
                groupmols=[mol for mol in groupmols if mol is not None]
                for groupmol in groupmols:
                    Chem.SanitizeMol(groupmol)

                legends=[str(group_smis_label_dict[key][smi]) for smi in group_smis_label_dict[key].keys()]
                img=Draw.MolsToGridImage(groupmols,molsPerRow=5,subImgSize=(250,250),legends=legends)
                img.save(f'{SavePaths}/PNG_{cover:.3f}/group/{key}.png')
            except Exception as e:
                print (f'{key} draw mols failed due to {e}')
        
        for key in ring_smis_label_dict.keys():
            f.write(f'{key} {ring_type_label_dict[key]}\n')
            try:
                ringmols=[Chem.MolFromSmiles(smi) for smi in ring_smis_label_dict[key]]
                print (key,len(ringmols))
                ringmols =[mol for mol in ringmols if mol is not None] 
                for ringmol in ringmols:
                    Chem.SanitizeMol(ringmol)
                if len(ringmols)==0:
                    continue
                legends=[str(ring_smis_label_dict[key][smi]) for smi in ring_smis_label_dict[key].keys()]
                img=Draw.MolsToGridImage(ringmols,molsPerRow=5,subImgSize=(250,250),legends=legends)
                img.save(f'{SavePaths}/PNG_{cover:.3f}/ring/{key}.png')
            except Exception as e:
                print (f'{key} draw mols failed due to {e}')
    return 
    


def Gen_Group_Class_Systems(SavePaths='./datasets/STAT',cover=0.98):  
    os.system(f"mkdir -p {SavePaths}/PNG_{cover:.3f}/ring")
    os.system(f"mkdir -p {SavePaths}/PNG_{cover:.3f}/group")  
    
    with open (f'{SavePaths}/ring_type_statistics.pkl','rb') as f:
        ring_type_statistics=pickle.load(f)
    with open (f'{SavePaths}/ring_smis_statistics.pkl','rb') as f:
        ring_smis_statistics=pickle.load(f) 
        
    with open (f'{SavePaths}/group_type_statistics.pkl','rb') as f:
        group_type_statistics=pickle.load(f)
    with open (f'{SavePaths}/group_smis_statistics.pkl','rb') as f:
        group_smis_statistics=pickle.load(f)
    
    single_atom_descriptors=[]
    pharm_group_descriptors=[]
    for key in group_type_statistics.keys():
        print (key)
        vars=key.split('-')
        Besnum=int(vars[10][3:])
        if Besnum==1:
            single_atom_descriptors.append(key)
        else:
            pharm_group_descriptors.append(key)
    
    single_atom_descriptors.sort(reverse=True)
    pharm_group_descriptors.sort(reverse=True)
    
    with open(f'{SavePaths}/descriptor_{cover:.3f}.csv','w') as f:
        for key in single_atom_descriptors:
            f.write(f'{key} {group_type_statistics[key]}\n')
            
        for key in pharm_group_descriptors:
            f.write(f'{key} {group_type_statistics[key]}\n')
            try:
                groupmols=[Chem.MolFromSmiles(smi) for smi in group_smis_statistics[key].keys()]
                groupmols=[mol for mol in groupmols if mol is not None]
                for groupmol in groupmols:
                    Chem.SanitizeMol(groupmol)
                legends=[str(group_smis_statistics[key][smi]) for smi in group_smis_statistics[key].keys()]
                
                img=Draw.MolsToGridImage(groupmols,molsPerRow=5,subImgSize=(250,250),legends=legends)
                
                img.save(f'{SavePaths}/PNG_{cover:.3f}/group/{key}.png')
                
            except Exception as e:
                print (f'{key} draw mols failed due to {e}')
                
        for key in ring_type_statistics.keys():
            f.write(f'{key} {ring_type_statistics[key]}\n')
            try:
            #if True:
                ringmols=[Chem.MolFromSmiles(smi) for smi in ring_smis_statistics[key]]
                print (key,len(ringmols))
                ringmols =[mol for mol in ringmols if mol is not None] 
                for ringmol in ringmols:
                    Chem.SanitizeMol(ringmol)
                if len(ringmols)==0:
                    continue
                legends=[str(ring_smis_statistics[key][smi]) for smi in ring_smis_statistics[key].keys()]
                img=Draw.MolsToGridImage(ringmols,molsPerRow=5,subImgSize=(250,250),legends=legends)
                img.save(f'{SavePaths}/PNG_{cover:.3f}/ring/{key}.png')
                
            except Exception as e:
                print (f'{key} draw mols failed due to {e}')
    return 