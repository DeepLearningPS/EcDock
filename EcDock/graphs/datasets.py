import pickle 
import os 
import numpy as np 
from rdkit import Chem
from EcDock.graphs import * 
from EcDock.utils import *
import random 
import math 
from tqdm import tqdm 
from torch.utils.data import Dataset,DataLoader
from ..comparm import *
import copy
import pickle
from pprint import pprint

def Deal_GEOM_Dataset(flist,max_conf_per_conf,savepath='molgraphs.pickle'):
    
    #raise Exception('stop2')
    
    #GeoDiffEcConf
    
    molgraphs=[]
    
    '''
    #PyG Data，Datalist，
    ：
    Data(atom_type=[19], boltzmannweight=[1], edge_index=[2, 40], edge_type=[40], idx=[1], nx=, pos=[19, 3], rdmol=<rdkit.Chem.rdchem.Mol object at 0x000001360B760DF0>,\
        smiles="C[C@@]1(N2CC2)C[C@H]1O", totalenergy=[1])
    
    '''
    
    #PyG Data
    #with open(flist_file, 'rb')as f:
        #flist = pickle.load(f)
        
    
    #PyGMolgraph
    node_noexit    = []
    natoms_noequal = []
    node_noexit_e    = []
    natoms_noequal_e = []
    error_conf_list = []
    true_conf_list  = []
    error_smiles_list = []
    true_smiles_list  = []
    for conf in tqdm(flist): #PyG DATA，#，rd_mol
        
        boltzmannweight = conf.boltzmannweight
        totalenergy     = conf.totalenergy
        atom_type       = conf.atom_type
        edge_index      = conf.edge_index #，0|all_atoms| - 1, ，，
        edge_type       = conf.edge_type
        smiles          = conf.smiles
        mol             = conf.rdmol
        idx             = conf.idx
        try:
            mol_noH=Chem.rdmolops.RemoveHs(mol,sanitize=True) #
            mol_noH=Neutralize_atoms(mol_noH)
            Chem.Kekulize(mol_noH)
            atoms=[atom.GetAtomicNum() for atom in mol_noH.GetAtoms()] #mol_noH.GetAtoms()，GetAtomicNum()：，
            natoms=sum([1 for i in atoms if i!=1]) #1
            #smiles=Chem.MolToSmiles(mol_noH)
            
            '''
            #
            if natoms>3 and '.' not in smiles:
                molgraph=Molgraph(mol_noH, boltzmannweight, totalenergy, atom_type, edge_index, edge_type, smiles=smiles)
                molgraphs.append(molgraph)
                if molgraph.natoms!=len(molgraph.atoms):
                    #print (e,smi,fname)
                    raise Exception(f'{molgraph.natoms}!={len(molgraph.atoms)}')
            
            else:
                error_list.append(conf)
            '''
            try:
                molgraph=Molgraph(mol_noH, boltzmannweight, totalenergy, atom_type, edge_index, edge_type, idx, smiles=smiles) #，
                #networkx.exception.NetworkXError: The node 7 is not in the graph.
            except Exception as e:
                node_noexit.append(conf)
                node_noexit_e.append(e)
                error_smiles_list.append(smiles)
                error_conf_list.append(conf)
                continue
                
            if molgraph.natoms!=len(molgraph.atoms):  #__init__，，，Molgraph？
                #raise Exception(f'{molgraph.natoms}!={len(molgraph.atoms)}')
                natoms_noequal.append(conf)
                natoms_noequal_e.append('molgraph.natoms!=len(molgraph.atoms)')
                error_smiles_list.append(smiles)
                error_conf_list.append(conf)
                continue
            
            molgraphs.append(molgraph)
            true_smiles_list.append(smiles)
            true_conf_list.append(conf)
            
        except Exception as e:
            error_smiles_list.append(smiles)
            error_conf_list.append(conf)
            #raise Exception('error')
            
    #print('node_noexit_num: {}'.format(len(node_noexit))) #，，
    #print('natoms_noequal_num: {}'.format(len(natoms_noequal)))
    #print('node_noexit_e_num: {}'.format(len(node_noexit_e)))
    #print('natoms_noequal_e_num: {}'.format(len(natoms_noequal_e)))
    #print('errors_num: {}'.format(len(error_smiles_list)))
    #print('true_num: {}'.format(len(true_smiles_list)))
    #print('all_num: {}'.format(len(flist)))
    
    with open(f'{savepath}/node_noexit_part.pickle','wb') as f:
        pickle.dump(node_noexit,f)
    with open(f'{savepath}/node_noexit_reason_part.txt','w') as f:
        for i in node_noexit_e:
            f.write(str(i) + '\n')

    with open(f'{savepath}/natoms_noequal_part.pickle','wb') as f:
        pickle.dump(natoms_noequal,f)
    with open(f'{savepath}/natoms_noequal_reason_part.txt','w') as f:
        for i in natoms_noequal_e:
            f.write(str(i) + '\n') 
        
    with open(f'{savepath}/errors_smiles.txt','w') as f:
        for i in error_smiles_list:
            f.write(i + '\n')
    with open(f'{savepath}/true_smiles.txt','w') as f:
        for i in true_smiles_list:
            f.write(i + '\n')
            
    with open(f'{savepath}/error_conf.pickle','wb') as f:
        pickle.dump(error_conf_list,f)
    with open(f'{savepath}/true_conf.pickle','wb') as f: #GeoDiffSDEGen
        pickle.dump(true_conf_list,f)
    
    with open(f'{savepath}/ecconf_all.pickle','wb') as f:
        pickle.dump(molgraphs,f)
        
    with open(f'{savepath}/data_readme.txt','w') as f:
        
        f.write('node_noexit_num: {}\n'.format(len(node_noexit)))
        f.write('natoms_noequal_num: {}\n'.format(len(natoms_noequal)))
        f.write('node_noexit_e_num: {}\n'.format(len(node_noexit_e)))
        f.write('natoms_noequal_e_num: {}\n'.format(len(natoms_noequal_e)))
        f.write('errors_num: {}\n'.format(len(error_smiles_list)))
        f.write('true_num: {}\n'.format(len(true_smiles_list)))
        
        f.write(f'all_conf_num:{len(flist)}, drop_conf_num:{len(error_smiles_list)}, save_conf_num:{len(true_smiles_list)}\n')
        
        f.write('true_conf.pickleGeoDiffSDEGen，ecconf_part.pickleEcConf')
        
    return 




def Deal_GEOM_Dataset_old(flist,max_conf_per_conf,save_path='molgraphs.pickle'):
    molgraphs=[]
    for fname in tqdm(flist):
        with open(f'{fname}','rb') as f:
            
            """
            #try，
            a=pickle.load(f)
            conformers=a['conformers']
            smiles=[]
            energies=[]
            for conf in conformers:
                energies.append(conf['totalenergy'])
            energies=np.array(energies)
            #lowest_ids=np.argsort(energies)[:max(max_conf_per_conf,len(conformers))]
            lowest_ids=np.argsort(energies)[:max(0,len(conformers))] #，，
            selected_mols=[conformers[i]['rd_mol'] for i in lowest_ids]
            for mol in selected_mols:
                #try:
                    mol_noH=Chem.rdmolops.RemoveHs(mol,sanitize=True)
                    mol_noH=Neutralize_atoms(mol_noH)
                    Chem.Kekulize(mol_noH)
                    atoms=[atom.GetAtomicNum() for atom in mol_noH.GetAtoms()]
                    natoms=sum([1 for i in atoms if i!=1])
                    smi=Chem.MolToSmiles(mol_noH)
                    
                    #
                    '''
                    if natoms>3 and '.' not in smi:
                        molgraph=Molgraph(mol_noH,smiles=smi)
                        molgraphs.append(molgraph)
                        if molgraph.natoms!=len(molgraph.atoms):
                            print (e,smi,fname)
                    '''
                    
                    molgraph=Molgraph(mol_noH,smiles=smi)
                    molgraphs.append(molgraph)
                    #if molgraph.natoms!=len(molgraph.atoms):
                        #print (e,smi,fname)
                #except Exception as e:
                    #pass
                """
            
            
            try:
                a=pickle.load(f)
                conformers=a['conformers']
                smiles=[]
                energies=[]
                for conf in conformers:
                    energies.append(conf['totalenergy'])
                energies=np.array(energies)
                lowest_ids=np.argsort(energies)[:min(max_conf_per_conf,len(conformers))]
                #lowest_ids=np.argsort(energies)[:max(0,len(conformers))] #，，
                selected_mols=[conformers[i]['rd_mol'] for i in lowest_ids]
                for mol in selected_mols:
                    #try:
                        mol_noH=Chem.rdmolops.RemoveHs(mol,sanitize=True)
                        mol_noH=Neutralize_atoms(mol_noH)
                        Chem.Kekulize(mol_noH)
                        atoms=[atom.GetAtomicNum() for atom in mol_noH.GetAtoms()]
                        natoms=sum([1 for i in atoms if i!=1])
                        smi=Chem.MolToSmiles(mol_noH)
                        
                        #
                        
                        if natoms>3 and '.' not in smi:
                            molgraph=Molgraph(mol_noH,smiles=smi)
                            molgraphs.append(molgraph)
                            if molgraph.natoms!=len(molgraph.atoms):
                                print (e,smi,fname)
                        
                        
                        molgraph=Molgraph(mol_noH,smiles=smi)
                        molgraphs.append(molgraph)
                        if molgraph.natoms!=len(molgraph.atoms):
                            print (e,smi,fname)
                    #except Exception as e:
                        #pass
            except:
                print (fname)
            
            
    with open(save_path,'wb') as f:
        pickle.dump(molgraphs,f)
    return 




def Multi_Process_Creat_GEOM_Molgraphs(flist,nmols_per_process,max_conf_per_mol,savepath='./molgraphs',nprocs=14):
    if not os.path.exists(savepath):
        os.system(f'mkdir -p {savepath}')
    
    '''
    with open(f'{savepath}/flist.csv','w') as f:
        for fname in flist:
            f.write(fname+'\n')
    '''
    #def Deal_GEOM_Dataset(flist,max_conf_per_conf,save_path='molgraphs.pickle'):
    #Deal_GEOM_Dataset(flist,max_conf_per_mol,f'{savepath}/part.pickle')
    Deal_GEOM_Dataset(flist,max_conf_per_mol,savepath)
    return 




def Multi_Process_Creat_GEOM_Molgraphs_old(flist,nmols_per_process,max_conf_per_mol,savepath='./molgraphs',nprocs=14):
    from multiprocessing import Pool,Queue,Manager,Process
    manager=Manager()
    DQueue=manager.Queue()
    nmols=len(flist)
    njobs=math.ceil(nmols/nmols_per_process)

    if not os.path.exists(savepath):
        os.system(f'mkdir -p {savepath}')
    
    '''
    with open(f'{savepath}/flist.csv','w') as f:
        for fname in flist:
            f.write(fname+'\n')
    '''

    p=Pool(nprocs)
    resultlist=[]
    p#print(f'njobs: {njobs}')
    for i in range(njobs):
        #result=p.apply_async(Deal_GEOM_Dataset,(flist[i*nmols_per_process:(i+1)*nmols_per_process],max_conf_per_mol,f'{savepath}/part_{i}.pickle'))
        result=p.apply_async(Deal_GEOM_Dataset,(flist,max_conf_per_mol,f'{savepath}/part_{i}.pickle'))
        resultlist.append(result)
    for i in range(len(resultlist)):
        tmp=resultlist[i].get()
        print (tmp)
    p.terminate()
    p.join()
    print (f'Mols have all trans to Molgraphs in {savepath}')
    return 




def Multi_Process_Creat_Test_GEOM_Molgraphs(flist,nmols_per_process,max_conf_per_mol,savepath='./molgraphs',nprocs=14):
    from multiprocessing import Pool,Queue,Manager,Process
    manager=Manager()
    DQueue=manager.Queue()
    nmols=len(flist)
    njobs=math.ceil(nmols/nmols_per_process)

    if not os.path.exists(savepath):
        os.system(f'mkdir -p {savepath}')
    with open(f'{savepath}/flist.csv','w') as f:
        for fname in flist:
            f.write(fname+'\n')

    p=Pool(nprocs)
    resultlist=[]
    for i in range(njobs): #，{i}.pickle
        result=p.apply_async(Deal_GEOM_Dataset,(flist[i*nmols_per_process:(i+1)*nmols_per_process],max_conf_per_mol,f'{savepath}/part_{i}.pickle'))
        resultlist.append(result)
    for i in range(len(resultlist)): #？
        tmp=resultlist[i].get()
        print (tmp)
    p.terminate()
    p.join()
    print (f'Mols have all trans to Molgraphs in {savepath}')
    return 


def Statistic_GPARAMS(MGFiles):
    params={'atom_types':[],'max_atoms':0}
    Hmols=0
    for fname in MGFiles:
        with open(fname,'rb') as f:
            mgs=pickle.load(f)
        for mg in tqdm(mgs):
            if mg.natoms > params["max_atoms"]:
                params["max_atoms"]=mg.natoms
            if 1 in mg.atoms:
                Hmols+=1
            for a in mg.atoms:
                if a not in params["atom_types"]:
                    params["atom_types"].append(a)
    params["atom_types"]=np.sort(params["atom_types"])
    print (Hmols)
    return params


class MG_Dataset(Dataset):
    def __init__(self,MGlist,name):
        super(Dataset,self).__init__()
        self.mglist=MGlist
        self.name=name
        self.nmols=len(self.mglist)
        self.max_atoms=GP.max_atoms
        return 
    def __len__(self):
        return len(self.mglist)
    def __getitem__(self,idx):
        return self.getitem__(idx)
    def getitem__(self,idx):
        mg=copy.deepcopy(self.mglist[idx])
        ##print('mg:', mg)
        ##print('type(mg):', type(mg))
        #raise Exception('stop')
        
        #mg.RemoveHs()
        atoms,chiraltags,adjs,coords,zmats,masks=mg.Get_3D_Graph_Tensor(max_atoms=self.max_atoms)
        ##print('zmats1.shape:', zmats.shape) #torch.Size([9, 4])
        ##print('===============================')
        ##print('zmats.shape1:', zmats.shape)
        #atoms,chiraltags,adjs,coords,zmats,masks=mg.Get_3D_Graph_Tensor(max_atoms=None)
        ##print('zmats.shape2:', zmats.shape)
        #raise Exception('stop')
        ##print('*****************************************')
        ##print('adjs.shape:', adjs.shape)
        if not GP.if_chiral:
            feats=atoms
        else:
            feats=torch.concat((atoms,chiraltags),axis=-1)
        return {
                'Feats':feats,
                "Adjs":adjs,
                "Coords":coords,
                "Zmats":zmats,
                "Masks":masks
                }



        