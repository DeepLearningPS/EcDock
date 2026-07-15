import numpy as np 
from rdkit import Chem
from tqdm import tqdm 
from torch.utils.data import Dataset,DataLoader
import copy,time,math,random,os,pickle 

from ..comparm import *
from .molecule import * 
from .complex import * 
from ..utils.utils_plt import *
from ..utils import *

def Deal_GEOM_Dataset(flist,max_conf_per_mol,save_path='molgraphs.pickle',max_atoms=60,job_id=0):
    with open('error.list','a') as errf:
        for fid,fname in tqdm(enumerate(flist)):
            with open(f'{fname}','rb') as f:
                try:
                    a=pickle.load(f)
                    conformers=a['conformers']
                    
                    boltzmannweights=[]
                    for conf in conformers:
                        boltzmannweights.append(conf['boltzmannweight'])
                    boltzmannweights=np.array(boltzmannweights)
                    descent_conf_ids=np.argsort(-boltzmannweights)[:min(max_conf_per_mol,len(conformers))]
                    #print (descent_conf_ids)
                    #lowest_ids=np.argsort(energies)[:min(max_conf_per_conf,len(conformers))]
                    selected_mols=[conformers[i]['rd_mol'] for i in descent_conf_ids]
                    molsupp=Chem.SDWriter(fname.strip('\.pickle')+'.sdf')
                    os.system(f'mkdir -p {save_path}/{job_id}/{fid}')
                    for mid,mol in enumerate(selected_mols):
                        molsupp.write(mol)
                        mol_noH=Chem.rdmolops.RemoveHs(mol,sanitize=True)
                        mol_noH=Neutralize_atoms(mol)
                        Chem.Kekulize(mol_noH)
                        atoms=[atom.GetAtomicNum() for atom in mol_noH.GetAtoms()]
                        natoms=sum([1 for i in atoms if i!=1])
                        smi=Chem.MolToSmiles(mol_noH)
                        if natoms>3 and natoms < max_atoms and '.' not in smi:
                            fstr=fname.strip('\.pickle').split('/')[-1]
                            c=Molgraph(mol_noH,savepath=f'{save_path}/{job_id}/{fid}')
                            c.standardrize(ligand_rearrange_mode='fix',debug=True)
                            if max(sum(np.where(c.l_adjs,1,0)))<=4:
                                c.save(f'{save_path}/{job_id}/{fid}/{mid}.pkl')
                            else:
                                print (fname)
                                errf.write(f'{fname}\n')
                    molsupp.close()
                except Exception as e:
                    print (fname,e)
                    errf.write(f'{fname}\n')
    return 

def Multi_Deal_GEOM_Datasets(flist,nmols_per_process,max_conf_per_mol,savepath='./molgraphs',max_atoms=50,nprocs=14):
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
    for i in range(njobs):
        result=p.apply_async(Deal_GEOM_Dataset,(flist[i*nmols_per_process:(i+1)*nmols_per_process],max_conf_per_mol,savepath,max_atoms,i))
        resultlist.append(result)
    for i in range(len(resultlist)):
        tmp=resultlist[i].get()
        print (tmp)
    p.terminate()
    p.join()
    print (f'Mols have all trans to Molgraphs in {savepath}')
    return 

def Deal_GuacaMol_Dataset(SDFname,GuacaMol_SDFPATH='./datasets'):
    molgraphs=[]
    with open('GuacaMol_ERRORs.log','a') as ErrF:
        guacamols=Chem.SDMolSupplier(f'{GuacaMol_SDFPATH}/{SDFname}',removeHs=True)
        for mid,mol in tqdm(enumerate(guacamols)):
            try:
                if mol is not None:
                    natoms=mol.GetNumAtoms()
                    if natoms <=60 and natoms>=3:
                        c=Molgraph(LigMol=mol,savepath=f'{GuacaMol_SDFPATH}/{SDFname[:-4]}/{mid}')
                        c.standardrize(ligand_rearrange_mode='fix',debug=True)
                        c.save(f'{GuacaMol_SDFPATH}/{SDFname[:-4]}/{mid}/{mid}.pkl')
            except Exception as e:
                ErrF.write(f'{SDFname},{mid}\n')
                ErrF.flush()
                print (f'Create Molecule OBJ in {SDFname},{mid} failded due to {e}')
    return 

def Multi_Deal_GuacaMol_Datasets(SDFnames,GuacaMol_SDFPATH='./datasets',nprocs=20):
    from multiprocessing import Pool,Queue,Manager,Process
    manager=Manager()
    DQueue=manager.Queue()
    njobs=len(SDFnames)

    p=Pool(nprocs)
    resultlist=[]
    for i in range(njobs):
        result=p.apply_async(Deal_GuacaMol_Dataset,(SDFnames[i],GuacaMol_SDFPATH))
        resultlist.append(result)
    
    for i in range(len(resultlist)):
        tmp=resultlist[i].get()
        print (tmp)
    
    p.terminate()
    p.join()
    print (f'Done')
    return 

def Deal_PDBBind_Dataset(pdbids,PDBBind_Path='./datasets',mode='int-only',max_patoms=150,max_latoms=60):
    molgraphs=[]
    with open('PDBBind_ERRORs.log','a') as ErrF:
        for pdbid in tqdm(pdbids):
            try:
                start_time=time.time()
                Protein_PDBFile=f'{PDBBind_Path}/{pdbid}/{pdbid}_protein_processed.pdb'
                Ligand_SDFFile =f'{PDBBind_Path}/{pdbid}/{pdbid}_ligand.sdf'
                c=Complex(Protein_PDBFile,Ligand_SDFFile,PDB_SOURCE='PDBBind-2017')
                c.standardrize(max_patoms=max_patoms,keep_pocket_mode=mode,ligand_rearrange_mode='fix',debug=True)
                c.save(f'{PDBBind_Path}/{pdbid}/{pdbid}_complex.pkl')
            except Exception as e:
                ErrF.write(f'{pdbid}\n')
                ErrF.flush()
                print (f'Create Complex OBJ for {pdbid} failded due to {e}')
    return 

def Multi_Deal_PDBBind_Datasets(PDBIDs,nmols_per_process,sourcepath='',nprocs=14,mode='int-only',max_patoms=150):
    from multiprocessing import Pool,Queue,Manager,Process
    manager=Manager()
    DQueue=manager.Queue()
    nmols=len(PDBIDs)
    njobs=math.ceil(nmols/nmols_per_process)

    p=Pool(nprocs)
    resultlist=[]
    for i in range(njobs):
        result=p.apply_async(Deal_PDBBind_Dataset,(PDBIDs[i*nmols_per_process:(i+1)*nmols_per_process],sourcepath,mode,max_patoms))
        resultlist.append(result)
    
    for i in range(len(resultlist)):
        tmp=resultlist[i].get()
        print (tmp)
    
    p.terminate()
    p.join()
    
    return 



def MGFielter(MG,params={'atom_types':[1,6,7,8,9,15,16,17,35,53],'max_atoms':210,'max_latoms':60,'max_patoms':150,"formal_charge_types":[-2,-1,0,1,2],"max_single_ring_size":8,"max_group_size":30},group_types=None):
    Flag=True
    if isinstance(MG,Molgraph):
        if MG.l_natoms>params["max_atoms"]:
            print(MG.savepath,"'s has more atoms")
            return False
        
    elif isinstance(MG,Complex):
        if MG.l_natoms>params["max_latoms"]:
            return False
        
        if MG.p_natoms>params["max_patoms"]:
            return False
        
        if MG.l_natoms+MG.p_natoms> params["max_atoms"]:
            return False
            
    for atom in MG.l_atoms:
        if atom not in params['atom_types']:
            return False
    
    for fc in MG.l_fcs:
        if fc not in params['formal_charge_types']:
            return False
    
    for i in range(MG.l_ngroups):
        descriptor=MG.get_l_group_descriptor(gid=i)
        if group_types is not None:
            if descriptor not in group_types:
                print (f'Unknown group type of {i} with descriptor {descriptor}')
                return False
                
        if MG.l_groups_max_ring_size[i]>params["max_single_ring_size"]:
            return False
        
        if len(MG.l_groups[i])>params["max_group_size"]:
            return False
    return True

def replicate_list(Alist,rep_num=1):
    Copylist=[]
    for a in Alist:
        for i in range(rep_num):
            Copylist.append(a)
    return Copylist

class Lig_Dataset(Dataset):
    def __init__(self,MGlist,mode,name):
        self.mglist=MGlist
        self.mode=mode
        self.nadd_states=[]
        self.rgen_states=[]
        self.nconn_states=[]
        self.coord_states=[]
        self.name=name
        self.nmols=len(self.mglist)
        for gid,G in enumerate(self.mglist):
            nadd_nstates=len(G.nadd_lg_states[0])
            for i in range(nadd_nstates):
                self.nadd_states.append((gid,i))
                
            rgen_nstates=len(G.rgen_lg_states[0])
            for i in range(rgen_nstates):
                self.rgen_states.append((gid,i))
                
            nconn_nstates=len(G.nconn_lg_states[0])
            for i in range(nconn_nstates):
                self.nconn_states.append((gid,i))
                
            coord_nstates=len(G.crd_gen_states[0])
            self.coord_states.append((gid,coord_nstates-1))
            #sampled_states=random.sample(range(coord_nstates-1),min(2,coord_nstates-1))
            for i in range(coord_nstates):
                self.coord_states.append((gid,i))
        return 
    def __len__(self):
        assert self.mode in ['nadd','rgen','nconn','coords'], 'Mode of Datasets not supported, please select from nadd,rgen,nconn,nint,coord'
        if self.mode=='nadd':
            return len(self.nadd_states)
        if self.mode=='rgen':
            return len(self.rgen_states)
        if self.mode=='nconn':  
            return len(self.nconn_states)
        if self.mode=='coords':
            return len(self.coord_states)
    def __getitem__(self,idx):
        return self.getitem__(idx)
    
    def getitem__(self,idx):
        if self.mode=='nadd':
            gid,stepid=self.nadd_states[idx]
            MG=copy.deepcopy(self.mglist[gid])
            vars=MG.sample_leaf_nadd_states(stepid,max_latoms=FGP.max_latoms,
                                                   noise_std=FGP.noise_std_for_graph)
            l_atoms,l_atoms_onehot,l_fcs,l_fcs_onehot,l_adjs,l_adjs_mat,l_coords,l_masks,l_apds_tensor=vars
            l_feats=torch.cat((l_atoms_onehot,l_fcs_onehot),axis=-1)

            return {'L_Feats':l_feats,'L_Atoms':l_atoms,'L_Adjs_Mat':l_adjs_mat,'L_Coords':l_coords,'L_Masks':l_masks,'L_Apds':l_apds_tensor}
            
        elif self.mode=='rgen':
            gid,stepid=self.rgen_states[idx]
            MG=copy.deepcopy(self.mglist[gid])
            vars=MG.sample_leaf_rgen_states(stepid,max_latoms=FGP.max_latoms,
                                                   max_ringsize=FGP.max_lgroup_size,
                                                   noise_std=FGP.noise_std_for_graph)
            
            l_atoms,l_atoms_onehot,l_fcs,l_fcs_onehot,l_adjs,l_adjs_mat,l_coords,l_masks,\
                r_atoms,r_atoms_onehot,r_fcs,r_fcs_onehot,r_adjs,r_adjs_mat,r_coords,r_masks,\
                r_ftypes,r_apds=vars
            l_feats=torch.cat((l_atoms_onehot,l_fcs_onehot),axis=-1)
            r_feats=torch.cat((r_atoms_onehot,r_fcs_onehot),axis=-1)
            return {'L_Feats':l_feats,'L_Atoms':l_atoms,'L_Adjs_Mat':l_adjs_mat,'L_Coords':l_coords,'L_Masks':l_masks,
                    'R_Feats':r_feats,'R_Atoms':r_atoms,'R_Adjs_Mat':r_adjs_mat,'R_Masks':r_masks,'R_Ftypes':r_ftypes,'R_Apds':r_apds}
                
        elif self.mode=='nconn':
            gid,stepid=self.nconn_states[idx]
            MG=copy.deepcopy(self.mglist[gid])
            vars=MG.sample_leaf_nconn_states(stepid,max_latoms=FGP.max_latoms,
                                                    max_ringsize=FGP.max_lgroup_size,
                                                    noise_std=FGP.noise_std_for_graph)
            
            l_atoms,l_atoms_onehot,l_fcs,l_fcs_onehot,l_adjs,l_adjs_mat,l_coords,l_masks,\
                r_atoms,r_atoms_onehot,r_fcs,r_fcs_onehot,r_adjs,r_adjs_mat,r_coords,r_masks,\
                focus_atom_idx,l_apds=vars
                
            l_feats=torch.cat((l_atoms_onehot,l_fcs_onehot),axis=-1)
            r_feats=torch.cat((r_atoms_onehot,r_fcs_onehot),axis=-1)
            return {'L_Feats':l_feats,'L_Atoms':l_atoms,'L_Adjs_Mat':l_adjs_mat,'L_Coords':l_coords,'L_Masks':l_masks,
                    'R_Feats':r_feats,'R_Atoms':r_atoms,'R_Adjs_Mat':r_adjs_mat,'R_Masks':r_masks,
                    'Focus_Atom':focus_atom_idx,'L_Apds':l_apds}

        elif self.mode=='coords':    
            gid,stepid=self.coord_states[idx]
            MG=copy.deepcopy(self.mglist[gid])
            vars=MG.sample_l_coord_gen_states(stepid,max_latoms=FGP.max_latoms,
                                                     max_lbonds=FGP.max_lbonds,
                                                     max_langles=FGP.max_langles,
                                                     max_ldihedrals=FGP.max_ldihedrals,
                                                     l_mode=FGP.l_mode)
            
            l_atoms,l_atoms_onehot,l_fcs,l_fcs_onehot,l_adjs,l_adjs_mat,l_coords,l_masks,l_fix_masks,l_flexible_masks,\
                l_zbs,l_zas,l_zds,l_zbmask,l_zamask,l_zdmask,\
                        l_pocket_labels,l_ligand_labels=vars
               
            l_feats=torch.cat((l_atoms_onehot,l_fcs_onehot),axis=-1)
            
            return {'L_Feats':l_feats,'L_Atoms':l_atoms,'L_Adjs_Mat':l_adjs_mat,'L_Coords':l_coords,'L_Masks':l_masks,'L_Fix_Masks':l_fix_masks,'L_Flexible_Masks':l_flexible_masks,
                    'L_Zbs':l_zbs,'L_Zas':l_zas,'L_Zds':l_zds,'L_Zbmask':l_zbmask,'L_Zamask':l_zamask,'L_Zdmask':l_zdmask,
                    'L_Pocket_Labels':l_pocket_labels,'L_Ligand_Labels':l_ligand_labels}
    

class MG_Dataset(Dataset):
    
    def __init__(self,MGlist,mode,name):
        super(Dataset,self).__init__()
        self.mglist=MGlist
        self.mode=mode
        self.nadd_states=[]
        self.rgen_states=[]
        self.nconn_states=[]
        self.nint_states=[]
        self.coord_states=[]
        self.name=name
        self.nmols=len(self.mglist)
        for gid,G in enumerate(self.mglist):
            nadd_nstates=len(G.nadd_lg_states[0])
            for i in range(nadd_nstates):
                self.nadd_states.append((gid,i))
                
            rgen_nstates=len(G.rgen_lg_states[0])
            for i in range(rgen_nstates):
                self.rgen_states.append((gid,i))
                
            nconn_nstates=len(G.nconn_lg_states[0])
            for i in range(nconn_nstates):
                self.nconn_states.append((gid,i))
                
            if isinstance(G,Complex):
                nint_nstates=len(G.nint_lg_states[0])
                for i in range(nint_nstates):
                    self.nint_states.append((gid,i))
                                            
            coord_nstates=len(G.crd_gen_states[0])
            self.coord_states.append((gid,coord_nstates-1))
            sampled_states=random.sample(range(coord_nstates-1),min(10,coord_nstates-1))
            for i in sampled_states:
                self.coord_states.append((gid,i))
            #for i in range(coord_nstates):
            #    self.coord_states.append((gid,i))
        return
    
    def check(self) :
        for i in range(len(self)):
            vars=self.getitem__(i)
            adjs_mat=vars['C_Adjs_Mat']
            #print (adjs_mat.shape)
            for a in range(adjs_mat.shape[0]):
                if adjs_mat[a,a].sum()>0:
                    print (i,self.mglist[self.coord_states[i][0]].protein_pdb,self.coord_states[i],a)
        return
    
    def repulicate(self,rep_num=1):
        self.nadd_states=replicate_list(self.nadd_states,rep_num)
        self.rgen_states=replicate_list(self.rgen_states,rep_num)
        self.nconn_states=replicate_list(self.nconn_states,rep_num)
        self.nint_states=replicate_list(self.nint_states,rep_num)
        self.coord_states=replicate_list(self.coord_states,rep_num)
            
    def __len__(self):
        assert self.mode in ['nadd','rgen','nconn','nint','coords','pocket_only'], 'Mode of Datasets not supported, please select from nadd,rgen,nconn,nint,coord'
        if self.mode=='nadd':
            return len(self.nadd_states)
        if self.mode=='rgen':
            return len(self.rgen_states)
        if self.mode=='nconn':  
            return len(self.nconn_states)
        if self.mode=='nint':
            return len(self.nint_states)
        if self.mode=='coords':
            return len(self.coord_states)
        if self.mode=='pocket_only':
            return self.nmols

    def __getitem__(self,idx):
        return self.getitem__(idx)
    
    def getitem__(self,idx):
        if self.mode=='nadd':
            gid,stepid=self.nadd_states[idx]
            MG=copy.deepcopy(self.mglist[gid])
            vars=MG.sample_leaf_nadd_states(stepid,max_patoms=FGP.max_patoms,
                                                   max_latoms=FGP.max_latoms,
                                                   noise_std=FGP.noise_std_for_graph)
            c_atoms,c_atoms_onehot,c_fcs,c_fcs_onehot,c_adjs,c_adjs_mat,c_coords,c_masks,l_apds_tensor=vars
            c_feats=torch.cat((c_atoms_onehot,c_fcs_onehot),axis=-1)
            gid_=torch.Tensor([gid]).long()
            stepid_=torch.Tensor([stepid]).long()
            #print (c_adjs_mat.shape,c_adjs_mat.type(),c_adjs_mat.device)
            #l_feats=torch.cat((l_atoms_onehot,l_fcs_onehot),axis=-1)
            #print (stepid,c_masks[FGP.max_patoms:])
            return {'C_Feats':c_feats,'C_Atoms':c_atoms,'C_Adjs_Mat':c_adjs_mat,'C_Coords':c_coords,'C_Masks':c_masks,'L_Apds':l_apds_tensor}
            
        elif self.mode=='rgen':
            gid,stepid=self.rgen_states[idx]
            MG=copy.deepcopy(self.mglist[gid])
            vars=MG.sample_leaf_rgen_states(stepid,max_patoms=FGP.max_patoms,
                                                   max_latoms=FGP.max_latoms,
                                                   max_ringsize=FGP.max_lgroup_size,
                                                   noise_std=FGP.noise_std_for_graph)
            c_atoms,c_atoms_onehot,c_fcs,c_fcs_onehot,c_adjs,c_adjs_mat,c_coords,c_masks,\
                r_atoms,r_atoms_onehot,r_fcs,r_fcs_onehot,r_adjs,r_adjs_mat,r_coords,r_masks,\
                r_ftypes,r_apds=vars
            c_feats=torch.cat((c_atoms_onehot,c_fcs_onehot),axis=-1)
            r_feats=torch.cat((r_atoms_onehot,r_fcs_onehot),axis=-1)
            return {'C_Feats':c_feats,'C_Atoms':c_atoms,'C_Adjs_Mat':c_adjs_mat,'C_Coords':c_coords,'C_Masks':c_masks,
                    'R_Feats':r_feats,'R_Atoms':r_atoms,'R_Adjs_Mat':r_adjs_mat,'R_Masks':r_masks,'R_Ftypes':r_ftypes,'R_Apds':r_apds}
                
        elif self.mode=='nconn':
            gid,stepid=self.nconn_states[idx]
            MG=copy.deepcopy(self.mglist[gid])
            vars=MG.sample_leaf_nconn_states(stepid,max_patoms=FGP.max_patoms,
                                                    max_latoms=FGP.max_latoms,
                                                    max_ringsize=FGP.max_lgroup_size,
                                                    noise_std=FGP.noise_std_for_graph)
            
            c_atoms,c_atoms_onehot,c_fcs,c_fcs_onehot,c_adjs,c_adjs_mat,c_coords,c_masks,\
                r_atoms,r_atoms_onehot,r_fcs,r_fcs_onehot,r_adjs,r_adjs_mat,r_coords,r_masks,\
                focus_atom_idx,l_apds=vars
                
            c_feats=torch.cat((c_atoms_onehot,c_fcs_onehot),axis=-1)
            r_feats=torch.cat((r_atoms_onehot,r_fcs_onehot),axis=-1)
            return {'C_Feats':c_feats,'C_Atoms':c_atoms,'C_Adjs_Mat':c_adjs_mat,'C_Coords':c_coords,'C_Masks':c_masks,
                    'R_Feats':r_feats,'R_Atoms':r_atoms,'R_Adjs_Mat':r_adjs_mat,'R_Masks':r_masks,
                    'Focus_Atom':focus_atom_idx,'L_Apds':l_apds}
            
        elif self.mode=='nint':
            
            gid,stepid=self.nint_states[idx]
            MG=copy.deepcopy(self.mglist[gid])
            #print (MG.p_groups,MG.savepath)
            vars=MG.sample_leaf_nint_states(stepid,max_patoms=FGP.max_patoms,\
                                                   max_latoms=FGP.max_latoms,
                                                   max_pgroups=FGP.max_pgroups,
                                                   max_lgroups=FGP.max_lgroups,
                                                   max_ringsize=FGP.max_lgroup_size,
                                                   max_pgsize=FGP.max_pgroup_size,
                                                   noise_std=FGP.noise_std_for_graph,)
            cg_atoms,cg_atoms_onehot,cg_fcs,cg_fcs_onehot,cg_adjs,cg_adjs_mat,cg_coords,cg_masks,\
                cd_atoms,cd_atoms_onehot,cd_fcs,cd_fcs_onehot,cd_adjs,cd_adjs_mat,cd_coords,cd_masks,\
                focus_lgroups,focus_lgroups_masks,p_groups,p_groups_masks,p_int_groups_masks,focus_ftypes,l_apds=vars
                
            cg_feats=torch.cat((cg_atoms_onehot,cg_fcs_onehot),axis=-1)
            cd_feats=torch.cat((cd_atoms_onehot,cd_fcs_onehot),axis=-1)
            return {'CG_Feats':cg_feats,'CG_Atoms':cg_atoms,'CG_Adjs_Mat':cg_adjs_mat,'CG_Coords':cg_coords,'CG_Masks':cg_masks,
                    'CD_Feats':cd_feats,'CD_Atoms':cd_atoms,'CD_Adjs_Mat':cd_adjs_mat,'CD_Coords':cd_coords,'CD_Masks':cd_masks,
                    'Focus_Lgroups':focus_lgroups,'Focus_Lgroups_Masks':focus_lgroups_masks,'P_Groups':p_groups,'P_Groups_Masks':p_groups_masks,'P_INT_Groups_Masks':p_int_groups_masks,'Ftypes':focus_ftypes,'L_Apds':l_apds}
            
        elif self.mode=='coords':    
            gid,stepid=self.coord_states[idx]
            MG=copy.deepcopy(self.mglist[gid])
            vars=MG.sample_l_coord_gen_states(stepid,max_patoms=FGP.max_patoms,
                                                     max_latoms=FGP.max_latoms,
                                                     max_pbonds=FGP.max_pbonds,
                                                     max_lbonds=FGP.max_lbonds,
                                                     max_pangles=FGP.max_pangles,
                                                     max_langles=FGP.max_langles,
                                                     max_pdihedrals=FGP.max_pdihedrals,
                                                     max_ldihedrals=FGP.max_ldihedrals,
                                                     pl_mask_rate=FGP.pl_mask_rate,
                                                     p_mode=FGP.p_mode,
                                                     l_mode=FGP.l_mode,
                                                     int_mode=FGP.int_mode)
            
            c_atoms,c_atoms_onehot,c_fcs,c_fcs_onehot,c_adjs,c_adjs_mat,c_coords,c_masks,c_fix_masks,c_flexible_masks,\
                l_zbs,l_zas,l_zds,l_zbmask,l_zamask,l_zdmask,\
                    p_zbs,p_zas,p_zds,p_zbmask,p_zamask,p_zdmask,\
                        c_pocket_labels,c_ligand_labels,step_id=vars
               
            c_feats=torch.cat((c_atoms_onehot,c_fcs_onehot),axis=-1)
            
            return {'C_Feats':c_feats,'C_Atoms':c_atoms,'C_Adjs_Mat':c_adjs_mat,'C_Coords':c_coords,'C_Masks':c_masks,'C_Fix_Masks':c_fix_masks,'C_Flexible_Masks':c_flexible_masks,
                    'L_Zbs':l_zbs,'L_Zas':l_zas,'L_Zds':l_zds,'L_Zbmask':l_zbmask,'L_Zamask':l_zamask,'L_Zdmask':l_zdmask,
                    'P_Zbs':p_zbs,'P_Zas':p_zas,'P_Zds':p_zds,'P_Zbmask':p_zbmask,'P_Zamask':p_zamask,'P_Zdmask':p_zdmask,
                    'C_Pocket_Labels':c_pocket_labels,'C_Ligand_Labels':c_ligand_labels,"Step_id":step_id}
            
        elif self.mode=='pocket_only':
            MG=copy.deepcopy(self.mglist[idx])
            vars=MG.pocket_tensors_only(max_patoms=FGP.max_patoms,max_latoms=FGP.max_latoms,max_pgroups=FGP.max_pgroups,max_pgroup_size=FGP.max_pgroup_size,p_mode=FGP.p_mode)
            
            c_atoms,c_atoms_onehot,c_fcs,c_fcs_onehot,c_adjs,c_adjs_mat,c_coords,c_masks,c_gmasks,c_dmasks,c_fix_masks,c_flexible_masks,p_groups,p_groups_masks,p_group_int_masks,p_group_atom_int_masks,p_atom_int_masks,c_pocket_labels,n_pgroups=vars
            #print (c_adjs_mat.shape,c_adjs_mat.type(),c_adjs_mat.device) 
            c_feats=torch.cat((c_atoms_onehot,c_fcs_onehot),axis=-1)
            
            return {'C_Feats':c_feats,'C_Atoms':c_atoms,'C_Adjs_Mat':c_adjs_mat,'C_Coords':c_coords,
                    'C_Masks':c_masks,'C_GMasks':c_gmasks,'C_DMasks':c_dmasks,
                    'C_Fix_Masks':c_fix_masks,'C_Flexible_Masks':c_flexible_masks,
                    'P_Groups':p_groups,'P_Groups_Masks':p_groups_masks,'C_Pocket_Labels':c_pocket_labels,
                    'P_INT_Groups_Masks':p_group_int_masks,'P_INT_Groups_Atom_Masks':p_group_atom_int_masks,'P_INT_Atom_Masks':p_atom_int_masks,
                    'N_PGroups':n_pgroups}
        

        
