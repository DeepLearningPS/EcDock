import pickle 
from rdkit import Chem
#from rdkit.Chem import AllChem,ChemicalFeatures,Draw
from rdkit.Chem import AllChem,ChemicalFeatures
from rdkit import RDConfig
import numpy as np
import scipy,os,random,copy
from tqdm import tqdm 
import networkx as nx
import torch 
#import cairosvg
from ..comparm import * 
from ..utils.utils_os import * 
from ..utils.utils_interactions import detect_interactions
from ..utils.utils_graphroute import bfs_seq,Merge_single_rings_to_nodes
from ..utils.utils_rdkit import Change_mol_xyz,Drawmols,Neutralize_atoms,MolToXYZ
from ..utils.utils_np import Adjs_to_IC_targets
from .protein import * 
from ..utils.utils_rdkit import mol_with_atom_index_and_formal_charge,standardize

def isRingAromatic(mol, bondRing):
        for id in bondRing:
            if not mol.GetBondWithIdx(id).GetIsAromatic():
                return False
        return True

def Adjs_to_Onek(adjs,nchannels=3):
    #nchannels=np.max(adjs)-1
    adjs_onek=np.zeros((adjs.shape[0],adjs.shape[0],nchannels))
    idx1,idx2=np.where(adjs)
    channel_idx=adjs[idx1,idx2].astype(int)-1
    for id1,id2,cid in zip(idx1,idx2,channel_idx):
        adjs_onek[id1,id2,cid]=1
    return adjs_onek.astype(int)

def Atoms_to_Idx(atoms,possible_atom_types=[1,5,6,7,8,9,15,16,17,35,53]):
    atom_idx_=[possible_atom_types.index(int(a)) for a in atoms] 
    return atom_idx_

def Atoms_to_onehot(atoms,possible_atom_types=[1,5,6,7,8,9,15,16,17,35,53]):
    atom_one_hot=[]
    for a in atoms:
        onehot=np.zeros(len(possible_atom_types))
        atom_idx=possible_atom_types.index(int(a))
        onehot[atom_idx]=1
        atom_one_hot.append(onehot)
    return np.array(atom_one_hot)

def isRingAromatic(mol, bondRing):
        for id in bondRing:
            if not mol.GetBondWithIdx(id).GetIsAromatic():
                return False
        return True

def group_descriptor_to_puzzled_group_feats(descriptor):
    desvars=descriptor.split('-')
    Rnum=int(desvars[0][1:])
    Besnum=int(desvars[10][3:])
    if not (Rnum==0 and Besnum>1):
        Cnum=int(desvars[1][1:])
        Nnum=int(desvars[2][1:])
        Onum=int(desvars[3][1:])
        Fnum=int(desvars[4][1:])
        Pnum=int(desvars[5][1:])
        Snum=int(desvars[6][1:])
        Clnum=int(desvars[7][2:])
        Brnum=int(desvars[8][2:])
        Inum=int(desvars[9][1:])
    else:
        Cnum,Nnum,Onum,Fnum,Pnum,Snum,Clnum,Brnum,Inum=-1,-1,-1,-1,-1,-1,-1,-1,-1
    
    if Rnum==0:
        ARnum=-1
    else:
        ARnum=int(desvars[11][2:])
        
    if Rnum==0 and Besnum==1:
        HDnum,HAnum,Negnum,Posnum,Aronum,Hydnum,LHydnum=-1,-1,-1,-1,-1,-1,-1
    else:
        HDnum=int(desvars[12][2:])
        HAnum=int(desvars[13][2:])
        Negnum=int(desvars[14][3:])
        Posnum=int(desvars[15][3:])
        Aronum=int(desvars[16][3:])
        Hydnum=int(desvars[17][3:])
        LHydnum=int(desvars[18][4:])
    
    feat=np.array([Rnum,Cnum,Nnum,Onum,Fnum,Pnum,Snum,Clnum,Brnum,Inum,Besnum,ARnum,HDnum,HAnum,Negnum,Posnum,Aronum,Hydnum,LHydnum])
    return feat

def group_fid_to_puzzled_group_feats(fid):
    descriptor=FGP.group_index_type_dict[fid]
    return group_descriptor_to_puzzled_group_feats(descriptor)


 
class Molgraph:    
    def __init__(self,LigMol,Ligand_SDFFile=None,savepath='./datasets'):
        if LigMol is None:
            assert Ligand_SDFFile is not None, "Ligand_SDFFile should not be None if you didn't provide rdkit mol object!"
            self.ligand_sdf=Ligand_SDFFile
            #print (self.savepath)
            self.ligmol=Chem.rdmolfiles.SDMolSupplier(self.ligand_sdf,removeHs=False)[0]
        else:
            self.ligmol=LigMol    
        #assert not check_warning_structures(self.ligmol), f"Warning structures in {self.ligand_sdf}!"
        self.ligmol=standardize(self.ligmol)
        Chem.Kekulize(self.ligmol)
        
        self.savepath=savepath
        if not os.path.exists(savepath):
            os.system(f'mkdir -p {self.savepath}')
        #print (self.ligmol)
        return 
        
    def standardrize(self,ligand_rearrange_mode='fix',debug=False):
        self.prepare_ligand()
        self.remove_l_Hs()
        self.devide_l_into_groups()
        self.gen_ligand_tree_graph()
        self.gen_ligand_leaf_graph()
        self.rearrange_ligand(ligand_rearrange_mode,debug=debug)
        return
        
    def prepare_ligand(self):
        self.l_atoms=np.array([atom.GetAtomicNum() for atom in self.ligmol.GetAtoms()])
        for atom in self.l_atoms:
            assert atom in [1]+FGP.atom_types, f"Atom type {atom} not in the standard atom types for {self.savepath}!"
                
        self.l_natoms=len(self.l_atoms)
        self.l_fcs=np.array([atom.GetFormalCharge() for atom in self.ligmol.GetAtoms()])

        for fc in self.l_fcs:
            assert fc==0, f"Formal charge {fc} not in the standard formal charge types for {self.savepath}!"
        
        self.l_coords=np.array(self.ligmol.GetConformer(0).GetPositions())
        self.l_adjs=np.zeros((self.l_natoms,self.l_natoms))
                #print (self.l_pharmacophores) 
        for bond in self.ligmol.GetBonds():
            a1=bond.GetBeginAtom().GetIdx()
            a2=bond.GetEndAtom().GetIdx()
            bt=bond.GetBondType() 
            ch=FGP.bond_types.index(bt)
            self.l_adjs[a1,a2]=ch+1
            self.l_adjs[a2,a1]=ch+1
        fdefName = os.path.join(RDConfig.RDDataDir,'BaseFeatures.fdef')
        factory = ChemicalFeatures.BuildFeatureFactory(fdefName) 
        feats = factory.GetFeaturesForMol(self.ligmol)
        
        self.l_pharmacophores={'Donor':[],
                               'Acceptor':[],
                               'NegIonizable':[],
                               'PosIonizable':[],
                               'Aromatic':[],
                               'Hydrophobe':[],
                               'LumpedHydrophobe':[]} 
        
        self.pharm_groups=[]
        for feat in feats:
            family=feat.GetFamily()
            atomids=list(feat.GetAtomIds())
            
            if family in self.l_pharmacophores.keys():
                self.l_pharmacophores[family].append(atomids)
                
                flag=True
                for group in self.pharm_groups:
                    if set(atomids).issubset(group):
                        flag=False
                        break
                if flag:
                    self.pharm_groups.append(list(atomids))
        
        ringinfo=self.ligmol.GetRingInfo()
        self.rings=[list(x) for x in ringinfo.AtomRings()]
        self.aromatic_ring_labels=[isRingAromatic(self.ligmol,ring) for ring in ringinfo.BondRings()]
        return 

    
    def remove_l_Hs(self):
        l_noH_idx=np.array([i for i in range(len(self.l_atoms)) if self.l_atoms[i]!=1])
        traverse_dict={}
        for aid,i in enumerate(l_noH_idx):
            traverse_dict[i]=aid
            
        self.l_atoms=self.l_atoms[np.ix_(l_noH_idx)]
        self.l_fcs=self.l_fcs[np.ix_(l_noH_idx)]
        self.l_adjs=self.l_adjs[np.ix_(l_noH_idx,l_noH_idx)]
        self.l_coords=self.l_coords[np.ix_(l_noH_idx)]

        molecule=Chem.RWMol()
        for atom in self.l_atoms:
            new_atom=Chem.Atom(int(atom))
            molecule_idx=molecule.AddAtom(new_atom)
        
        row,col=np.diag_indices_from(self.l_adjs)
        self.l_adjs[row,col]=0
        idx1,idx2=np.where(self.l_adjs!=0)
        
        for id1,id2 in zip(idx1,idx2):
            if id1<id2:
                #print (id1,id2,FGP.bond_types[int(adjs[id1,id2])-1])
                molecule.AddBond(int(id1),int(id2),FGP.bond_types[int(self.l_adjs[id1,id2])-1])

        for aid,at in enumerate(molecule.GetAtoms()):
            at.SetFormalCharge(int(self.l_fcs[aid]))
        mol=molecule.GetMol()
        #Chem.SanitizeMol(mol)
        AllChem.Compute2DCoords(mol)
        self.ligmol=Change_mol_xyz(mol,self.l_coords)
        self.ligmol=Neutralize_atoms(self.ligmol)
        #Chem.SanitizeMol(self.ligmol)
        self.l_natoms=len(self.l_atoms)
        for key in self.l_pharmacophores.keys():
            for i in range(len(self.l_pharmacophores[key])):
                self.l_pharmacophores[key][i]=[traverse_dict[c] for c in self.l_pharmacophores[key][i] if c in l_noH_idx]
                
        pharm_groups=[]
        for group in self.pharm_groups:
            pharm_groups.append([traverse_dict[c] for c in group if c in l_noH_idx])
        self.pharm_groups=pharm_groups
        
        rings=[]
        for ring in self.rings:
            rings.append([traverse_dict[c] for c in ring if c in l_noH_idx])
        self.rings=rings
        #aromatic_ring_labels don't need to be changed
        return
    
    def devide_l_into_groups(self):
        self.l_groups=Merge_single_rings_to_nodes(adjs=self.l_adjs,rings=self.rings,atomics=self.l_atoms,pharm_groups=self.pharm_groups)
        #print (self.l_groups)
        self.l_ngroups=len(self.l_groups)
        self.l_atom_gid=np.zeros(self.l_natoms).astype(int)
        for i in range(self.l_ngroups):
            for j in self.l_groups[i]:
                self.l_atom_gid[j]=i
        return 
    
    def gen_ligand_tree_graph(self):
        self.l_groups_adjs=np.zeros((self.l_ngroups,self.l_ngroups))
        for i in range(self.l_ngroups-1):
            for j in range(i+1,self.l_ngroups):
                for atom_i in self.l_groups[i]:
                    for atom_j in self.l_groups[j]:
                        if self.l_adjs[atom_i,atom_j]!=0:
                            self.l_groups_adjs[i,j]=1
                            self.l_groups_adjs[j,i]=1
        
        self.l_groups_feats=np.zeros((self.l_ngroups,FGP.n_group_feats),dtype=int)
        self.l_groups_max_ring_size=np.zeros(self.l_ngroups,dtype=int)
        self.l_groups_coords=np.zeros((self.l_ngroups,3))
        
        for i in range(self.l_ngroups):
            ring_num=0
            group_atomics=self.l_atoms[np.ix_(self.l_groups[i])]
            aromatic_ring_num=0
            
            max_single_ring_size=0
            
            ring_atomids=[]
            for rid,ring in enumerate(self.rings):
                if set(ring).issubset(self.l_groups[i]):
                    ring_num+=1
                    if len(ring)>max_single_ring_size:
                        max_single_ring_size=len(ring)
                    ring_atomids+=ring
                    if self.aromatic_ring_labels[rid]:
                        aromatic_ring_num+=1

            ring_atomids=list(set(ring_atomids))
            non_ring_num=len(self.l_groups[i])-len(ring_atomids)
            
            CNOFPSClBrI=[list(group_atomics).count(i) for i in FGP.atom_types]
            pharm_fp=np.zeros(7)
            for key in  ['Donor','Acceptor','NegIonizable','PosIonizable','Aromatic','Hydrophobe','LumpedHydrophobe']:
                for atomids in self.l_pharmacophores[key]:
                    if set(atomids).issubset(self.l_groups[i]):
                        pharm_fp[FGP.pharm_types.index(key)]=1
                        break

            self.l_groups_feats[i]=np.array([ring_num]+CNOFPSClBrI+[non_ring_num]+[aromatic_ring_num]+list(pharm_fp))
            self.l_groups_max_ring_size[i]=max_single_ring_size
            
            assert self.l_groups_feats[i][0]<=4, f"Ring nums of {self.l_groups_feats[i][0]} in node exceeds the maximum size of 4" 
            assert np.sum(self.l_groups_feats[i][1:10])<=FGP.max_lgroup_size, f"atom nums in ring system exceeds the maximum size of {FGP.max_lgroup_size}, feats is {self.l_groups_feats[i]}"
            assert self.l_groups_max_ring_size[i]<=8, f"Ring size of {self.l_groups_max_ring_size[i]} in node exceeds the maximum size of 8"
            atomics=self.l_atoms[np.ix_(self.l_groups[i])].reshape(-1,1)
            coords=self.l_coords[np.ix_(self.l_groups[i])]
            self.l_groups_coords[i]=np.mean(coords*atomics,axis=0)/np.sum(self.l_atoms[np.ix_(self.l_groups[i])])
        return
    
    def gen_ligand_leaf_graph(self):
        self.l_groups_inner_atoms=[]
        self.l_groups_inner_coords=[]
        self.l_groups_inner_fcs=[]
        self.l_groups_inner_adjs=[]
        
        for i in range(self.l_ngroups):
            atoms=self.l_atoms[np.ix_(self.l_groups[i])]
            fcs=self.l_fcs[np.ix_(self.l_groups[i])]
            adjs=self.l_adjs[np.ix_(self.l_groups[i],self.l_groups[i])]
            coords=self.l_coords[np.ix_(self.l_groups[i])]
            self.l_groups_inner_atoms.append(atoms)
            self.l_groups_inner_fcs.append(fcs)
            self.l_groups_inner_adjs.append(adjs)
            self.l_groups_inner_coords.append(coords)
        return 

    def reindex_tree_graph(self,mode='random'):
        if self.l_ngroups>1:
            if mode=='random':
                root_gid=random.choice([i for i in range(self.l_groups)])
            else:
                root_gid=self.l_atom_gid[0]
                
            tree_graph=nx.Graph(self.l_groups_adjs)
            
            group_order=bfs_seq(tree_graph,root_gid)
        else:
            group_order=np.array([0])
        return group_order
        
    def reindex_atom_graph_and_leaf_graph(self,group_order):
        atom_order=[]
        group_inner_order=[]
        for i in group_order:
            if len(self.l_groups[i])>1:
                group_inner_graph=nx.Graph(self.l_groups_inner_adjs[i])
                candidate_start_ids=[]
                if len(atom_order)==0:
                    candidate_start_ids=[si for si in range(len(self.l_groups[i]))]
                else:
                    for si in range(len(self.l_groups[i])):
                        for j in range(len(atom_order)):
                            if self.l_adjs[self.l_groups[i][si],atom_order[j]]>0:
                                candidate_start_ids.append(si)
                    set(candidate_start_ids)
                group_start_id=random.choice(candidate_start_ids)
                inner_order=bfs_seq(group_inner_graph,group_start_id)
                group_inner_order.append(inner_order)
                for si in inner_order:
                    if self.l_groups[i][si] not in atom_order:
                        atom_order.append(self.l_groups[i][si])
            else:
                group_inner_order.append([0])
                atom_order.append(self.l_groups[i][0])
        atom_order=np.array(atom_order)
        group_inner_order=[np.array(order) for order in group_inner_order]
        return atom_order,group_inner_order
    
    def rearrange_ligand(self,mode='fixed',debug=False):
        group_order=self.reindex_tree_graph(mode)
        atom_order,group_inner_order=self.reindex_atom_graph_and_leaf_graph(group_order)
        if debug:
            assert self.ligmol is not None, "ligmol should not be None for debug option!"
            Drawmols(self.ligmol,permindex=atom_order,filename=f'{self.savepath}/lig.png',cliques=self.l_groups)
            
            tmpmol=mol_with_atom_index_and_formal_charge(self.ligmol)
            AllChem.Compute2DCoords(tmpmol)
            Draw.MolToFile(tmpmol,f"{self.savepath}/lig_fc.png")
            """
            input_svg = f"{self.savepath}/lig_fc.svg"
            output_png = f"{self.savepath}/lig_fc.png"
            dpi = 300
            output_width = 800  # 
            output_height = 800  # 
            cairosvg.svg2png(
                url=input_svg, write_to=output_png,
                dpi=dpi, output_width=output_width, 
                output_height=output_height)
            """
            
        #create index corresponse between old and new index
        b2n_index={}
        n2b_index={}
        for aid,a in enumerate(atom_order):
            b2n_index[a]=aid
            n2b_index[aid]=a
        #atom_reorder=[int(i) for i in np.argsort(atom_order)]
        # rearrage the ligand graph
        self.l_atoms=self.l_atoms[np.ix_(atom_order)]
        self.l_natoms=len(self.l_atoms)
        self.l_coords=self.l_coords[np.ix_(atom_order)]
        self.l_fcs=self.l_fcs[np.ix_(atom_order)]
        self.l_adjs=self.l_adjs[np.ix_(atom_order,atom_order)]
        # rearrange the ligand tree graph
        self.l_groups_adjs=self.l_groups_adjs[np.ix_(group_order,group_order)]
        self.l_groups=[self.l_groups[i] for i in group_order]
        self.l_groups_feats=self.l_groups_feats[np.ix_(group_order)]
        self.l_groups_max_ring_size=self.l_groups_max_ring_size[np.ix_(group_order)]
        self.l_groups_coords=self.l_groups_coords[np.ix_(group_order)]
        
        # rearrange the ligand leaf graphs
        for gid,inner_order in enumerate(group_inner_order):
            self.l_groups[gid]=[b2n_index[c] for c in self.l_groups[gid][np.ix_(inner_order)]]
            
        self.l_atom_gid=np.zeros(self.l_natoms).astype(int)
        
        for gid in range(self.l_ngroups):
            for i in self.l_groups[gid]:
                self.l_atom_gid[i]=gid
                
        for key in self.l_pharmacophores.keys():
            for i in range(len(self.l_pharmacophores[key])):
                self.l_pharmacophores[key][i]=[b2n_index[c] for c in self.l_pharmacophores[key][i]]
        
        self.gen_ligand_leaf_graph()
        self.ligmol=Chem.rdmolops.RenumberAtoms(self.ligmol,list([int(i) for i in atom_order]))
        return   
        
    def get_ligand_mass_center(self):
        l_mc=np.sum(self.l_atoms.reshape(-1,1)*self.l_coords,axis=0)/np.sum(self.l_atoms,axis=0)
        return l_mc

    def gen_ligand_ic_lists(self):
        adjs_=np.where(self.l_adjs>0,1,0)
        self.l_zb,self.l_za,self.l_zd=Adjs_to_IC_targets(adjs_)
        self.l_nbonds=len(self.l_zb)
        self.l_nangles=len(self.l_za)
        self.l_ndihedrals=len(self.l_zd)
        return 
    
    def gen_l_coord_gen_states(self): 
        l_mg_gmasks=[]
        l_mg_dmasks=[]

        for step_id in range(self.l_ngroups):
            l_all_atoms=np.array(Flatten(list(self.l_groups[:step_id+1])))
            l_fixed_atoms=np.array(Flatten(list(self.l_groups[:step_id])))
            
            l_gmasks=np.zeros(self.l_natoms)
            l_gmasks[np.ix_(l_all_atoms)]=1
            l_fixmasks=np.zeros(self.l_natoms)
            if step_id>0:
                l_fixmasks[np.ix_(l_fixed_atoms)]=1
             
            l_mg_gmasks.append(l_gmasks)
            l_mg_dmasks.append(l_fixmasks)
            

        self.crd_gen_states= (l_mg_gmasks,l_mg_dmasks)
        return
    
    def sample_l_coord_gen_states(self,stepid,max_latoms=None,max_lbonds=None,max_langles=None,max_ldihedrals=None,l_mode='flex-new'):
        
        assert stepid < len(self.crd_gen_states[0]), "stepid should be less than the number of crd gen steps"
        
        if max_latoms is None:
            max_latoms=self.l_natoms
        
        l_gmasks=self.crd_gen_states[0][stepid]
        l_gmasks_2D=l_gmasks.reshape(-1,1)*l_gmasks.reshape(1,-1)
        l_fixmasks=self.crd_gen_states[1][stepid]
        
        l_atoms,l_atoms_onehot,l_fcs,l_fcs_onehot,l_adjs,l_adjs_mat,l_coords,l_masks=self.gen_masked_l_infos(l_gmasks,\
                                                                                                             l_gmasks_2D,\
                                                                                                             max_latoms=max_latoms)
        l_fix_masks=np.zeros(max_latoms).astype(int)
        l_flexible_masks=np.ones(max_latoms).astype(int)
        
        if l_mode=='fix-previous':
            l_fix_masks[:self.l_natoms]=l_fixmasks
            l_flexible_masks[:self.l_natoms]=(1-l_fixmasks)*l_gmasks
        else:
            l_flexible_masks[:self.l_natoms]=l_gmasks
        
        l_adjs=self.l_adjs*l_gmasks_2D
        l_adjs_=np.where(l_adjs>0,1,0)
        l_zb,l_za,l_zd=Adjs_to_IC_targets(l_adjs_)
        l_nbonds=len(l_zb)
        l_nangles=len(l_za)
        l_ndihedrals=len(l_zd)
        
        if max_lbonds is None:
            max_lbonds=l_nbonds
        if max_langles is None:
            max_langles=l_nangles
        if max_ldihedrals is None:
            max_ldihedrals=l_ndihedrals
            
        l_zbs=torch.zeros((max_lbonds,2)).long()
        l_zas=torch.zeros((max_langles,3)).long()
        l_zds=torch.zeros((max_ldihedrals,4)).long()
        l_zbmask=torch.zeros(max_lbonds).long()
        l_zamask=torch.zeros(max_langles).long()
        l_zdmask=torch.zeros(max_ldihedrals).long()
        
        l_fix_masks=torch.from_numpy(l_fix_masks)
        l_flexible_masks=torch.from_numpy(l_flexible_masks)
        
        l_zbs[:l_nbonds]=torch.from_numpy(l_zb)
        l_zas[:l_nangles]=torch.from_numpy(l_za)
        l_zds[:l_ndihedrals]=torch.from_numpy(l_zd)
        l_zbmask[:l_nbonds]=1
        l_zamask[:l_nangles]=1
        l_zdmask[:l_ndihedrals]=1
        l_pocket_labels=torch.zeros(max_latoms).long()
        l_ligand_labels=torch.zeros(max_latoms).long()
        l_ligand_labels[:self.l_natoms]=torch.Tensor(l_gmasks).long()
        
        return l_atoms,l_atoms_onehot,l_fcs,l_fcs_onehot,l_adjs,l_adjs_mat,l_coords,l_masks,l_fix_masks,l_flexible_masks,\
               l_zbs,l_zas,l_zds,l_zbmask,l_zamask,l_zdmask,l_pocket_labels,l_ligand_labels            
    
    def gen_l_leaf_graph_states(self):
        nadd_mg_masks,nadd_mg_masks_2D,nadd_apds=[],[],[]
        rgen_mg_masks,rgen_mg_masks_2D,rgen_rg_masks,rgen_rg_masks_2D,rgen_ftypes,rgen_apds,rgen_focus_groups=[],[],[],[],[],[],[]
        nconn_mg_masks,nconn_mg_masks_2D,nconn_rg_masks,nconn_rg_masks_2D,nconn_focus_ids,nconn_apds,nconn_focus_groups=[],[],[],[],[],[],[]
        
        for i in range(self.l_ngroups):
            graph_atoms=np.array(Flatten(list(self.l_groups[:i])))
            l_masks=np.zeros(self.l_natoms)

            if i>0:
                l_masks[np.ix_(graph_atoms)]=1

            l_masks_2D=l_masks.reshape(-1,1)*l_masks.reshape(1,-1)
            fid=self.get_l_group_fid(gid=i)
            
            APD_add=np.zeros(FGP.leaf_add_dim)
            APD_add[fid]=1
            
            nadd_mg_masks.append(np.array(l_masks))
            nadd_mg_masks_2D.append(np.array(l_masks_2D))
            
            nadd_apds.append(APD_add)
            f_node=group_fid_to_puzzled_group_feats(fid)
            
            if not (f_node[0]==0 and f_node[10]==1):
                for ii in range(len(self.l_groups[i])):
                    r_masks=np.zeros(len(self.l_groups[i]))
                    r_masks[:ii]=1
                    r_masks_2D=r_masks.reshape(-1,1)*r_masks.reshape(1,-1)
                    
                    r_add=np.zeros(FGP.r_add_dim)
                    r_conn=np.zeros(FGP.r_conn_dim)
                    r_term=np.zeros(1)
                    
                    atype_id=FGP.atom_types.index(self.l_atoms[self.l_groups[i][ii]])
                    fc_id=FGP.formal_charge_types.index(self.l_fcs[self.l_groups[i][ii]])
                    if ii!=0:
                        conn_atom_id=np.where(self.l_groups_inner_adjs[i][ii,:ii]>0)[0][-1]
                        bond_type_id=int(self.l_groups_inner_adjs[i][ii,conn_atom_id])-1
                    else:
                        conn_atom_id=0
                        bond_type_id=0
                        
                    r_add[conn_atom_id,atype_id,fc_id,bond_type_id]=1
                    r_add=r_add.reshape(-1)
                    r_conn=r_conn.reshape(-1)
                    r_term=r_term.reshape(-1)
                    
                    rgen_mg_masks.append(np.array(l_masks))
                    rgen_mg_masks_2D.append(np.array(l_masks_2D))
                    rgen_rg_masks.append(np.array(r_masks))
                    rgen_rg_masks_2D.append(np.array(r_masks_2D))
                    rgen_focus_groups.append(i)
                    rgen_ftypes.append(f_node)
                    
                    APD_ring=np.concatenate((r_add,r_conn,r_term))

                    rgen_apds.append(APD_ring)
                    if len(np.where(self.l_groups_inner_adjs[i][ii,:ii]>0)[0])>1:
                        remain_conn_atoms=np.where(self.l_groups_inner_adjs[i][ii,:ii]>0)[0][:-1]
                        r_masks[ii]=1
                        r_masks_2D[ii,conn_atom_id]=1
                        r_masks_2D[conn_atom_id,ii]=1
                        for jj in range(len(remain_conn_atoms)):
                            if jj!=0:
                                r_masks_2D[ii,remain_conn_atoms[jj-1]]=1
                                r_masks_2D[remain_conn_atoms[jj-1],ii]=1
                            r_add=np.zeros(FGP.r_add_dim)
                            r_conn=np.zeros(FGP.r_conn_dim)
                            r_term=np.zeros(1)
                            atom_j=self.l_groups[i][remain_conn_atoms[jj]]
                            bond_type_id=int(self.l_groups_inner_adjs[i][ii,remain_conn_atoms[jj]])-1
                            r_conn[remain_conn_atoms[jj],bond_type_id]=1
                            r_add=r_add.reshape(-1)
                            r_conn=r_conn.reshape(-1)
                            r_term=r_term.reshape(-1)
                            
                            rgen_mg_masks.append(np.array(l_masks))
                            rgen_mg_masks_2D.append(np.array(l_masks_2D))

                            rgen_rg_masks.append(np.array(r_masks))
                            rgen_rg_masks_2D.append(np.array(r_masks_2D))
                            rgen_focus_groups.append(i)
                            rgen_ftypes.append(f_node)
                            APD_ring=np.concatenate((r_add,r_conn,r_term))
                            rgen_apds.append(APD_ring)
            
                r_add=np.zeros(FGP.r_add_dim)
                r_conn=np.zeros(FGP.r_conn_dim)
                r_term=np.ones(1)
                r_add=r_add.reshape(-1)
                r_conn=r_conn.reshape(-1)
                r_term=r_term.reshape(-1)
                r_masks=np.ones(len(self.l_groups[i])) 
                r_masks_2D=np.ones((len(self.l_groups[i]),len(self.l_groups[i])))

                rgen_mg_masks.append(np.array(l_masks))
                rgen_mg_masks_2D.append(np.array(l_masks_2D))
                
                rgen_rg_masks.append(np.array(r_masks))
                rgen_rg_masks_2D.append(np.array(r_masks_2D))
                rgen_focus_groups.append(i)
                rgen_ftypes.append(f_node)
                APD_ring=np.concatenate((r_add,r_conn,r_term))
                rgen_apds.append(APD_ring)
            else:
                r_masks=np.zeros(len(self.l_groups[i]))
                r_masks[:1]=1
                r_masks_2D=r_masks.reshape(-1,1)*r_masks.reshape(1,-1)
                
            if i>0:
                APD_conn=np.zeros(FGP.leaf_conn_dim) 
                conn_atom_id=np.where(self.l_adjs[self.l_groups[i][0],:self.l_groups[i][0]]>0)[0][-1]
                bond_type_id=int(self.l_adjs[self.l_groups[i][0],conn_atom_id])-1
                if bond_type_id<0:
                    print (self.savepath,' is wrong !!!')
                APD_conn[conn_atom_id,bond_type_id]=1
                nconn_mg_masks.append(np.array(l_masks))
                nconn_mg_masks_2D.append(np.array(l_masks_2D))
                nconn_focus_groups.append(i)

                nconn_rg_masks.append(np.array(r_masks))
                nconn_rg_masks_2D.append(np.array(r_masks_2D))
                nconn_focus_ids.append(np.array(0))
                nconn_apds.append(APD_conn.reshape(-1))
                
        nadd_mg_masks.append(np.ones(self.l_natoms))
        nadd_mg_masks_2D.append(np.ones((self.l_natoms,self.l_natoms)))

        APD_add=np.zeros(FGP.leaf_add_dim)
        APD_add[-1]=1
        
        nadd_apds.append(APD_add)
        self.nadd_lg_states=(nadd_mg_masks,nadd_mg_masks_2D,nadd_apds)
        self.rgen_lg_states=(rgen_mg_masks,rgen_mg_masks_2D,rgen_rg_masks,rgen_rg_masks_2D,rgen_focus_groups,rgen_ftypes,rgen_apds)
        self.nconn_lg_states=(nconn_mg_masks,nconn_mg_masks_2D,nconn_rg_masks,nconn_rg_masks_2D,nconn_focus_groups,nconn_focus_ids,nconn_apds)
        return 
    
    def sample_leaf_nadd_states(self,stepid,max_patoms=None,max_latoms=None,noise_std=0):
            
        if max_latoms is None:
            max_latoms=self.l_natoms
        
        l_gmasks=self.nadd_lg_states[0][stepid]
        l_gmasks_2D=self.nadd_lg_states[1][stepid]
        l_apds=self.nadd_lg_states[2][stepid]
        lmc=self.get_ligand_mass_center()
        self.l_coords=self.l_coords-lmc

        l_atoms,l_atoms_onehot,l_fcs,l_fcs_onehot,l_adjs,l_adjs_mat,l_coords,l_masks=self.gen_masked_l_infos(lmasks=l_gmasks,
                                                                                                            lmasks_2D=l_gmasks_2D,
                                                                                                            max_latoms=max_latoms,
                                                                                                            noise_std=noise_std)

        
        l_apds_tensor=torch.from_numpy(l_apds)
        
        return l_atoms,l_atoms_onehot,l_fcs,l_fcs_onehot,l_adjs,l_adjs_mat,l_coords,l_masks,l_apds_tensor

    def sample_leaf_rgen_states(self,stepid, max_latoms=None, max_ringsize=None,noise_std=0):
        if max_latoms is None:
            max_latoms=self.l_natoms
        
        l_gmasks=self.rgen_lg_states[0][stepid]
        l_gmasks_2D=self.rgen_lg_states[1][stepid]
        r_masks=self.rgen_lg_states[2][stepid]
        r_masks_2D=self.rgen_lg_states[3][stepid]

        focus_group_idx=self.rgen_lg_states[4][stepid]
        r_feats=self.rgen_lg_states[5][stepid]
        r_apds=self.rgen_lg_states[6][stepid]
        lmc=self.get_ligand_mass_center()
        self.l_coords=self.l_coords-lmc
        
        l_atoms,l_atoms_onehot,l_fcs,l_fcs_onehot,l_adjs,l_adjs_mat,l_coords,l_masks=\
                                                    self.gen_masked_l_infos(lmasks=l_gmasks,
                                                                            lmasks_2D=l_gmasks_2D,
                                                                            max_latoms=max_latoms,
                                                                            noise_std=noise_std)
                                                    
        r_atoms,r_atoms_onehot,r_fcs,r_fcs_onehot,r_adjs,r_adjs_mat,r_coords,r_masks=\
                                                    self.gen_masked_r_infos(gid=focus_group_idx,
                                                                            rmasks=r_masks,
                                                                            rmasks_2D=r_masks_2D,
                                                                            max_ratoms=max_ringsize)
        r_feats=torch.from_numpy(r_feats)
        r_apds=torch.from_numpy(r_apds)
        
        return l_atoms,l_atoms_onehot,l_fcs,l_fcs_onehot,l_adjs,l_adjs_mat,l_coords,l_masks,\
                r_atoms,r_atoms_onehot,r_fcs,r_fcs_onehot,r_adjs,r_adjs_mat,r_coords,r_masks,\
                r_feats,r_apds
    
    def sample_leaf_nconn_states(self, stepid, max_latoms=None, max_ringsize=None,noise_std=0):
        
        if max_latoms is None:
            max_latoms=self.l_natoms
        
        l_gmasks=self.nconn_lg_states[0][stepid]
        l_gmasks_2D=self.nconn_lg_states[1][stepid]
        r_masks=self.nconn_lg_states[2][stepid]
        r_masks_2D=self.nconn_lg_states[3][stepid]

        focus_group_idx=self.nconn_lg_states[4][stepid]
        focus_atom_idx=self.nconn_lg_states[5][stepid]
        l_apds=self.nconn_lg_states[6][stepid]
        lmc=self.get_ligand_mass_center()
        self.l_coords=self.l_coords-lmc
        l_atoms,l_atoms_onehot,l_fcs,l_fcs_onehot,l_adjs,l_adjs_mat,l_coords,l_masks=\
                                                    self.gen_masked_l_infos(lmasks=l_gmasks,
                                                                            lmasks_2D=l_gmasks_2D,
                                                                            max_latoms=max_latoms,
                                                                            noise_std=noise_std)
                                                                            
        r_atoms,r_atoms_onehot,r_fcs,r_fcs_onehot,r_adjs,r_adjs_mat,r_coords,r_masks=\
                                                    self.gen_masked_r_infos(
                                                                            gid=focus_group_idx,
                                                                            rmasks=r_masks,
                                                                            rmasks_2D=r_masks_2D,
                                                                            max_ratoms=max_ringsize)
        
        focus_atom_idx=torch.Tensor(np.array([focus_atom_idx])).long()
        l_apds=torch.from_numpy(l_apds)
        
        return l_atoms,l_atoms_onehot,l_fcs,l_fcs_onehot,l_adjs,l_adjs_mat,l_coords,l_masks,\
                r_atoms,r_atoms_onehot,r_fcs,r_fcs_onehot,r_adjs,r_adjs_mat,r_coords,r_masks,\
                focus_atom_idx,l_apds
    
    def gen_masked_r_infos(self,gid,rmasks,rmasks_2D,max_ratoms=None):
        if max_ratoms is None:
            max_ratoms=len(self.l_groups[gid])
        r_atoms,r_atoms_onehot,r_fcs,r_fcs_onehot=self.padding_group_feat_tensors(gid=gid,rmasks=rmasks,max_ratoms=max_ratoms)
        r_adjs,r_adjs_mat=self.padding_group_adjs_tensors(gid=gid,rmasks_2D=rmasks_2D,max_ratoms=max_ratoms)
        r_coords=self.padding_group_coords_tensors(gid=gid,rmasks=rmasks,max_ratoms=max_ratoms)
        r_masks=np.zeros(max_ratoms)
        r_masks[:len(self.l_groups[gid])]=rmasks
        return r_atoms,r_atoms_onehot,r_fcs,r_fcs_onehot,r_adjs,r_adjs_mat,r_coords,r_masks
    
    def gen_masked_l_infos(self,lmasks,lmasks_2D,max_latoms=None,noise_std=0):
        if max_latoms is None:
            max_latoms=self.l_natoms
        l_atoms,l_atoms_onehot,l_fcs,l_fcs_onehot=self.padding_ligand_feat_tensors(lmasks=lmasks,max_latoms=max_latoms)
        l_adjs,l_adjs_mat=self.padding_ligand_adjs_tensors(lmasks_2D=lmasks_2D,max_latoms=max_latoms)
        l_coords=self.padding_ligand_coords_tensors(lmasks=lmasks,max_latoms=max_latoms,noise_std=noise_std) 
        l_masks=np.zeros(max_latoms)
        l_masks[:self.l_natoms]=lmasks
        l_masks=torch.from_numpy(l_masks)
        return l_atoms,l_atoms_onehot,l_fcs,l_fcs_onehot,l_adjs,l_adjs_mat,l_coords,l_masks
        
    def padding_group_feat_tensors(self,gid,rmasks=None,max_ratoms=None):
        if max_ratoms is None:
            max_ratoms=self.l_groups[gid]
        if rmasks is None:
            rmasks=np.ones(len(self.l_groups[gid]))
            
        r_atoms_=np.zeros(max_ratoms)
        r_fcs_=np.zeros(max_ratoms)
        r_atoms_onehot_=np.zeros((max_ratoms,len(FGP.atom_types_for_feats)))
        r_fcs_onehot_=np.zeros((max_ratoms,len(FGP.formal_charge_types)))
        r_natoms=len(self.l_groups[gid])
        r_atoms_[:r_natoms]=self.l_groups_inner_atoms[gid]*rmasks
        r_fcs_[:r_natoms]=self.l_groups_inner_fcs[gid]*rmasks
        
        r_atoms_onehot_[:r_natoms]=Atoms_to_onehot(self.l_groups_inner_atoms[gid],FGP.atom_types_for_feats)*rmasks.reshape(-1,1)
        r_fcs_onehot_[:r_natoms]=Atoms_to_onehot(self.l_groups_inner_fcs[gid],FGP.formal_charge_types)*rmasks.reshape(-1,1)
        r_atoms=torch.from_numpy(r_atoms_)
        r_atoms_onehot=torch.from_numpy(r_atoms_onehot_)
        r_fcs=torch.from_numpy(r_fcs_)
        r_fcs_onehot=torch.from_numpy(r_fcs_onehot_)
        return r_atoms, r_atoms_onehot, r_fcs, r_fcs_onehot
        
    def padding_group_adjs_tensors(self,gid,rmasks_2D=None,max_ratoms=None):
        if max_ratoms is None:
            max_ratoms=len(self.l_groups[gid])
        if rmasks_2D is None:
            rmasks_2D=np.ones((len(self.l_groups[gid]),len(self.l_groups[gid])))
            
        r_adjs_=np.zeros((max_ratoms,max_ratoms))
        r_adjs_[:len(self.l_groups[gid]),:len(self.l_groups[gid])]=self.l_groups_inner_adjs[gid]*rmasks_2D
        r_adjs_mat_=np.zeros((max_ratoms,max_ratoms,len(FGP.bond_types)+1))
        idx1,idx2=np.where(r_adjs_!=0)
        for id1,id2 in zip(idx1,idx2):
            r_adjs_mat_[id1,id2,int(r_adjs_[id1,id2])]=1
        r_adjs=torch.from_numpy(r_adjs_).long()
        r_adjs_mat=torch.from_numpy(r_adjs_mat_).long()
        return r_adjs,r_adjs_mat
    
    def padding_group_coords_tensors(self,gid,rmasks=None,max_ratoms=None) :
        if max_ratoms is None:
            max_ratoms=len(self.l_groups[gid])
        if rmasks is None:
            rmasks=np.ones(len(self.l_groups[gid]))
        r_coords_=np.zeros((max_ratoms,3))
        r_coords_[:len(self.l_groups[gid])]=self.l_groups_inner_coords[gid]*rmasks.reshape(-1,1)
        r_coords=torch.from_numpy(r_coords_)
        return r_coords
        
    def padding_ligand_feat_tensors(self,lmasks=None,max_latoms=None):
        if max_latoms is None:
            max_latoms=self.l_natoms
        if lmasks is None:
            lmasks=np.ones(self.l_natoms)
            
        l_atoms_=np.zeros(max_latoms)
        l_fcs_=np.zeros(max_latoms)
        l_atoms_onehot_=np.zeros((max_latoms,len(FGP.atom_types_for_feats)))
        l_fcs_onehot_=np.zeros((max_latoms,len(FGP.formal_charge_types)))
                
        l_atoms_[:self.l_natoms]=self.l_atoms*lmasks
        l_fcs_[:self.l_natoms]=self.l_fcs*lmasks
        l_atoms_onehot_[:self.l_natoms]=Atoms_to_onehot(self.l_atoms,FGP.atom_types_for_feats)*lmasks.reshape(-1,1)
        l_fcs_onehot_[:self.l_natoms]=Atoms_to_onehot(self.l_fcs,FGP.formal_charge_types)*lmasks.reshape(-1,1)
        l_atoms=torch.from_numpy(l_atoms_)
        l_atoms_onehot=torch.from_numpy(l_atoms_onehot_)
        l_fcs=torch.from_numpy(l_fcs_)
        l_fcs_onehot=torch.from_numpy(l_fcs_onehot_)
        return l_atoms, l_atoms_onehot, l_fcs, l_fcs_onehot

    def padding_ligand_adjs_tensors(self,lmasks_2D=None,max_latoms=None):
        if max_latoms is None:
            max_latoms=self.l_natoms
        if lmasks_2D is None:
            lmasks_2D=np.ones((self.l_natoms,self.l_natoms))
        l_adjs_=np.zeros((max_latoms,max_latoms))
        l_adjs_[:self.l_natoms,:self.l_natoms]=self.l_adjs*lmasks_2D
        l_adjs_mat_=np.zeros((max_latoms,max_latoms,len(FGP.bond_types)+1))
        idx1,idx2=np.where(l_adjs_!=0)
        for id1,id2 in zip(idx1,idx2):
            l_adjs_mat_[id1,id2,int(l_adjs_[id1,id2])]=1
        l_adjs=torch.from_numpy(l_adjs_).long()
        l_adjs_mat=torch.from_numpy(l_adjs_mat_).long()
        
        return l_adjs,l_adjs_mat
        
    def padding_ligand_coords_tensors(self,lmasks=None,max_latoms=None, noise_std=0):
        if max_latoms is None:
            max_latoms=self.l_natoms
            
        if lmasks is None:
            lmasks=np.ones(self.l_natoms)
        noise=np.random.normal(0,noise_std,(self.l_natoms,3)) 
        l_coords_=np.zeros((max_latoms,3))
        l_coords_[:self.l_natoms]=(self.l_coords+noise)*lmasks.reshape(-1,1)
        l_coords=torch.from_numpy(l_coords_)
        return l_coords
    
    def Trans_ring_to_Mol(self,gid,rmasks=None):
        if rmasks is None:
            rmasks=np.ones(len(self.l_groups[gid]))
        rmasks=rmasks.astype(bool)
        molecule=Chem.RWMol()
        
        atoms=self.l_groups_inner_atoms[gid][rmasks]
        adjs=self.l_groups_inner_adjs[gid][rmasks][:,rmasks]
        fcs=self.l_groups_inner_fcs[gid][rmasks]
        coords=self.l_groups_inner_coords[gid][rmasks]
        
        for atom in atoms:
            new_atom=Chem.Atom(int(atom))
            molecule_idx=molecule.AddAtom(new_atom)
        
        row,col=np.diag_indices_from(adjs)
        adjs[row,col]=0
        idx1,idx2=np.where(adjs!=0)
        
        for id1,id2 in zip(idx1,idx2):
            if id1<id2:
                molecule.AddBond(int(id1),int(id2),FGP.bond_types[int(adjs[id1,id2])-1])

        for aid,at in enumerate(molecule.GetAtoms()):
            at.SetFormalCharge(int(fcs[aid]))
        mol=molecule.GetMol()
        #Chem.SanitizeMol(mol)
        AllChem.Compute2DCoords(mol)
        mol=Change_mol_xyz(mol,coords)
        return mol
     
    def Trans_Ligand_to_Mol(self,lmasks=None):
        if lmasks is None:
            lmasks=np.ones(self.l_natoms)
        lmasks=lmasks.astype(bool)
        molecule=Chem.RWMol()
        
        atoms=self.l_atoms[lmasks]
        adjs=self.l_adjs[lmasks][:,lmasks]
        fcs=self.l_fcs[lmasks]
        coords=self.l_coords[lmasks]
        
        for atom in atoms:
            new_atom=Chem.Atom(int(atom))
            molecule_idx=molecule.AddAtom(new_atom)
        
        row,col=np.diag_indices_from(adjs)
        adjs[row,col]=0
        idx1,idx2=np.where(adjs!=0)
        
        for id1,id2 in zip(idx1,idx2):
            if id1<id2:
                molecule.AddBond(int(id1),int(id2),FGP.bond_types[int(adjs[id1,id2])-1])

        for aid,at in enumerate(molecule.GetAtoms()):
            at.SetFormalCharge(int(fcs[aid]))
        mol=molecule.GetMol()
        #Chem.SanitizeMol(mol)
        AllChem.Compute2DCoords(mol)
        mol=Change_mol_xyz(mol,coords)
        return mol

    def get_l_group_descriptor(self,gid):
        f=self.l_groups_feats[gid]
        if f[0]>0:
            descriptor=f'R{f[0]}-C{f[1]}-N{f[2]}-O{f[3]}-F{f[4]}-P{f[5]}-S{f[6]}-Cl{f[7]}-Br{f[8]}-I{f[9]}-Bes{f[10]}-AR{f[11]}-'+\
                        f'HD{f[12]}-HA{f[13]}-Neg{f[14]}-Pos{f[15]}-Aro{f[16]}-Hyd{f[17]}-LHyd{f[18]}'
        else:
            if f[10]>1:
                descriptor=f'R{f[0]}-C*-N*-O*-F*-P*-S*-Cl*-Br*-I*-Bes{f[10]}-AR*-'+\
                            f'HD{f[12]}-HA{f[13]}-Neg{f[14]}-Pos{f[15]}-Aro{f[16]}-Hyd{f[17]}-LHyd{f[18]}'
            else:
                descriptor=f'R{f[0]}-C{f[1]}-N{f[2]}-O{f[3]}-F{f[4]}-P{f[5]}-S{f[6]}-Cl{f[7]}-Br{f[8]}-I{f[9]}-Bes{f[10]}-AR*-'+\
                        f'HD*-HA*-Neg*-Pos*-Aro*-Hyd*-LHyd*'
        return descriptor
    
    def get_l_group_fid(self,gid):
        descriptor=self.get_l_group_descriptor(gid)
        try:
            if descriptor not in FGP.group_type_index_dict.keys():
                #print (descriptor)
                raise ValueError (f'Unsupported Node type for Complex ligand {self.idx} with descriptor {descriptor}')
            fid=FGP.group_type_index_dict[descriptor] 
            return fid
        except ValueError as e:
            print (repr(e))
            return None

    def save(self,path):
        with open(path,'wb') as f:
            pickle.dump(self,f)
        return
    
    def Trans_to_constrains(self,
                            keep_total_atom_num=True,
                            keep_ring_topology=True,
                            keep_non_c_atoms=True,
                            keep_non_hydro_pharm_info=True,
                            ):
        
        constrains={}
        for i in range(self.l_ngroups):
            group_feats=self.l_groups_feats[i]
            ring_num,C_num,N_num,O_num,F_num,P_num,S_num,Cl_num,Br_num,I_num,non_ring_num,\
                AR_num,HD_flag,HA_flag,Neg_flag,Pos_flag,Aro_flag,Hyd_flag,LHyd_flag=group_feats
            element_num_dict={'C':C_num,'N':N_num,'O':O_num,'F':F_num,'P':P_num,'S':S_num,'Cl':Cl_num,'Br':Br_num,'I':I_num}
            pharm_type_dict={'Donor':HD_flag,'Acceptor':HA_flag,'Neg':Neg_flag,'Pos':Pos_flag,'Aro':Aro_flag,'Hyd':Hyd_flag,'LHyd':LHyd_flag}
            nadd_ct=nadd_constrain()
            if keep_total_atom_num:
                total_atnum=sum(element_num_dict.values())
                nadd_ct.total_atnum_range=(total_atnum,total_atnum)

            if keep_ring_topology:
                nadd_ct.ringnum_range=(ring_num,ring_num)
                nadd_ct.ar_ringnum_range=(AR_num,AR_num)
                nadd_ct.branchnum_range=(non_ring_num,non_ring_num)
                
            if keep_non_c_atoms:
                for key in ['N','O','F','P','S','Cl','Br','I']:
                    if element_num_dict[key]>0:
                        nadd_ct.atnum_range_dict[key]=(element_num_dict[key],element_num_dict[key])
            
            if keep_non_hydro_pharm_info:
                for key in ['Donor','Acceptor','Neg','Pos','Aro']:
                    if pharm_type_dict[key]>0:
                        nadd_ct.pharm_types[key]=pharm_type_dict[key]
            
            if keep_non_hydro_pharm_info:
                for key in ['Hyd','LHyd']:
                    if pharm_type_dict[key]>0:
                        nadd_ct.pharm_types[key]=pharm_type_dict[key]
            
            nadd_ct.force_step=True
            
            nconn_ct=nconn_constrain()
            nconn_ct.constrain_connect_node_id=[np.nonzero(self.l_groups_adjs[i,:i])[0]]  
            constrains[str(i)]={"node add":nadd_ct,"node conn":nconn_ct}
            
        return constrains

                
                        
                
    






        
            
        
                    
        
            
            
    
