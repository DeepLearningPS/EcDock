import pickle 
from rdkit import Chem
from rdkit.Chem import AllChem,ChemicalFeatures
from rdkit import RDConfig
import numpy as np
import scipy,os,random,copy
from tqdm import tqdm 
import networkx as nx
import torch 

from ..comparm import * 
from ..utils.utils_os import * 
from ..utils.utils_interactions import detect_interactions
from ..utils.utils_graphroute import bfs_seq,Merge_single_rings_to_nodes
from ..utils.utils_rdkit import Change_mol_xyz,Drawmols,MolToXYZ,Neutralize_atoms,Check_warning_structures
from ..utils.utils_np import Adjs_to_IC_targets,np_adjs_to_zmat

from .molecule import *
from .protein import * 


class Complex:    
    def __init__(self,Protein_PDBFile,Ligand_SDFFile,PDB_SOURCE='PDBBind-2017'):
        self.protein_pdb=Protein_PDBFile
        self.ligand_sdf=Ligand_SDFFile
        self.pdb_source=PDB_SOURCE
        
        vars=self.protein_pdb.split('/')[:-1]
        self.savepath='/'.join(vars)
        
        
    def standardrize(self,max_patoms=150,keep_pocket_mode='int-only',ligand_rearrange_mode='fix',debug=False):
        self.prepare_ligand()
        
        self.prepare_protein()

        self.get_pl_int_adjs()
        
        self.remove_l_Hs()
        self.devide_l_into_groups()
        
        self.remove_p_Hs()

        self.get_pl_group_adjs()
        
        self.get_pl_atom_adjs()
        
        self.keep_pocket_only(max_patoms=max_patoms,mode=keep_pocket_mode)
        
        self.gen_ligand_tree_graph()
        
        self.gen_ligand_leaf_graph()

        self.rearrange_ligand(ligand_rearrange_mode,debug=debug)
        
        self.add_protein_group_connections()
        
        self.gen_pocket_ic_lists()
        
        ligmol=self.Trans_Ligand_to_Mol()
        promol=self.Trans_Pocket_to_Mol()
        try:
            molsupp=Chem.SDWriter(f'{self.savepath}/lig.sdf')
            molsupp.write(ligmol)
        except:
            #print (f'Complex in {self.savepath} failed to write ligand sdf!' )
            pass
        #print (self.l_smi,Chem.MolToSmiles(ligmol))
        MolToXYZ(ligmol,self.savepath+'/lig.xyz')
        MolToXYZ(promol,self.savepath+f'/pocket_{keep_pocket_mode}_{max_patoms}.xyz')
        return 
    
    def prepare_protein(self):
        protein=Protein(self.protein_pdb,source=self.pdb_source)
        #protein.write_group_info()
        self.p_groups=protein.groups
        self.p_ngroups=protein.ngroups 
        self.p_atoms=np.array(protein.atoms)
        self.p_natoms=protein.natoms
        self.p_adjs=np.array(protein.adjs)
        self.p_atom_name=np.array(protein.atom_name)
        self.p_atom_rname=np.array(protein.atom_rname)
        self.p_atom_rid=np.array(protein.atom_rid)
        self.p_fcs=np.array(protein.formal_charges)
        self.p_atom_gid=protein.atom_gids.astype(int)
        self.p_backbone_labels=np.array(protein.backbone_labels)
        self.p_coords=np.array(protein.coords)
        
    def prepare_ligand(self):
        if self.ligand_sdf[-4:]=='.sdf':
            self.ligmol=Chem.rdmolfiles.SDMolSupplier(self.ligand_sdf,removeHs=False)[0]
        elif self.ligand_sdf[-4:]=='.pdb':
            self.ligmol=Chem.MolFromPDBFile(self.ligand_sdf,removeHs=False)
        
        if self.ligmol is None:
            self.ligmol=Chem.MolFromMol2File(self.ligand_sdf[:-4]+'.mol2',removeHs=False)
            if self.ligmol is None:
                print (f'Error in both reading {self.ligand_sdf[:-4]+".mol2"} and {self.ligand_sdf}')
                return
            
        #assert not Check_warning_structures(self.ligmol), f"Warning structures in {self.ligand_sdf}!"
        self.ligmol=standardize(self.ligmol) 
        #self.ligmol=Neutralize_atoms(self.ligmol)
        Chem.Kekulize(self.ligmol)
         
        assert self.ligmol is not None, f"Complex in {self.savepath} prepare failed due to ligand reading error"
        
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
    
    def gen_complex_pdb(self,Complex_PDBFile='./complex.pdb'):
        #ligmol=Chem.rdmolfiles.SDMolSupplier(self.ligand_sdf,removeHs=True)[0]
        #Chem.Kekulize(ligmol)
        self.ligand_pdb=self.ligand_sdf[:-4]+'.pdb'
        Chem.MolToPDBFile(self.ligmol,self.ligand_pdb)
        self.complex_pdb=Complex_PDBFile
        with open (self.protein_pdb,'r') as protein_f:
            protein_lines= [line.strip() for line in protein_f.readlines()]
        with open(self.ligand_pdb,'r') as ligand_f:
            ligand_lines=[line.strip() for line in ligand_f.readlines()]
        with open (self.complex_pdb,'w') as f:
            for line in protein_lines+ligand_lines:
                if line[:6]=='ATOM  ' or line[:6]=='HETATM':
                    f.write(line+'\n')
        return 

    def analysis_interactions(self):
        coords=np.concatenate([self.p_coords,self.l_coords],axis=0)
        interactions=detect_interactions(self.complex_pdb,coords=coords)
        return interactions
    
    def get_pl_int_adjs(self):
        self.gen_complex_pdb(Complex_PDBFile=f'{self.savepath}/complex.pdb')
        interactions=self.analysis_interactions()
        
        self.pl_adjs=np.zeros((self.p_natoms,self.l_natoms))
        self.lp_adjs=np.zeros((self.l_natoms,self.p_natoms))
        interactions_to_intbond={
                                 'pistacking':'PISTACKING',
                                 'hbond_ldon':'HBOND','hbond_lacc':'HBOND',
                                 'hydrophobic':'HYDROPHOBIC',
                                 'pication_lpi':'PICATION','pication_lcation':'PICATION',
                                 'halogen_bonds_lacc':'HALOGEN','halogen_bonds_ldon':'HALOGEN',
                                 'saltbridge_lneg':'SALTBRIDGE','saltbridge_lpos':'SALTBRIDGE',
                                 'waterbridge_lacc':'WATERBRIDGE','waterbridge_ldon':'WATERBRIDGE',
                                 }
        for key in interactions_to_intbond.keys():
            for pair in interactions[key]:
                for ai in pair[0]:
                    for aj in pair[1]:
                        p_a=np.min([ai,aj])
                        l_a=np.max([ai,aj])-self.p_natoms
                        #print (p_a,l_a)
                        idx=FGP.bond_types.index(interactions_to_intbond[key])+1
                        self.pl_adjs[p_a][l_a]=idx
                        self.lp_adjs[l_a][p_a]=idx
        
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
        self.lp_adjs=self.lp_adjs[np.ix_(l_noH_idx)]
        self.pl_adjs=self.lp_adjs.T
        
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
        Neutralize_atoms(self.ligmol)
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
        #print (self.ligmol)

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
    
    def remove_p_Hs(self):
        p_noH_idx=np.array([i for i in range(len(self.p_atoms)) if self.p_atoms[i]!=1])
        traverse_dict={}

        for aid,i in enumerate(p_noH_idx):
            traverse_dict[i]=aid
        
        for i in range(self.p_ngroups):
            self.p_groups[i]=[traverse_dict[j] for j in self.p_groups[i] if j in p_noH_idx]
        self.p_ngroups=len(self.p_groups)
        
        #print (self.groups)
        self.p_atoms=self.p_atoms[np.ix_(p_noH_idx)]
        self.p_fcs=self.p_fcs[np.ix_(p_noH_idx)]        
        self.p_adjs=self.p_adjs[np.ix_(p_noH_idx,p_noH_idx)]
        self.p_atom_name=self.p_atom_name[np.ix_(p_noH_idx)]
        self.p_atom_rname=self.p_atom_rname[np.ix_(p_noH_idx)]
        self.p_atom_rid=self.p_atom_rid[np.ix_(p_noH_idx)]
        self.p_atom_gid=self.p_atom_gid[np.ix_(p_noH_idx)]
        self.p_coords=self.p_coords[np.ix_(p_noH_idx)]
        self.p_natoms=len(self.p_atoms)
        self.p_backbone_labels=self.p_backbone_labels[np.ix_(p_noH_idx)]
        self.pl_adjs=self.pl_adjs[np.ix_(p_noH_idx)]
        self.lp_adjs=self.pl_adjs.T

        return
    
    def get_pl_group_adjs(self):
        self.pl_groups_adjs=np.zeros((self.p_ngroups,self.l_ngroups))
        self.lp_groups_adjs=np.zeros((self.l_ngroups,self.p_ngroups))
        self.pl_int_groups=[]
        
        for i in range(self.p_ngroups):
            for j in range(self.l_ngroups):
                flag=False
                for ai in self.p_groups[i]:
                    for aj in self.l_groups[j]:
                        if self.pl_adjs[ai,aj]>0:
                            self.pl_groups_adjs[i,j]=self.pl_adjs[ai,aj]
                            self.lp_groups_adjs[j,i]=self.lp_adjs[aj,ai]
                            flag=True
                if flag:
                    #print (i,j,self.p_groups[i],self.l_groups[j],flag)
                    self.pl_int_groups.append(i)
                    
        self.pl_int_groups=list(set(self.pl_int_groups))

        return
    
    def get_pl_atom_adjs(self):
        self.pl_atom_adjs=np.zeros((self.p_natoms,self.l_natoms)) # atom-based group level interaction adjs
        self.lp_atom_adjs=np.zeros((self.l_natoms,self.p_natoms)) 
        idx1,idx2=np.where(self.pl_groups_adjs>0)
        for id1,id2 in zip(idx1,idx2):
            self.pl_atom_adjs[np.ix_(self.p_groups[id1],self.l_groups[id2])]=self.pl_groups_adjs[id1,id2]
            self.lp_atom_adjs[np.ix_(self.l_groups[id2],self.p_groups[id1])]=self.lp_groups_adjs[id2,id1]
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
            sub_order=[]
            sub_bonds=[]
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
            #print (atom_order,len(atom_order),self.ligmol,Chem.MolToSmiles(self.ligmol),self.l_groups)
            Drawmols(self.ligmol,permindex=atom_order,filename=f'{self.savepath}/lig.png',cliques=self.l_groups)
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
        # rearrange the pl interactions 
        
        self.lp_groups_adjs=self.lp_groups_adjs[np.ix_(group_order)]
        self.pl_groups_adjs=self.lp_groups_adjs.T
        
        #tmp_lp_adjs=np.zeros((self.l_natoms,self.p_natoms))
        #for i in range(self.l_natoms):
        #    for j in range(self.p_natoms):
        #        l_gid=self.l_atom_gid[i]
        #        p_gid=self.p_atom_gid[j]
        #        tmp_lp_adjs[i,j]=self.lp_groups_adjs[l_gid,p_gid]
                
        self.lp_adjs=self.lp_adjs[np.ix_(atom_order)]
        self.pl_adjs=self.lp_adjs.T
        self.lp_atom_adjs=self.lp_atom_adjs[np.ix_(atom_order)]
        self.pl_atom_adjs=self.lp_atom_adjs.T
        
        #print (atom_order)
        #print ('lp_adjs reordered check',np.sum(self.lp_atom_adjs-tmp_lp_adjs))
        self.ligmol=Chem.rdmolops.RenumberAtoms(self.ligmol,list([int(i) for i in atom_order]))
        return   

    def get_pocket_atoms(self,max_patoms=150,mode='int-only'):
        self.p_int_labels=np.zeros(self.p_natoms)

        pocket_groups=self.pl_int_groups
        #print (pocket_groups)
        pocket_atoms=list(np.sort(Flatten([self.p_groups[i] for i in pocket_groups])))   
        
        #print ('int_pocket_atoms',len(pocket_atoms))

        if mode!='int-only':
            l_to_p=np.zeros(self.p_ngroups)
            for i in range(self.p_ngroups):
                flag=False
                gi_coords=self.p_coords[np.ix_(self.p_groups[i])]
                dismat=scipy.spatial.distance.cdist(self.l_coords,gi_coords)
                #print (i,self.p_groups[i])
                min_dis=np.min(dismat)    
                l_to_p[i]=min_dis
             
            p_groups_near_to_l=np.argsort(l_to_p)
            #print(p_groups_near_to_l)
            
            for i in p_groups_near_to_l:
                pocket_natoms=len(pocket_atoms)
                if i not in pocket_groups and pocket_natoms+len(self.p_groups[i])<max_patoms:
                    pocket_groups.append(i)
                    pocket_atoms+=self.p_groups[i]
                if pocket_natoms>max_patoms:
                    break

            pocket_groups=np.sort(pocket_groups)
            pocket_atoms=np.sort(pocket_atoms)
        #print ('Final pockets',pocket_groups,pocket_atoms)
        return pocket_groups,pocket_atoms
        
    def get_pocket_mass_center(self):
        p_mc=np.sum(self.p_atoms.reshape(-1,1)*self.p_coords,axis=0)/np.sum(self.p_atoms,axis=0)
        return p_mc
        
    def get_ligand_mass_center(self):
        l_mc=np.sum(self.l_atoms.reshape(-1,1)*self.l_coords,axis=0)/np.sum(self.l_atoms,axis=0)
        return l_mc

    def keep_pocket_only(self,max_patoms=150,mode='int-only',add_protein_connections=False):
        pocket_groups,pocket_atoms=self.get_pocket_atoms(max_patoms=max_patoms,mode=mode)
        
        traverse_dict={}
        for aid,i in enumerate(pocket_atoms):
            traverse_dict[i]=aid
            
        self.p_groups=[self.p_groups[i] for i in pocket_groups]
        self.p_ngroups=len(self.p_groups)
        
        for i in range(self.p_ngroups):
            self.p_groups[i]=[traverse_dict[j] for j in self.p_groups[i]]
        
        self.p_atoms=self.p_atoms[np.ix_(pocket_atoms)]
        
        self.p_atom_name=self.p_atom_name[np.ix_(pocket_atoms)]
        self.p_atom_rname=self.p_atom_rname[np.ix_(pocket_atoms)]
        self.p_atom_rid=self.p_atom_rid[np.ix_(pocket_atoms)]
        
        self.p_adjs=self.p_adjs[np.ix_(pocket_atoms,pocket_atoms)]
        self.p_fcs=self.p_fcs[np.ix_(pocket_atoms)]
        self.p_coords=self.p_coords[np.ix_(pocket_atoms)]
        self.p_natoms=len(self.p_atoms)
        self.p_backbone_labels=self.p_backbone_labels[np.ix_(pocket_atoms)]
        
        self.p_atom_gid=np.zeros(self.p_natoms).astype(int)
        for i in range(self.p_ngroups):
            for j in  self.p_groups[i]:
                self.p_atom_gid[j]=i

        
        self.pl_groups_adjs=self.pl_groups_adjs[np.ix_(pocket_groups)]
        self.lp_groups_adjs=self.pl_groups_adjs.T 
        self.pl_adjs=self.pl_adjs[np.ix_(pocket_atoms)]
        self.lp_adjs=self.pl_adjs.T
        self.pl_atom_adjs=self.pl_atom_adjs[np.ix_(pocket_atoms)]
        self.lp_atom_adjs=self.pl_atom_adjs.T
        
        return

    def add_protein_group_connections(self):
        self.pp_adjs=np.zeros((self.p_natoms,self.p_natoms))
        group_backbone_cdist=np.zeros((self.p_ngroups,self.p_ngroups))+1e4
        for i in range(self.p_ngroups):
            for j in range(i+1,self.p_ngroups):
                mindis=1e4
                for a in self.p_groups[i]:
                    if self.p_backbone_labels[a]==1:
                        for b in self.p_groups[j]:
                            if self.p_backbone_labels[b]==1:
                                dis=np.linalg.norm(self.p_coords[a]-self.p_coords[b])
                                if dis< mindis:
                                    mindis=dis
                group_backbone_cdist[i][j]=mindis
                group_backbone_cdist[j][i]=mindis
        #print (group_backbone_cdist)
        for i in range(self.p_ngroups):    
            near_groups=np.argsort(group_backbone_cdist[i])[:2]
            #print (i,near_groups)
            for j in near_groups:
                for a in self.p_groups[i]:
                    for b in self.p_groups[j]:
                        self.pp_adjs[a,b]=1
                        self.pp_adjs[b,a]=1
        return                                                            

    def gen_pocket_ic_lists(self):
        adjs_=np.where(self.p_adjs>0,1,0)
        
        self.p_zb,self.p_za,self.p_zd=Adjs_to_IC_targets(adjs_)
        self.p_nbonds=len(self.p_zb)
        self.p_nangles=len(self.p_za)
        self.p_ndihedrals=len(self.p_zd)
        return 

    def gen_ligand_ic_lists(self):
        adjs_=np.where(self.l_adjs>0,1,0)
        self.l_zb,self.l_za,self.l_zd=Adjs_to_IC_targets(adjs_)
        #self.l_zmats=np_adjs_to_zmat(adjs_)
        
        self.l_nbonds=len(self.l_zb)
        self.l_nangles=len(self.l_za)
        self.l_ndihedrals=len(self.l_zd)
        return 
    
    def mask_pl_adjs(self,pl_adjs,rate=0):
        pl_adjs_=copy.deepcopy(pl_adjs)
        pl_edge_indexes=np.where(pl_adjs_!=0)
        pl_edge_indexes=np.vstack(pl_edge_indexes).T
        #print (pl_edge_indexes)
        for edge in pl_edge_indexes:
            c=random.choices([0,1],weights=[rate,1-rate],k=1)
            if c==0:
                pl_adjs_[edge[0],edge[1]]=0
        return pl_adjs_
        
    def mask_pl_atom_adjs(self,pl_groups_adjs,rate=0):
        pl_groups_adjs_=copy.deepcopy(pl_groups_adjs)
        pl_atoms_adjs_=np.zeros(self.pl_atom_adjs.shape)
        
        idx1,idx2=np.where(pl_groups_adjs_!=0)
        for i,j in zip(idx1,idx2):
            c=random.choices([0,1],weights=[rate,1-rate],k=1)
            if c==0:
                pl_groups_adjs_[i,j]=0
                
        idx1,idx2=np.where(pl_groups_adjs_!=0)
        for i,j in zip(idx1,idx2):
            pl_atoms_adjs_[np.ix_(self.p_groups[i],self.l_groups[j])]=pl_groups_adjs_[i,j]
            
        return pl_groups_adjs_,pl_atoms_adjs_
    
    def get_p_group_atom_int_masks(self):
        # in coords generations, p_group_atom_int_labels are used to select flexible atoms in pocket
        masks=np.sum(self.pl_atom_adjs,axis=1)>0
        #print (labels)
        return masks

    def get_p_group_int_masks(self):
        # in nint actions, p_group_int_masks is designed to force the model only predict the interactions between masked pocket groups and ligand groups
        masks=np.sum(self.pl_groups_adjs,axis=1)>0
        #print (labels)
        return masks
    
    def get_p_atom_int_masks(self):
        # in complex adjs, p_atom_int_masks is designed to only connect the pocket atoms with pl interactions to ligands
        masks=np.sum(self.pl_adjs,axis=1)>0
        #print (masks)
        return masks
        
    def gen_l_coord_gen_states(self): 
        l_mg_gmasks=[]
        l_mg_dmasks=[]
        pl_int_masks=[]
        pl_atom_int_masks=[]
        for step_id in range(self.l_ngroups):
            l_all_atoms=np.array(Flatten(list(self.l_groups[:step_id+1])))
            l_fixed_atoms=np.array(Flatten(list(self.l_groups[:step_id])))
            
            l_gmasks=np.zeros(self.l_natoms)
            l_gmasks[np.ix_(l_all_atoms)]=1
            l_fixmasks=np.zeros(self.l_natoms)
            if step_id>0:
                l_fixmasks[np.ix_(l_fixed_atoms)]=1
                
            lp_int_masks=np.zeros((self.l_ngroups,self.p_ngroups))
            lp_int_masks[:step_id+1]=1
            lp_atom_int_masks=np.zeros((self.l_natoms,self.p_natoms))
            lp_atom_int_masks[np.ix_(l_all_atoms)]=1
            l_mg_gmasks.append(l_gmasks)
            l_mg_dmasks.append(l_fixmasks)
            pl_int_masks.append(lp_int_masks.T)
            pl_atom_int_masks.append(lp_atom_int_masks.T)
            
        self.crd_gen_states= (l_mg_gmasks,l_mg_dmasks,pl_int_masks,pl_atom_int_masks)
        return
    
    def sample_l_coord_gen_states(self,stepid,max_patoms=None,max_latoms=None,max_lbonds=None,max_langles=None,max_ldihedrals=None,max_pbonds=None,max_pangles=None,max_pdihedrals=None,pl_mask_rate=0,p_mode='fix-all',l_mode='flex-new',int_mode='group-based'):
        #print (stepid,len(self.crd_gen_states[0]))
        assert stepid < len(self.crd_gen_states[0]), "stepid should be less than the number of crd gen steps"
         
        if max_patoms is None:
            max_patoms=self.p_natoms
        if max_latoms is None:
            max_latoms=self.l_natoms
        
        max_catoms=max_patoms+max_latoms
        # remove pocket mass center
        pmc=self.get_pocket_mass_center()
        self.p_coords=self.p_coords-pmc
        self.l_coords=self.l_coords-pmc
        
        l_gmasks=self.crd_gen_states[0][stepid]
        l_gmasks_2D=l_gmasks.reshape(-1,1)*l_gmasks.reshape(1,-1)
        l_fixmasks=self.crd_gen_states[1][stepid]
        pl_int_masks=self.crd_gen_states[2][stepid]
        pl_atom_int_masks=self.crd_gen_states[3][stepid]
        #print ('******',stepid,l_fixmasks)
        if int_mode=='group-based':
            pl_atom_int_masks=None
        else:
            pl_int_masks=None
        c_atoms,c_atoms_onehot,c_fcs,c_fcs_onehot,c_adjs,c_adjs_mat,c_coords,c_masks=self.gen_masked_c_infos(l_gmasks,\
                                                                                                            l_gmasks_2D,
                                                                                                            pl_group_masks=pl_int_masks,
                                                                                                            pl_atom_masks=pl_atom_int_masks,
                                                                                                            pl_mask_rate=pl_mask_rate,
                                                                                                            max_patoms=max_patoms,
                                                                                                            max_latoms=max_latoms)
        
        c_fix_masks=np.zeros(max_catoms).astype(int)
        c_flexible_masks=np.ones(max_catoms).astype(int)
        p_group_atom_int_masks=self.get_p_group_atom_int_masks()
        
        if p_mode=='fix-all':
            c_fix_masks[:self.p_natoms]=1
            c_flexible_masks[:self.p_natoms]=0
            
        elif p_mode=='fix-backbone':
            c_fix_masks[:self.p_natoms]=self.p_backbone_labels
            c_flexible_masks[:self.p_natoms]=1-self.p_backbone_labels
            
        elif p_mode=='fix-nonint':
            for i in range(self.p_natoms):
                if not (p_group_atom_int_masks[i] and self.p_backbone_labels[i]!=1):
                    c_fix_masks[i]=1
                    c_flexible_masks[i]=0
        else:
            pass
        
        c_fix_masks[self.p_natoms:max_patoms]=0
        c_flexible_masks[self.p_natoms:max_patoms]=0
        
        if l_mode=='fix-previous':
            c_fix_masks[max_patoms:max_patoms+self.l_natoms]=l_fixmasks
            c_flexible_masks[max_patoms:max_patoms+self.l_natoms]=(1-l_fixmasks)*l_gmasks
        else:
            c_flexible_masks[max_patoms:max_patoms+self.l_natoms]=l_gmasks
            
        c_fix_masks[max_patoms+self.l_natoms:]=0
        c_flexible_masks[max_patoms+self.l_natoms:]=0
        
        p_zb,p_za,p_zd=self.p_zb,self.p_za,self.p_zd
        p_nbonds,p_nangles,p_ndihedrals=self.p_nbonds,self.p_nangles,self.p_ndihedrals
        
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
        
        if max_pbonds is None:
            max_pbonds=self.p_nbonds
        if max_pangles is None:
            max_pangles=self.p_nangles
        if max_pdihedrals is None:
            max_pdihedrals=self.p_ndihedrals
            
        p_zbs=torch.zeros((max_pbonds,2)).long()
        p_zas=torch.zeros((max_pangles,3)).long()
        p_zds=torch.zeros((max_pdihedrals,4)).long()
        p_zbmask=torch.zeros(max_pbonds).long()
        p_zamask=torch.zeros(max_pangles).long()
        p_zdmask=torch.zeros(max_pdihedrals).long()

        c_pocket_labels=torch.zeros(max_catoms).long()
        c_ligand_labels=torch.zeros(max_catoms).long()
        
        c_fix_masks=torch.from_numpy(c_fix_masks)
        c_flexible_masks=torch.from_numpy(c_flexible_masks)
        
        l_zbs[:l_nbonds]=torch.from_numpy(l_zb)
        l_zas[:l_nangles]=torch.from_numpy(l_za)
        l_zds[:l_ndihedrals]=torch.from_numpy(l_zd)
        l_zbmask[:l_nbonds]=1
        l_zamask[:l_nangles]=1
        l_zdmask[:l_ndihedrals]=1
        
        p_zbs[:p_nbonds]=torch.from_numpy(p_zb)
        p_zas[:p_nangles]=torch.from_numpy(p_za)
        p_zds[:p_ndihedrals]=torch.from_numpy(p_zd)
        p_zbmask[:p_nbonds]=1
        p_zamask[:p_nangles]=1
        p_zdmask[:p_ndihedrals]=1        

        c_pocket_labels[:self.p_natoms]=1
        c_ligand_labels[max_patoms:max_patoms+self.l_natoms]=torch.from_numpy(l_gmasks).long()

        stepid=torch.Tensor([stepid]).long()

        return  c_atoms,c_atoms_onehot,c_fcs,c_fcs_onehot,c_adjs,c_adjs_mat,c_coords,c_masks,c_fix_masks,c_flexible_masks,\
                l_zbs,l_zas,l_zds,l_zbmask,l_zamask,l_zdmask,\
                p_zbs,p_zas,p_zds,p_zbmask,p_zamask,p_zdmask,c_pocket_labels,c_ligand_labels,stepid
    
    def gen_l_leaf_graph_states(self):
        nadd_mg_masks,nadd_mg_masks_2D,nadd_pl_masks,nadd_apds=[],[],[],[]
        rgen_mg_masks,rgen_mg_masks_2D,rgen_rg_masks,rgen_rg_masks_2D,rgen_pl_masks,rgen_ftypes,rgen_apds,rgen_focus_groups=[],[],[],[],[],[],[],[]
        nconn_mg_masks,nconn_mg_masks_2D,nconn_rg_masks,nconn_rg_masks_2D,nconn_pl_masks,nconn_focus_ids,nconn_apds,nconn_focus_groups=[],[],[],[],[],[],[],[]
        nint_mg_gmasks,nint_mg_gmasks_2D,nint_mg_dmasks,nint_mg_dmasks_2D,nint_pl_gmasks,nint_pl_dmasks,nint_focus_groups,nint_focus_ftypes,nint_apds=[],[],[],[],[],[],[],[],[]
        rgen_step_id=0
        for i in range(self.l_ngroups):
            graph_atoms=np.array(Flatten(list(self.l_groups[:i])))
            #print (i,graph_atoms)
            
            l_masks=np.zeros(self.l_natoms)
            if i>0:
                l_masks[np.ix_(graph_atoms)]=1
            #print (i,l_masks)
            
            l_masks_2D=l_masks.reshape(-1,1)*l_masks.reshape(1,-1)
            lp_groups_masks=np.zeros((self.l_ngroups,self.p_ngroups))
            lp_groups_masks[:i]=1
            fid=self.get_l_group_fid(gid=i)
            
            APD_add=np.zeros(FGP.leaf_add_dim)
            APD_add[fid]=1
            
            nadd_mg_masks.append(np.array(l_masks))
            nadd_mg_masks_2D.append(np.array(l_masks_2D))
            nadd_pl_masks.append(np.array(lp_groups_masks.T))
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
                        
                    lp_groups_masks=np.zeros((self.l_ngroups,self.p_ngroups))
                    lp_groups_masks[:i]=1
                    r_add[conn_atom_id,atype_id,fc_id,bond_type_id]=1
                    r_add=r_add.reshape(-1)
                    r_conn=r_conn.reshape(-1)
                    r_term=r_term.reshape(-1)
                    #print (i,rgen_step_id,l_masks)
                    rgen_mg_masks.append(np.array(l_masks))
                    rgen_mg_masks_2D.append(np.array(l_masks_2D))
                    rgen_rg_masks.append(np.array(r_masks))
                    rgen_rg_masks_2D.append(np.array(r_masks_2D))
                    rgen_pl_masks.append(np.array(lp_groups_masks.T))
                    rgen_focus_groups.append(i)
                    rgen_ftypes.append(f_node)
                    
                    APD_ring=np.concatenate((r_add,r_conn,r_term))

                    rgen_apds.append(APD_ring)
                    rgen_step_id+=1

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
                            #print (self.l_groups_inner_adjs[i])
                            #print (rgen_step_id,ii,remain_conn_atoms[jj],bond_type_id,self.l_groups_inner_adjs[i][ii,remain_conn_atoms[jj]])
                            r_conn[remain_conn_atoms[jj],bond_type_id]=1
                            r_add=r_add.reshape(-1)
                            r_conn=r_conn.reshape(-1)
                            r_term=r_term.reshape(-1)
                            
                            lp_groups_masks=np.zeros((self.l_ngroups,self.p_ngroups))
                            lp_groups_masks[:i]=1
                            #print (i,rgen_step_id,l_masks)
                            rgen_mg_masks.append(np.array(l_masks))
                            rgen_mg_masks_2D.append(np.array(l_masks_2D))
                            rgen_pl_masks.append(np.array(lp_groups_masks.T))
                            rgen_rg_masks.append(np.array(r_masks))
                            rgen_rg_masks_2D.append(np.array(r_masks_2D))
                            rgen_focus_groups.append(i)
                            rgen_ftypes.append(f_node)
                            APD_ring=np.concatenate((r_add,r_conn,r_term))
                            rgen_apds.append(APD_ring)
                            rgen_step_id+=1
            
                r_add=np.zeros(FGP.r_add_dim)
                r_conn=np.zeros(FGP.r_conn_dim)
                r_term=np.ones(1)
                r_add=r_add.reshape(-1)
                r_conn=r_conn.reshape(-1)
                r_term=r_term.reshape(-1)
                r_masks=np.ones(len(self.l_groups[i])) 
                r_masks_2D=np.ones((len(self.l_groups[i]),len(self.l_groups[i])))
                lp_groups_masks=np.zeros((self.l_ngroups,self.p_ngroups))
                lp_groups_masks[:i]=1
                #print (i,rgen_step_id,l_masks)
                rgen_mg_masks.append(np.array(l_masks))
                rgen_mg_masks_2D.append(np.array(l_masks_2D))
                rgen_pl_masks.append(np.array(lp_groups_masks.T))
                rgen_rg_masks.append(np.array(r_masks))
                rgen_rg_masks_2D.append(np.array(r_masks_2D))
                rgen_focus_groups.append(i)
                rgen_ftypes.append(f_node)
                APD_ring=np.concatenate((r_add,r_conn,r_term))
                rgen_apds.append(APD_ring)
                rgen_step_id+=1
            else:
                r_masks=np.zeros(len(self.l_groups[i]))
                r_masks[:1]=1
                r_masks_2D=r_masks.reshape(-1,1)*r_masks.reshape(1,-1)
                
            if i>0:
                APD_conn=np.zeros(FGP.leaf_conn_dim) 
                conn_atom_id=np.where(self.l_adjs[self.l_groups[i][0],:self.l_groups[i][0]]>0)[0][-1]
                bond_type_id=int(self.l_adjs[self.l_groups[i][0],conn_atom_id])-1

                APD_conn[conn_atom_id,bond_type_id]=1
                nconn_mg_masks.append(np.array(l_masks))
                nconn_mg_masks_2D.append(np.array(l_masks_2D))
                nconn_focus_groups.append(i)
                lp_groups_masks=np.zeros((self.l_ngroups,self.p_ngroups))
                lp_groups_masks[:i]=1
                nconn_pl_masks.append(np.array(lp_groups_masks.T))
                nconn_rg_masks.append(np.array(r_masks))
                nconn_rg_masks_2D.append(np.array(r_masks_2D))
                nconn_focus_ids.append(np.array(0))
                nconn_apds.append(APD_conn.reshape(-1))
            
            graph_atoms=np.array(Flatten(list(self.l_groups[:i+1])))
            l_gmasks=np.zeros(self.l_natoms)
            l_gmasks[np.ix_(graph_atoms)]=1
            
            d3_atoms=np.array(Flatten(list(self.l_groups[:i])))
            l_dmasks=np.zeros(self.l_natoms)
            if i>0:
                l_dmasks[np.ix_(d3_atoms)]=1
            
            l_gmasks_2D=l_gmasks.reshape(-1,1)*l_gmasks.reshape(1,-1)
            l_dmasks_2D=l_dmasks.reshape(-1,1)*l_dmasks.reshape(1,-1)
            
            lp_groups_gmasks=np.zeros((self.l_ngroups,self.p_ngroups))
            lp_groups_gmasks[:i]=1
            
            lp_groups_dmasks=np.zeros((self.l_ngroups,self.p_ngroups))
            lp_groups_dmasks[:i]=1
            
            pint_groups=np.where(self.lp_groups_adjs[i]>0)[0]
            if len(pint_groups)>0:
                for pgid in pint_groups:
                    leaf_int_add=np.zeros(FGP.leaf_int_add_dim)
                    leaf_int_term=np.zeros(1)
                    int_idx=int(self.lp_atom_adjs[self.l_groups[i][0],self.p_groups[pgid][0]]-5)
                    leaf_int_add[pgid,int_idx]=1 
                    leaf_int_add=leaf_int_add.reshape(-1)
                    APD_int=np.concatenate((leaf_int_add,leaf_int_term))
                    
                    nint_mg_gmasks.append(np.array(l_gmasks))
                    nint_mg_gmasks_2D.append(np.array(l_gmasks_2D))
                    nint_mg_dmasks.append(np.array(l_dmasks))
                    nint_mg_dmasks_2D.append(np.array(l_dmasks_2D))
                    nint_pl_dmasks.append(np.array(lp_groups_dmasks.T))
                    nint_pl_gmasks.append(np.array(lp_groups_gmasks.T))
                    focus_groups=i
                    nint_focus_groups.append(focus_groups)
                    nint_focus_ftypes.append(f_node)
                    nint_apds.append(APD_int)
                    lp_groups_gmasks[i,:pgid+1]=1
                    
                leaf_int_add=np.zeros(FGP.leaf_int_add_dim)
                leaf_int_term=np.ones(1)
                leaf_int_add=leaf_int_add.reshape(-1)
                APD_int=np.concatenate((leaf_int_add,leaf_int_term))
                nint_mg_gmasks.append(np.array(l_gmasks))
                nint_mg_gmasks_2D.append(np.array(l_gmasks_2D))
                
                nint_mg_dmasks.append(np.array(l_dmasks))
                nint_mg_dmasks_2D.append(np.array(l_dmasks_2D))
                
                nint_pl_dmasks.append(np.array(lp_groups_dmasks.T))
                nint_pl_gmasks.append(np.array(lp_groups_gmasks.T))
                
                focus_groups=i
                nint_focus_groups.append(focus_groups)
                nint_focus_ftypes.append(f_node)
                nint_apds.append(APD_int)
            else:
                leaf_int_add=np.zeros(FGP.leaf_int_add_dim)
                leaf_int_term=np.ones(1)
                leaf_int_add=leaf_int_add.reshape(-1)
                
                APD_int=np.concatenate((leaf_int_add,leaf_int_term))
                nint_mg_gmasks.append(np.array(l_gmasks))
                nint_mg_gmasks_2D.append(np.array(l_gmasks_2D))
                nint_mg_dmasks.append(np.array(l_dmasks))
                nint_mg_dmasks_2D.append(np.array(l_dmasks_2D))
                nint_pl_dmasks.append(np.array(lp_groups_dmasks.T))
                nint_pl_gmasks.append(np.array(lp_groups_gmasks.T))
                focus_groups=i
                nint_focus_groups.append(focus_groups)
                nint_focus_ftypes.append(f_node)
                nint_apds.append(APD_int)
                
        nadd_mg_masks.append(np.ones(self.l_natoms))
        nadd_mg_masks_2D.append(np.ones((self.l_natoms,self.l_natoms)))
        lp_groups_masks=np.zeros((self.l_ngroups,self.p_ngroups))
        lp_groups_masks[:self.l_ngroups]=1
        nadd_pl_masks.append(np.array(lp_groups_masks.T))
        APD_add=np.zeros(FGP.leaf_add_dim)
        APD_add[-1]=1
        
        nadd_apds.append(APD_add)
        self.nadd_lg_states=(nadd_mg_masks,nadd_mg_masks_2D,nadd_pl_masks,nadd_apds)
        self.rgen_lg_states=(rgen_mg_masks,rgen_mg_masks_2D,rgen_rg_masks,rgen_rg_masks_2D,rgen_pl_masks,rgen_focus_groups,rgen_ftypes,rgen_apds)
        self.nconn_lg_states=(nconn_mg_masks,nconn_mg_masks_2D,nconn_rg_masks,nconn_rg_masks_2D,nconn_pl_masks,nconn_focus_groups,nconn_focus_ids,nconn_apds)
        self.nint_lg_states=(nint_mg_gmasks,nint_mg_gmasks_2D,nint_mg_dmasks,nint_mg_dmasks_2D,nint_pl_gmasks,nint_pl_dmasks,nint_focus_groups,nint_focus_ftypes,nint_apds)
        return 
    
    def sample_leaf_nadd_states(self,stepid,max_patoms=None,max_latoms=None,noise_std=0):
        if max_patoms is None:
            max_patoms=self.p_natoms
            
        if max_latoms is None:
            max_latoms=self.l_natoms
        
        l_gmasks=self.nadd_lg_states[0][stepid]
        l_gmasks_2D=self.nadd_lg_states[1][stepid]
        pl_group_masks=self.nadd_lg_states[2][stepid]
        l_apds=self.nadd_lg_states[3][stepid]
        pmc=self.get_pocket_mass_center()
        self.p_coords = self.p_coords - pmc
        self.l_coords = self.l_coords - pmc
        c_atoms,c_atoms_onehot,c_fcs,c_fcs_onehot,c_adjs,c_adjs_mat,c_coords,c_masks=self.gen_masked_c_infos(lmasks=l_gmasks,
                                                                          lmasks_2D=l_gmasks_2D,
                                                                          pl_group_masks=pl_group_masks,
                                                                          max_patoms=max_patoms,
                                                                          max_latoms=max_latoms,
                                                                          noise_std=noise_std)
        
        l_apds_tensor=torch.from_numpy(l_apds)
        
        return  c_atoms,c_atoms_onehot,c_fcs,c_fcs_onehot,c_adjs,c_adjs_mat,c_coords,c_masks,l_apds_tensor
                

    def sample_leaf_rgen_states(self,stepid,max_patoms=None, max_latoms=None, max_ringsize=None,noise_std=0):
        if max_patoms is None:
            max_patoms=self.p_natoms
        if max_latoms is None:
            max_latoms=self.l_natoms
        l_gmasks=self.rgen_lg_states[0][stepid]
        #print (l_gmasks) 
        l_gmasks_2D=self.rgen_lg_states[1][stepid]
        r_masks=self.rgen_lg_states[2][stepid]
        r_masks_2D=self.rgen_lg_states[3][stepid]
        pl_group_masks=self.rgen_lg_states[4][stepid]
        focus_group_idx=self.rgen_lg_states[5][stepid]
        r_feats=self.rgen_lg_states[6][stepid]
        r_apds=self.rgen_lg_states[7][stepid]
        pmc=self.get_pocket_mass_center()
        self.p_coords = self.p_coords - pmc
        self.l_coords = self.l_coords - pmc
        c_atoms,c_atoms_onehot,c_fcs,c_fcs_onehot,c_adjs,c_adjs_mat,c_coords,c_masks=\
                                                    self.gen_masked_c_infos(lmasks=l_gmasks,
                                                                            lmasks_2D=l_gmasks_2D,
                                                                            pl_group_masks=pl_group_masks,
                                                                            max_patoms=max_patoms,
                                                                            max_latoms=max_latoms,
                                                                            noise_std=noise_std)
        r_atoms,r_atoms_onehot,r_fcs,r_fcs_onehot,r_adjs,r_adjs_mat,r_coords,r_masks=\
                                                    self.gen_masked_r_infos(gid=focus_group_idx,
                                                                            rmasks=r_masks,
                                                                            rmasks_2D=r_masks_2D,
                                                                            max_ratoms=max_ringsize)
        r_feats=torch.from_numpy(r_feats)
        r_apds=torch.from_numpy(r_apds)
        
        return c_atoms,c_atoms_onehot,c_fcs,c_fcs_onehot,c_adjs,c_adjs_mat,c_coords,c_masks,\
                r_atoms,r_atoms_onehot,r_fcs,r_fcs_onehot,r_adjs,r_adjs_mat,r_coords,r_masks,\
                r_feats,r_apds
    
    def sample_leaf_nconn_states(self,stepid,max_patoms=None, max_latoms=None, max_ringsize=None,noise_std=0):
        
        if max_patoms is None:
            max_patoms=self.p_natoms
        if max_latoms is None:
            max_latoms=self.l_natoms
        
        l_gmasks=self.nconn_lg_states[0][stepid]
        l_gmasks_2D=self.nconn_lg_states[1][stepid]
        r_masks=self.nconn_lg_states[2][stepid]
        r_masks_2D=self.nconn_lg_states[3][stepid]
        pl_group_masks=self.nconn_lg_states[4][stepid]
        focus_group_idx=self.nconn_lg_states[5][stepid]
        focus_atom_idx=self.nconn_lg_states[6][stepid]
        l_apds=self.nconn_lg_states[7][stepid].reshape(-1)
        pmc=self.get_pocket_mass_center()
        self.p_coords = self.p_coords - pmc
        self.l_coords = self.l_coords - pmc
        c_atoms,c_atoms_onehot,c_fcs,c_fcs_onehot,c_adjs,c_adjs_mat,c_coords,c_masks=\
                                                    self.gen_masked_c_infos(lmasks=l_gmasks,
                                                                            lmasks_2D=l_gmasks_2D,
                                                                            pl_group_masks=pl_group_masks,
                                                                            max_patoms=max_patoms,
                                                                            max_latoms=max_latoms,
                                                                            noise_std=noise_std)
                                                                            
        r_atoms,r_atoms_onehot,r_fcs,r_fcs_onehot,r_adjs,r_adjs_mat,r_coords,r_masks=\
                                                    self.gen_masked_r_infos(
                                                                            gid=focus_group_idx,
                                                                            rmasks=r_masks,
                                                                            rmasks_2D=r_masks_2D,
                                                                            max_ratoms=max_ringsize)
        #print (focus_atom_idx) 
        focus_atom_idx=torch.Tensor(np.array([focus_atom_idx])).long()
        l_apds=torch.from_numpy(l_apds)
        
        return  c_atoms,c_atoms_onehot,c_fcs,c_fcs_onehot,c_adjs,c_adjs_mat,c_coords,c_masks,\
                r_atoms,r_atoms_onehot,r_fcs,r_fcs_onehot,r_adjs,r_adjs_mat,r_coords,r_masks,\
                focus_atom_idx,l_apds
    
    def sample_leaf_nint_states(self,stepid,max_patoms=None,max_latoms=None,max_pgroups=None,max_lgroups=None,max_ringsize=None,max_pgsize=None,noise_std=0):
        if max_patoms is None:
            max_patoms=self.p_natoms
        if max_latoms is None:
            max_latoms=self.l_natoms
        if max_lgroups is None:
            max_lgroups=self.l_ngroups
        if max_pgroups is None:
            max_pgroups=self.p_ngroups
        if max_ringsize is None:
            max_ringsize=np.max([len(self.l_groups[i]) for i in range(self.l_ngroups)])
        if max_pgsize is None:
            max_pgsize=np.max([len(self.p_groups[i]) for i in range(self.p_ngroups)])
        
        pmc=self.get_pocket_mass_center()
        self.p_coords = self.p_coords - pmc
        self.l_coords = self.l_coords - pmc
        l_gmasks=self.nint_lg_states[0][stepid]
        l_gmasks_2D=self.nint_lg_states[1][stepid]
        l_dmasks=self.nint_lg_states[2][stepid]
        l_dmasks_2D=self.nint_lg_states[3][stepid]
        pl_group_gmasks=self.nint_lg_states[4][stepid]
        pl_group_dmasks=self.nint_lg_states[5][stepid]
        focus_group_idx=self.nint_lg_states[6][stepid]
        focus_ftypes=self.nint_lg_states[7][stepid]
        l_apds=self.nint_lg_states[8][stepid]

        cg_atoms,cg_atoms_onehot,cg_fcs,cg_fcs_onehot,cg_adjs,cg_adjs_mat,cg_coords,cg_masks=\
                                                    self.gen_masked_c_infos(lmasks=l_gmasks,
                                                                            lmasks_2D=l_gmasks_2D,
                                                                            pl_group_masks=pl_group_gmasks,
                                                    
                                                                            max_patoms=max_patoms,
                                                                            max_latoms=max_latoms,
                                                                            noise_std=noise_std)
                                                                            
        cd_atoms,cd_atoms_onehot,cd_fcs,cd_fcs_onehot,cd_adjs,cd_adjs_mat,cd_coords,cd_masks=\
                                                    self.gen_masked_c_infos(lmasks=l_dmasks,
                                                                            lmasks_2D=l_dmasks_2D,
                                                                            pl_group_masks=pl_group_dmasks,
                                                                            max_patoms=max_patoms,
                                                                            max_latoms=max_latoms,
                                                                            noise_std=noise_std)
                                                    
        p_groups,p_groups_masks=self.padding_pgroups_tensors(max_pgroups=max_pgroups,max_pgsize=max_pgsize)
        
        p_int_groups_masks=np.zeros(max_pgroups)
        p_int_groups_masks_=torch.Tensor(self.get_p_group_int_masks())
        p_int_groups_masks[:self.p_ngroups]=p_int_groups_masks_.long()
        
        focus_group_idx=torch.Tensor([focus_group_idx]).long()
        focus_lgroups=np.zeros(max_ringsize)
        focus_lgroups_masks=np.zeros(max_ringsize)
        focus_lgroups[:len(self.l_groups[focus_group_idx])]=self.l_groups[focus_group_idx]
        focus_lgroups_masks[:len(self.l_groups[focus_group_idx])]=1
        focus_lgroups=torch.from_numpy(focus_lgroups)
        focus_lgroups_masks=torch.from_numpy(focus_lgroups_masks)
        focus_ftypes=torch.from_numpy(focus_ftypes)
        l_apds=torch.from_numpy(l_apds)
        
        return  cg_atoms,cg_atoms_onehot,cg_fcs,cg_fcs_onehot,cg_adjs,cg_adjs_mat,cg_coords,cg_masks,\
                cd_atoms,cd_atoms_onehot,cd_fcs,cd_fcs_onehot,cd_adjs,cd_adjs_mat,cd_coords,cd_masks,\
                focus_lgroups,focus_lgroups_masks,p_groups,p_groups_masks,p_int_groups_masks,focus_ftypes,l_apds
    
    def gen_masked_r_infos(self,gid,rmasks,rmasks_2D,max_ratoms=None):
        if max_ratoms is None:
            max_ratoms=len(self.l_groups[gid])
        r_atoms,r_atoms_onehot,r_fcs,r_fcs_onehot=self.padding_group_feat_tensors(gid=gid,rmasks=rmasks,max_ratoms=max_ratoms)
        r_adjs,r_adjs_mat=self.padding_group_adjs_tensors(gid=gid,rmasks_2D=rmasks_2D,max_ratoms=max_ratoms)
        r_coords=self.padding_group_coords_tensors(gid=gid,rmasks=rmasks,max_ratoms=max_ratoms)
        r_masks=np.zeros(max_ratoms)
        r_masks[:len(self.l_groups[gid])]=rmasks
        return r_atoms,r_atoms_onehot,r_fcs,r_fcs_onehot,r_adjs,r_adjs_mat,r_coords,r_masks
    
    def gen_masked_l_infos(self,lmasks,lmasks_2D,max_latoms=None):
        if max_latoms is None:
            max_latoms=self.l_natoms
        l_atoms,l_atoms_onehot,l_fcs,l_fcs_onehot=self.padding_ligand_feat_tensors(lmasks=lmasks,max_latoms=max_latoms)
        l_adjs,l_adjs_mat=self.padding_ligand_adjs_tensors(lmasks_2D=lmasks_2D,max_latoms=max_latoms)
        l_coords=self.padding_ligand_coords_tensors(lmasks=lmasks,max_latoms=max_latoms) 
        l_masks=np.zeros(max_latoms)
        l_masks[:self.l_natoms]=lmasks
        l_masks=torch.from_numpy(l_masks)
        return l_atoms,l_atoms_onehot,l_fcs,l_fcs_onehot,l_adjs,l_adjs_mat,l_coords,l_masks
        
    def gen_masked_c_infos(self,lmasks,lmasks_2D,pl_group_masks=None,pl_atom_masks=None,pl_mask_rate=0,max_patoms=None,max_latoms=None,noise_std=0):
        if max_latoms is None:
            max_latoms=self.l_natoms
        if max_patoms is None: 
            max_patoms=self.p_natoms
        
        c_atoms,c_atoms_onehot,c_fcs,c_fcs_onehot=self.concat_complex_feats(lmasks,max_patoms=max_patoms,
                                                                            max_latoms=max_latoms)
        c_adjs,c_adjs_mat=self.concat_complex_adjs_mat(lmasks,lmasks_2D,pl_group_masks=pl_group_masks,pl_atom_masks=pl_atom_masks,pl_mask_rate=pl_mask_rate,max_patoms=max_patoms,max_latoms=max_latoms)
        c_coords=self.concat_complex_coords(lmasks,max_latoms=max_latoms,max_patoms=max_patoms,noise_std=noise_std)
        c_masks=np.zeros(max_patoms+max_latoms)
        c_masks[:self.p_natoms]=1
        c_masks[max_patoms:max_patoms+self.l_natoms]=lmasks
        c_masks=torch.from_numpy(c_masks)
        return c_atoms,c_atoms_onehot,c_fcs,c_fcs_onehot,c_adjs,c_adjs_mat,c_coords,c_masks

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
        
    def padding_ligand_coords_tensors(self,lmasks=None,max_latoms=None) :
        if max_latoms is None:
            max_latoms=self.l_natoms
        if lmasks is None:
            lmasks=np.ones(self.l_natoms)
            
        l_coords_=np.zeros((max_latoms,3))
        l_coords_[:self.l_natoms]=self.l_coords*lmasks.reshape(-1,1)
        l_coords=torch.from_numpy(l_coords_)
        return l_coords
    
    def padding_pgroups_tensors(self,max_pgroups=None,max_pgsize=None):
        if max_pgroups is None:
            max_pgroups=self.p_ngroups
        if max_pgsize is None:
            max_pgsize=np.max([len(self.p_groups[i]) for i in range(self.p_ngroups)])
            
        p_groups_=np.zeros((max_pgroups,max_pgsize))
        p_groups_masks=np.zeros((max_pgroups,max_pgsize))
        #print (self.protein_pdb,np.max([len(g) for g in self.p_groups])) 
        for i in range(self.p_ngroups):
            p_groups_[i,:len(self.p_groups[i])]=self.p_groups[i]
            p_groups_masks[i,:len(self.p_groups[i])]=1
        
        p_groups=torch.from_numpy(p_groups_)
        p_groups_masks=torch.from_numpy(p_groups_masks)
        
        return p_groups,p_groups_masks    
    
    def concat_complex_feats(self,lmasks=None,max_patoms=None,max_latoms=None,with_ligand=True):

        if max_patoms is None:
            max_patoms=self.p_natoms

        if max_latoms is None:
            max_latoms=self.l_natoms

        if lmasks is None:
            lmasks=np.ones(self.l_natoms)
            
        max_catoms=max_patoms+max_latoms

        c_atoms_=np.zeros(max_catoms)
        c_atoms_[:self.p_natoms]=self.p_atoms
        c_atoms_[max_patoms:max_patoms+self.l_natoms]=self.l_atoms*lmasks
        
        c_atoms_onehot_=np.zeros((max_catoms,len(FGP.atom_types_for_feats)))
        p_atoms_onehot_=Atoms_to_onehot(self.p_atoms,FGP.atom_types_for_feats)

        if with_ligand:
            l_atoms_onehot_=Atoms_to_onehot(self.l_atoms,FGP.atom_types_for_feats)
        
        c_atoms_onehot_[:self.p_natoms]=p_atoms_onehot_
        
        if with_ligand:
            c_atoms_onehot_[max_patoms:max_patoms+self.l_natoms]=l_atoms_onehot_*lmasks.reshape(-1,1)
        
        c_fcs_=np.zeros(max_catoms)
        c_fcs_[:self.p_natoms]=self.p_fcs
        if with_ligand:
            c_fcs_[max_patoms:max_patoms+self.l_natoms]=self.l_fcs*lmasks 

        c_fcs_onehot_=np.zeros((max_catoms,len(FGP.formal_charge_types)))
        p_fcs_onehot_=Atoms_to_onehot(self.p_fcs,FGP.formal_charge_types)
        if with_ligand:
            l_fcs_onehot_=Atoms_to_onehot(self.l_fcs,FGP.formal_charge_types)
        c_fcs_onehot_[:self.p_natoms]=p_fcs_onehot_
        if with_ligand:
            c_fcs_onehot_[max_patoms:max_patoms+self.l_natoms]=l_fcs_onehot_*lmasks.reshape(-1,1)
        
        c_atoms=torch.from_numpy(c_atoms_)
        c_atoms_onehot=torch.from_numpy(c_atoms_onehot_)
        c_fcs=torch.from_numpy(c_fcs_)
        c_fcs_onehot=torch.from_numpy(c_fcs_onehot_)
        
        return c_atoms,c_atoms_onehot,c_fcs,c_fcs_onehot
        
    def concat_complex_adjs_mat(self,lmasks=None,lmasks_2D=None,pl_group_masks=None,pl_atom_masks=None,pl_mask_rate=0,max_patoms=None,max_latoms=None,with_ligand=True):
        if max_patoms is None :
            max_patoms=self.p_natoms
            
        if max_latoms is None:
            max_latoms=self.l_natoms
            
        if lmasks_2D is None:
            lmasks_2D=np.ones((self.l_natoms,self.l_natoms))

        if lmasks is None:
            lmasks=np.ones(self.l_natoms)     
        
        max_catoms=max_patoms+max_latoms
        adjs_=np.zeros((max_catoms,max_catoms))
        adjs_mat_=np.zeros((max_catoms,max_catoms,len(FGP.bond_types)+1))

        # add protein adjs
        adjs_[:self.p_natoms,:self.p_natoms]=self.p_adjs
        
        p_atom_int_masks=self.get_p_atom_int_masks()
        
        if with_ligand:
            # add ligand adjs
            adjs_[max_patoms:max_patoms+self.l_natoms,max_patoms:max_patoms+self.l_natoms]=self.l_adjs*lmasks_2D
            if pl_group_masks is not None:
                #print (self.savepath,self.p_ngroups,self.l_ngroups,pl_group_masks.shape,self.pl_groups_adjs.shape)
                pl_groups_adjs_=self.pl_groups_adjs*pl_group_masks
                
                # add protein-ligand interaction adjs
                pl_groups_adjs_,pl_atom_adjs_=self.mask_pl_atom_adjs(pl_groups_adjs_,pl_mask_rate)
                pl_atom_adjs_=pl_atom_adjs_*p_atom_int_masks.reshape(-1,1)
                adjs_[:self.p_natoms,max_patoms:max_patoms+self.l_natoms]=pl_atom_adjs_
                adjs_[max_patoms:max_patoms+self.l_natoms,:self.p_natoms]=pl_atom_adjs_.T
            else:
                print ('Here we used atom-based adjs')
                assert pl_atom_masks is not None, 'pl_group_masks or pl_atom_masks should be provided!'
                pl_adjs_=self.pl_adjs*pl_atom_masks
                pl_adjs_=self.mask_pl_adjs(pl_adjs_,pl_mask_rate)
                adjs_[:self.p_natoms,max_patoms:max_patoms+self.l_natoms]=pl_adjs_
                adjs_[max_patoms:max_patoms+self.l_natoms,:self.p_natoms]=pl_adjs_.T
                
        idx1,idx2=np.where(adjs_>0)
        for id1,id2 in zip(idx1,idx2):
            adjs_mat_[id1,id2,int(adjs_[id1,id2])]=1

        # add p-p neighbor adjs
        idx1,idx2=np.where(self.pp_adjs>0)
        
        for id1,id2 in zip(idx1,idx2):
            adjs_mat_[id1,id2,0]=1
            
        if with_ligand:
            lp_full_adjs=np.zeros((self.l_natoms,self.p_natoms)) # ligand only connect to protein atoms with interaction labels
            lp_full_adjs[:,p_atom_int_masks]=1
            lp_full_adjs=lp_full_adjs*lmasks.reshape(-1,1)
            
            # add p-l full adjs
            adjs_mat_[:self.p_natoms,max_patoms:max_patoms+self.l_natoms,-1]=lp_full_adjs.T
            adjs_mat_[max_patoms:max_patoms+self.l_natoms,:self.p_natoms,-1]=lp_full_adjs

        adjs=torch.from_numpy(adjs_)
        adjs_mat=torch.from_numpy(adjs_mat_)
        #print (np.where(adjs_mat_>0))
        return adjs,adjs_mat
            
    def concat_complex_coords(self,lmasks=None,max_patoms=None,max_latoms=None,with_ligand=True,noise_std=0):
        if max_patoms is None:
            max_patoms=self.p_natoms
        if max_latoms is None:
            max_latoms=self.l_natoms
        
        max_catoms=max_patoms+max_latoms
        
        c_coords=np.zeros((max_catoms,3))
        c_coords[:self.p_natoms]=self.p_coords
        if with_ligand:
            noise=np.random.normal(0,noise_std,(self.l_natoms,3))
            c_coords[max_patoms:max_patoms+self.l_natoms]=(self.l_coords+noise)*lmasks.reshape(-1,1)
            c_coords=torch.from_numpy(c_coords)
        return c_coords 
    
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
                #print (id1,id2,FGP.bond_types[int(adjs[id1,id2])-1])
                molecule.AddBond(int(id1),int(id2),FGP.bond_types[int(adjs[id1,id2])-1])

        for aid,at in enumerate(molecule.GetAtoms()):
            at.SetFormalCharge(int(fcs[aid]))
        mol=molecule.GetMol()
        #Chem.SanitizeMol(mol)
        AllChem.Compute2DCoords(mol)
        mol=Change_mol_xyz(mol,coords)
        return mol
    
    def Trans_Pocket_to_Mol(self):
        molecule=Chem.RWMol()
        for j in range(self.p_natoms):
            new_atom=Chem.Atom(int(self.p_atoms[j]))
            molecule_idx=molecule.AddAtom(new_atom)
            
        row,col=np.diag_indices_from(self.p_adjs)
        adjs=self.p_adjs.copy()
        adjs[row,col]=0
        idx1,idx2=np.where(adjs!=0)
        for id1,id2 in zip(idx1,idx2):
            if id1<id2:
                molecule.AddBond(int(id1),int(id2),FGP.bond_types[int(adjs[id1,id2])-1])
        
        mol=molecule.GetMol()
        for aid,at in enumerate(mol.GetAtoms()):
            at.SetFormalCharge(int(self.p_fcs[aid]))
        #Chem.SanitizeMol(mol)
        AllChem.Compute2DCoords(mol)
        mol=Change_mol_xyz(mol,self.p_coords)
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
    
    def pocket_tensors_only(self,max_patoms,max_latoms,max_pgroups,max_pgroup_size,p_mode='fix-all'):
        max_catoms=max_patoms+max_latoms
        pmc=self.get_pocket_mass_center()
        self.p_coords = self.p_coords - pmc
        c_atoms,c_atoms_onehot,c_fcs,c_fcs_onehot=self.concat_complex_feats(max_patoms=max_patoms,
                                                                            max_latoms=max_latoms,with_ligand=False)
        c_adjs,c_adjs_mat=self.concat_complex_adjs_mat(max_patoms=max_patoms,max_latoms=max_latoms,with_ligand=False)
        c_coords=self.concat_complex_coords(max_patoms=max_patoms,max_latoms=max_latoms,with_ligand=False)
        c_masks=np.zeros(max_patoms+max_latoms)
        c_masks[:self.p_natoms]=1
        c_masks=torch.from_numpy(c_masks)
        c_gmasks=c_masks.clone().detach()
        c_dmasks=c_masks.clone().detach()
        p_groups,p_groups_masks=self.padding_pgroups_tensors(max_pgroups=max_pgroups,max_pgsize=max_pgroup_size)
        
        p_group_atom_int_masks=self.get_p_group_atom_int_masks()
        
        p_group_int_masks=torch.zeros(max_pgroups)
        p_group_int_masks_=self.get_p_group_int_masks()
        p_group_int_masks[:self.p_ngroups]=torch.from_numpy(p_group_int_masks_)
        
        p_atom_int_masks=torch.zeros(max_patoms)
        p_atom_int_masks_=self.get_p_atom_int_masks()
        p_atom_int_masks[:self.p_natoms]=torch.from_numpy(p_atom_int_masks_)
        
        n_pgroups=torch.Tensor([self.p_ngroups]).long()
        
        c_fix_masks=np.zeros(max_catoms).astype(int)
        c_flexible_masks=np.ones(max_catoms).astype(int)
        
        if p_mode=='fix-all':
            c_fix_masks[:self.p_natoms]=1
            c_flexible_masks[:self.p_natoms]=0
            
        elif p_mode=='fix-backbone':
            c_fix_masks[:self.p_natoms]=self.p_backbone_labels
            c_flexible_masks[:self.p_natoms]=1-self.p_backbone_labels
            
        elif p_mode=='fix-nonint':
            for i in range(self.p_natoms):
                if not (p_group_atom_int_labels[i] and self.p_backbone_labels[i]!=1):
                    c_fix_masks[i]=1
                    c_flexible_masks[i]=0
                    
        else:
            pass
        
        c_fix_masks[self.p_natoms:max_patoms]=0
        c_flexible_masks[self.p_natoms:max_patoms]=0
        c_pocket_labels=np.zeros(max_catoms)
        c_pocket_labels[:self.p_natoms]=1
        

        return c_atoms,c_atoms_onehot,c_fcs,c_fcs_onehot,c_adjs,c_adjs_mat,c_coords,c_masks,c_gmasks,c_dmasks,c_fix_masks,c_flexible_masks,\
                    p_groups, p_groups_masks, p_group_int_masks, p_group_atom_int_masks, p_atom_int_masks, c_pocket_labels, n_pgroups

        

                
                        
                
    






        
            
        
                    
        
            
            
    
