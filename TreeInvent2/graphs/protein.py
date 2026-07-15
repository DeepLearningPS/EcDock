import pickle 
from rdkit import Chem
import numpy as np
import scipy 
from ..comparm import * 
from ..utils.utils_os import * 
import os
from tqdm import tqdm 
from ..utils.utils_interactions import * 
import networkx as nx
from .molecule import *
import random 
from ..utils.utils_graphroute import bfs_seq
from rdkit.Chem import AllChem
from ..utils.utils_rdkit import Change_mol_xyz
import copy
from ..utils.utils_np import Adjs_to_IC_targets
chardict={'A':1,'B':2,'C':3,'D':4,'E':5,'F':6,'G':7,'H':8,'I':9,'J':10,'K':11,'L':12,'M':13,'N':14,'O':15,'P':16,'Q':17,'R':18,'S':19,'T':20,'U':21,'V':22,'W':23,'X':24,'Y':25,'Z':26}
def preprocess_pdb(PDBFileName):
    PDBlines=[]
    with open(PDBFileName,'r') as f:
        line=f.readline()
        while line:
            if line[:6]=="ATOM  " and (line[16]==' ' or line[16]=='A'):
                PDBlines.append(line)
            if line[:6]=="HETATM":
                PDBlines.append(line)
            line=f.readline()
    with open(PDBFileName.strip('.pdb')+'_preprocess.pdb','w') as f:
        for line in PDBlines:
            f.write(line)
    os.system(f"babel {PDBFileName.strip('.pdb')+'_preprocess.pdb'} -o pdb {PDBFileName.strip('.pdb')+'_dnH.pdb'} --DelNonPolarH")
    return PDBFileName.strip('.pdb')+'_dnH.pdb'

def ligsdf_to_pdb(SDFFileName):
    supp=Chem.rdmolfiles.SDMolSupplier(SDFFileName)
    for mid,mol in tqdm (enumerate(supp)):
        if mol:
            if mid>0:
                PDBFileName=SDFFileName[:-4]+'_{mid}.pdb'
            else:
                PDBFileName=SDFFileName[:-4]+'.pdb'
            Chem.MolToPDBFile(mol,PDBFileName)
        else:
            print (f'Invalid {mid}th Mol in {SDFFileName}')
    return  

Standardresidue=['ALA','GLY','PRO','GLN','ASP','ASH','ARG','LYS','ILE','VAL','PHE','MET','CYS','CYM','CYX','HIE','LEU','TRP','TYR','SER','ASN','GLU','GLH','THR','HIS','HID']

def add_nitrogen_charges(m):
    m.UpdatePropertyCache(strict=False)
    ps = Chem.DetectChemistryProblems(m)
    if not ps:
        Chem.SanitizeMol(m)
        return m
    for p in ps:
        if p.GetType()=='AtomValenceException':
            at = m.GetAtomWithIdx(p.GetAtomIdx())
            if at.GetAtomicNum()==7 and at.GetFormalCharge()==0 and at.GetExplicitValence()==4:
                at.SetFormalCharge(1)
    Chem.SanitizeMol(m)
    return m

class Protein:
    def __init__(self,PDBFileName,source='PDBBind-2017'):
        #pdbdict=rscbpdb(filepath=PDBFileName)
        pro_mol=Chem.MolFromPDBFile(PDBFileName,removeHs=False,sanitize=False)
        
        #pro_mol=add_nitrogen_charges(pro_mol)
        Chem.Kekulize(pro_mol)
        
        self.atoms=[atom.GetAtomicNum() for atom in pro_mol.GetAtoms()] 
        self.formal_charges=np.array([atom.GetFormalCharge() for atom in pro_mol.GetAtoms()])
        self.atom_name=[atom.GetPDBResidueInfo().GetName().strip() for atom in pro_mol.GetAtoms()]
        self.atom_rname=[atom.GetPDBResidueInfo().GetResidueName().strip() for atom in pro_mol.GetAtoms()]
        if source!='PDBBind-2017':
            self.atom_rid=[(atom.GetPDBResidueInfo().GetResidueNumber(),0) for atom in pro_mol.GetAtoms()]
        else:
            self.atom_rid=[]
            with open(PDBFileName,'r') as f:
                lines=f.readlines()
                for line in lines:
                    if line[:6]=='ATOM  ' or line[:6]=='HETATM':
                        rid=line[22:27].strip()
                        try:
                            idx1=int(rid)
                            idx2=0
                        except:
                            idx1=int(rid[:-1])
                            idx2=chardict[rid[-1]]
                    
                        self.atom_rid.append((idx1,idx2))
        #print (len(self.atom_rid),len(self.atom_rname))
        #print (self.atom_rid,self.atom_rname)
        assert len(self.atom_rid) == len(self.atom_rname), "Length of Atom Resid != its of Atom ResName"
        #print (self.atom_rid)
        self.atom_cid=[atom.GetPDBResidueInfo().GetChainId() for atom in pro_mol.GetAtoms()]
        self.nchains=len(self.atom_cid)
        #print (self.atom_cid)
        self.coords=np.array(pro_mol.GetConformer(0).GetPositions())
        self.natoms=len(self.atoms)
        self.fix_labels=np.zeros(self.natoms)
        self.backbone_labels=np.zeros(self.natoms)
        self.atom_gids=np.zeros(self.natoms)
        self.respt_before=[0]
        
        for i in range(1,self.natoms):
            #print (self.atom_rid[i],self.atom_rid[i-1],self.atom_rid[i]!=self.atom_rid[i-1])
            if self.atom_rid[i]!=self.atom_rid[i-1]:
                self.respt_before.append(i)

        self.respt_after=[]
        for i in self.respt_before[1:]:
            self.respt_after.append(i)
        #print (self.respt_before)
        self.respt_after.append(self.natoms)
        self.nres=len(self.respt_before)
        self.atom_real_rid=[]
        for i in range(self.nres):
            for j in range(self.respt_before[i],self.respt_after[i]):
                self.atom_real_rid.append(i)
        
        self.resname=[self.atom_rname[i] for i in self.respt_before]
        self.resid=[self.atom_real_rid[i] for i in self.respt_before]
        self.rtype=[]

        for i in range(self.nres):
            if self.resname[i] in Standardresidue:
                self.rtype.append('P')
            else:
                self.rtype.append('C')
        
        self.cpt_before=[0]
        self.cpt_after=[]
        #print (self.respt_before,self.respt_after)
        for i in range(1,self.nres):
            if self.atom_cid[self.respt_before[i]]!=self.atom_cid[self.respt_after[i-1]-1]:
                #print (i,self.respt_before[i],self.atom_cid[self.respt_before[i]],self.respt_after[i-1],self.atom_cid[self.respt_after[i-1]-1])
                self.cpt_before.append(i)
            else:
                if self.atom_rid[self.respt_before[i]][0]>self.atom_rid[self.respt_after[i-1]-1][0]+1:
                    self.cpt_before.append(i)
                
        for i in self.cpt_before[1:]:
            self.cpt_after.append(i)
        self.cpt_after.append(self.nres)
        self.nchains=len(self.cpt_before)

        groups=[[]]
        for i in range(self.natoms):
            if self.atom_rname[i] in ['PRO','ACE','NME']:
                self.backbone_labels[i]=1
                self.fix_labels[i]=1
            else:
                if self.atom_name[i] in ['N','C','O','CA','H','HA']:
                    self.backbone_labels[i]=1
                    if self.atom_name[i] in ['N','CA','C']:
                        self.fix_labels[i]=1

        tmp_=np.zeros(self.natoms)
        for i in range(self.nres):
            if len(groups[-1])>0:
                groups.append([])
            for j in range(self.respt_before[i],self.respt_after[i]):
                if self.backbone_labels[j]==0:
                    groups[-1].append(j)
                    tmp_[j]=1
                if (self.atom_name[j]=='CA' or 'HA' in self.atom_name[j]) and self.atom_rname[j] not in ['PRO', 'ACE', 'NME']:
                    groups[-1].append(j)
                    tmp_[j]=1

        if len(groups[-1])==0:
            groups.pop()

        #for i in range(len(groups)):
        #    aid=groups[i][0]
        #    print (i,groups[i],[self.atom_name[j] for j in groups[i]],self.atom_rname[aid],self.atom_rid[aid],self.atom_cid[aid])
        #print ('remain after sidechain groups',np.sum(1-tmp_))

        backbone_groups=[] 

        for c in range(self.nchains):
            #print (self.cpt_before[c],self.cpt_after[c])
            for i in range(self.cpt_before[c],self.cpt_after[c]):
                if i==self.cpt_before[c]:
                    backbone_groups.append([])
                    for j in range(self.respt_before[i],self.respt_after[i]):
                        if (self.atom_name[j]!='C' and self.atom_name[j]!='O') and self.backbone_labels[j]==1 and tmp_[j]==0:
                            backbone_groups[-1].append(j)                
                            tmp_[j]=1
                else:
                    if i!=self.cpt_before[c]+1:
                        backbone_groups.append([])
                    for j in range(self.respt_before[i-1],self.respt_after[i-1]):
                        if self.backbone_labels[j]==1 and tmp_[j]==0:
                            backbone_groups[-1].append(j)
                            tmp_[j]=1

                    if self.resname[i]=='PRO':
                        for j in range(self.respt_before[i],self.respt_after[i]):
                            if (self.atom_name[j]!='C' and self.atom_name[j]!='O') and self.backbone_labels[j]==1 and tmp_[j]==0:
                                backbone_groups[-1].append(j)
                                tmp_[j]=1
                    else:
                        for j in range(self.respt_before[i],self.respt_after[i]):
                            if (self.atom_name[j]=='N' or self.atom_name[j]=='H') and self.backbone_labels[j]==1 and tmp_[j]==0:
                                backbone_groups[-1].append(j)
                                tmp_[j]=1
            for j in range(self.respt_before[i],self.respt_after[i]):
                if self.backbone_labels[j]==1 and tmp_[j]==0:
                    backbone_groups[-1].append(j)
                    tmp_[j]=1

        #for i in range(len(backbone_groups)):
        #    print (i,backbone_groups[i],[self.atom_name[j] for j in backbone_groups[i]],self.atom_rname[backbone_groups[i][0]],self.atom_rid[backbone_groups[i][0]],self.atom_cid[groups[-1][0]])
        
        self.groups=groups+backbone_groups
        self.groups=[group for group in self.groups if len(group)>0]

        #for group in self.groups:
        #    print (group)

        self.ngroups=len(self.groups)
        self.adjs=np.zeros((self.natoms,self.natoms))
        for bond in pro_mol.GetBonds():
            a1=bond.GetBeginAtom().GetIdx()
            a2=bond.GetEndAtom().GetIdx()
            bt=bond.GetBondType() 
            ch=FGP.bond_types.index(bt)
            self.adjs[a1,a2]=ch+1
            self.adjs[a2,a1]=ch+1

        #print (self.groups[0])
        for gid,group in enumerate(self.groups):
            for i in group:
                self.atom_gids[i]=int(gid)
                
        self.atom_gids=np.array(self.atom_gids)
        return 
    def write_group_info(self,fname='./group.info'):
        with open(fname,'w') as f:
            for i in range(self.ngroups):
                f.write(f'---group {i}\n')
                for j in self.groups[i]:
                    f.write(f'{self.atoms[j]},{self.atom_name[j]},{self.atom_rname[j]},{self.atom_rid[j]}\n')
        


        
        
            

            


        
    


        
        
        
        


                
                        
                
    






        
            
        
                    
        
            
            
    
