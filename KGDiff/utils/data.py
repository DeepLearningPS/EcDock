import os
import numpy as np
from rdkit import Chem
from rdkit.Chem.rdchem import BondType
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
import copy
from rdkit.Chem import AllChem
import torch
from rdkit.Geometry.rdGeometry import Point3D


from EcDock.utils.utils_np import *
from EcDock.comparm import *
import networkx as nx
from EcDock.utils.utils_graphroute import *
from EcDock.utils.utils_rdkit import *
import random 
from rdkit.Chem import rdDistGeom
import dill
from collections import defaultdict
from ordered_set import OrderedSet
from biopandas.pdb import PandasPdb
import pandas as pd

#from CFM.m6_optim_rdkit import get_lig_graph_with_matching
#from CFM.cfm_comparm import CFMGP


ATOM_FAMILIES = ['Acceptor', 'Donor', 'Aromatic', 'Hydrophobe', 'LumpedHydrophobe', 'NegIonizable', 'PosIonizable',
                 'ZnBinder']
ATOM_FAMILIES_ID = {s: i for i, s in enumerate(ATOM_FAMILIES)}
BOND_TYPES = {
    BondType.UNSPECIFIED: 0,
    BondType.SINGLE: 1,
    BondType.DOUBLE: 2,
    BondType.TRIPLE: 3,
    BondType.AROMATIC: 4,
}
BOND_NAMES = {v: str(k) for k, v in BOND_TYPES.items()}
HYBRIDIZATION_TYPE = ['S', 'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2']
HYBRIDIZATION_TYPE_ID = {s: i for i, s in enumerate(HYBRIDIZATION_TYPE)}


def Change_Mol_D3coord(inputmol,coords):
    '''
        ，inputmol，coords：
    '''
    molobj=copy.deepcopy(inputmol)
    conformer=molobj.GetConformer()
    id=conformer.GetId()
    for cid,xyz in enumerate(coords):
        ##print(xyz[0],xyz[1],xyz[2],type(xyz))
        conformer.SetAtomPosition(cid,Point3D(float(xyz[0]),float(xyz[1]),float(xyz[2]))) #
    conf_id=molobj.AddConformer(conformer)
    molobj.RemoveConformer(id)
    return molobj


def generate_3d_conformer_from_smiles(smiles):
    #  SMILES 
    ##print(smiles)
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    ##print(mol)
    flag = False

    # 
    AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    try:
        AllChem.UFFOptimizeMolecule(mol)
    except Exception as e:
        ##print(e)
        flag = True

    Chem.SanitizeMol(mol) #
    mol = Chem.RemoveHs(mol) #

    return mol


def save_sdf(mol, output_sdf):
    #  SDF 
    writer = Chem.SDWriter(output_sdf)
    #  SDF 
    writer.write(mol)
    #  SDF 
    writer.close()
    ##print(f"SDF ：{output_sdf}")


def add_knn_ligand_pos(pos, smiles):
    #rdkit3DKNN
    # 
    ##print('dt:', dt)
    mol = generate_3d_conformer_from_smiles(smiles)

    conformer      = mol.GetConformer()
    knn_ligand_pos = conformer.GetPositions() #KNN
    knn_ligand_pos = np.array(knn_ligand_pos)
    centor = knn_ligand_pos.mean(axis = 0) - pos.mean(axis = 0)
    knn_ligand_pos = knn_ligand_pos - centor

    return mol, knn_ligand_pos


def calculate_distance_matrix_torch(A, B):
    """
    

    :
    A (torch.Tensor):  (n, 3) 
    B (torch.Tensor):  (m, 3) 

    :
    torch.Tensor:  (n, m) 
    """
    # A  (n, 3)，B  (m, 3)
    #  A  B 
    diff = A.unsqueeze(1) - B.unsqueeze(0)  # diff  (n, m, 3)
    dist_matrix = torch.sqrt(torch.sum(diff**2, dim=2))  # dist_matrix  (n, m)
    return dist_matrix



import numpy as np

def calculate_distance_matrix_numpy(A, B):
    """
    
    
    :
    A (np.ndarray):  (n, 3) 
    B (np.ndarray):  (m, 3) 
    
    :
    np.ndarray:  (n, m) 
    """
    # A  (n, 3)，B  (m, 3)
    #  A  B 
    diff = A[:, np.newaxis, :] - B[np.newaxis, :, :]  # diff  (n, m, 3)
    dist_matrix = np.sqrt(np.sum(diff**2, axis=2))  # dist_matrix  (n, m)
    
    return dist_matrix




class PDBProtein(object):

    #，20
    AA_NAME_SYM = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H',
        'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q',
        'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
    }

    #,postbuster
    new_AA_NAME_SYM = {'CME': 'CME', 'PTR': 'PTR', 'SEP': 'SEP', 'MHS': 'MHS', 'AGM': 'AGM', 'SMC': 'SMC', 'I2M': 'I2M',
                    'MGN': 'MGN', 'CSD': 'CSD', 'ALY': 'ALY', 'KCX': 'KCX', 'MSE': 'MSE', 'OCS': 'OCS', 'SNN': 'SNN',
                    'MLY': 'MLY', 'TPO': 'TPO', 'LLP': 'LLP', '4OG': '4OG', 'CSO': 'CSO', 'FME': 'FME', 'HIC': 'HIC', 'CSX': 'CSX', 'DYA': 'DYA'
                    
                    } #？？

    #AA_NAME_SYM.update(new_AA_NAME_SYM) #43，，，
    print('len(AA_NAME_SYM):', len(AA_NAME_SYM))

    AA_NAME_NUMBER = {
        k: i for i, (k, _) in enumerate(AA_NAME_SYM.items())
    }

    BACKBONE_NAMES = ["CA", "C", "N", "O"] #

    def __init__(self, data, ligand_centor = None, ligand_dict = None, data_flag = None, cross_distance_num = None, unimol_pcoords = None, mode='auto'):
        super().__init__()
        self.data_flag = data_flag
        if (data[-4:].lower() == '.pdb' and mode == 'auto') or mode == 'path':
            self.protein_file = data 
            with open(data, 'r') as f:
                self.block = f.read() #
        else:
            self.block = data

        self.ligand_centor  = ligand_centor
        self.ligand_dict    = ligand_dict
        self.unimol_pcoords = unimol_pcoords
        self.cross_distance_num = cross_distance_num

        self.ptable = Chem.GetPeriodicTable()

        # Molecule properties
        self.title = None
        # Atom properties
        self.atoms = []
        self.element = []
        self.atomic_weight = []
        self.pos = []
        self.atom_name = []
        self.is_backbone = []
        self.atom_to_aa_type = []
        # Residue properties
        self.residues = []
        self.amino_acid = []
        self.center_of_mass = []
        self.pos_CA = []
        self.pos_C = []
        self.pos_N = []
        self.pos_O = []

        self._parse()

    def _enum_formatted_atom_lines_text(self):
        #O,N, ，rdkitpdb，，rdkit，
        #。，，rdkit，
        for line in self.block.splitlines(): #pdb， 
            if line[0:6].strip() == 'ATOM':
                element_symb = line[76:78].strip().capitalize() #，。capitalize() ，。
                if len(element_symb) == 0:
                    element_symb = line[13:14]
                
                #？？_enum_formatted_atom_lines()
                yield {
                    'line': line,
                    'type': 'ATOM',
                    'atom_id': int(line[6:11]),
                    'atom_name': line[12:16].strip(), #，
                    'res_name': line[17:20].strip(), #
                    'chain': line[21:22].strip(), #
                    'res_id': int(line[22:26]),
                    'res_insert_id': line[26:27].strip(), #
                    'x': float(line[30:38]),
                    'y': float(line[38:46]),
                    'z': float(line[46:54]),
                    'occupancy': float(line[54:60]), #1.0
                    'segment': line[72:76].strip(), #
                    'element_symb': element_symb,
                    'charge': line[78:80].strip(), #
                }
            elif line[0:6].strip() == 'HEADER':
                yield {
                    'type': 'HEADER',
                    'value': line[10:].strip()
                }
            elif line[0:6].strip() == 'ENDMDL':
                break  # Some PDBs have more than 1 model.


    def _enum_formatted_atom_lines_rdkitmol(self):
        #O,N, ，rdkitpdb，，rdkit，
        #。，，rdkit，
        #rdkitmol
        pro_mol=Chem.MolFromPDBFile(self.protein_file,removeHs=False,sanitize=False)
        conf = pro_mol.GetConformer() #

        # 
        self.title = 'POCKET'.lower() #
        if pro_mol is not None:
            # 
            for atom in pro_mol.GetAtoms():
                atom_index = atom.GetIdx()              # 
                atom_symb = atom.GetSymbol()            # 
                atom_name = atom.GetPDBResidueInfo().GetName().strip()   # ，，
                residue_name = atom.GetPDBResidueInfo().GetResidueName().strip()
                residue_number = atom.GetPDBResidueInfo().GetResidueNumber() # 
                chain_id = atom.GetPDBResidueInfo().GetChainId().strip()     # ID
                atom_coords = conf.GetAtomPosition(atom_index)
                x_coord, y_coord, z_coord = atom_coords.x, atom_coords.y, atom_coords.z
                occupancy = atom.GetPropsAsDict().get('_occupancy')           # 
                temp_factor = atom.GetPropsAsDict().get('_bfactor')           # 

                yield {
                    'line': '',
                    'type': 'ATOM',
                    'atom_id': atom_index,
                    'atom_name': atom_name, #，
                    'res_name': residue_name, #
                    'chain': chain_id, #
                    'res_id': residue_number, #
                    'res_insert_id': '', #
                    'x': x_coord,
                    'y': y_coord,
                    'z': z_coord,
                    'occupancy': '', #1.0
                    'segment': '', #
                    'element_symb': atom_symb,
                    'charge': '', #
                }
                
                '''
                # 
                print(f"ATOM : {atom_index}")
                print(f": {atom_name}")
                print(f"：{atom_symb}")
                print(f": {residue_name}") #TRP
                print(f"ID: {chain_id}")
                print(f": {residue_number}")
                print(f" (X, Y, Z): ({x_coord}, {y_coord}, {z_coord})")
                print(f": {occupancy}")
                print(f": {temp_factor}")
                print("----------------------------------")
                '''

        """
        for line in self.block.splitlines(): #pdb， 
            if line[0:6].strip() == 'ATOM':
                element_symb = line[76:78].strip().capitalize() #，。capitalize() ，。
                if len(element_symb) == 0:
                    element_symb = line[13:14]
                
                #？？_enum_formatted_atom_lines()
                #rdkit，
                yield {
                    'line': line,
                    'type': 'ATOM',
                    'atom_id': int(line[6:11]),
                    'atom_name': line[12:16].strip(), #，
                    'res_name': line[17:20].strip(), #
                    'chain': line[21:22].strip(), #
                    'res_id': int(line[22:26]),
                    'res_insert_id': line[26:27].strip(), #
                    'x': float(line[30:38]),
                    'y': float(line[38:46]),
                    'z': float(line[46:54]),
                    'occupancy': float(line[54:60]), #1.0
                    'segment': line[72:76].strip(), #
                    'element_symb': element_symb,
                    'charge': line[78:80].strip(), #
                }
            elif line[0:6].strip() == 'HEADER':
                yield {
                    'type': 'HEADER',
                    'value': line[10:].strip()
                }
            elif line[0:6].strip() == 'ENDMDL':
                break  # Some PDBs have more than 1 model.
        """




    def _enum_formatted_atom_lines_biopandas(self):
        #O,N, ，rdkitpdb，，rdkit，
        #。，，rdkit，
        #biopands，rdkit，

        #  PDB 
        pdb = PandasPdb().read_pdb(self.protein_file)
        #  ATOM 
        atom_df = pdb.df['ATOM']

        #
        '''
        # 
        # ，
        atom_df['coords'] = atom_df[['x_coord', 'y_coord', 'z_coord']].apply(tuple, axis=1)

        # ，
        unique_atom_df = atom_df.drop_duplicates(subset='coords')

        # 
        unique_atom_df = unique_atom_df.drop(columns=['coords'])

        # 
        unique_atom_df = unique_atom_df.reset_index(drop=True)  

        #  'line_idx' ， 0 
        unique_atom_df['line_idx'] = range(len(unique_atom_df))

        #  PDB 
        #pdb.df['ATOM'] = unique_atom_df
        #pdb.to_pdb(path='output_protein.pdb', records=None)

        atom_df = unique_atom_df
        '''




        # 
        self.title = 'POCKET'.lower() #

        for i, row in atom_df.iterrows():
            yield {
                'line': '',
                'type': 'ATOM',
                'atom_id': row['atom_number'],
                'atom_name': row['atom_name'],  # 
                'res_name': row['residue_name'],  # 
                'chain': row['chain_id'],  # 
                'res_id': row['residue_number'],  # 
                'res_insert_id': '',  # ，
                'x': row['x_coord'],  # x
                'y': row['y_coord'],  # y
                'z': row['z_coord'],  # z
                'occupancy': '',  # ，
                'segment': '',  # ，
                'element_symb': row['element_symbol'],  # 
                'charge': '',  # ，
            }



    def _read_pdb_ONRing_biopandas(self, pdb_file):
        #pdbON
        #  PDB 
        #print('pdb_file:', pdb_file)
        pdb = PandasPdb().read_pdb(pdb_file)

        #  ATOM 
        atom_df = pdb.df['ATOM']

        '''
        # 
        # ，
        atom_df['coords'] = atom_df[['x_coord', 'y_coord', 'z_coord']].apply(tuple, axis=1)

        # ，
        unique_atom_df = atom_df.drop_duplicates(subset='coords')

        # 
        unique_atom_df = unique_atom_df.drop(columns=['coords'])

        # 
        unique_atom_df = unique_atom_df.reset_index(drop=True)

        #  'line_idx' ， 0 
        unique_atom_df['line_idx'] = range(len(unique_atom_df))

        # 
        #print(unique_atom_df.head())


        #  PDB 
        #pdb.df['ATOM'] = unique_atom_df
        #pdb.to_pdb(path='output_protein.pdb', records=None)

        atom_df = unique_atom_df
        '''

        # 
        oxygen_atoms = []
        nitrogen_atoms = []
        ring_atoms = []
        xyzs = []
        atom_index = []
        atom_isspecial = []

        # 
        for i, row in atom_df.iterrows():
            atom_name    = row['atom_name'].strip()
            element_symb = row['element_symbol'].strip()
            residue_name = row['residue_name'].strip()

            #
            atom_index.append(row['line_idx'])
            
            #
            x, y, z = row['x_coord'], row['y_coord'], row['z_coord']
            xyzs.append([x, y, z])

            # 
            oxygen_atoms.append(element_symb == 'O')
            o_flag = element_symb == 'O'

            # 
            nitrogen_atoms.append(element_symb == 'N')
            n_flag = element_symb == 'N'
    

            # 
            # 
            ring_atom_dict = {
                'PHE': ['CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'],
                'TYR': ['CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'],
                'TRP': ['CG', 'CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2'],
                'HIS': ['CG', 'ND1', 'CD2', 'CE1', 'NE2'],
                'PRO': ['CG', 'CD'],
                # 
            }

            # 
            if residue_name in ring_atom_dict and atom_name in ring_atom_dict[residue_name]:
                ring_atoms.append(True)
                r_flag = True
            else:
                ring_atoms.append(False)
                r_flag = False

            atom_isspecial.append(o_flag or n_flag or r_flag)

        # 
        assert len(oxygen_atoms) == len(nitrogen_atoms) and len(ring_atoms) == len(atom_isspecial) and len(oxygen_atoms) == len(atom_index)
        print(f'{max(atom_index)} <= {len(atom_index) - 1}')
        assert max(atom_index) <= len(atom_index) - 1

        #if not np.array(xyzs).shape == np.unique(np.array(xyzs), axis = 0).shape: #，，
            #raise Exception(f'{np.array(xyzs).shape} == {np.unique(np.array(xyzs), axis = 0).shape}')


        return np.array(oxygen_atoms), np.array(nitrogen_atoms), np.array(ring_atoms), np.array(atom_isspecial), np.array(atom_index), np.array(xyzs, dtype=np.float32)

    def compare_atom_order(self, mol1, mol2):
        # 
        mol1_no_h = Chem.RemoveHs(mol1)
        mol2_no_h = Chem.RemoveHs(mol2)
        
        # 
        atoms1 = [atom.GetSymbol() for atom in mol1_no_h.GetAtoms()]
        atoms2 = [atom.GetSymbol() for atom in mol2_no_h.GetAtoms()]
        
        # 
        return atoms1 == atoms2



    def _parse(self):
        # Process atoms
        residues_tmp = {}
        for atom in self._enum_formatted_atom_lines_biopandas(): #，,
            if atom['type'] == 'HEADER':
                self.title = atom['value'].lower() #POCKET，，
                continue
            self.atoms.append(atom)
            atomic_number = self.ptable.GetAtomicNumber(atom['element_symb']) #，atom['atom_id']
            next_ptr = len(self.element) #0
            self.element.append(atomic_number)
            self.atomic_weight.append(self.ptable.GetAtomicWeight(atomic_number))
            self.pos.append(np.array([atom['x'], atom['y'], atom['z']], dtype=np.float32))
            #self.pos.append(np.array([atom['x'], atom['y'], atom['z']]))
            self.atom_name.append(atom['atom_name'])
            self.is_backbone.append(atom['atom_name'] in self.BACKBONE_NAMES) # boolean
            self.atom_to_aa_type.append(self.AA_NAME_NUMBER[atom['res_name']]) #

            chain_res_id = '%s_%s_%d_%s' % (atom['chain'], atom['segment'], atom['res_id'], atom['res_insert_id'])
            if chain_res_id not in residues_tmp:
                residues_tmp[chain_res_id] = {
                    'name': atom['res_name'],
                    'atoms': [next_ptr],
                    'chain': atom['chain'],
                    'segment': atom['segment'],
                }
            else:
                #print('chain_res_id:', chain_res_id)
                #print(f"{residues_tmp[chain_res_id]['name']} == {atom['res_name']}")
                #print(f"{residues_tmp[chain_res_id]['chain']} == {atom['chain']}")

                try:
                    assert residues_tmp[chain_res_id]['name'] == atom['res_name']
                    assert residues_tmp[chain_res_id]['chain'] == atom['chain']
                except AssertionError as e:
                    print(e)
                    #raise SystemExit(f"chain_res_id]['name'] == atom['res_name']")
                residues_tmp[chain_res_id]['atoms'].append(next_ptr)

        
        


        # Process residues
        self.residues = [r for _, r in residues_tmp.items()]
        for residue in self.residues:
            sum_pos = np.zeros([3], dtype=np.float32)
            sum_mass = 0.0
            for atom_idx in residue['atoms']:
                sum_pos += self.pos[atom_idx] * self.atomic_weight[atom_idx]
                sum_mass += self.atomic_weight[atom_idx]
                if self.atom_name[atom_idx] in self.BACKBONE_NAMES:
                    residue['pos_%s' % self.atom_name[atom_idx]] = self.pos[atom_idx]
            residue['center_of_mass'] = sum_pos / sum_mass

        # Process backbone atoms of residues
        for residue in self.residues:
            self.amino_acid.append(self.AA_NAME_NUMBER[residue['name']])
            self.center_of_mass.append(residue['center_of_mass'])
            for name in self.BACKBONE_NAMES:
                pos_key = 'pos_%s' % name  # pos_CA, pos_C, pos_N, pos_O
                if pos_key in residue:
                    getattr(self, pos_key).append(residue[pos_key])
                else:
                    getattr(self, pos_key).append(residue['center_of_mass'])



    def to_dict_atom_interaction_old(self):
        #，，6ai，

        old_data = {
            'element': np.array(self.element, dtype=np.int64), #
            'molecule_name': self.title, # pocket
            'pos': np.array(self.pos, dtype=np.float32), #
            'is_backbone': np.array(self.is_backbone, dtype=np.bool_), #Boolean，
            'atom_name': self.atom_name, #
            'atom_to_aa_type': np.array(self.atom_to_aa_type, dtype=np.int64) ##
        }

        #，400

        if 'pdbbind2020' in self.protein_file:
            file_path = os.path.join(os.path.dirname(self.protein_file), 'interaction_' + os.path.basename(self.protein_file).split('_')[0] + '_v2.pkl')   #5S8I_protein.pdb
        else:
            file_path = os.path.join(os.path.dirname(self.protein_file), 'interaction_' + os.path.basename(self.protein_file).split('_')[0] + '.pkl')   #5S8I_protein.pdb
        with open(file_path, 'rb') as file:
            interaction_data = dill.load(file)
        holo_coords_list = interaction_data['holo_coords_list']
        coords_predict_list = interaction_data['coords_predict_list']
        pocket_coords_list = interaction_data['pocket_coords_list']
        cross_distance_list = interaction_data['cross_distance_list']

        assert np.allclose(holo_coords_list[0], holo_coords_list[-1], atol=0.02)

        #print('pocket_coords_list[0]:', pocket_coords_list[0][:])
        #np.set_printoptions(suppress=True, precision=4)
        #print('pocket_coords_list[-1]:', pocket_coords_list[-1]) #，？？，
        #print('pocket_coords_list[0].shape:', pocket_coords_list[0].shape)
        #print('pocket_coords_list[-1].shape:', pocket_coords_list[-1].shape) #，？？
        #assert np.allclose(pocket_coords_list[0], pocket_coords_list[-1], atol=0.02)

        assert len(holo_coords_list) == len(coords_predict_list) and len(holo_coords_list) == len(pocket_coords_list) and len(holo_coords_list) == len(cross_distance_list)

        #rmsd，，
        rmsd_list = []
        for pre_pos, holo_pos in zip(coords_predict_list, holo_coords_list):
            assert pre_pos.shape == holo_pos.shape   #"Coordinate matrices must have the same shape"
            rmsd = np.sqrt(np.mean(np.sum((pre_pos - holo_pos) ** 2, axis=1)))
            rmsd_list.append(rmsd)
        
        sorted_indices = np.argsort(rmsd_list)
        best_index = sorted_indices[0]

        if self.cross_distance_num == None or self.cross_distance_num == 'best':
            holo_coords = holo_coords_list[best_index]
            coords_predict = coords_predict_list[best_index]
            pocket_coords = pocket_coords_list[best_index]
            cross_distance = cross_distance_list[best_index]
        else:            
            holo_coords = holo_coords_list[self.cross_distance_num]
            coords_predict = coords_predict_list[self.cross_distance_num]
            pocket_coords = pocket_coords_list[self.cross_distance_num]
            cross_distance = cross_distance_list[self.cross_distance_num]

        #unimol，，
        unique_index_dict       = {}
        for j, ps in enumerate(pocket_coords):
            unique_index_dict[tuple(ps)] = j
            #，
        
        unique_index_list   = list(unique_index_dict.values())
        pocket_coords       = pocket_coords[unique_index_list]
        cross_distance      = cross_distance[:, unique_index_list]

        #，unimolrdkit，，
        #cross_distanceO,N,
        #，
        pro_isring_flag = {}
        pro_isO_flag = {}
        pro_isN_flag = {}

        '''
        pro_mol=Chem.MolFromPDBFile(self.protein_file,removeHs=False,sanitize=False) #，, 
        #print('self.protein_file:', self.protein_file)
        #print('ok1')
        atom_isring=np.array([atom.IsInRing() for atom in pro_mol.GetAtoms()])    #
        atom_isO=np.array([atom.GetSymbol() == 'O'  for atom in pro_mol.GetAtoms()])    #O
        atom_isN=np.array([atom.GetSymbol() == 'N'  for atom in pro_mol.GetAtoms()])    #N
        coords=np.array(pro_mol.GetConformer(0).GetPositions(), dtype=np.float32) #,float32，float64
        #coords=np.array(pro_mol.GetConformer(0).GetPositions())
        '''
        atom_isO, atom_isN, atom_isring, atom_isspecial, atom_id, coords = self._read_pdb_ONRing_biopandas(self.protein_file)
        count = 0
        coords_atom_dict = defaultdict(list)
        for r, o, n, c in zip(atom_isring, atom_isO, atom_isN, coords):
            count += 1
            k = torch.FloatTensor(c)
            tg = ''
            for ii in k:
                #tg += str(round(i.item(), 4)) + '_'
                tg += str(self.truncate(ii.item(), 3)) + '_' #，2，
            c_sum = str(tg)

            coords_atom_dict[c_sum].append(c)

            pro_isring_flag[c_sum] = r
            pro_isO_flag[c_sum]    = o
            pro_isN_flag[c_sum]    = n
        
        assert len(pro_isring_flag) == len(atom_isring)

        #print('pro_isring_flag.keys:', pro_isring_flag.keys())
        
        for k in coords_atom_dict:
            if len(coords_atom_dict[k]) > 1:
                print('exist repeated coordinates')
                print('k:', k)
                print('v:', coords_atom_dict[k])

        print('：', count)
        if len(pro_isring_flag) != len(coords) or len(pro_isO_flag) != len(coords) or len(pro_isN_flag) != len(coords):
            print(f'{len(pro_isring_flag)} != {len(coords)} or {len(pro_isO_flag)} != {len(coords)} or {len(pro_isN_flag)} != {len(coords)}')
            #979 != 990 or 979 != 990 or 979 != 990 #？
            raise Exception("pro atom num is error")


        #cross_distance, cross_ligand, cross_protein, cross_ligand_atom_flag, cross_protein_atom_flag
        #，
        pro_isring_flag_list = []
        pro_isO_flag_list = []
        pro_isN_flag_list = []




        drop_protein_ids = torch.ones(pocket_coords.shape[0], dtype = torch.bool)
        test_dict = {}

        #unimolrdkit，rdkit，，unimolrdkit，
        for ids, c in enumerate(pocket_coords):
            k = torch.FloatTensor(c)
            tg = ''
            for ii in k:
                #tg += str(round(i.item(), 4)) + '_'
                tg += str(self.truncate(ii.item(), 3)) + '_' #，2，
            c_sum = str(tg)
            test_dict[c_sum] = ids

            try:
                pro_isring_flag_list.append(pro_isring_flag[c_sum])
                pro_isO_flag_list.append(pro_isO_flag[c_sum])
                pro_isN_flag_list.append(pro_isN_flag[c_sum])
            except KeyError as e:
                print('error:', e)
                #drop_protein_ids.append(ids) #，
                drop_protein_ids[ids] = False
                continue
        

        assert len(test_dict) == len(pocket_coords)

        #assert len(pro_isring_flag_list) == len(pocket_coords) #，rdkit，

        

        cross_pro_isring_flag = np.stack(pro_isring_flag_list, axis = 0)
        cross_pro_isO_flag = np.stack(pro_isO_flag_list, axis = 0)
        cross_pro_isN_flag = np.stack(pro_isN_flag_list, axis = 0)


        #，,
        #，unimolrdkit，unimolrdkit，sdf，rdkit mol，rdkit
        #，unimol，rdkit
        lig_isring_flag = {}
        lig_isO_flag = {}
        lig_isN_flag = {}

        lig_mol = copy.deepcopy(self.ligand_dict['mol']) #
        lig_mol = lig_mol #，？cross，，
        #print('ok1')
        atom_isring=np.array([atom.IsInRing() for atom in lig_mol.GetAtoms()])    #
        atom_isO=np.array([atom.GetSymbol() == 'O'  for atom in lig_mol.GetAtoms()])    #O
        atom_isN=np.array([atom.GetSymbol() == 'N'  for atom in lig_mol.GetAtoms()])    #N
        coords=np.array(lig_mol.GetConformer(0).GetPositions(), dtype=np.float32) #,float32，float64
        #coords=np.array(lig_mol.GetConformer(0).GetPositions())

        for r, o, n, c in zip(atom_isring, atom_isO, atom_isN, coords):
            k = torch.FloatTensor(c)
            tg = ''
            for ii in k:
                #tg += str(round(i.item(), 4)) + '_'
                tg += str(self.truncate(ii.item(), 3)) + '_' #，2，
            c_sum = str(tg)

            lig_isring_flag[c_sum] = r
            lig_isO_flag[c_sum]    = o
            lig_isN_flag[c_sum]    = n
        
        assert len(lig_isring_flag) == len(atom_isring)

        if len(lig_isring_flag) != len(coords) or len(lig_isO_flag) != len(coords) or len(lig_isN_flag) != len(coords) or len(coords) != len(holo_coords):
            print(f'{len(coords)} != {len(holo_coords)}') #19 != 18, rdkit，crosss_liand，
            raise Exception("ligand atom num is error") #，

        
        assert len(coords) == len(holo_coords) #，rkdit，，，

        #unimolrdkit, 0.02
        #assert torch.allclose(torch.FloatTensor(coords), torch.FloatTensor(holo_coords), atol=0.00)

        #，holo_coordsrdkit，
        #rdkitholo，
        #print('coords:', torch.FloatTensor(coords))
        #print('holo_coords:', torch.FloatTensor(holo_coords))
        #if not torch.allclose(torch.FloatTensor(coords), torch.FloatTensor(holo_coords), atol=0.000000001):
        if not torch.equal(torch.FloatTensor(coords), torch.FloatTensor(holo_coords)):
            print('，rdkit')
            holo_coords_ids_dict  = {}
            rdkit_coords_ids_dict = {}

            for ids, c in enumerate(holo_coords):
                k = torch.FloatTensor(c)
                tg = ''
                for ii in k:
                    #tg += str(round(i.item(), 4)) + '_'
                    tg += str(self.truncate(ii.item(), 1)) + '_' #，2，
                c_sum = str(tg)
                holo_coords_ids_dict[c_sum] = ids

            assert len(holo_coords_ids_dict) == len(holo_coords)

            for ids, c in enumerate(coords):
                k = torch.FloatTensor(c)
                tg = ''
                for ii in k:
                    #tg += str(round(i.item(), 4)) + '_'
                    tg += str(self.truncate(ii.item(), 1)) + '_' #，2，
                c_sum = str(tg)
                rdkit_coords_ids_dict[c_sum] = ids

            assert len(rdkit_coords_ids_dict) == len(coords)

            unimol_true_index = []
            for k in holo_coords_ids_dict:
                v = rdkit_coords_ids_dict[k]
                unimol_true_index.append(v)
            
            #
            coords_predict  = coords_predict[unimol_true_index]
            holo_coords     = copy.deepcopy(coords) #rdkit    #holo_coords[unimol_true_index]
            cross_distance  = cross_distance[unimol_true_index]

        #cross_distance, cross_ligand, cross_protein, cross_ligand_atom_flag, cross_protein_atom_flag
        #，
        lig_isring_flag_list = []
        lig_isO_flag_list = []
        lig_isN_flag_list = []

        test_dict = {}
        for c in holo_coords:
            k = torch.FloatTensor(c)
            tg = ''
            for ii in k:
                #tg += str(round(i.item(), 4)) + '_'
                tg += str(self.truncate(ii.item(), 3)) + '_' #，2，
            c_sum = str(tg)
            test_dict[c_sum] = c

            try:
                lig_isring_flag_list.append(lig_isring_flag[c_sum])
                lig_isO_flag_list.append(lig_isO_flag[c_sum])
                lig_isN_flag_list.append(lig_isN_flag[c_sum])
            except KeyError as e:
                print('ligand error:', e)
                print('c:', c)
                print('lig_isring_flag.keys:', list(lig_isring_flag.keys()))
                #drop_protein_ids.append(ids) #，
                #drop_protein_ids[ids] = False
                raise Exception('ligand atom key error')
                continue
        
        assert len(holo_coords) == len(test_dict)

        
        cross_lig_isring_flag = np.stack(lig_isring_flag_list, axis = 0)
        cross_lig_isO_flag = np.stack(lig_isO_flag_list, axis = 0)
        cross_lig_isN_flag = np.stack(lig_isN_flag_list, axis = 0)

        #drop_protein_idsholo_coords, coords_predict, pocket_coords, cross_distance
        pocket_coords  = pocket_coords[drop_protein_ids]
        cross_distance = cross_distance[:, drop_protein_ids]
        print('pocket_coords:', pocket_coords.shape)
        print('cross_distance:', cross_distance.shape)

        
        if old_data['pos'].shape[0] <= 0:
            print("old_data['pos'].size(0):", old_data['pos'].shape[0])
            print("old_data['pos']:", old_data['pos'].shape)
            return old_data
        else:
            data = {}
            #self.ligand_centor
            dis   = np.linalg.norm(old_data['pos'] - self.ligand_centor, axis = 1) #

            try:
                #，12ai，400，400

                if pocket_coords.shape[0] < 300:
                    cutoff_num = 1000
                else:
                    cutoff_num = 400

                #，，rdkit
                #print('self.protein_file:', self.protein_file)
                '''
                pro_mol=Chem.MolFromPDBFile(self.protein_file,removeHs=False,sanitize=False)
                atom_isspecial = np.array([atom.IsInRing() or atom.GetSymbol() == 'O' or atom.GetSymbol() == 'N' for atom in pro_mol.GetAtoms()]) #
                atom_id = np.array([atom.GetIdx() for atom in pro_mol.GetAtoms()])
                #coords=np.array(pro_mol.GetConformer(0).GetPositions(), dtype=np.float32) #,float32，float64
                all_coords=np.array(pro_mol.GetConformer(0).GetPositions(), dtype=np.float32)
                '''
                atom_isO, atom_isN, atom_isring, atom_isspecial, atom_id, all_coords = self._read_pdb_ONRing_biopandas(self.protein_file)
                cutoff_indices = np.argsort(dis)[:cutoff_num] #,cutoff_num，，
                coords = all_coords[cutoff_indices] #，

                #，pocket_coords，400
                new_indices = []
                pro_flag_dict = {}

                for j, c in zip(cutoff_indices, coords): #j，，
                    k = torch.FloatTensor(c)
                    tg = ''
                    for ii in k:
                        #tg += str(round(i.item(), 4)) + '_'
                        tg += str(self.truncate(ii.item(), 3)) + '_' #，2，
                    k = str(tg)

                    pro_flag_dict[k] = j
                

                if len(pro_flag_dict) != len(coords):
                    raise Exception(f"{len(pro_flag_dict)} != {len(coords)}")

                cutoff_protein_ids = torch.ones(pocket_coords.shape[0], dtype = torch.bool)
                
                #rdkitholo，，rdkit
                for ids, c in enumerate(pocket_coords):
                    k = torch.FloatTensor(c)
                    tg = ''
                    for ii in k:
                        #tg += str(round(i.item(), 4)) + '_'
                        tg += str(self.truncate(ii.item(), 3)) + '_' #，2，
                    k = str(tg)

                    #pocket_coords，
                    #if cross_pro_flag_dict.get(c_sum):
                    try:
                        new_indices.append(pro_flag_dict[k]) #，unimol，，
                    except KeyError as e:
                        cutoff_protein_ids[ids] = False
                        print('key error, skip')
                        continue
                
                #，，
                pocket_coords  = pocket_coords[cutoff_protein_ids]
                cross_distance = cross_distance[:, cutoff_protein_ids]

                cross_pro_isring_flag = cross_pro_isring_flag[cutoff_protein_ids]
                cross_pro_isO_flag    = cross_pro_isO_flag[cutoff_protein_ids]
                cross_pro_isN_flag    = cross_pro_isN_flag[cutoff_protein_ids]
                
                #if len(new_indices) != len(self.unimol_pcoords[0]):
                    #raise Exception(f'{len(new_indices)} <= 0')

                print('len(new_indices):', len(new_indices))
                #print('new_indices:', new_indices)

                new_atom_isspecial = atom_isspecial[new_indices] #atom_isspecial
                new_atom_id = atom_id[new_indices] #atom_id
                nonzero_indices = new_indices

                #，
                atom_isspecial_index = np.nonzero(new_atom_isspecial == True)[0]
                atom_isgeneral_index = np.nonzero(new_atom_isspecial == False)[0]
                

                #rdkit，，
                #new_atom_id
                sub_protein_file = os.path.join(os.path.dirname(self.protein_file), os.path.splitext(os.path.basename(self.protein_file))[0] + '_400.pdb')

                def keep_atoms_by_id(mol, atom_ids):
                    """ID"""
                    editable_mol = Chem.EditableMol(mol)
                    all_atoms = list(editable_mol.GetMol().GetAtoms())
                    atoms_to_keep = {atom.GetIdx() for atom in all_atoms if atom.GetIdx() in atom_ids}
                    
                    atoms_to_remove = [atom.GetIdx() for atom in all_atoms if atom.GetIdx() not in atom_ids]
                    atoms_to_remove.sort(reverse=True)  # ，

                    for atom_id in atoms_to_remove:
                        editable_mol.RemoveAtom(atom_id)

                    return editable_mol.GetMol()
                

                # ID（0）
                atom_ids_to_keep = new_atom_id  # ID，
                pro_mol=Chem.MolFromPDBFile(self.protein_file,removeHs=False,sanitize=False)
                # 
                new_pro_mol = keep_atoms_by_id(pro_mol, atom_ids_to_keep)
                Chem.MolToPDBFile(new_pro_mol, sub_protein_file)



                #，
                #400pdb，,# 5l8c/5l8c_pocket10.pdb，5l8c/5l8c_pocket10_400.pdb
                #nonzero_indices
                #: '../CrossDocked2020/data/pdbbind2020_r10/v2020-other-PL/5qb1/5qb1_pocket10_400.pdb'

                '''
                sub_protein_file = os.path.join(os.path.dirname(self.protein_file), os.path.splitext(os.path.basename(self.protein_file))[0] + '_400.pdb')
                sub_protein_w = open(sub_protein_file, 'w')

                org_pro_list = []
                count = 0
                with open(self.protein_file, 'r')as f:
                    for i, line in enumerate(f):
                        if line[0:6].strip() == 'ATOM' and i in nonzero_indices:
                            sub_protein_w.write(line)
                    

                    sub_protein_w.write('END')
                sub_protein_w.close()

                '''
            
                #print('ok3')

                #O,N, ，rdkitpdb，，rdkit，
                #。，，rdkit，
                #，，，rdkit
                '''
                pro_mol=Chem.MolFromPDBFile(self.protein_file,removeHs=False,sanitize=False)
                #print('ok1')
                atom_isring=np.array([atom.IsInRing() for atom in pro_mol.GetAtoms()])    #
                atom_isO=np.array([atom.GetSymbol() == 'O'  for atom in pro_mol.GetAtoms()])    #O
                atom_isN=np.array([atom.GetSymbol() == 'N'  for atom in pro_mol.GetAtoms()])    #N
                coords=np.array(pro_mol.GetConformer(0).GetPositions(), dtype=np.float32) #,float32，float64
                #coords=np.array(pro_mol.GetConformer(0).GetPositions())
                #，，-，
                #indexs = []  #, 
                '''
                atom_isO, atom_isN, atom_isring, atom_isspecial, atom_id, coords = self._read_pdb_ONRing_biopandas(self.protein_file)
                target_xyz =  np.array(self.pos) #xyz
                #target_xyz_sum =  np.round(np.array(self.pos, dtype=np.float32).sum(axis = -1), 2)

                #assert np.array_equal(target_xyz, coords) #，, postbus, assert，try, except

                if not np.array_equal(target_xyz, coords):
                    print('target_xyz.shape:', target_xyz.shape)
                    print('coords.shape:', coords.shape)
                    raise Exception('np.array_equal(target_xyz, coords):', np.array_equal(target_xyz, coords))

                new_atom_isring = atom_isring
                new_atom_isO = atom_isO
                new_atom_isN = atom_isN

                assert len(new_atom_isN[nonzero_indices]) == len(np.array(self.element, dtype=np.int64)[nonzero_indices])
                #print('ok4')

                data = {
                    'element': np.array(self.element, dtype=np.int64)[nonzero_indices], #
                    'molecule_name': self.title, # pocket
                    #'pos': np.array(self.pos, dtype=np.float32)[nonzero_indices], #
                    'pos': pocket_coords, #，pocket_coords
                    'is_backbone': np.array(self.is_backbone, dtype=np.bool_)[nonzero_indices], #Boolean，
                    'atom_name': [self.atom_name[i] for i in nonzero_indices], #
                    'atom_to_aa_type': np.array(self.atom_to_aa_type, dtype=np.int64)[nonzero_indices], ##
                    'atom_isring': new_atom_isring[nonzero_indices],
                    'atom_isO': new_atom_isO[nonzero_indices],
                    'atom_isN': new_atom_isN[nonzero_indices],

                    'cross_lig_isring_flag': cross_lig_isring_flag,
                    'cross_lig_isO_flag': cross_lig_isO_flag,
                    'cross_lig_isN_flag': cross_lig_isN_flag,

                    'cross_pro_isring_flag': cross_pro_isring_flag,
                    'cross_pro_isO_flag': cross_pro_isO_flag,
                    'cross_pro_isN_flag': cross_pro_isN_flag,

                    'cross_ligand': holo_coords,
                    'cross_protein': pocket_coords,
                    'cross_distance': set(torch.from_numpy(cross_distance)),   #npset，tensor, torch.from_numpy
                    #cross_distance，PyG，，set，
                    #，，，
                    #
                }

                #print('ok5')
                return data
            
            
            except Exception as e:
                print('protein error:', e)
                print('self.protein_file:', self.protein_file)
                #raise Exception('protein error, stop')
                #exit()
                return None
            








    def to_dict_atom_interaction(self):
        #，，6ai，

        old_data = {
            'element': np.array(self.element, dtype=np.int64), #
            'molecule_name': self.title, # pocket
            'pos': np.array(self.pos, dtype=np.float32), #
            'is_backbone': np.array(self.is_backbone, dtype=np.bool_), #Boolean，
            'atom_name': self.atom_name, #
            'atom_to_aa_type': np.array(self.atom_to_aa_type, dtype=np.int64) ##
        }

        #，400

        if 'pdbbind2020' in self.protein_file:
            file_path = os.path.join(os.path.dirname(self.protein_file), 'interaction_' + os.path.basename(self.protein_file).split('_')[0] + '_v2.pkl')   #5S8I_protein.pdb
        else:
            file_path = os.path.join(os.path.dirname(self.protein_file), 'interaction_' + os.path.basename(self.protein_file).split('_')[0] + '.pkl')   #5S8I_protein.pdb
        with open(file_path, 'rb') as file:
            interaction_data = dill.load(file)
        holo_coords_list = interaction_data['holo_coords_list']
        coords_predict_list = interaction_data['coords_predict_list']
        pocket_coords_list = interaction_data['pocket_coords_list']
        cross_distance_list = interaction_data['cross_distance_list']

        assert np.allclose(holo_coords_list[0], holo_coords_list[-1], atol=0.02)

        #print('pocket_coords_list[0]:', pocket_coords_list[0][:])
        #np.set_printoptions(suppress=True, precision=4)
        #print('pocket_coords_list[-1]:', pocket_coords_list[-1]) #，？？，
        #print('pocket_coords_list[0].shape:', pocket_coords_list[0].shape)
        #print('pocket_coords_list[-1].shape:', pocket_coords_list[-1].shape) #，？？
        #assert np.allclose(pocket_coords_list[0], pocket_coords_list[-1], atol=0.02)

        assert len(holo_coords_list) == len(coords_predict_list) and len(holo_coords_list) == len(pocket_coords_list) and len(holo_coords_list) == len(cross_distance_list)

        
        #rmsd，，
        rmsd_list = []
        for pre_pos, holo_pos in zip(coords_predict_list, holo_coords_list):
            assert pre_pos.shape == holo_pos.shape   #"Coordinate matrices must have the same shape"
            rmsd = np.sqrt(np.mean(np.sum((pre_pos - holo_pos) ** 2, axis=1)))
            rmsd_list.append(rmsd)
        
        sorted_indices = np.argsort(rmsd_list)
        best_index = sorted_indices[0]
        
        #
        #best_index  = random.choice(list(range(len(holo_coords_list))))



        if self.cross_distance_num == None or self.cross_distance_num == 'best':
            holo_coords = holo_coords_list[best_index]
            coords_predict = coords_predict_list[best_index]
            pocket_coords = pocket_coords_list[best_index]
            cross_distance = cross_distance_list[best_index]
        else:            
            holo_coords = holo_coords_list[self.cross_distance_num]
            coords_predict = coords_predict_list[self.cross_distance_num]
            pocket_coords = pocket_coords_list[self.cross_distance_num]
            cross_distance = cross_distance_list[self.cross_distance_num]

        #unimol，，
        unique_index_dict       = {}
        for j, ps in enumerate(pocket_coords):
            unique_index_dict[tuple(ps)] = j
            #，
        
        unique_index_list   = list(unique_index_dict.values())
        pocket_coords       = pocket_coords[unique_index_list]
        cross_distance      = cross_distance[:, unique_index_list]






        #，unimolrdkit，，
        #cross_distanceO,N,
        #，
        pro_isring_flag = {}
        pro_isO_flag = {}
        pro_isN_flag = {}

        '''
        pro_mol=Chem.MolFromPDBFile(self.protein_file,removeHs=False,sanitize=False) #，, 
        #print('self.protein_file:', self.protein_file)
        #print('ok1')
        atom_isring=np.array([atom.IsInRing() for atom in pro_mol.GetAtoms()])    #
        atom_isO=np.array([atom.GetSymbol() == 'O'  for atom in pro_mol.GetAtoms()])    #O
        atom_isN=np.array([atom.GetSymbol() == 'N'  for atom in pro_mol.GetAtoms()])    #N
        #coords=np.array(pro_mol.GetConformer(0).GetPositions(), dtype=np.float32) #,float32，float64
        coords=np.array(pro_mol.GetConformer(0).GetPositions())
        '''

        atom_isO, atom_isN, atom_isring, atom_isspecial, atom_id, coords = self._read_pdb_ONRing_biopandas(self.protein_file)
        #

        count = 0
        coords_atom_dict = defaultdict(list)
        for r, o, n, c in zip(atom_isring, atom_isO, atom_isN, coords):
            count += 1
            k = torch.FloatTensor(c)
            tg = ''
            for ii in k:
                #tg += str(round(i.item(), 4)) + '_'
                tg += str(self.truncate(ii.item(), 3)) + '_' #，2，
            c_sum = str(tg)

            coords_atom_dict[c_sum].append(c)

            pro_isring_flag[c_sum] = r
            pro_isO_flag[c_sum]    = o
            pro_isN_flag[c_sum]    = n
        

        print('：', count)
        if len(pro_isring_flag) != len(coords) or len(pro_isO_flag) != len(coords) or len(pro_isN_flag) != len(coords):
            print(f'{len(pro_isring_flag)} != {len(coords)} or {len(pro_isO_flag)} != {len(coords)} or {len(pro_isN_flag)} != {len(coords)}')
            #979 != 990 or 979 != 990 or 979 != 990 #？
            raise Exception("pro atom num is error")


        #cross_distance, cross_ligand, cross_protein, cross_ligand_atom_flag, cross_protein_atom_flag
        #，
        pro_isring_flag_list = []
        pro_isO_flag_list = []
        pro_isN_flag_list = []




        drop_protein_ids = torch.ones(pocket_coords.shape[0], dtype = torch.bool)
        test_dict = {}

        #unimolrdkit，rdkit，，unimolrdkit，
        for ids, c in enumerate(pocket_coords):
            k = torch.FloatTensor(c)
            tg = ''
            for ii in k:
                #tg += str(round(i.item(), 4)) + '_'
                tg += str(self.truncate(ii.item(), 3)) + '_' #，2，
            c_sum = str(tg)
            test_dict[c_sum] = ids

            try:
                pro_isring_flag_list.append(pro_isring_flag[c_sum])
                pro_isO_flag_list.append(pro_isO_flag[c_sum])
                pro_isN_flag_list.append(pro_isN_flag[c_sum])
            except KeyError as e:
                print('error:', e)
                #drop_protein_ids.append(ids) #，
                drop_protein_ids[ids] = False
                continue
        

        assert len(test_dict) == len(pocket_coords)

        #assert len(pro_isring_flag_list) == len(pocket_coords) #，rdkit，

        

        cross_pro_isring_flag = np.stack(pro_isring_flag_list, axis = 0)
        cross_pro_isO_flag = np.stack(pro_isO_flag_list, axis = 0)
        cross_pro_isN_flag = np.stack(pro_isN_flag_list, axis = 0)


        #，,
        #，unimolrdkit，unimolrdkit，sdf，rdkit mol，rdkit
        #，unimol，rdkit
        lig_isring_flag = {}
        lig_isO_flag = {}
        lig_isN_flag = {}


        #，rdkit3d，，
        if self.data_flag == 'new_test':
            lig_mol = copy.deepcopy(self.ligand_dict['mol']) #
            #，？cross，，
            lig_rdkit_mol = copy.deepcopy(self.ligand_dict['rd_mol'])

            unimol_pos  = torch.FloatTensor(holo_coords)
            rdkit_pos   = torch.FloatTensor(lig_rdkit_mol.GetConformer(0).GetPositions())
            ground_pos  = torch.FloatTensor(lig_mol.GetConformer(0).GetPositions())

            #
            assert len(ground_pos) == len(unimol_pos) #，rkdit，，，

            #，rdkit, rdkitground
            if self.compare_atom_order(lig_mol, lig_rdkit_mol) and torch.allclose(ground_pos, unimol_pos, atol=0.02):
                lig_mol     = copy.deepcopy(lig_rdkit_mol)
                holo_coords = np.array(lig_rdkit_mol.GetConformer(0).GetPositions())
        
        else:
            lig_mol = copy.deepcopy(self.ligand_dict['mol']) #





        #print('ok1')
        atom_isring=np.array([atom.IsInRing() for atom in lig_mol.GetAtoms()])    #
        atom_isO=np.array([atom.GetSymbol() == 'O'  for atom in lig_mol.GetAtoms()])    #O
        atom_isN=np.array([atom.GetSymbol() == 'N'  for atom in lig_mol.GetAtoms()])    #N
        #coords=np.array(lig_mol.GetConformer(0).GetPositions(), dtype=np.float32) #,float32，float64
        coords=np.array(lig_mol.GetConformer(0).GetPositions())

        for r, o, n, c in zip(atom_isring, atom_isO, atom_isN, coords):
            k = torch.FloatTensor(c)
            tg = ''
            for ii in k:
                #tg += str(round(i.item(), 4)) + '_'
                tg += str(self.truncate(ii.item(), 3)) + '_' #，2，
            c_sum = str(tg)

            lig_isring_flag[c_sum] = r
            lig_isO_flag[c_sum]    = o
            lig_isN_flag[c_sum]    = n
        
        assert len(lig_isring_flag) == len(atom_isring)

        assert len(coords) == len(holo_coords) #，rkdit，，，

        if len(lig_isring_flag) != len(coords) or len(lig_isO_flag) != len(coords) or len(lig_isN_flag) != len(coords) or len(coords) != len(holo_coords):
            print(f'{len(coords)} != {len(holo_coords)}') #19 != 18, rdkit，crosss_liand，
            raise Exception("lig atom num is error") #，

        
        assert len(coords) == len(holo_coords) #，rkdit，，，

        #unimolrdkit, 0.02
        assert torch.allclose(torch.FloatTensor(coords), torch.FloatTensor(holo_coords), atol=0.02)

        #，holo_coordsrdkit，
        if not torch.allclose(torch.FloatTensor(coords), torch.FloatTensor(holo_coords), atol=0.02):
            holo_coords_ids_dict  = {}
            rdkit_coords_ids_dict = {}

            for ids, c in enumerate(holo_coords):
                k = torch.FloatTensor(c)
                tg = ''
                for ii in k:
                    #tg += str(round(i.item(), 4)) + '_'
                    tg += str(self.truncate(ii.item(), 3)) + '_' #，2，
                c_sum = str(tg)
                holo_coords_ids_dict[c_sum] = ids

            assert len(holo_coords_ids_dict) == len(holo_coords)

            for ids, c in enumerate(coords):
                k = torch.FloatTensor(c)
                tg = ''
                for ii in k:
                    #tg += str(round(i.item(), 4)) + '_'
                    tg += str(self.truncate(ii.item(), 3)) + '_' #，2，
                c_sum = str(tg)
                rdkit_coords_ids_dict[c_sum] = ids

            assert len(rdkit_coords_ids_dict) == len(coords)

            unimol_true_index = []
            for k in holo_coords_ids_dict:
                v = rdkit_coords_ids_dict[k]
                unimol_true_index.append(v)
            
            #
            coords_predict  = coords_predict[unimol_true_index]
            holo_coords     = holo_coords[unimol_true_index]
            cross_distance  = cross_distance[unimol_true_index]

        #cross_distance, cross_ligand, cross_protein, cross_ligand_atom_flag, cross_protein_atom_flag
        #，
        lig_isring_flag_list = []
        lig_isO_flag_list = []
        lig_isN_flag_list = []

        test_dict = {}
        for c in holo_coords:
            k = torch.FloatTensor(c)
            tg = ''
            for ii in k:
                #tg += str(round(i.item(), 4)) + '_'
                tg += str(self.truncate(ii.item(), 3)) + '_' #，2，
            c_sum = str(tg)
            test_dict[c_sum] = c

            try:
                lig_isring_flag_list.append(lig_isring_flag[c_sum])
                lig_isO_flag_list.append(lig_isO_flag[c_sum])
                lig_isN_flag_list.append(lig_isN_flag[c_sum])
            except KeyError as e:
                print('ligand error:', e)
                print('c:', c)
                print('lig_isring_flag.keys:', list(lig_isring_flag.keys()))
                #drop_protein_ids.append(ids) #，
                #drop_protein_ids[ids] = False
                raise Exception('ligand atom key error')
                continue
        
        assert len(holo_coords) == len(test_dict)

        
        cross_lig_isring_flag = np.stack(lig_isring_flag_list, axis = 0)
        cross_lig_isO_flag = np.stack(lig_isO_flag_list, axis = 0)
        cross_lig_isN_flag = np.stack(lig_isN_flag_list, axis = 0)

        #drop_protein_idsholo_coords, coords_predict, pocket_coords, cross_distance
        pocket_coords  = pocket_coords[drop_protein_ids]
        cross_distance = cross_distance[:, drop_protein_ids]
        print('pocket_coords:', pocket_coords.shape)
        print('cross_distance:', cross_distance.shape)

        
        if old_data['pos'].shape[0] <= 0:
            print("old_data['pos'].size(0):", old_data['pos'].shape[0])
            print("old_data['pos']:", old_data['pos'].shape)
            return old_data
        else:
            data = {}
            #self.ligand_centor
            dis   = np.linalg.norm(old_data['pos'] - self.ligand_centor, axis = 1) #

        #try:
            #，12ai，400，400

            if pocket_coords.shape[0] < 300:
                cutoff_num = 1000
            else:
                cutoff_num = 400
            
            '''
            #，，rdkit
            #print('self.protein_file:', self.protein_file)
            pro_mol=Chem.MolFromPDBFile(self.protein_file,removeHs=False,sanitize=False)
            atom_isspecial = np.array([atom.IsInRing() or atom.GetSymbol() == 'O' or atom.GetSymbol() == 'N' for atom in pro_mol.GetAtoms()]) #
            atom_id = np.array([atom.GetIdx() for atom in pro_mol.GetAtoms()])
            #coords=np.array(pro_mol.GetConformer(0).GetPositions(), dtype=np.float32) #,float32，float64
            all_coords=np.array(pro_mol.GetConformer(0).GetPositions())
            '''

            atom_isO, atom_isN, atom_isring, atom_isspecial, atom_id, all_coords = self._read_pdb_ONRing_biopandas(self.protein_file)
            
            cutoff_indices = np.argsort(dis)[:cutoff_num] #,cutoff_num，，
            coords = all_coords[cutoff_indices] #，

            #，pocket_coords，400
            new_indices = []
            pro_flag_dict = {}

            for j, c in zip(cutoff_indices, coords): #j，，
                k = torch.FloatTensor(c)
                tg = ''
                for ii in k:
                    #tg += str(round(i.item(), 4)) + '_'
                    tg += str(self.truncate(ii.item(), 3)) + '_' #，2，
                k = str(tg)

                pro_flag_dict[k] = j
            

            if len(pro_flag_dict) != len(coords):
                raise Exception(f"{len(pro_flag_dict)} != {len(coords)}")

            cutoff_protein_ids = torch.ones(pocket_coords.shape[0], dtype = torch.bool)
            for ids, c in enumerate(pocket_coords):
                k = torch.FloatTensor(c)
                tg = ''
                for ii in k:
                    #tg += str(round(i.item(), 4)) + '_'
                    tg += str(self.truncate(ii.item(), 3)) + '_' #，2，
                k = str(tg)

                #pocket_coords，
                #if cross_pro_flag_dict.get(c_sum):
                try:
                    new_indices.append(pro_flag_dict[k]) #，unimol，，
                except KeyError as e:
                    cutoff_protein_ids[ids] = False
                    print('key error, skip')
                    continue
            
            #，，
            pocket_coords  = pocket_coords[cutoff_protein_ids]
            cross_distance = cross_distance[:, cutoff_protein_ids]

            cross_pro_isring_flag = cross_pro_isring_flag[cutoff_protein_ids]
            cross_pro_isO_flag    = cross_pro_isO_flag[cutoff_protein_ids]
            cross_pro_isN_flag    = cross_pro_isN_flag[cutoff_protein_ids]
            
            #if len(new_indices) != len(self.unimol_pcoords[0]):
                #raise Exception(f'{len(new_indices)} <= 0')

            print('len(new_indices):', len(new_indices))
            #print('new_indices:', new_indices)

            new_atom_isspecial = atom_isspecial[new_indices] #atom_isspecial
            new_atom_id = atom_id[new_indices] #atom_id
            nonzero_indices = new_indices

            #，
            atom_isspecial_index = np.nonzero(new_atom_isspecial == True)[0]
            atom_isgeneral_index = np.nonzero(new_atom_isspecial == False)[0]
            

            #rdkit，，
            #new_atom_id
            sub_protein_file = os.path.join(os.path.dirname(self.protein_file), os.path.splitext(os.path.basename(self.protein_file))[0] + '_400.pdb')

            def keep_atoms_by_id(mol, atom_ids):
                """ID"""
                editable_mol = Chem.EditableMol(mol)
                all_atoms = list(editable_mol.GetMol().GetAtoms())
                atoms_to_keep = {atom.GetIdx() for atom in all_atoms if atom.GetIdx() in atom_ids}
                
                atoms_to_remove = [atom.GetIdx() for atom in all_atoms if atom.GetIdx() not in atom_ids]
                atoms_to_remove.sort(reverse=True)  # ，

                for atom_id in atoms_to_remove:
                    editable_mol.RemoveAtom(atom_id)

                return editable_mol.GetMol()
            

            # ID（0）
            atom_ids_to_keep = new_atom_id  # ID，

            pro_mol=Chem.MolFromPDBFile(self.protein_file,removeHs=False,sanitize=False)
            # 
            new_pro_mol = keep_atoms_by_id(pro_mol, atom_ids_to_keep)
            Chem.MolToPDBFile(new_pro_mol, sub_protein_file)



            #，
            #400pdb，,# 5l8c/5l8c_pocket10.pdb，5l8c/5l8c_pocket10_400.pdb
            #nonzero_indices
            #: '../CrossDocked2020/data/pdbbind2020_r10/v2020-other-PL/5qb1/5qb1_pocket10_400.pdb'

            '''
            sub_protein_file = os.path.join(os.path.dirname(self.protein_file), os.path.splitext(os.path.basename(self.protein_file))[0] + '_400.pdb')
            sub_protein_w = open(sub_protein_file, 'w')

            org_pro_list = []
            count = 0
            with open(self.protein_file, 'r')as f:
                for i, line in enumerate(f):
                    if line[0:6].strip() == 'ATOM' and i in nonzero_indices:
                        sub_protein_w.write(line)
                

                sub_protein_w.write('END')
            sub_protein_w.close()

            '''
        
            #print('ok3')

            #O,N, ，rdkitpdb，，rdkit，
            #。，，rdkit，
            #，，，rdkit
            '''
            pro_mol=Chem.MolFromPDBFile(self.protein_file,removeHs=False,sanitize=False)
            #print('ok1')
            atom_isring=np.array([atom.IsInRing() for atom in pro_mol.GetAtoms()])    #
            atom_isO=np.array([atom.GetSymbol() == 'O'  for atom in pro_mol.GetAtoms()])    #O
            atom_isN=np.array([atom.GetSymbol() == 'N'  for atom in pro_mol.GetAtoms()])    #N
            coords=np.array(pro_mol.GetConformer(0).GetPositions(), dtype=np.float32) #,float32，float64
            #coords=np.array(pro_mol.GetConformer(0).GetPositions())
            '''
            atom_isO, atom_isN, atom_isring, atom_isspecial, atom_id, coords = self._read_pdb_ONRing_biopandas(self.protein_file)

            #，，-，
            #indexs = []  #, 
            target_xyz =  np.array(self.pos) #xyz
            #target_xyz_sum =  np.round(np.array(self.pos, dtype=np.float32).sum(axis = -1), 2)

            #assert np.array_equal(target_xyz, coords) #，, postbus, assert，try, except

            if not np.array_equal(target_xyz, coords):
                print('target_xyz.shape:', target_xyz.shape)
                print('coords.shape:', coords.shape)
                raise Exception('np.array_equal(target_xyz, coords):', np.array_equal(target_xyz, coords))

            new_atom_isring = atom_isring
            new_atom_isO = atom_isO
            new_atom_isN = atom_isN

            assert len(new_atom_isN[nonzero_indices]) == len(np.array(self.element, dtype=np.int64)[nonzero_indices])
            #print('ok4')

            data = {
                'element': np.array(self.element, dtype=np.int64)[nonzero_indices], #
                'molecule_name': self.title, # pocket
                'pos': np.array(self.pos, dtype=np.float32)[nonzero_indices], #
                'is_backbone': np.array(self.is_backbone, dtype=np.bool_)[nonzero_indices], #Boolean，
                'atom_name': [self.atom_name[i] for i in nonzero_indices], #
                'atom_to_aa_type': np.array(self.atom_to_aa_type, dtype=np.int64)[nonzero_indices], ##
                'atom_isring': new_atom_isring[nonzero_indices],
                'atom_isO': new_atom_isO[nonzero_indices],
                'atom_isN': new_atom_isN[nonzero_indices],

                'cross_lig_isring_flag': cross_lig_isring_flag,
                'cross_lig_isO_flag': cross_lig_isO_flag,
                'cross_lig_isN_flag': cross_lig_isN_flag,

                'cross_pro_isring_flag': cross_pro_isring_flag,
                'cross_pro_isO_flag': cross_pro_isO_flag,
                'cross_pro_isN_flag': cross_pro_isN_flag,

                'cross_ligand': holo_coords,
                'cross_protein': pocket_coords,

                #'cross_distance': set(torch.from_numpy(cross_distance)), #，，？，
                #'cross_distance': list(torch.from_numpy(cross_distance)), #pyglist，
                #'cross_distance': tuple(torch.from_numpy(cross_distance)), #tuple，pyg
                #'cross_distance': OrderedSet(torch.from_numpy(cross_distance)), #，pyg，numpy

                'cross_distance': cross_distance,

                'link_e': np.zeros([10, 2], dtype = int),
                'link_t': np.zeros([10], dtype = int),
                'link_e_reverse': np.zeros([10, 2], dtype = int),
                'link_t_reverse': np.zeros([10], dtype = int),

            }

            #print('ok5')
            return data
        
        '''
        except Exception as e:
            print('protein error:', e)
            print('self.protein_file:', self.protein_file)
            #raise Exception('protein error, stop')
            #exit()
            return None
        '''
            

    def to_dict_atom_interaction_org(self):
        #，，6ai，

        old_data = {
            'element': np.array(self.element, dtype=np.int64), #
            'molecule_name': self.title, # pocket
            'pos': np.array(self.pos, dtype=np.float32), #
            'is_backbone': np.array(self.is_backbone, dtype=np.bool_), #Boolean，
            'atom_name': self.atom_name, #
            'atom_to_aa_type': np.array(self.atom_to_aa_type, dtype=np.int64) ##
        }

        #，400

        if 'pdbbind2020' in self.protein_file:
            file_path = os.path.join(os.path.dirname(self.protein_file), 'interaction_' + os.path.basename(self.protein_file).split('_')[0] + '_v2.pkl')   #5S8I_protein.pdb
        else:
            file_path = os.path.join(os.path.dirname(self.protein_file), 'interaction_' + os.path.basename(self.protein_file).split('_')[0] + '.pkl')   #5S8I_protein.pdb
        with open(file_path, 'rb') as file:
            interaction_data = dill.load(file)
        holo_coords_list = interaction_data['holo_coords_list']
        coords_predict_list = interaction_data['coords_predict_list']
        pocket_coords_list = interaction_data['pocket_coords_list']
        cross_distance_list = interaction_data['cross_distance_list']

        assert np.allclose(holo_coords_list[0], holo_coords_list[-1], atol=0.02)

        #print('pocket_coords_list[0]:', pocket_coords_list[0][:])
        #np.set_printoptions(suppress=True, precision=4)
        #print('pocket_coords_list[-1]:', pocket_coords_list[-1]) #，？？，
        #print('pocket_coords_list[0].shape:', pocket_coords_list[0].shape)
        #print('pocket_coords_list[-1].shape:', pocket_coords_list[-1].shape) #，？？
        #assert np.allclose(pocket_coords_list[0], pocket_coords_list[-1], atol=0.02)

        assert len(holo_coords_list) == len(coords_predict_list) and len(holo_coords_list) == len(pocket_coords_list) and len(holo_coords_list) == len(cross_distance_list)

        
        #rmsd，，
        rmsd_list = []
        for pre_pos, holo_pos in zip(coords_predict_list, holo_coords_list):
            assert pre_pos.shape == holo_pos.shape   #"Coordinate matrices must have the same shape"
            rmsd = np.sqrt(np.mean(np.sum((pre_pos - holo_pos) ** 2, axis=1)))
            rmsd_list.append(rmsd)
        
        sorted_indices = np.argsort(rmsd_list)
        best_index = sorted_indices[0]
        
        #
        #best_index  = random.choice(list(range(len(holo_coords_list))))



        holo_coords = holo_coords_list[best_index]
        coords_predict = coords_predict_list[best_index]
        pocket_coords = pocket_coords_list[best_index]
        #cross_distance = cross_distance_list[best_index]


        #
        # A  (n, 3)，B  (m, 3)
        #  A  B 
        #numpy
        diff = np.expand_dims(holo_coords, axis=1) - np.expand_dims(pocket_coords, axis=0)
        cross_distance = np.sqrt(np.sum(diff**2, axis=2))  # dist_matrix  (n, m)

        #torch
        #diff = holo_coords.unsqueeze(1) - pocket_coords.unsqueeze(0)  # diff  (n, m, 3)
        #cross_distance = torch.sqrt(torch.sum(diff**2, dim=2))  # dist_matrix  (n, m)

        #unimol，，
        unique_index_dict       = {}
        for j, ps in enumerate(pocket_coords):
            unique_index_dict[tuple(ps)] = j
            #，
        
        unique_index_list   = list(unique_index_dict.values())
        pocket_coords       = pocket_coords[unique_index_list]
        cross_distance      = cross_distance[:, unique_index_list]



        #，unimolrdkit，，
        #cross_distanceO,N,
        #，
        pro_isring_flag = {}
        pro_isO_flag = {}
        pro_isN_flag = {}

        '''
        pro_mol=Chem.MolFromPDBFile(self.protein_file,removeHs=False,sanitize=False) #，, 
        #print('self.protein_file:', self.protein_file)
        #print('ok1')
        atom_isring=np.array([atom.IsInRing() for atom in pro_mol.GetAtoms()])    #
        atom_isO=np.array([atom.GetSymbol() == 'O'  for atom in pro_mol.GetAtoms()])    #O
        atom_isN=np.array([atom.GetSymbol() == 'N'  for atom in pro_mol.GetAtoms()])    #N
        #coords=np.array(pro_mol.GetConformer(0).GetPositions(), dtype=np.float32) #,float32，float64
        coords=np.array(pro_mol.GetConformer(0).GetPositions())
        '''
        atom_isO, atom_isN, atom_isring, atom_isspecial, atom_id, coords = self._read_pdb_ONRing_biopandas(self.protein_file)

        count = 0
        coords_atom_dict = defaultdict(list)
        for r, o, n, c in zip(atom_isring, atom_isO, atom_isN, coords):
            count += 1
            k = torch.FloatTensor(c)
            tg = ''
            for ii in k:
                #tg += str(round(i.item(), 4)) + '_'
                tg += str(self.truncate(ii.item(), 3)) + '_' #，2，
            c_sum = str(tg)

            coords_atom_dict[c_sum].append(c)

            pro_isring_flag[c_sum] = r
            pro_isO_flag[c_sum]    = o
            pro_isN_flag[c_sum]    = n
        

        print('：', count)
        if len(pro_isring_flag) != len(coords) or len(pro_isO_flag) != len(coords) or len(pro_isN_flag) != len(coords):
            print(f'{len(pro_isring_flag)} != {len(coords)} or {len(pro_isO_flag)} != {len(coords)} or {len(pro_isN_flag)} != {len(coords)}')
            #979 != 990 or 979 != 990 or 979 != 990 #？
            raise Exception("pro atom num is error")


        #cross_distance, cross_ligand, cross_protein, cross_ligand_atom_flag, cross_protein_atom_flag
        #，
        pro_isring_flag_list = []
        pro_isO_flag_list = []
        pro_isN_flag_list = []




        drop_protein_ids = torch.ones(pocket_coords.shape[0], dtype = torch.bool)
        test_dict = {}

        #unimolrdkit，rdkit，，unimolrdkit，
        for ids, c in enumerate(pocket_coords):
            k = torch.FloatTensor(c)
            tg = ''
            for ii in k:
                #tg += str(round(i.item(), 4)) + '_'
                tg += str(self.truncate(ii.item(), 3)) + '_' #，2，
            c_sum = str(tg)
            test_dict[c_sum] = ids

            try:
                pro_isring_flag_list.append(pro_isring_flag[c_sum])
                pro_isO_flag_list.append(pro_isO_flag[c_sum])
                pro_isN_flag_list.append(pro_isN_flag[c_sum])
            except KeyError as e:
                print('error:', e)
                #drop_protein_ids.append(ids) #，
                drop_protein_ids[ids] = False
                continue
        

        assert len(test_dict) == len(pocket_coords)

        #assert len(pro_isring_flag_list) == len(pocket_coords) #，rdkit，

        

        cross_pro_isring_flag = np.stack(pro_isring_flag_list, axis = 0)
        cross_pro_isO_flag = np.stack(pro_isO_flag_list, axis = 0)
        cross_pro_isN_flag = np.stack(pro_isN_flag_list, axis = 0)


        #，,
        #，unimolrdkit，unimolrdkit，sdf，rdkit mol，rdkit
        #，unimol，rdkit
        lig_isring_flag = {}
        lig_isO_flag = {}
        lig_isN_flag = {}


        #，rdkit3d，，
        if self.data_flag == 'new_test':
            lig_mol = copy.deepcopy(self.ligand_dict['mol']) #
            #，？cross，，
            lig_rdkit_mol = copy.deepcopy(self.ligand_dict['rd_mol'])

            unimol_pos  = torch.FloatTensor(holo_coords)
            rdkit_pos   = torch.FloatTensor(lig_rdkit_mol.GetConformer(0).GetPositions())
            ground_pos  = torch.FloatTensor(lig_mol.GetConformer(0).GetPositions())

            #
            assert len(ground_pos) == len(unimol_pos) #，rkdit，，，

            #，rdkit, rdkitground
            if self.compare_atom_order(lig_mol, lig_rdkit_mol) and torch.allclose(ground_pos, unimol_pos, atol=0.02):
                lig_mol     = copy.deepcopy(lig_rdkit_mol)
                holo_coords = np.array(lig_rdkit_mol.GetConformer(0).GetPositions())
        
        else:
            lig_mol = copy.deepcopy(self.ligand_dict['mol']) #





        #print('ok1')
        atom_isring=np.array([atom.IsInRing() for atom in lig_mol.GetAtoms()])    #
        atom_isO=np.array([atom.GetSymbol() == 'O'  for atom in lig_mol.GetAtoms()])    #O
        atom_isN=np.array([atom.GetSymbol() == 'N'  for atom in lig_mol.GetAtoms()])    #N
        #coords=np.array(lig_mol.GetConformer(0).GetPositions(), dtype=np.float32) #,float32，float64
        coords=np.array(lig_mol.GetConformer(0).GetPositions())

        for r, o, n, c in zip(atom_isring, atom_isO, atom_isN, coords):
            k = torch.FloatTensor(c)
            tg = ''
            for ii in k:
                #tg += str(round(i.item(), 4)) + '_'
                tg += str(self.truncate(ii.item(), 3)) + '_' #，2，
            c_sum = str(tg)

            lig_isring_flag[c_sum] = r
            lig_isO_flag[c_sum]    = o
            lig_isN_flag[c_sum]    = n
        
        assert len(lig_isring_flag) == len(atom_isring)

        assert len(coords) == len(holo_coords) #，rkdit，，，

        if len(lig_isring_flag) != len(coords) or len(lig_isO_flag) != len(coords) or len(lig_isN_flag) != len(coords) or len(coords) != len(holo_coords):
            print(f'{len(coords)} != {len(holo_coords)}') #19 != 18, rdkit，crosss_liand，
            raise Exception("lig atom num is error") #，

        
        assert len(coords) == len(holo_coords) #，rkdit，，，

        #unimolrdkit, 0.02
        assert torch.allclose(torch.FloatTensor(coords), torch.FloatTensor(holo_coords), atol=0.02)

        #，holo_coordsrdkit，
        if not torch.allclose(torch.FloatTensor(coords), torch.FloatTensor(holo_coords), atol=0.02):
            holo_coords_ids_dict  = {}
            rdkit_coords_ids_dict = {}

            for ids, c in enumerate(holo_coords):
                k = torch.FloatTensor(c)
                tg = ''
                for ii in k:
                    #tg += str(round(i.item(), 4)) + '_'
                    tg += str(self.truncate(ii.item(), 3)) + '_' #，2，
                c_sum = str(tg)
                holo_coords_ids_dict[c_sum] = ids

            assert len(holo_coords_ids_dict) == len(holo_coords)

            for ids, c in enumerate(coords):
                k = torch.FloatTensor(c)
                tg = ''
                for ii in k:
                    #tg += str(round(i.item(), 4)) + '_'
                    tg += str(self.truncate(ii.item(), 3)) + '_' #，2，
                c_sum = str(tg)
                rdkit_coords_ids_dict[c_sum] = ids

            assert len(rdkit_coords_ids_dict) == len(coords)

            unimol_true_index = []
            for k in holo_coords_ids_dict:
                v = rdkit_coords_ids_dict[k]
                unimol_true_index.append(v)
            
            #
            coords_predict  = coords_predict[unimol_true_index]
            holo_coords     = holo_coords[unimol_true_index]
            cross_distance  = cross_distance[unimol_true_index]

        #cross_distance, cross_ligand, cross_protein, cross_ligand_atom_flag, cross_protein_atom_flag
        #，
        lig_isring_flag_list = []
        lig_isO_flag_list = []
        lig_isN_flag_list = []

        test_dict = {}
        for c in holo_coords:
            k = torch.FloatTensor(c)
            tg = ''
            for ii in k:
                #tg += str(round(i.item(), 4)) + '_'
                tg += str(self.truncate(ii.item(), 3)) + '_' #，2，
            c_sum = str(tg)
            test_dict[c_sum] = c

            try:
                lig_isring_flag_list.append(lig_isring_flag[c_sum])
                lig_isO_flag_list.append(lig_isO_flag[c_sum])
                lig_isN_flag_list.append(lig_isN_flag[c_sum])
            except KeyError as e:
                print('ligand error:', e)
                print('c:', c)
                print('lig_isring_flag.keys:', list(lig_isring_flag.keys()))
                #drop_protein_ids.append(ids) #，
                #drop_protein_ids[ids] = False
                raise Exception('ligand atom key error')
                continue
        
        assert len(holo_coords) == len(test_dict)

        
        cross_lig_isring_flag = np.stack(lig_isring_flag_list, axis = 0)
        cross_lig_isO_flag = np.stack(lig_isO_flag_list, axis = 0)
        cross_lig_isN_flag = np.stack(lig_isN_flag_list, axis = 0)

        #drop_protein_idsholo_coords, coords_predict, pocket_coords, cross_distance
        pocket_coords  = pocket_coords[drop_protein_ids]
        cross_distance = cross_distance[:, drop_protein_ids]
        print('pocket_coords:', pocket_coords.shape)
        print('cross_distance:', cross_distance.shape)

        
        if old_data['pos'].shape[0] <= 0:
            print("old_data['pos'].size(0):", old_data['pos'].shape[0])
            print("old_data['pos']:", old_data['pos'].shape)
            return old_data
        else:
            data = {}
            #self.ligand_centor
            dis   = np.linalg.norm(old_data['pos'] - self.ligand_centor, axis = 1) #

            try:
                #，12ai，400，400

                if pocket_coords.shape[0] < 300:
                    cutoff_num = 1000
                else:
                    cutoff_num = 400

                #，，rdkit
                #print('self.protein_file:', self.protein_file)
                '''
                pro_mol=Chem.MolFromPDBFile(self.protein_file,removeHs=False,sanitize=False)
                atom_isspecial = np.array([atom.IsInRing() or atom.GetSymbol() == 'O' or atom.GetSymbol() == 'N' for atom in pro_mol.GetAtoms()]) #
                atom_id = np.array([atom.GetIdx() for atom in pro_mol.GetAtoms()])
                #coords=np.array(pro_mol.GetConformer(0).GetPositions(), dtype=np.float32) #,float32，float64
                all_coords=np.array(pro_mol.GetConformer(0).GetPositions())
                '''
                atom_isO, atom_isN, atom_isring, atom_isspecial, atom_id, all_coords = self._read_pdb_ONRing_biopandas(self.protein_file)
                cutoff_indices = np.argsort(dis)[:cutoff_num] #,cutoff_num，，
                coords = all_coords[cutoff_indices] #，

                #，pocket_coords，400
                new_indices = []
                pro_flag_dict = {}

                for j, c in zip(cutoff_indices, coords): #j，，
                    k = torch.FloatTensor(c)
                    tg = ''
                    for ii in k:
                        #tg += str(round(i.item(), 4)) + '_'
                        tg += str(self.truncate(ii.item(), 3)) + '_' #，2，
                    k = str(tg)

                    pro_flag_dict[k] = j
                

                if len(pro_flag_dict) != len(coords):
                    raise Exception(f"{len(pro_flag_dict)} != {len(coords)}")

                cutoff_protein_ids = torch.ones(pocket_coords.shape[0], dtype = torch.bool)
                for ids, c in enumerate(pocket_coords):
                    k = torch.FloatTensor(c)
                    tg = ''
                    for ii in k:
                        #tg += str(round(i.item(), 4)) + '_'
                        tg += str(self.truncate(ii.item(), 3)) + '_' #，2，
                    k = str(tg)

                    #pocket_coords，
                    #if cross_pro_flag_dict.get(c_sum):
                    try:
                        new_indices.append(pro_flag_dict[k]) #，unimol，，
                    except KeyError as e:
                        cutoff_protein_ids[ids] = False
                        print('key error, skip')
                        continue
                
                #，，
                pocket_coords  = pocket_coords[cutoff_protein_ids]
                cross_distance = cross_distance[:, cutoff_protein_ids]

                cross_pro_isring_flag = cross_pro_isring_flag[cutoff_protein_ids]
                cross_pro_isO_flag    = cross_pro_isO_flag[cutoff_protein_ids]
                cross_pro_isN_flag    = cross_pro_isN_flag[cutoff_protein_ids]
                
                #if len(new_indices) != len(self.unimol_pcoords[0]):
                    #raise Exception(f'{len(new_indices)} <= 0')

                print('len(new_indices):', len(new_indices))
                #print('new_indices:', new_indices)

                new_atom_isspecial = atom_isspecial[new_indices] #atom_isspecial
                new_atom_id = atom_id[new_indices] #atom_id
                nonzero_indices = new_indices

                #，
                atom_isspecial_index = np.nonzero(new_atom_isspecial == True)[0]
                atom_isgeneral_index = np.nonzero(new_atom_isspecial == False)[0]
                

                #rdkit，，
                #new_atom_id
                sub_protein_file = os.path.join(os.path.dirname(self.protein_file), os.path.splitext(os.path.basename(self.protein_file))[0] + '_400.pdb')

                def keep_atoms_by_id(mol, atom_ids):
                    """ID"""
                    editable_mol = Chem.EditableMol(mol)
                    all_atoms = list(editable_mol.GetMol().GetAtoms())
                    atoms_to_keep = {atom.GetIdx() for atom in all_atoms if atom.GetIdx() in atom_ids}
                    
                    atoms_to_remove = [atom.GetIdx() for atom in all_atoms if atom.GetIdx() not in atom_ids]
                    atoms_to_remove.sort(reverse=True)  # ，

                    for atom_id in atoms_to_remove:
                        editable_mol.RemoveAtom(atom_id)

                    return editable_mol.GetMol()
                

                # ID（0）
                atom_ids_to_keep = new_atom_id  # ID，
                pro_mol=Chem.MolFromPDBFile(self.protein_file,removeHs=False,sanitize=False)
                # 
                new_pro_mol = keep_atoms_by_id(pro_mol, atom_ids_to_keep)
                Chem.MolToPDBFile(new_pro_mol, sub_protein_file)



                #，
                #400pdb，,# 5l8c/5l8c_pocket10.pdb，5l8c/5l8c_pocket10_400.pdb
                #nonzero_indices
                #: '../CrossDocked2020/data/pdbbind2020_r10/v2020-other-PL/5qb1/5qb1_pocket10_400.pdb'

                '''
                sub_protein_file = os.path.join(os.path.dirname(self.protein_file), os.path.splitext(os.path.basename(self.protein_file))[0] + '_400.pdb')
                sub_protein_w = open(sub_protein_file, 'w')

                org_pro_list = []
                count = 0
                with open(self.protein_file, 'r')as f:
                    for i, line in enumerate(f):
                        if line[0:6].strip() == 'ATOM' and i in nonzero_indices:
                            sub_protein_w.write(line)
                    

                    sub_protein_w.write('END')
                sub_protein_w.close()

                '''
            
                #print('ok3')

                #O,N, ，rdkitpdb，，rdkit，
                #。，，rdkit，
                #，，，rdkit
                '''
                pro_mol=Chem.MolFromPDBFile(self.protein_file,removeHs=False,sanitize=False)
                #print('ok1')
                atom_isring=np.array([atom.IsInRing() for atom in pro_mol.GetAtoms()])    #
                atom_isO=np.array([atom.GetSymbol() == 'O'  for atom in pro_mol.GetAtoms()])    #O
                atom_isN=np.array([atom.GetSymbol() == 'N'  for atom in pro_mol.GetAtoms()])    #N
                coords=np.array(pro_mol.GetConformer(0).GetPositions(), dtype=np.float32) #,float32，float64
                #coords=np.array(pro_mol.GetConformer(0).GetPositions())
                '''
                atom_isO, atom_isN, atom_isring, atom_isspecial, atom_id, coords = self._read_pdb_ONRing_biopandas(self.protein_file)
                #，，-，
                #indexs = []  #, 
                target_xyz =  np.array(self.pos) #xyz
                #target_xyz_sum =  np.round(np.array(self.pos, dtype=np.float32).sum(axis = -1), 2)

                #assert np.array_equal(target_xyz, coords) #，, postbus, assert，try, except

                if not np.array_equal(target_xyz, coords):
                    print('target_xyz.shape:', target_xyz.shape)
                    print('coords.shape:', coords.shape)
                    raise Exception('np.array_equal(target_xyz, coords):', np.array_equal(target_xyz, coords))

                new_atom_isring = atom_isring
                new_atom_isO = atom_isO
                new_atom_isN = atom_isN

                assert len(new_atom_isN[nonzero_indices]) == len(np.array(self.element, dtype=np.int64)[nonzero_indices])
                #print('ok4')

                data = {
                    'element': np.array(self.element, dtype=np.int64)[nonzero_indices], #
                    'molecule_name': self.title, # pocket
                    'pos': np.array(self.pos, dtype=np.float32)[nonzero_indices], #
                    'is_backbone': np.array(self.is_backbone, dtype=np.bool_)[nonzero_indices], #Boolean，
                    'atom_name': [self.atom_name[i] for i in nonzero_indices], #
                    'atom_to_aa_type': np.array(self.atom_to_aa_type, dtype=np.int64)[nonzero_indices], ##
                    'atom_isring': new_atom_isring[nonzero_indices],
                    'atom_isO': new_atom_isO[nonzero_indices],
                    'atom_isN': new_atom_isN[nonzero_indices],

                    'cross_lig_isring_flag': cross_lig_isring_flag,
                    'cross_lig_isO_flag': cross_lig_isO_flag,
                    'cross_lig_isN_flag': cross_lig_isN_flag,

                    'cross_pro_isring_flag': cross_pro_isring_flag,
                    'cross_pro_isO_flag': cross_pro_isO_flag,
                    'cross_pro_isN_flag': cross_pro_isN_flag,

                    'cross_ligand': holo_coords,
                    'cross_protein': pocket_coords,

                    #'cross_distance': set(torch.from_numpy(cross_distance)), #，，？，
                    #'cross_distance': list(torch.from_numpy(cross_distance)), #pyglist，
                    #'cross_distance': tuple(torch.from_numpy(cross_distance)), #tuple，pyg
                    #'cross_distance': OrderedSet(torch.from_numpy(cross_distance)), #，pyg，numpy

                    'cross_distance': cross_distance,

                    'link_e': np.zeros([10, 2], dtype = int),
                    'link_t': np.zeros([10], dtype = int),
                    'link_e_reverse': np.zeros([10, 2], dtype = int),
                    'link_t_reverse': np.zeros([10], dtype = int),

                }

                #print('ok5')
                return data
            
            
            except Exception as e:
                print('protein error:', e)
                print('self.protein_file:', self.protein_file)
                #raise Exception('protein error, stop')
                #exit()
                return None




    def to_dict_atom_interaction_v2(self):
        #，，6ai，

        old_data = {
            'element': np.array(self.element, dtype=np.int64), #
            'molecule_name': self.title, # pocket
            'pos': np.array(self.pos, dtype=np.float32), #
            'is_backbone': np.array(self.is_backbone, dtype=np.bool_), #Boolean，
            'atom_name': self.atom_name, #
            'atom_to_aa_type': np.array(self.atom_to_aa_type, dtype=np.int64) ##
        }

        #，400

        if 'pdbbind2020' in self.protein_file:
            file_path = os.path.join(os.path.dirname(self.protein_file), 'interaction_' + os.path.basename(self.protein_file).split('_')[0] + '_v2.pkl')   #5S8I_protein.pdb
        else:
            file_path = os.path.join(os.path.dirname(self.protein_file), 'interaction_' + os.path.basename(self.protein_file).split('_')[0] + '.pkl')   #5S8I_protein.pdb
        with open(file_path, 'rb') as file:
            interaction_data = dill.load(file)
        holo_coords_list = interaction_data['holo_coords_list']
        coords_predict_list = interaction_data['coords_predict_list']
        pocket_coords_list = interaction_data['pocket_coords_list']
        cross_distance_list = interaction_data['cross_distance_list']

        assert np.allclose(holo_coords_list[0], holo_coords_list[-1], atol=0.02)

        #print('pocket_coords_list[0]:', pocket_coords_list[0][:])
        #np.set_printoptions(suppress=True, precision=4)
        #print('pocket_coords_list[-1]:', pocket_coords_list[-1]) #，？？，
        #print('pocket_coords_list[0].shape:', pocket_coords_list[0].shape)
        #print('pocket_coords_list[-1].shape:', pocket_coords_list[-1].shape) #，？？
        #assert np.allclose(pocket_coords_list[0], pocket_coords_list[-1], atol=0.02)

        assert len(holo_coords_list) == len(coords_predict_list) and len(holo_coords_list) == len(pocket_coords_list) and len(holo_coords_list) == len(cross_distance_list)

        
        #rmsd，，
        rmsd_list = []
        for pre_pos, holo_pos in zip(coords_predict_list, holo_coords_list):
            assert pre_pos.shape == holo_pos.shape   #"Coordinate matrices must have the same shape"
            rmsd = np.sqrt(np.mean(np.sum((pre_pos - holo_pos) ** 2, axis=1)))
            rmsd_list.append(rmsd)
        
        sorted_indices = np.argsort(rmsd_list)
        best_index = sorted_indices[0]
        
        #
        #best_index  = random.choice(list(range(len(holo_coords_list))))



        if self.cross_distance_num == None or self.cross_distance_num == 'best':
            holo_coords = holo_coords_list[best_index]
            coords_predict = coords_predict_list[best_index]
            pocket_coords = pocket_coords_list[best_index]
            cross_distance = cross_distance_list[best_index]
        else:            
            holo_coords = holo_coords_list[self.cross_distance_num]
            coords_predict = coords_predict_list[self.cross_distance_num]
            pocket_coords = pocket_coords_list[self.cross_distance_num]
            cross_distance = cross_distance_list[self.cross_distance_num]



        #unimol，，
        unique_index_dict       = {}
        for j, ps in enumerate(pocket_coords):
            unique_index_dict[tuple(ps)] = j
            #，
        
        unique_index_list   = list(unique_index_dict.values())
        pocket_coords       = pocket_coords[unique_index_list]
        cross_distance      = cross_distance[:, unique_index_list]


        #，unimolrdkit，，
        #cross_distanceO,N,
        #，
        pro_isring_flag = {}
        pro_isO_flag = {}
        pro_isN_flag = {}

        '''
        pro_mol=Chem.MolFromPDBFile(self.protein_file,removeHs=False,sanitize=False) #，, 
        #print('self.protein_file:', self.protein_file)
        #print('ok1')
        atom_isring=np.array([atom.IsInRing() for atom in pro_mol.GetAtoms()])    #
        atom_isO=np.array([atom.GetSymbol() == 'O'  for atom in pro_mol.GetAtoms()])    #O
        atom_isN=np.array([atom.GetSymbol() == 'N'  for atom in pro_mol.GetAtoms()])    #N
        #coords=np.array(pro_mol.GetConformer(0).GetPositions(), dtype=np.float32) #,float32，float64
        coords=np.array(pro_mol.GetConformer(0).GetPositions())
        '''
        atom_isO, atom_isN, atom_isring, atom_isspecial, atom_id, coords = self._read_pdb_ONRing_biopandas(self.protein_file)
        count = 0
        coords_atom_dict = defaultdict(list)
        for r, o, n, c in zip(atom_isring, atom_isO, atom_isN, coords):
            count += 1
            k = torch.FloatTensor(c)
            tg = ''
            for ii in k:
                #tg += str(round(i.item(), 4)) + '_'
                tg += str(self.truncate(ii.item(), 3)) + '_' #，2，
            c_sum = str(tg)

            coords_atom_dict[c_sum].append(c)

            pro_isring_flag[c_sum] = r
            pro_isO_flag[c_sum]    = o
            pro_isN_flag[c_sum]    = n
        
        assert len(pro_isring_flag) == len(atom_isring)
        
        print('：', count)
        if len(pro_isring_flag) != len(coords) or len(pro_isO_flag) != len(coords) or len(pro_isN_flag) != len(coords):
            print(f'{len(pro_isring_flag)} != {len(coords)} or {len(pro_isO_flag)} != {len(coords)} or {len(pro_isN_flag)} != {len(coords)}')
            #979 != 990 or 979 != 990 or 979 != 990 #？
            raise Exception("pro atom num is error")


        cross_pro_isring_flag = copy.deepcopy(atom_isring)
        cross_pro_isO_flag = copy.deepcopy(atom_isO)
        cross_pro_isN_flag = copy.deepcopy(atom_isN)



        #，,
        #，unimolrdkit，unimolrdkit，sdf，rdkit mol，rdkit
        #，unimol，rdkit
        lig_isring_flag = {}
        lig_isO_flag = {}
        lig_isN_flag = {}


        #，rdkit3d，，
        if self.data_flag == 'new_test':
            
            lig_mol = copy.deepcopy(self.ligand_dict['mol']) #
            #，？cross，，
            lig_rdkit_mol = copy.deepcopy(self.ligand_dict['rd_mol'])

            unimol_pos  = torch.FloatTensor(holo_coords)
            rdkit_pos   = torch.FloatTensor(lig_rdkit_mol.GetConformer(0).GetPositions())
            ground_pos  = torch.FloatTensor(lig_mol.GetConformer(0).GetPositions())

            #
            assert len(ground_pos) == len(unimol_pos) #，rkdit，，，

            #，rdkit, rdkitsmiles3dground，3d，
            if self.compare_atom_order(lig_mol, lig_rdkit_mol) and torch.allclose(ground_pos, unimol_pos, atol=0.02):
                lig_mol     = copy.deepcopy(lig_rdkit_mol) #，org_liandlig_rdkit_mol, 
                holo_coords = np.array(lig_rdkit_mol.GetConformer(0).GetPositions()) 
            
            #lig_mol = copy.deepcopy(self.ligand_dict['mol']) 
        else:
            lig_mol = copy.deepcopy(self.ligand_dict['mol']) #





        #print('ok1')
        atom_isring=np.array([atom.IsInRing() for atom in lig_mol.GetAtoms()])    #
        atom_isO=np.array([atom.GetSymbol() == 'O'  for atom in lig_mol.GetAtoms()])    #O
        atom_isN=np.array([atom.GetSymbol() == 'N'  for atom in lig_mol.GetAtoms()])    #N
        #coords=np.array(lig_mol.GetConformer(0).GetPositions(), dtype=np.float32) #,float32，float64
        coords=np.array(lig_mol.GetConformer(0).GetPositions())

        for r, o, n, c in zip(atom_isring, atom_isO, atom_isN, coords):
            k = torch.FloatTensor(c)
            tg = ''
            for ii in k:
                #tg += str(round(i.item(), 4)) + '_'
                tg += str(self.truncate(ii.item(), 3)) + '_' #，2，
            c_sum = str(tg)

            lig_isring_flag[c_sum] = r
            lig_isO_flag[c_sum]    = o
            lig_isN_flag[c_sum]    = n
        
        assert len(lig_isring_flag) == len(atom_isring)
        assert len(coords) == len(holo_coords) #，rkdit，，，

        if len(lig_isring_flag) != len(coords) or len(lig_isO_flag) != len(coords) or len(lig_isN_flag) != len(coords) or len(coords) != len(holo_coords):
            print(f'{len(coords)} != {len(holo_coords)}') #19 != 18, rdkit，crosss_liand，
            raise Exception("lig atom num is error") #，

    
        #unimolrdkit, 0.02
        assert torch.allclose(torch.FloatTensor(coords), torch.FloatTensor(holo_coords), atol=0.02)

        holo_coords = copy.deepcopy(coords)

        #，rdkitsdf，，，，，，holo_coords = coords
        #cross_distance, cross_ligand, cross_protein, cross_ligand_atom_flag, cross_protein_atom_flag

        cross_lig_isring_flag = copy.deepcopy(atom_isring)
        cross_lig_isO_flag = copy.deepcopy(atom_isO)
        cross_lig_isN_flag = copy.deepcopy(atom_isN)

        
        if old_data['pos'].shape[0] <= 0:
            print("old_data['pos'].size(0):", old_data['pos'].shape[0])
            print("old_data['pos']:", old_data['pos'].shape)
            return old_data
        else:
            data = {}
            #self.ligand_centor
            dis   = np.linalg.norm(old_data['pos'] - self.ligand_centor, axis = 1) #

        #try:
            #，12ai，400，400

            if pocket_coords.shape[0] < 300:
                cutoff_num = 2000
            else:
                cutoff_num = 400

            #，，rdkit
            #print('self.protein_file:', self.protein_file)
            '''
            pro_mol=Chem.MolFromPDBFile(self.protein_file,removeHs=False,sanitize=False)
            atom_isspecial = np.array([atom.IsInRing() or atom.GetSymbol() == 'O' or atom.GetSymbol() == 'N' for atom in pro_mol.GetAtoms()]) #
            atom_id = np.array([atom.GetIdx() for atom in pro_mol.GetAtoms()])
            #coords=np.array(pro_mol.GetConformer(0).GetPositions(), dtype=np.float32) #,float32，float64
            all_coords=np.array(pro_mol.GetConformer(0).GetPositions())
            '''
            atom_isO, atom_isN, atom_isring, atom_isspecial, atom_id, all_coords = self._read_pdb_ONRing_biopandas(self.protein_file)
            cutoff_indices = np.argsort(dis)[:] #,cutoff_num，，
            coords = all_coords[cutoff_indices] #，

            assert old_data['pos'].shape == coords.shape

            assert cutoff_indices.shape == np.unique(cutoff_indices).shape

            #，pocket_coords，400
            new_indices = []
            pro_flag_dict = {} #value
            pro_flag_dict2 = {}

            for j, c in zip(cutoff_indices, coords): #j，，
                k = torch.FloatTensor(c)
                tg = ''
                for ii in k:
                    #tg += str(round(i.item(), 4)) + '_'
                    tg += str(self.truncate(ii.item(), 3)) + '_' #，2，
                k = str(tg)

                pro_flag_dict[k] = j

                k2 = torch.FloatTensor(c)
                tg2 = ''
                for ii in k2:
                    #tg += str(round(i.item(), 4)) + '_'
                    tg2 += str(self.truncate2(ii.item(), 3)) + '_' #，2，
                k2 = str(tg2)

                pro_flag_dict2[k2] = j

            

            if len(pro_flag_dict) != len(coords):
                raise Exception(f"{len(pro_flag_dict)} != {len(coords)}")


            if len(pro_flag_dict2) != len(coords):
                raise Exception(f"{len(pro_flag_dict2)} != {len(coords)}")
            
            #cutoff_protein_ids = torch.ones(pocket_coords.shape[0], dtype = torch.bool)
            cutoff_protein_ids = torch.zeros(pocket_coords.shape[0], dtype = torch.bool)

            assert pocket_coords.shape == np.unique(pocket_coords, axis = 0).shape

            for ids, c in enumerate(pocket_coords): #pocket_coords
                k = torch.FloatTensor(c)
                tg = ''
                for ii in k:
                    #tg += str(round(i.item(), 4)) + '_'
                    tg += str(self.truncate(ii.item(), 3)) + '_' #，2，
                k = str(tg)

                #pocket_coords，
                #if cross_pro_flag_dict.get(c_sum):

                
                try:
                    new_indices.append(pro_flag_dict[k]) #，unimol，，, 
                    cutoff_protein_ids[ids] = True
                except Exception as e:
                    print(e)
                    #cutoff_protein_ids[ids] = False
                    #print('key error, skip:', k)
                    #print('key all:', list(pro_flag_dict.keys()))
                    #print('-----------------------------------------')
                    continue
                    
            #，，
            pocket_coords  = pocket_coords[cutoff_protein_ids]
            cross_distance = cross_distance[:, cutoff_protein_ids]

            cross_pro_isring_flag = cross_pro_isring_flag[new_indices]
            cross_pro_isO_flag    = cross_pro_isO_flag[new_indices]
            cross_pro_isN_flag    = cross_pro_isN_flag[new_indices]

            #print('cutoff_protein_ids num:', torch.sum(cutoff_protein_ids)) # tensor(252)
            #print('len(new_indices):', np.sum(np.array(new_indices))) #len(new_indices): 251

            #print('new_indices1:', sorted(np.nonzero(np.array(new_indices))[0]))
            #print('new_indices2:', sorted(np.array(new_indices2))) #id，
            
            if len(cross_pro_isring_flag) != len(pocket_coords):
                raise Exception(f'{len(cross_pro_isring_flag)} != {len(pocket_coords)}') #Exception: 251 != 252
            
            #if len(new_indices) != len(self.unimol_pcoords[0]):
                #raise Exception(f'{len(new_indices)} <= 0')

            #print('len(new_indices):', len(new_indices)) #
            #print('new_indices:', new_indices)

            new_atom_isspecial = atom_isspecial[new_indices] #atom_isspecial
            new_atom_id = atom_id[new_indices] #atom_id
            nonzero_indices = new_indices

            #，
            atom_isspecial_index = np.nonzero(new_atom_isspecial == True)[0]
            atom_isgeneral_index = np.nonzero(new_atom_isspecial == False)[0]
            

            #rdkit，，
            #new_atom_id
            sub_protein_file = os.path.join(os.path.dirname(self.protein_file), os.path.splitext(os.path.basename(self.protein_file))[0] + '_400.pdb')

            def keep_atoms_by_id(mol, atom_ids):
                """ID"""
                editable_mol = Chem.EditableMol(mol)
                all_atoms = list(editable_mol.GetMol().GetAtoms())
                atoms_to_keep = {atom.GetIdx() for atom in all_atoms if atom.GetIdx() in atom_ids}
                
                atoms_to_remove = [atom.GetIdx() for atom in all_atoms if atom.GetIdx() not in atom_ids]
                atoms_to_remove.sort(reverse=True)  # ，

                for atom_id in atoms_to_remove:
                    editable_mol.RemoveAtom(atom_id)

                return editable_mol.GetMol()
            

            # ID（0）
            atom_ids_to_keep = new_atom_id  # ID，
            pro_mol=Chem.MolFromPDBFile(self.protein_file,removeHs=False,sanitize=False)
            # 
            new_pro_mol = keep_atoms_by_id(pro_mol, atom_ids_to_keep)
            Chem.MolToPDBFile(new_pro_mol, sub_protein_file)



            #，
            #400pdb，,# 5l8c/5l8c_pocket10.pdb，5l8c/5l8c_pocket10_400.pdb
            #nonzero_indices
            #: '../CrossDocked2020/data/pdbbind2020_r10/v2020-other-PL/5qb1/5qb1_pocket10_400.pdb'

            '''
            sub_protein_file = os.path.join(os.path.dirname(self.protein_file), os.path.splitext(os.path.basename(self.protein_file))[0] + '_400.pdb')
            sub_protein_w = open(sub_protein_file, 'w')

            org_pro_list = []
            count = 0
            with open(self.protein_file, 'r')as f:
                for i, line in enumerate(f):
                    if line[0:6].strip() == 'ATOM' and i in nonzero_indices:
                        sub_protein_w.write(line)
                

                sub_protein_w.write('END')
            sub_protein_w.close()

            '''
        
            #print('ok3')

            #O,N, ，rdkitpdb，，rdkit，
            #。，，rdkit，
            #，，，rdkit
            '''
            pro_mol=Chem.MolFromPDBFile(self.protein_file,removeHs=False,sanitize=False)
            #print('ok1')
            atom_isring=np.array([atom.IsInRing() for atom in pro_mol.GetAtoms()])    #
            atom_isO=np.array([atom.GetSymbol() == 'O'  for atom in pro_mol.GetAtoms()])    #O
            atom_isN=np.array([atom.GetSymbol() == 'N'  for atom in pro_mol.GetAtoms()])    #N
            coords=np.array(pro_mol.GetConformer(0).GetPositions(), dtype=np.float32) #,float32，float64
            #coords=np.array(pro_mol.GetConformer(0).GetPositions())
            '''
            atom_isO, atom_isN, atom_isring, atom_isspecial, atom_id, coords = self._read_pdb_ONRing_biopandas(self.protein_file)
            #，，-，
            #indexs = []  #, 
            target_xyz =  np.array(self.pos) #xyz
            #target_xyz_sum =  np.round(np.array(self.pos, dtype=np.float32).sum(axis = -1), 2)

            #assert np.array_equal(target_xyz, coords) #，, postbus, assert，try, except

            if not np.array_equal(target_xyz, coords):
                print('target_xyz.shape:', target_xyz.shape)
                print('coords.shape:', coords.shape)
                raise Exception('np.array_equal(target_xyz, coords):', np.array_equal(target_xyz, coords))

            new_atom_isring = atom_isring
            new_atom_isO = atom_isO
            new_atom_isN = atom_isN

            assert len(new_atom_isN[nonzero_indices]) == len(np.array(self.element, dtype=np.int64)[nonzero_indices])
            #print('ok4')



            #

            l2p = [] #
            p2l = [] #
            l2p_type = [] #
            p2l_type = [] #

            #
            ligand_atom_isring = copy.deepcopy(cross_lig_isring_flag)
            ligand_atom_isO    = copy.deepcopy(cross_lig_isO_flag)
            ligand_atom_isN    = copy.deepcopy(cross_lig_isN_flag)

            protein_atom_isring = copy.deepcopy(cross_pro_isring_flag)
            protein_atom_isO    = copy.deepcopy(cross_pro_isO_flag)
            protein_atom_isN    = copy.deepcopy(cross_pro_isN_flag)


            ligand_cross_isring_flag = copy.deepcopy(cross_lig_isring_flag)
            ligand_cross_isO_flag    = copy.deepcopy(cross_lig_isO_flag)
            ligand_cross_isN_flag    = copy.deepcopy(cross_lig_isN_flag)
            ligand_cross_lp_pos      = copy.deepcopy(holo_coords)

            protein_cross_isring_flag = copy.deepcopy(cross_pro_isring_flag)
            protein_cross_isO_flag    = copy.deepcopy(cross_pro_isO_flag) 
            protein_cross_isN_flag    = copy.deepcopy(cross_pro_isN_flag) 
            protein_cross_lp_pos      = copy.deepcopy(pocket_coords)

            cross_distance_matrix = copy.deepcopy(cross_distance) #cross_distancelist，cross_distance

            #[2, n*m,]
            #
            centor = holo_coords.mean(axis = 0)
            l_combinations_isring = self.combinations_optim(torch.from_numpy(holo_coords), torch.from_numpy(pocket_coords), torch.from_numpy(np.array(list(range(len(ligand_atom_isring))))), 
                    torch.from_numpy(np.array(list(range(len(protein_atom_isring))))), torch.from_numpy(ligand_atom_isring), torch.from_numpy(protein_atom_isring), 
                    torch.from_numpy(centor), 
                    torch.from_numpy(ligand_cross_isring_flag), torch.from_numpy(protein_cross_isring_flag), torch.from_numpy(cross_distance_matrix), 
                    torch.from_numpy(ligand_cross_lp_pos), torch.from_numpy(protein_cross_lp_pos), flag = 'ligand')
            
            l_combinations_isO    = self.combinations_optim(torch.from_numpy(holo_coords), torch.from_numpy(pocket_coords), torch.from_numpy(np.array(list(range(len(ligand_atom_isring))))), 
                    torch.from_numpy(np.array(list(range(len(protein_atom_isring))))), torch.from_numpy(ligand_atom_isO), torch.from_numpy(protein_atom_isN), 
                    torch.from_numpy(centor), 
                    torch.from_numpy(ligand_cross_isO_flag), torch.from_numpy(protein_cross_isN_flag), torch.from_numpy(cross_distance_matrix), 
                    torch.from_numpy(ligand_cross_lp_pos), torch.from_numpy(protein_cross_lp_pos), flag = 'ligand')
            
            l_combinations_isN    = self.combinations_optim(torch.from_numpy(holo_coords), torch.from_numpy(pocket_coords), torch.from_numpy(np.array(list(range(len(ligand_atom_isring))))), 
                    torch.from_numpy(np.array(list(range(len(protein_atom_isring))))), torch.from_numpy(ligand_atom_isN), torch.from_numpy(protein_atom_isO), 
                    torch.from_numpy(centor), 
                    torch.from_numpy(ligand_cross_isN_flag), torch.from_numpy(protein_cross_isO_flag), torch.from_numpy(cross_distance_matrix), 
                    torch.from_numpy(ligand_cross_lp_pos), torch.from_numpy(protein_cross_lp_pos), flag = 'ligand')

            if l_combinations_isring.shape[0] != 0:
                l2p.append(l_combinations_isring.numpy()) #2 * N
                l2p_type.append(np.full(l_combinations_isring.shape[1], 5))  #，，2*N,N*2？cat
            if l_combinations_isO.shape[0] != 0:
                l2p.append(l_combinations_isO.numpy())
                l2p_type.append(np.full(l_combinations_isO.shape[1], 6))
            if l_combinations_isN.shape[0] != 0:
                l2p.append(l_combinations_isN.numpy())
                l2p_type.append(np.full(l_combinations_isN.shape[1], 7))

            #print('l_combinations_isring:', l_combinations_isring.shape)
            #print('l_combinations_isO:', l_combinations_isO.shape)
            #print('l_combinations_isN:', l_combinations_isN.shape)

            #
            p_combinations_isring = self.combinations_optim(torch.from_numpy(pocket_coords), torch.from_numpy(holo_coords), torch.from_numpy(np.array(list(range(len(protein_atom_isring))))), 
                    torch.from_numpy(np.array(list(range(len(ligand_atom_isring))))), torch.from_numpy(protein_atom_isring), torch.from_numpy(ligand_atom_isring), 
                    torch.from_numpy(centor), 
                    torch.from_numpy(protein_cross_isring_flag), torch.from_numpy(ligand_cross_isring_flag), torch.from_numpy(cross_distance_matrix), 
                    torch.from_numpy(ligand_cross_lp_pos), torch.from_numpy(protein_cross_lp_pos), flag = 'protein')
            
            p_combinations_isO    = self.combinations_optim(torch.from_numpy(pocket_coords), torch.from_numpy(holo_coords), torch.from_numpy(np.array(list(range(len(protein_atom_isring))))), 
                    torch.from_numpy(np.array(list(range(len(ligand_atom_isring))))), torch.from_numpy(protein_atom_isN), torch.from_numpy(ligand_atom_isO), 
                    torch.from_numpy(centor), 
                    torch.from_numpy(protein_cross_isN_flag), torch.from_numpy(ligand_cross_isO_flag), torch.from_numpy(cross_distance_matrix), 
                    torch.from_numpy(ligand_cross_lp_pos),  torch.from_numpy(protein_cross_lp_pos), flag = 'protein')
            
            p_combinations_isN    = self.combinations_optim(torch.from_numpy(pocket_coords), torch.from_numpy(holo_coords), torch.from_numpy(np.array(list(range(len(protein_atom_isring))))), 
                    torch.from_numpy(np.array(list(range(len(ligand_atom_isring))))), torch.from_numpy(protein_atom_isO), torch.from_numpy(ligand_atom_isN), 
                    torch.from_numpy(centor), 
                    torch.from_numpy(protein_cross_isO_flag), torch.from_numpy(ligand_cross_isN_flag), torch.from_numpy(cross_distance_matrix), 
                    torch.from_numpy(ligand_cross_lp_pos), torch.from_numpy(protein_cross_lp_pos), flag = 'protein')
            
            if p_combinations_isring.shape[0] != 0:
                p2l.append(p_combinations_isring)
                p2l_type.append(np.full(p_combinations_isring.shape[1], 8))  #，，2*N,N*2？cat
            if p_combinations_isO.shape[0] != 0:
                p2l.append(p_combinations_isO)
                p2l_type.append(np.full(p_combinations_isO.shape[1], 9))
            if p_combinations_isN.shape[0] != 0:
                p2l.append(p_combinations_isN)
                p2l_type.append(np.full(p_combinations_isN.shape[1], 10))

            #print('p_combinations_isring:', p_combinations_isring.shape)
            #print('p_combinations_isO:', p_combinations_isO.shape)
            #print('p_combinations_isN:', p_combinations_isN.shape)

            #ON,NO,，，，，knn
            if len(l2p) != 0:
                l2p_edge_index = np.concatenate(l2p, axis = -1, dtype = int)
                l2p_type = np.concatenate(l2p_type, axis = -1, dtype = int)
            else:
                print('')
                exit()
                l2p_edge_index = np.empty((0, 0))
                l2p_type = np.empty((0, 0))
            
            if len(p2l) != 0:
                p2l_edge_index = np.concatenate(p2l, axis = -1, dtype = int)
                p2l_type = np.concatenate(p2l_type, axis = -1, dtype = int)
            else:
                print('')
                exit()
                p2l_edge_index = np.empty((0, 0))
                p2l_type = np.empty((0, 0))

            #id，id，cross_bond_index, cross_bond_type， cross_bond_index_reverse, cross_bond_type_reverse
            #cross_bond_index_reverse, cross_bond_type_reverse, [5,6,7],[8,9,10]
            

            data = {
                'element': np.array(self.element, dtype=np.int64)[nonzero_indices], #
                'molecule_name': self.title, # pocket
                'pos': np.array(self.pos, dtype=np.float32)[nonzero_indices], #
                'is_backbone': np.array(self.is_backbone, dtype=np.bool_)[nonzero_indices], #Boolean，
                'atom_name': [self.atom_name[i] for i in nonzero_indices], #
                'atom_to_aa_type': np.array(self.atom_to_aa_type, dtype=np.int64)[nonzero_indices], ##
                'atom_isring': new_atom_isring[nonzero_indices],
                'atom_isO': new_atom_isO[nonzero_indices],
                'atom_isN': new_atom_isN[nonzero_indices],

                'cross_lig_isring_flag': cross_lig_isring_flag,
                'cross_lig_isO_flag': cross_lig_isO_flag,
                'cross_lig_isN_flag': cross_lig_isN_flag,

                'cross_pro_isring_flag': cross_pro_isring_flag,
                'cross_pro_isO_flag': cross_pro_isO_flag,
                'cross_pro_isN_flag': cross_pro_isN_flag,

                'cross_ligand': holo_coords,
                'cross_protein': pocket_coords,

                #'cross_distance': set(torch.from_numpy(cross_distance)), #，，？，
                #'cross_distance': list(torch.from_numpy(cross_distance)), #pyglist，
                #'cross_distance': tuple(torch.from_numpy(cross_distance)), #tuple，pyg
                #'cross_distance': OrderedSet(torch.from_numpy(cross_distance)), #，pyg，numpy

                'cross_distance': cross_distance,

                #'cross_bond_index': np.ones_like(l2p_edge_index),
                #'cross_distance2': np.ones_like(l2p_edge_index), #？？，，
                #Batch.from_data_list([data.clone() for _ in range(n_data)], follow_batch=FOLLOW_BATCH, exclude_keys = collate_exclude_keys).to(device)，
                #bond？pyg？?

                #'cross_distance1': l2p_edge_index,
                #'cross_distance2': l2p_type,
                #'cross_distance3': p2l_edge_index,
                #'cross_distance4': p2l_type,
                

                'link_e': l2p_edge_index.T,
                'link_t': l2p_type,
                'link_e_reverse': p2l_edge_index.T,
                'link_t_reverse': p2l_type,

                'l_combinations_isring': l_combinations_isring.numpy(),
                'l_combinations_O': l_combinations_isO.numpy(),
                'l_combinations_N': l_combinations_isN.numpy(),


                'p_combinations_isring': p_combinations_isring.numpy(),
                'p_combinations_O': p_combinations_isO.numpy(),
                'p_combinations_N': p_combinations_isN.numpy(),


                #def torchify_dict(data):
            }

            #print('ok5')
            return data
        
        
        #except Exception as e:
            #print('protein error:', e)
            #print('self.protein_file:', self.protein_file)
            #raise Exception('protein error, stop')
            #exit()
            #return None
        
            





    def to_dict_atom_interaction_v2_org(self):
        #

        old_data = {
            'element': np.array(self.element, dtype=np.int64), #
            'molecule_name': self.title, # pocket
            'pos': np.array(self.pos, dtype=np.float32), #
            'is_backbone': np.array(self.is_backbone, dtype=np.bool_), #Boolean，
            'atom_name': self.atom_name, #
            'atom_to_aa_type': np.array(self.atom_to_aa_type, dtype=np.int64) ##
        }

        #，400

        if 'pdbbind2020' in self.protein_file:
            file_path = os.path.join(os.path.dirname(self.protein_file), 'interaction_' + os.path.basename(self.protein_file).split('_')[0] + '_v2.pkl')   #5S8I_protein.pdb
        else:
            file_path = os.path.join(os.path.dirname(self.protein_file), 'interaction_' + os.path.basename(self.protein_file).split('_')[0] + '.pkl')   #5S8I_protein.pdb
        with open(file_path, 'rb') as file:
            interaction_data = dill.load(file)
        holo_coords_list = interaction_data['holo_coords_list']
        coords_predict_list = interaction_data['coords_predict_list']
        pocket_coords_list = interaction_data['pocket_coords_list']
        cross_distance_list = interaction_data['cross_distance_list']

        assert np.allclose(holo_coords_list[0], holo_coords_list[-1], atol=0.02)

        #print('pocket_coords_list[0]:', pocket_coords_list[0][:])
        #np.set_printoptions(suppress=True, precision=4)
        #print('pocket_coords_list[-1]:', pocket_coords_list[-1]) #，？？，
        #print('pocket_coords_list[0].shape:', pocket_coords_list[0].shape)
        #print('pocket_coords_list[-1].shape:', pocket_coords_list[-1].shape) #，？？
        #assert np.allclose(pocket_coords_list[0], pocket_coords_list[-1], atol=0.02)

        assert len(holo_coords_list) == len(coords_predict_list) and len(holo_coords_list) == len(pocket_coords_list) and len(holo_coords_list) == len(cross_distance_list)

        
        #rmsd，，
        rmsd_list = []
        for pre_pos, holo_pos in zip(coords_predict_list, holo_coords_list):
            assert pre_pos.shape == holo_pos.shape   #"Coordinate matrices must have the same shape"
            rmsd = np.sqrt(np.mean(np.sum((pre_pos - holo_pos) ** 2, axis=1)))
            rmsd_list.append(rmsd)
        
        sorted_indices = np.argsort(rmsd_list)
        best_index = sorted_indices[0]
        
        #
        #best_index  = random.choice(list(range(len(holo_coords_list))))



        holo_coords = holo_coords_list[best_index]
        coords_predict = coords_predict_list[best_index]
        pocket_coords = pocket_coords_list[best_index]
        #cross_distance = cross_distance_list[best_index]

        #
        # A  (n, 3)，B  (m, 3)
        #  A  B 
        #numpy
        diff = np.expand_dims(holo_coords, axis=1) - np.expand_dims(pocket_coords, axis=0)
        cross_distance = np.sqrt(np.sum(diff**2, axis=2))  # dist_matrix  (n, m)


        #unimol，，
        unique_index_dict       = {}
        for j, ps in enumerate(pocket_coords):
            unique_index_dict[tuple(ps)] = j
            #，
        
        unique_index_list   = list(unique_index_dict.values())
        pocket_coords       = pocket_coords[unique_index_list]
        cross_distance      = cross_distance[:, unique_index_list]


        #，unimolrdkit，，
        #cross_distanceO,N,
        #，
        pro_isring_flag = {}
        pro_isO_flag = {}
        pro_isN_flag = {}

        '''
        pro_mol=Chem.MolFromPDBFile(self.protein_file,removeHs=False,sanitize=False) #，, 
        #print('self.protein_file:', self.protein_file)
        #print('ok1')
        atom_isring=np.array([atom.IsInRing() for atom in pro_mol.GetAtoms()])    #
        atom_isO=np.array([atom.GetSymbol() == 'O'  for atom in pro_mol.GetAtoms()])    #O
        atom_isN=np.array([atom.GetSymbol() == 'N'  for atom in pro_mol.GetAtoms()])    #N
        #coords=np.array(pro_mol.GetConformer(0).GetPositions(), dtype=np.float32) #,float32，float64
        coords=np.array(pro_mol.GetConformer(0).GetPositions())
        '''
        atom_isO, atom_isN, atom_isring, atom_isspecial, atom_id, coords = self._read_pdb_ONRing_biopandas(self.protein_file)
        count = 0
        coords_atom_dict = defaultdict(list)
        for r, o, n, c in zip(atom_isring, atom_isO, atom_isN, coords):
            count += 1
            k = torch.FloatTensor(c)
            tg = ''
            for ii in k:
                #tg += str(round(i.item(), 4)) + '_'
                tg += str(self.truncate(ii.item(), 3)) + '_' #，2，
            c_sum = str(tg)

            coords_atom_dict[c_sum].append(c)

            pro_isring_flag[c_sum] = r
            pro_isO_flag[c_sum]    = o
            pro_isN_flag[c_sum]    = n
        
        assert len(pro_isring_flag) == len(atom_isring)
        
        print('：', count)
        if len(pro_isring_flag) != len(coords) or len(pro_isO_flag) != len(coords) or len(pro_isN_flag) != len(coords):
            print(f'{len(pro_isring_flag)} != {len(coords)} or {len(pro_isO_flag)} != {len(coords)} or {len(pro_isN_flag)} != {len(coords)}')
            #979 != 990 or 979 != 990 or 979 != 990 #？
            raise Exception("pro atom num is error")


        cross_pro_isring_flag = copy.deepcopy(atom_isring)
        cross_pro_isO_flag = copy.deepcopy(atom_isO)
        cross_pro_isN_flag = copy.deepcopy(atom_isN)



        #，,
        #，unimolrdkit，unimolrdkit，sdf，rdkit mol，rdkit
        #，unimol，rdkit
        lig_isring_flag = {}
        lig_isO_flag = {}
        lig_isN_flag = {}


        #，rdkit3d，，
        if self.data_flag == 'new_test':
            
            lig_mol = copy.deepcopy(self.ligand_dict['mol']) #
            #，？cross，，
            lig_rdkit_mol = copy.deepcopy(self.ligand_dict['rd_mol'])

            unimol_pos  = torch.FloatTensor(holo_coords)
            rdkit_pos   = torch.FloatTensor(lig_rdkit_mol.GetConformer(0).GetPositions())
            ground_pos  = torch.FloatTensor(lig_mol.GetConformer(0).GetPositions())

            #
            assert len(ground_pos) == len(unimol_pos) #，rkdit，，，

            #，rdkit, rdkitsmiles3dground，3d，
            if self.compare_atom_order(lig_mol, lig_rdkit_mol) and torch.allclose(ground_pos, unimol_pos, atol=0.02):
                lig_mol     = copy.deepcopy(lig_rdkit_mol) #，org_liandlig_rdkit_mol, 
                holo_coords = np.array(lig_rdkit_mol.GetConformer(0).GetPositions()) 
            
            #lig_mol = copy.deepcopy(self.ligand_dict['mol']) 
        else:
            lig_mol = copy.deepcopy(self.ligand_dict['mol']) #





        #print('ok1')
        atom_isring=np.array([atom.IsInRing() for atom in lig_mol.GetAtoms()])    #
        atom_isO=np.array([atom.GetSymbol() == 'O'  for atom in lig_mol.GetAtoms()])    #O
        atom_isN=np.array([atom.GetSymbol() == 'N'  for atom in lig_mol.GetAtoms()])    #N
        #coords=np.array(lig_mol.GetConformer(0).GetPositions(), dtype=np.float32) #,float32，float64
        coords=np.array(lig_mol.GetConformer(0).GetPositions())

        for r, o, n, c in zip(atom_isring, atom_isO, atom_isN, coords):
            k = torch.FloatTensor(c)
            tg = ''
            for ii in k:
                #tg += str(round(i.item(), 4)) + '_'
                tg += str(self.truncate(ii.item(), 3)) + '_' #，2，
            c_sum = str(tg)

            lig_isring_flag[c_sum] = r
            lig_isO_flag[c_sum]    = o
            lig_isN_flag[c_sum]    = n
        
        assert len(lig_isring_flag) == len(atom_isring)
        assert len(coords) == len(holo_coords) #，rkdit，，，

        if len(lig_isring_flag) != len(coords) or len(lig_isO_flag) != len(coords) or len(lig_isN_flag) != len(coords) or len(coords) != len(holo_coords):
            print(f'{len(coords)} != {len(holo_coords)}') #19 != 18, rdkit，crosss_liand，
            raise Exception("lig atom num is error") #，

    
        #unimolrdkit, 0.02
        assert torch.allclose(torch.FloatTensor(coords), torch.FloatTensor(holo_coords), atol=0.02)

        holo_coords = copy.deepcopy(coords)

        #，rdkitsdf，，，，，，holo_coords = coords
        #cross_distance, cross_ligand, cross_protein, cross_ligand_atom_flag, cross_protein_atom_flag

        cross_lig_isring_flag = copy.deepcopy(atom_isring)
        cross_lig_isO_flag = copy.deepcopy(atom_isO)
        cross_lig_isN_flag = copy.deepcopy(atom_isN)

        
        if old_data['pos'].shape[0] <= 0:
            print("old_data['pos'].size(0):", old_data['pos'].shape[0])
            print("old_data['pos']:", old_data['pos'].shape)
            return old_data
        else:
            data = {}
            #self.ligand_centor
            dis   = np.linalg.norm(old_data['pos'] - self.ligand_centor, axis = 1) #

        #try:
            #，12ai，400，400

            if pocket_coords.shape[0] < 300:
                cutoff_num = 2000
            else:
                cutoff_num = 400

            #，，rdkit
            #print('self.protein_file:', self.protein_file)
            '''
            pro_mol=Chem.MolFromPDBFile(self.protein_file,removeHs=False,sanitize=False)
            atom_isspecial = np.array([atom.IsInRing() or atom.GetSymbol() == 'O' or atom.GetSymbol() == 'N' for atom in pro_mol.GetAtoms()]) #
            atom_id = np.array([atom.GetIdx() for atom in pro_mol.GetAtoms()])
            #coords=np.array(pro_mol.GetConformer(0).GetPositions(), dtype=np.float32) #,float32，float64
            all_coords=np.array(pro_mol.GetConformer(0).GetPositions())
            '''
            atom_isO, atom_isN, atom_isring, atom_isspecial, atom_id, all_coords = self._read_pdb_ONRing_biopandas(self.protein_file)
            cutoff_indices = np.argsort(dis)[:] #,cutoff_num，，
            coords = all_coords[cutoff_indices] #，

            assert old_data['pos'].shape == coords.shape

            assert cutoff_indices.shape == np.unique(cutoff_indices).shape

            #，pocket_coords，400
            new_indices = []
            pro_flag_dict = {} #value
            pro_flag_dict2 = {}

            for j, c in zip(cutoff_indices, coords): #j，，
                k = torch.FloatTensor(c)
                tg = ''
                for ii in k:
                    #tg += str(round(i.item(), 4)) + '_'
                    tg += str(self.truncate(ii.item(), 3)) + '_' #，2，
                k = str(tg)

                pro_flag_dict[k] = j

                k2 = torch.FloatTensor(c)
                tg2 = ''
                for ii in k2:
                    #tg += str(round(i.item(), 4)) + '_'
                    tg2 += str(self.truncate2(ii.item(), 3)) + '_' #，2，
                k2 = str(tg2)

                pro_flag_dict2[k2] = j

            

            if len(pro_flag_dict) != len(coords):
                raise Exception(f"{len(pro_flag_dict)} != {len(coords)}")


            if len(pro_flag_dict2) != len(coords):
                raise Exception(f"{len(pro_flag_dict2)} != {len(coords)}")
            
            #cutoff_protein_ids = torch.ones(pocket_coords.shape[0], dtype = torch.bool)
            cutoff_protein_ids = torch.zeros(pocket_coords.shape[0], dtype = torch.bool)

            assert pocket_coords.shape == np.unique(pocket_coords, axis = 0).shape

            for ids, c in enumerate(pocket_coords): #pocket_coords
                k = torch.FloatTensor(c)
                tg = ''
                for ii in k:
                    #tg += str(round(i.item(), 4)) + '_'
                    tg += str(self.truncate(ii.item(), 3)) + '_' #，2，
                k = str(tg)

                #pocket_coords，
                #if cross_pro_flag_dict.get(c_sum):

                
                try:
                    new_indices.append(pro_flag_dict[k]) #，unimol，，, 
                    cutoff_protein_ids[ids] = True
                except Exception as e:
                    print(e)
                    #cutoff_protein_ids[ids] = False
                    #print('key error, skip:', k)
                    #print('key all:', list(pro_flag_dict.keys()))
                    #print('-----------------------------------------')
                    continue
                    
            #，，
            pocket_coords  = pocket_coords[cutoff_protein_ids]
            cross_distance = cross_distance[:, cutoff_protein_ids]

            cross_pro_isring_flag = cross_pro_isring_flag[new_indices]
            cross_pro_isO_flag    = cross_pro_isO_flag[new_indices]
            cross_pro_isN_flag    = cross_pro_isN_flag[new_indices]

            #print('cutoff_protein_ids num:', torch.sum(cutoff_protein_ids)) # tensor(252)
            #print('len(new_indices):', np.sum(np.array(new_indices))) #len(new_indices): 251

            #print('new_indices1:', sorted(np.nonzero(np.array(new_indices))[0]))
            #print('new_indices2:', sorted(np.array(new_indices2))) #id，
            
            if len(cross_pro_isring_flag) != len(pocket_coords):
                raise Exception(f'{len(cross_pro_isring_flag)} != {len(pocket_coords)}') #Exception: 251 != 252
            
            #if len(new_indices) != len(self.unimol_pcoords[0]):
                #raise Exception(f'{len(new_indices)} <= 0')

            #print('len(new_indices):', len(new_indices)) #
            #print('new_indices:', new_indices)

            new_atom_isspecial = atom_isspecial[new_indices] #atom_isspecial
            new_atom_id = atom_id[new_indices] #atom_id
            nonzero_indices = new_indices

            #，
            atom_isspecial_index = np.nonzero(new_atom_isspecial == True)[0]
            atom_isgeneral_index = np.nonzero(new_atom_isspecial == False)[0]
            

            #rdkit，，
            #new_atom_id
            sub_protein_file = os.path.join(os.path.dirname(self.protein_file), os.path.splitext(os.path.basename(self.protein_file))[0] + '_400.pdb')

            def keep_atoms_by_id(mol, atom_ids):
                """ID"""
                editable_mol = Chem.EditableMol(mol)
                all_atoms = list(editable_mol.GetMol().GetAtoms())
                atoms_to_keep = {atom.GetIdx() for atom in all_atoms if atom.GetIdx() in atom_ids}
                
                atoms_to_remove = [atom.GetIdx() for atom in all_atoms if atom.GetIdx() not in atom_ids]
                atoms_to_remove.sort(reverse=True)  # ，

                for atom_id in atoms_to_remove:
                    editable_mol.RemoveAtom(atom_id)

                return editable_mol.GetMol()
            

            # ID（0）
            atom_ids_to_keep = new_atom_id  # ID，
            pro_mol=Chem.MolFromPDBFile(self.protein_file,removeHs=False,sanitize=False)
            # 
            new_pro_mol = keep_atoms_by_id(pro_mol, atom_ids_to_keep)
            Chem.MolToPDBFile(new_pro_mol, sub_protein_file)



            #，
            #400pdb，,# 5l8c/5l8c_pocket10.pdb，5l8c/5l8c_pocket10_400.pdb
            #nonzero_indices
            #: '../CrossDocked2020/data/pdbbind2020_r10/v2020-other-PL/5qb1/5qb1_pocket10_400.pdb'

            '''
            sub_protein_file = os.path.join(os.path.dirname(self.protein_file), os.path.splitext(os.path.basename(self.protein_file))[0] + '_400.pdb')
            sub_protein_w = open(sub_protein_file, 'w')

            org_pro_list = []
            count = 0
            with open(self.protein_file, 'r')as f:
                for i, line in enumerate(f):
                    if line[0:6].strip() == 'ATOM' and i in nonzero_indices:
                        sub_protein_w.write(line)
                

                sub_protein_w.write('END')
            sub_protein_w.close()

            '''
        
            #print('ok3')

            #O,N, ，rdkitpdb，，rdkit，
            #。，，rdkit，
            #，，，rdkit
            '''
            pro_mol=Chem.MolFromPDBFile(self.protein_file,removeHs=False,sanitize=False)
            #print('ok1')
            atom_isring=np.array([atom.IsInRing() for atom in pro_mol.GetAtoms()])    #
            atom_isO=np.array([atom.GetSymbol() == 'O'  for atom in pro_mol.GetAtoms()])    #O
            atom_isN=np.array([atom.GetSymbol() == 'N'  for atom in pro_mol.GetAtoms()])    #N
            coords=np.array(pro_mol.GetConformer(0).GetPositions(), dtype=np.float32) #,float32，float64
            #coords=np.array(pro_mol.GetConformer(0).GetPositions())
            '''
            atom_isO, atom_isN, atom_isring, atom_isspecial, atom_id, coords = self._read_pdb_ONRing_biopandas(self.protein_file)
            #，，-，
            #indexs = []  #, 
            target_xyz =  np.array(self.pos) #xyz
            #target_xyz_sum =  np.round(np.array(self.pos, dtype=np.float32).sum(axis = -1), 2)

            #assert np.array_equal(target_xyz, coords) #，, postbus, assert，try, except

            if not np.array_equal(target_xyz, coords):
                print('target_xyz.shape:', target_xyz.shape)
                print('coords.shape:', coords.shape)
                raise Exception('np.array_equal(target_xyz, coords):', np.array_equal(target_xyz, coords))

            new_atom_isring = atom_isring
            new_atom_isO = atom_isO
            new_atom_isN = atom_isN

            assert len(new_atom_isN[nonzero_indices]) == len(np.array(self.element, dtype=np.int64)[nonzero_indices])
            #print('ok4')



            #

            l2p = [] #
            p2l = [] #
            l2p_type = [] #
            p2l_type = [] #

            #
            ligand_atom_isring = copy.deepcopy(cross_lig_isring_flag)
            ligand_atom_isO    = copy.deepcopy(cross_lig_isO_flag)
            ligand_atom_isN    = copy.deepcopy(cross_lig_isN_flag)

            protein_atom_isring = copy.deepcopy(cross_pro_isring_flag)
            protein_atom_isO    = copy.deepcopy(cross_pro_isO_flag)
            protein_atom_isN    = copy.deepcopy(cross_pro_isN_flag)


            ligand_cross_isring_flag = copy.deepcopy(cross_lig_isring_flag)
            ligand_cross_isO_flag    = copy.deepcopy(cross_lig_isO_flag)
            ligand_cross_isN_flag    = copy.deepcopy(cross_lig_isN_flag)
            ligand_cross_lp_pos      = copy.deepcopy(holo_coords)

            protein_cross_isring_flag = copy.deepcopy(cross_pro_isring_flag)
            protein_cross_isO_flag    = copy.deepcopy(cross_pro_isO_flag) 
            protein_cross_isN_flag    = copy.deepcopy(cross_pro_isN_flag) 
            protein_cross_lp_pos      = copy.deepcopy(pocket_coords)

            cross_distance_matrix = copy.deepcopy(cross_distance) #cross_distancelist，cross_distance

            #[2, n*m,]
            #
            centor = holo_coords.mean(axis = 0)
            l_combinations_isring = self.combinations_optim(torch.from_numpy(holo_coords), torch.from_numpy(pocket_coords), torch.from_numpy(np.array(list(range(len(ligand_atom_isring))))), 
                    torch.from_numpy(np.array(list(range(len(protein_atom_isring))))), torch.from_numpy(ligand_atom_isring), torch.from_numpy(protein_atom_isring), 
                    torch.from_numpy(centor), 
                    torch.from_numpy(ligand_cross_isring_flag), torch.from_numpy(protein_cross_isring_flag), torch.from_numpy(cross_distance_matrix), 
                    torch.from_numpy(ligand_cross_lp_pos), torch.from_numpy(protein_cross_lp_pos), flag = 'ligand')
            
            l_combinations_isO    = self.combinations_optim(torch.from_numpy(holo_coords), torch.from_numpy(pocket_coords), torch.from_numpy(np.array(list(range(len(ligand_atom_isring))))), 
                    torch.from_numpy(np.array(list(range(len(protein_atom_isring))))), torch.from_numpy(ligand_atom_isO), torch.from_numpy(protein_atom_isN), 
                    torch.from_numpy(centor), 
                    torch.from_numpy(ligand_cross_isO_flag), torch.from_numpy(protein_cross_isN_flag), torch.from_numpy(cross_distance_matrix), 
                    torch.from_numpy(ligand_cross_lp_pos), torch.from_numpy(protein_cross_lp_pos), flag = 'ligand')
            
            l_combinations_isN    = self.combinations_optim(torch.from_numpy(holo_coords), torch.from_numpy(pocket_coords), torch.from_numpy(np.array(list(range(len(ligand_atom_isring))))), 
                    torch.from_numpy(np.array(list(range(len(protein_atom_isring))))), torch.from_numpy(ligand_atom_isN), torch.from_numpy(protein_atom_isO), 
                    torch.from_numpy(centor), 
                    torch.from_numpy(ligand_cross_isN_flag), torch.from_numpy(protein_cross_isO_flag), torch.from_numpy(cross_distance_matrix), 
                    torch.from_numpy(ligand_cross_lp_pos), torch.from_numpy(protein_cross_lp_pos), flag = 'ligand')

            if l_combinations_isring.shape[0] != 0:
                l2p.append(l_combinations_isring.numpy()) #2 * N
                l2p_type.append(np.full(l_combinations_isring.shape[1], 5))  #，，2*N,N*2？cat
            if l_combinations_isO.shape[0] != 0:
                l2p.append(l_combinations_isO.numpy())
                l2p_type.append(np.full(l_combinations_isO.shape[1], 6))
            if l_combinations_isN.shape[0] != 0:
                l2p.append(l_combinations_isN.numpy())
                l2p_type.append(np.full(l_combinations_isN.shape[1], 7))

            #print('l_combinations_isring:', l_combinations_isring.shape)
            #print('l_combinations_isO:', l_combinations_isO.shape)
            #print('l_combinations_isN:', l_combinations_isN.shape)

            #
            p_combinations_isring = self.combinations_optim(torch.from_numpy(pocket_coords), torch.from_numpy(holo_coords), torch.from_numpy(np.array(list(range(len(protein_atom_isring))))), 
                    torch.from_numpy(np.array(list(range(len(ligand_atom_isring))))), torch.from_numpy(protein_atom_isring), torch.from_numpy(ligand_atom_isring), 
                    torch.from_numpy(centor), 
                    torch.from_numpy(protein_cross_isring_flag), torch.from_numpy(ligand_cross_isring_flag), torch.from_numpy(cross_distance_matrix), 
                    torch.from_numpy(ligand_cross_lp_pos), torch.from_numpy(protein_cross_lp_pos), flag = 'protein')
            
            p_combinations_isO    = self.combinations_optim(torch.from_numpy(pocket_coords), torch.from_numpy(holo_coords), torch.from_numpy(np.array(list(range(len(protein_atom_isring))))), 
                    torch.from_numpy(np.array(list(range(len(ligand_atom_isring))))), torch.from_numpy(protein_atom_isN), torch.from_numpy(ligand_atom_isO), 
                    torch.from_numpy(centor), 
                    torch.from_numpy(protein_cross_isN_flag), torch.from_numpy(ligand_cross_isO_flag), torch.from_numpy(cross_distance_matrix), 
                    torch.from_numpy(ligand_cross_lp_pos),  torch.from_numpy(protein_cross_lp_pos), flag = 'protein')
            
            p_combinations_isN    = self.combinations_optim(torch.from_numpy(pocket_coords), torch.from_numpy(holo_coords), torch.from_numpy(np.array(list(range(len(protein_atom_isring))))), 
                    torch.from_numpy(np.array(list(range(len(ligand_atom_isring))))), torch.from_numpy(protein_atom_isO), torch.from_numpy(ligand_atom_isN), 
                    torch.from_numpy(centor), 
                    torch.from_numpy(protein_cross_isO_flag), torch.from_numpy(ligand_cross_isN_flag), torch.from_numpy(cross_distance_matrix), 
                    torch.from_numpy(ligand_cross_lp_pos), torch.from_numpy(protein_cross_lp_pos), flag = 'protein')
            
            if p_combinations_isring.shape[0] != 0:
                p2l.append(p_combinations_isring)
                p2l_type.append(np.full(p_combinations_isring.shape[1], 8))  #，，2*N,N*2？cat
            if p_combinations_isO.shape[0] != 0:
                p2l.append(p_combinations_isO)
                p2l_type.append(np.full(p_combinations_isO.shape[1], 9))
            if p_combinations_isN.shape[0] != 0:
                p2l.append(p_combinations_isN)
                p2l_type.append(np.full(p_combinations_isN.shape[1], 10))

            #print('p_combinations_isring:', p_combinations_isring.shape)
            #print('p_combinations_isO:', p_combinations_isO.shape)
            #print('p_combinations_isN:', p_combinations_isN.shape)

            #ON,NO,，，，，knn
            if len(l2p) != 0:
                l2p_edge_index = np.concatenate(l2p, axis = -1, dtype = int)
                l2p_type = np.concatenate(l2p_type, axis = -1, dtype = int)
            else:
                print('')
                exit()
                l2p_edge_index = np.empty((0, 0))
                l2p_type = np.empty((0, 0))
            
            if len(p2l) != 0:
                p2l_edge_index = np.concatenate(p2l, axis = -1, dtype = int)
                p2l_type = np.concatenate(p2l_type, axis = -1, dtype = int)
            else:
                print('')
                exit()
                p2l_edge_index = np.empty((0, 0))
                p2l_type = np.empty((0, 0))

            #id，id，cross_bond_index, cross_bond_type， cross_bond_index_reverse, cross_bond_type_reverse
            #cross_bond_index_reverse, cross_bond_type_reverse, [5,6,7],[8,9,10]
            

            data = {
                'element': np.array(self.element, dtype=np.int64)[nonzero_indices], #
                'molecule_name': self.title, # pocket
                'pos': np.array(self.pos, dtype=np.float32)[nonzero_indices], #
                'is_backbone': np.array(self.is_backbone, dtype=np.bool_)[nonzero_indices], #Boolean，
                'atom_name': [self.atom_name[i] for i in nonzero_indices], #
                'atom_to_aa_type': np.array(self.atom_to_aa_type, dtype=np.int64)[nonzero_indices], ##
                'atom_isring': new_atom_isring[nonzero_indices],
                'atom_isO': new_atom_isO[nonzero_indices],
                'atom_isN': new_atom_isN[nonzero_indices],

                'cross_lig_isring_flag': cross_lig_isring_flag,
                'cross_lig_isO_flag': cross_lig_isO_flag,
                'cross_lig_isN_flag': cross_lig_isN_flag,

                'cross_pro_isring_flag': cross_pro_isring_flag,
                'cross_pro_isO_flag': cross_pro_isO_flag,
                'cross_pro_isN_flag': cross_pro_isN_flag,

                'cross_ligand': holo_coords,
                'cross_protein': pocket_coords,

                #'cross_distance': set(torch.from_numpy(cross_distance)), #，，？，
                #'cross_distance': list(torch.from_numpy(cross_distance)), #pyglist，
                #'cross_distance': tuple(torch.from_numpy(cross_distance)), #tuple，pyg
                #'cross_distance': OrderedSet(torch.from_numpy(cross_distance)), #，pyg，numpy

                'cross_distance': cross_distance,

                #'cross_bond_index': np.ones_like(l2p_edge_index),
                #'cross_distance2': np.ones_like(l2p_edge_index), #？？，，
                #Batch.from_data_list([data.clone() for _ in range(n_data)], follow_batch=FOLLOW_BATCH, exclude_keys = collate_exclude_keys).to(device)，
                #bond？pyg？?

                #'cross_distance1': l2p_edge_index,
                #'cross_distance2': l2p_type,
                #'cross_distance3': p2l_edge_index,
                #'cross_distance4': p2l_type,
                

                'link_e': l2p_edge_index.T,
                'link_t': l2p_type,
                'link_e_reverse': p2l_edge_index.T,
                'link_t_reverse': p2l_type,

                'l_combinations_isring': l_combinations_isring.numpy(),
                'l_combinations_O': l_combinations_isO.numpy(),
                'l_combinations_N': l_combinations_isN.numpy(),


                'p_combinations_isring': p_combinations_isring.numpy(),
                'p_combinations_O': p_combinations_isO.numpy(),
                'p_combinations_N': p_combinations_isN.numpy(),


                #def torchify_dict(data):
            }

            #print('ok5')
            return data
        
        
        #except Exception as e:
            #print('protein error:', e)
            #print('self.protein_file:', self.protein_file)
            #raise Exception('protein error, stop')
            #exit()
            #return None











    def to_dict_atom_interaction_gen_split3_5(self):
        #<3.53.5~4.5，，4，，，，20，2，
        # 4index：12,13,14,15

        old_data = {
            'element': np.array(self.element, dtype=np.int64), #
            'molecule_name': self.title, # pocket
            'pos': np.array(self.pos, dtype=np.float32), #
            'is_backbone': np.array(self.is_backbone, dtype=np.bool_), #Boolean，
            'atom_name': self.atom_name, #
            'atom_to_aa_type': np.array(self.atom_to_aa_type, dtype=np.int64) ##
        }

        #，400

        if 'pdbbind2020' in self.protein_file:
            file_path = os.path.join(os.path.dirname(self.protein_file), 'interaction_' + os.path.basename(self.protein_file).split('_')[0] + '_v2.pkl')   #5S8I_protein.pdb
        else:
            file_path = os.path.join(os.path.dirname(self.protein_file), 'interaction_' + os.path.basename(self.protein_file).split('_')[0] + '.pkl')   #5S8I_protein.pdb
        with open(file_path, 'rb') as file:
            interaction_data = dill.load(file)
        holo_coords_list = interaction_data['holo_coords_list']
        coords_predict_list = interaction_data['coords_predict_list']
        pocket_coords_list = interaction_data['pocket_coords_list']
        cross_distance_list = interaction_data['cross_distance_list']

        assert np.allclose(holo_coords_list[0], holo_coords_list[-1], atol=0.02)

        #print('pocket_coords_list[0]:', pocket_coords_list[0][:])
        #np.set_printoptions(suppress=True, precision=4)
        #print('pocket_coords_list[-1]:', pocket_coords_list[-1]) #，？？，
        #print('pocket_coords_list[0].shape:', pocket_coords_list[0].shape)
        #print('pocket_coords_list[-1].shape:', pocket_coords_list[-1].shape) #，？？
        #assert np.allclose(pocket_coords_list[0], pocket_coords_list[-1], atol=0.02)

        assert len(holo_coords_list) == len(coords_predict_list) and len(holo_coords_list) == len(pocket_coords_list) and len(holo_coords_list) == len(cross_distance_list)

        
        #rmsd，，
        rmsd_list = []
        for pre_pos, holo_pos in zip(coords_predict_list, holo_coords_list):
            assert pre_pos.shape == holo_pos.shape   #"Coordinate matrices must have the same shape"
            rmsd = np.sqrt(np.mean(np.sum((pre_pos - holo_pos) ** 2, axis=1)))
            rmsd_list.append(rmsd)
        
        sorted_indices = np.argsort(rmsd_list)
        best_index = sorted_indices[0]
        
        #
        #best_index  = random.choice(list(range(len(holo_coords_list))))


        if self.cross_distance_num == None or self.cross_distance_num == 'best':
            holo_coords = holo_coords_list[best_index]
            coords_predict = coords_predict_list[best_index]
            pocket_coords = pocket_coords_list[best_index]
            cross_distance = cross_distance_list[best_index]
        else:            
            holo_coords = holo_coords_list[self.cross_distance_num]
            coords_predict = coords_predict_list[self.cross_distance_num]
            pocket_coords = pocket_coords_list[self.cross_distance_num]
            cross_distance = cross_distance_list[self.cross_distance_num]



        #unimol，，
        unique_index_dict       = {}
        for j, ps in enumerate(pocket_coords):
            unique_index_dict[tuple(ps)] = j
            #，
        
        unique_index_list   = list(unique_index_dict.values())
        pocket_coords       = pocket_coords[unique_index_list]
        cross_distance      = cross_distance[:, unique_index_list]



        #，unimolrdkit，，
        #cross_distanceO,N,
        #，
        pro_isring_flag = {}
        pro_isO_flag = {}
        pro_isN_flag = {}

        '''
        pro_mol=Chem.MolFromPDBFile(self.protein_file,removeHs=False,sanitize=False) #，, 
        #print('self.protein_file:', self.protein_file)
        #print('ok1')
        atom_isring=np.array([atom.IsInRing() for atom in pro_mol.GetAtoms()])    #
        atom_isO=np.array([atom.GetSymbol() == 'O'  for atom in pro_mol.GetAtoms()])    #O
        atom_isN=np.array([atom.GetSymbol() == 'N'  for atom in pro_mol.GetAtoms()])    #N
        #coords=np.array(pro_mol.GetConformer(0).GetPositions(), dtype=np.float32) #,float32，float64
        coords=np.array(pro_mol.GetConformer(0).GetPositions())
        '''
        atom_isO, atom_isN, atom_isring, atom_isspecial, atom_id, coords = self._read_pdb_ONRing_biopandas(self.protein_file)
        count = 0
        coords_atom_dict = defaultdict(list)
        for r, o, n, c in zip(atom_isring, atom_isO, atom_isN, coords):
            count += 1
            k = torch.FloatTensor(c)
            tg = ''
            for ii in k:
                #tg += str(round(i.item(), 4)) + '_'
                tg += str(self.truncate(ii.item(), 3)) + '_' #，2，
            c_sum = str(tg)

            coords_atom_dict[c_sum].append(c)

            pro_isring_flag[c_sum] = r
            pro_isO_flag[c_sum]    = o
            pro_isN_flag[c_sum]    = n
        
        assert len(pro_isring_flag) == len(atom_isring)
        
        print('：', count)
        if len(pro_isring_flag) != len(coords) or len(pro_isO_flag) != len(coords) or len(pro_isN_flag) != len(coords):
            print(f'{len(pro_isring_flag)} != {len(coords)} or {len(pro_isO_flag)} != {len(coords)} or {len(pro_isN_flag)} != {len(coords)}')
            #979 != 990 or 979 != 990 or 979 != 990 #？
            raise Exception("pro atom num is error")


        cross_pro_isring_flag = copy.deepcopy(atom_isring)
        cross_pro_isO_flag = copy.deepcopy(atom_isO)
        cross_pro_isN_flag = copy.deepcopy(atom_isN)



        #，,
        #，unimolrdkit，unimolrdkit，sdf，rdkit mol，rdkit
        #，unimol，rdkit
        lig_isring_flag = {}
        lig_isO_flag = {}
        lig_isN_flag = {}


        #，rdkit3d，，
        if self.data_flag == 'new_test':
            
            lig_mol = copy.deepcopy(self.ligand_dict['mol']) #
            #，？cross，，
            lig_rdkit_mol = copy.deepcopy(self.ligand_dict['rd_mol'])

            unimol_pos  = torch.FloatTensor(holo_coords)
            rdkit_pos   = torch.FloatTensor(lig_rdkit_mol.GetConformer(0).GetPositions())
            ground_pos  = torch.FloatTensor(lig_mol.GetConformer(0).GetPositions())

            #
            assert len(ground_pos) == len(unimol_pos) #，rkdit，，，

            #，rdkit, rdkitsmiles3dground，3d，
            if self.compare_atom_order(lig_mol, lig_rdkit_mol) and torch.allclose(ground_pos, unimol_pos, atol=0.02):
                lig_mol     = copy.deepcopy(lig_rdkit_mol) #，org_liandlig_rdkit_mol, 
                holo_coords = np.array(lig_rdkit_mol.GetConformer(0).GetPositions()) 
            
            #lig_mol = copy.deepcopy(self.ligand_dict['mol']) 
        else:
            lig_mol = copy.deepcopy(self.ligand_dict['mol']) #





        #print('ok1')
        atom_isring=np.array([atom.IsInRing() for atom in lig_mol.GetAtoms()])    #
        atom_isO=np.array([atom.GetSymbol() == 'O'  for atom in lig_mol.GetAtoms()])    #O
        atom_isN=np.array([atom.GetSymbol() == 'N'  for atom in lig_mol.GetAtoms()])    #N
        #coords=np.array(lig_mol.GetConformer(0).GetPositions(), dtype=np.float32) #,float32，float64
        coords=np.array(lig_mol.GetConformer(0).GetPositions())

        for r, o, n, c in zip(atom_isring, atom_isO, atom_isN, coords):
            k = torch.FloatTensor(c)
            tg = ''
            for ii in k:
                #tg += str(round(i.item(), 4)) + '_'
                tg += str(self.truncate(ii.item(), 3)) + '_' #，2，
            c_sum = str(tg)

            lig_isring_flag[c_sum] = r
            lig_isO_flag[c_sum]    = o
            lig_isN_flag[c_sum]    = n
        
        assert len(lig_isring_flag) == len(atom_isring)
        assert len(coords) == len(holo_coords) #，rkdit，，，

        if len(lig_isring_flag) != len(coords) or len(lig_isO_flag) != len(coords) or len(lig_isN_flag) != len(coords) or len(coords) != len(holo_coords):
            print(f'{len(coords)} != {len(holo_coords)}') #19 != 18, rdkit，crosss_liand，
            raise Exception("lig atom num is error") #，

    
        #unimolrdkit, 0.02
        assert torch.allclose(torch.FloatTensor(coords), torch.FloatTensor(holo_coords), atol=0.02)

        holo_coords = copy.deepcopy(coords)

        #，rdkitsdf，，，，，，holo_coords = coords
        #cross_distance, cross_ligand, cross_protein, cross_ligand_atom_flag, cross_protein_atom_flag

        cross_lig_isring_flag = copy.deepcopy(atom_isring)
        cross_lig_isO_flag = copy.deepcopy(atom_isO)
        cross_lig_isN_flag = copy.deepcopy(atom_isN)

        
        if old_data['pos'].shape[0] <= 0:
            print("old_data['pos'].size(0):", old_data['pos'].shape[0])
            print("old_data['pos']:", old_data['pos'].shape)
            return old_data
        else:
            data = {}
            #self.ligand_centor
            dis   = np.linalg.norm(old_data['pos'] - self.ligand_centor, axis = 1) #

        #try:
            #，12ai，400，400

            if pocket_coords.shape[0] < 300:
                cutoff_num = 2000
            else:
                cutoff_num = 400

            #，，rdkit
            #print('self.protein_file:', self.protein_file)
            '''
            pro_mol=Chem.MolFromPDBFile(self.protein_file,removeHs=False,sanitize=False)
            atom_isspecial = np.array([atom.IsInRing() or atom.GetSymbol() == 'O' or atom.GetSymbol() == 'N' for atom in pro_mol.GetAtoms()]) #
            atom_id = np.array([atom.GetIdx() for atom in pro_mol.GetAtoms()])
            #coords=np.array(pro_mol.GetConformer(0).GetPositions(), dtype=np.float32) #,float32，float64
            all_coords=np.array(pro_mol.GetConformer(0).GetPositions())
            '''
            atom_isO, atom_isN, atom_isring, atom_isspecial, atom_id, all_coords = self._read_pdb_ONRing_biopandas(self.protein_file)
            cutoff_indices = np.argsort(dis)[:] #,cutoff_num，，
            coords = all_coords[cutoff_indices] #，

            assert old_data['pos'].shape == coords.shape

            assert cutoff_indices.shape == np.unique(cutoff_indices).shape

            #，pocket_coords，400
            new_indices = []
            pro_flag_dict = {} #value
            pro_flag_dict2 = {}

            for j, c in zip(cutoff_indices, coords): #j，，
                k = torch.FloatTensor(c)
                tg = ''
                for ii in k:
                    #tg += str(round(i.item(), 4)) + '_'
                    tg += str(self.truncate(ii.item(), 3)) + '_' #，2，
                k = str(tg)

                pro_flag_dict[k] = j

                k2 = torch.FloatTensor(c)
                tg2 = ''
                for ii in k2:
                    #tg += str(round(i.item(), 4)) + '_'
                    tg2 += str(self.truncate2(ii.item(), 3)) + '_' #，2，
                k2 = str(tg2)

                pro_flag_dict2[k2] = j

            

            if len(pro_flag_dict) != len(coords):
                raise Exception(f"{len(pro_flag_dict)} != {len(coords)}")


            if len(pro_flag_dict2) != len(coords):
                raise Exception(f"{len(pro_flag_dict2)} != {len(coords)}")
            
            #cutoff_protein_ids = torch.ones(pocket_coords.shape[0], dtype = torch.bool)
            cutoff_protein_ids = torch.zeros(pocket_coords.shape[0], dtype = torch.bool)

            assert pocket_coords.shape == np.unique(pocket_coords, axis = 0).shape

            for ids, c in enumerate(pocket_coords): #pocket_coords
                k = torch.FloatTensor(c)
                tg = ''
                for ii in k:
                    #tg += str(round(i.item(), 4)) + '_'
                    tg += str(self.truncate(ii.item(), 3)) + '_' #，2，
                k = str(tg)

                #pocket_coords，
                #if cross_pro_flag_dict.get(c_sum):

                
                try:
                    new_indices.append(pro_flag_dict[k]) #，unimol，，, 
                    cutoff_protein_ids[ids] = True
                except Exception as e:
                    print(e)
                    #cutoff_protein_ids[ids] = False
                    #print('key error, skip:', k)
                    #print('key all:', list(pro_flag_dict.keys()))
                    #print('-----------------------------------------')
                    continue
                    
            #，，
            pocket_coords  = pocket_coords[cutoff_protein_ids]
            cross_distance = cross_distance[:, cutoff_protein_ids]

            cross_pro_isring_flag = cross_pro_isring_flag[new_indices]
            cross_pro_isO_flag    = cross_pro_isO_flag[new_indices]
            cross_pro_isN_flag    = cross_pro_isN_flag[new_indices]

            #print('cutoff_protein_ids num:', torch.sum(cutoff_protein_ids)) # tensor(252)
            #print('len(new_indices):', np.sum(np.array(new_indices))) #len(new_indices): 251

            #print('new_indices1:', sorted(np.nonzero(np.array(new_indices))[0]))
            #print('new_indices2:', sorted(np.array(new_indices2))) #id，
            
            if len(cross_pro_isring_flag) != len(pocket_coords):
                raise Exception(f'{len(cross_pro_isring_flag)} != {len(pocket_coords)}') #Exception: 251 != 252
            
            #if len(new_indices) != len(self.unimol_pcoords[0]):
                #raise Exception(f'{len(new_indices)} <= 0')

            #print('len(new_indices):', len(new_indices)) #
            #print('new_indices:', new_indices)

            new_atom_isspecial = atom_isspecial[new_indices] #atom_isspecial
            new_atom_id = atom_id[new_indices] #atom_id
            nonzero_indices = new_indices

            #，
            atom_isspecial_index = np.nonzero(new_atom_isspecial == True)[0]
            atom_isgeneral_index = np.nonzero(new_atom_isspecial == False)[0]
            

            #rdkit，，
            #new_atom_id
            sub_protein_file = os.path.join(os.path.dirname(self.protein_file), os.path.splitext(os.path.basename(self.protein_file))[0] + '_400.pdb')

            def keep_atoms_by_id(mol, atom_ids):
                """ID"""
                editable_mol = Chem.EditableMol(mol)
                all_atoms = list(editable_mol.GetMol().GetAtoms())
                atoms_to_keep = {atom.GetIdx() for atom in all_atoms if atom.GetIdx() in atom_ids}
                
                atoms_to_remove = [atom.GetIdx() for atom in all_atoms if atom.GetIdx() not in atom_ids]
                atoms_to_remove.sort(reverse=True)  # ，

                for atom_id in atoms_to_remove:
                    editable_mol.RemoveAtom(atom_id)

                return editable_mol.GetMol()
            

            # ID（0）
            atom_ids_to_keep = new_atom_id  # ID，
            pro_mol=Chem.MolFromPDBFile(self.protein_file,removeHs=False,sanitize=False)
            # 
            new_pro_mol = keep_atoms_by_id(pro_mol, atom_ids_to_keep)
            Chem.MolToPDBFile(new_pro_mol, sub_protein_file)



            #，
            #400pdb，,# 5l8c/5l8c_pocket10.pdb，5l8c/5l8c_pocket10_400.pdb
            #nonzero_indices
            #: '../CrossDocked2020/data/pdbbind2020_r10/v2020-other-PL/5qb1/5qb1_pocket10_400.pdb'

            '''
            sub_protein_file = os.path.join(os.path.dirname(self.protein_file), os.path.splitext(os.path.basename(self.protein_file))[0] + '_400.pdb')
            sub_protein_w = open(sub_protein_file, 'w')

            org_pro_list = []
            count = 0
            with open(self.protein_file, 'r')as f:
                for i, line in enumerate(f):
                    if line[0:6].strip() == 'ATOM' and i in nonzero_indices:
                        sub_protein_w.write(line)
                

                sub_protein_w.write('END')
            sub_protein_w.close()

            '''
        
            #print('ok3')

            #O,N, ，rdkitpdb，，rdkit，
            #。，，rdkit，
            #，，，rdkit
            '''
            pro_mol=Chem.MolFromPDBFile(self.protein_file,removeHs=False,sanitize=False)
            #print('ok1')
            atom_isring=np.array([atom.IsInRing() for atom in pro_mol.GetAtoms()])    #
            atom_isO=np.array([atom.GetSymbol() == 'O'  for atom in pro_mol.GetAtoms()])    #O
            atom_isN=np.array([atom.GetSymbol() == 'N'  for atom in pro_mol.GetAtoms()])    #N
            coords=np.array(pro_mol.GetConformer(0).GetPositions(), dtype=np.float32) #,float32，float64
            #coords=np.array(pro_mol.GetConformer(0).GetPositions())
            '''
            atom_isO, atom_isN, atom_isring, atom_isspecial, atom_id, coords = self._read_pdb_ONRing_biopandas(self.protein_file)
            #，，-，
            #indexs = []  #, 
            target_xyz =  np.array(self.pos) #xyz
            #target_xyz_sum =  np.round(np.array(self.pos, dtype=np.float32).sum(axis = -1), 2)

            #assert np.array_equal(target_xyz, coords) #，, postbus, assert，try, except

            if not np.array_equal(target_xyz, coords):
                print('target_xyz.shape:', target_xyz.shape)
                print('coords.shape:', coords.shape)
                raise Exception('np.array_equal(target_xyz, coords):', np.array_equal(target_xyz, coords))

            new_atom_isring = atom_isring
            new_atom_isO = atom_isO
            new_atom_isN = atom_isN

            assert len(new_atom_isN[nonzero_indices]) == len(np.array(self.element, dtype=np.int64)[nonzero_indices])
            #print('ok4')



            #

            l2p = [] #
            p2l = [] #
            l2p_type = [] #
            p2l_type = [] #

            #
            ligand_atom_isring = copy.deepcopy(cross_lig_isring_flag)
            ligand_atom_isO    = copy.deepcopy(cross_lig_isO_flag)
            ligand_atom_isN    = copy.deepcopy(cross_lig_isN_flag)

            protein_atom_isring = copy.deepcopy(cross_pro_isring_flag)
            protein_atom_isO    = copy.deepcopy(cross_pro_isO_flag)
            protein_atom_isN    = copy.deepcopy(cross_pro_isN_flag)


            ligand_cross_isring_flag = copy.deepcopy(cross_lig_isring_flag)
            ligand_cross_isO_flag    = copy.deepcopy(cross_lig_isO_flag)
            ligand_cross_isN_flag    = copy.deepcopy(cross_lig_isN_flag)
            ligand_cross_lp_pos      = copy.deepcopy(holo_coords)

            protein_cross_isring_flag = copy.deepcopy(cross_pro_isring_flag)
            protein_cross_isO_flag    = copy.deepcopy(cross_pro_isO_flag) 
            protein_cross_isN_flag    = copy.deepcopy(cross_pro_isN_flag) 
            protein_cross_lp_pos      = copy.deepcopy(pocket_coords)

            cross_distance_matrix = copy.deepcopy(cross_distance) #cross_distancelist，cross_distance

            #[2, n*m,], 4.5O,N,，，3.5，2
            #
            centor = holo_coords.mean(axis = 0)
            l_combination_less3_5, l_combination_greater3_5 = self.combinations_optim_split_3_5(torch.from_numpy(holo_coords), torch.from_numpy(pocket_coords), 
                    torch.from_numpy(np.array(list(range(len(holo_coords))))), 
                    torch.from_numpy(np.array(list(range(len(pocket_coords))))), None, None, 
                    None, 
                    None, None, torch.from_numpy(cross_distance_matrix), 
                    torch.from_numpy(ligand_cross_lp_pos), torch.from_numpy(protein_cross_lp_pos), flag = 'ligand')

            #
            p_combination_less3_5, p_combination_greater3_5 = self.combinations_optim_split_3_5(torch.from_numpy(pocket_coords), torch.from_numpy(holo_coords), 
                    torch.from_numpy(np.array(list(range(len(pocket_coords))))), 
                    torch.from_numpy(np.array(list(range(len(holo_coords))))), None, None, 
                    None, 
                    None, None, torch.from_numpy(cross_distance_matrix), 
                    torch.from_numpy(ligand_cross_lp_pos), torch.from_numpy(protein_cross_lp_pos), flag = 'protein')


            if l_combination_less3_5.shape[0] != 0:
                l2p.append(l_combination_less3_5)
                l2p_type.append(np.full(l_combination_less3_5.shape[1], 12))  #，，2*N,N*2？cat
            if l_combination_greater3_5.shape[0] != 0:
                l2p.append(l_combination_greater3_5)
                l2p_type.append(np.full(l_combination_greater3_5.shape[1], 13))


            if p_combination_less3_5.shape[0] != 0:
                p2l.append(p_combination_less3_5)
                p2l_type.append(np.full(p_combination_less3_5.shape[1], 14))  #，，2*N,N*2？cat
            if p_combination_greater3_5.shape[0] != 0:
                p2l.append(p_combination_greater3_5)
                p2l_type.append(np.full(p_combination_greater3_5.shape[1], 15))



            #ON,NO,，，，，knn
            if len(l2p) != 0:
                l2p_edge_index = np.concatenate(l2p, axis = -1, dtype = int)
                l2p_type = np.concatenate(l2p_type, axis = -1, dtype = int)
            else:
                print('')
                exit()
                l2p_edge_index = np.empty((0, 0))
                l2p_type = np.empty((0, 0))
            
            if len(p2l) != 0:
                p2l_edge_index = np.concatenate(p2l, axis = -1, dtype = int)
                p2l_type = np.concatenate(p2l_type, axis = -1, dtype = int)
            else:
                print('')
                exit()
                p2l_edge_index = np.empty((0, 0))
                p2l_type = np.empty((0, 0))

            #id，id，cross_bond_index, cross_bond_type， cross_bond_index_reverse, cross_bond_type_reverse
            #cross_bond_index_reverse, cross_bond_type_reverse, [5,6,7],[8,9,10]
            

            data = {
                'element': np.array(self.element, dtype=np.int64)[nonzero_indices], #
                'molecule_name': self.title, # pocket
                'pos': np.array(self.pos, dtype=np.float32)[nonzero_indices], #
                'is_backbone': np.array(self.is_backbone, dtype=np.bool_)[nonzero_indices], #Boolean，
                'atom_name': [self.atom_name[i] for i in nonzero_indices], #
                'atom_to_aa_type': np.array(self.atom_to_aa_type, dtype=np.int64)[nonzero_indices], ##
                'atom_isring': new_atom_isring[nonzero_indices],
                'atom_isO': new_atom_isO[nonzero_indices],
                'atom_isN': new_atom_isN[nonzero_indices],

                'cross_lig_isring_flag': cross_lig_isring_flag,
                'cross_lig_isO_flag': cross_lig_isO_flag,
                'cross_lig_isN_flag': cross_lig_isN_flag,

                'cross_pro_isring_flag': cross_pro_isring_flag,
                'cross_pro_isO_flag': cross_pro_isO_flag,
                'cross_pro_isN_flag': cross_pro_isN_flag,

                'cross_ligand': holo_coords,
                'cross_protein': pocket_coords,

                #'cross_distance': set(torch.from_numpy(cross_distance)), #，，？，
                #'cross_distance': list(torch.from_numpy(cross_distance)), #pyglist，
                #'cross_distance': tuple(torch.from_numpy(cross_distance)), #tuple，pyg
                #'cross_distance': OrderedSet(torch.from_numpy(cross_distance)), #，pyg，numpy

                'cross_distance': cross_distance,

                #'cross_bond_index': np.ones_like(l2p_edge_index),
                #'cross_distance2': np.ones_like(l2p_edge_index), #？？，，
                #Batch.from_data_list([data.clone() for _ in range(n_data)], follow_batch=FOLLOW_BATCH, exclude_keys = collate_exclude_keys).to(device)，
                #bond？pyg？?

                #'cross_distance1': l2p_edge_index,
                #'cross_distance2': l2p_type,
                #'cross_distance3': p2l_edge_index,
                #'cross_distance4': p2l_type,
                

                'link_e': l2p_edge_index.T,
                'link_t': l2p_type,
                'link_e_reverse': p2l_edge_index.T,
                'link_t_reverse': p2l_type,


                #def torchify_dict(data):
            }

            #print('ok5')
            return data
        
        
        #except Exception as e:
            #print('protein error:', e)
            #print('self.protein_file:', self.protein_file)
            #raise Exception('protein error, stop')
            #exit()
            #return None




    def to_dict_atom_interaction_gen_split3_5_extend(self):
        #<3.53.5~4.5，，4，，，，20，2，
        # 4index：12,13,14,15

        old_data = {
            'element': np.array(self.element, dtype=np.int64), #
            'molecule_name': self.title, # pocket
            'pos': np.array(self.pos, dtype=np.float32), #
            'is_backbone': np.array(self.is_backbone, dtype=np.bool_), #Boolean，
            'atom_name': self.atom_name, #
            'atom_to_aa_type': np.array(self.atom_to_aa_type, dtype=np.int64) ##
        }

        #，400
        #file_path = os.path.join(os.path.dirname(self.protein_file), 'interaction_' + os.path.basename(self.protein_file).split('_')[0] + '_v2.pkl')   #5S8I_protein.pdb
        file_path = os.path.join(os.path.dirname(self.protein_file), 'interaction_' + os.path.basename(self.protein_file).rsplit("_protein", 1)[0] + '_v2.pkl')
        if not os.path.exists(file_path):
            file_path = os.path.join(os.path.dirname(self.protein_file), 'interaction_' + os.path.basename(self.protein_file).rsplit("_protein", 1)[0] + '_v2.pkl')
        
        with open(file_path, 'rb') as file:
            interaction_data = dill.load(file)

        if GP.glide_vina:
            #glidevina
            tmp             = os.path.basename(self.protein_file).split('_')[0]
            predict_file    = os.path.join(os.path.dirname(self.protein_file), f'{tmp}_ligand.sdf')
            predict_sup     = list(Chem.rdmolfiles.SDMolSupplier(predict_file)) #，

            if len(predict_sup) < 40:
                predict_sup.extend([predict_sup[0]] * (40 - len(predict_sup)))

            conf_num = len(predict_sup)
            coords_predict_list = []
            cross_distance_list = []

            #40, 40
            if conf_num >= 41:
                conf_num = 40


            #2, 
            holo_coords_list    = interaction_data['holo_coords_list'][:conf_num]
            pocket_coords_list  = interaction_data['pocket_coords_list'][:conf_num]
            #ligand_emb_list = interaction_data['ligand_emb_list'][:conf_num]
            #pocket_emb_list = interaction_data['pocket_emb_list'][:conf_num]
            

            for k in range(conf_num):
                coords_predict  = np.array(Chem.RemoveHs(predict_sup[k]).GetConformer(0).GetPositions(), dtype = np.float32)
                coords_predict_list.append(coords_predict)

                #, 40
                if k >= 40:
                    k = 39

                try:
                    cross_distance = calculate_distance_matrix_numpy(coords_predict, pocket_coords_list[k]) #
                except Exception as e:
                    print("interaction_data['pocket_coords_list'] num:", len(interaction_data['pocket_coords_list']))
                    print('pocket_coords_list:', len(pocket_coords_list))
                    print('conf_num:', conf_num)
                    print('k:', k)
                    raise Exception(e) #40
                cross_distance_list.append(cross_distance)




        else:
            holo_coords_list = interaction_data['holo_coords_list']
            coords_predict_list = interaction_data['coords_predict_list']
            pocket_coords_list = interaction_data['pocket_coords_list']
            cross_distance_list = interaction_data['cross_distance_list']
            #ligand_emb_list = interaction_data['ligand_emb_list']
            #pocket_emb_list = interaction_data['pocket_emb_list']

        assert np.allclose(holo_coords_list[0], holo_coords_list[-1], atol=0.02)

        #print('pocket_coords_list[0]:', pocket_coords_list[0][:])
        #np.set_printoptions(suppress=True, precision=4)
        #print('pocket_coords_list[-1]:', pocket_coords_list[-1]) #，？？，
        #print('pocket_coords_list[0].shape:', pocket_coords_list[0].shape)
        #print('pocket_coords_list[-1].shape:', pocket_coords_list[-1].shape) #，？？
        #assert np.allclose(pocket_coords_list[0], pocket_coords_list[-1], atol=0.02)

        assert len(holo_coords_list) == len(coords_predict_list) and len(holo_coords_list) == len(pocket_coords_list) and len(holo_coords_list) == len(cross_distance_list)

        if len(holo_coords_list) == 1:
            holo_coords_list     = holo_coords_list * 1000
            coords_predict_list  = coords_predict_list * 1000
            pocket_coords_list   = pocket_coords_list * 1000
            cross_distance_list  = cross_distance_list * 1000
            
            #ligand_emb_list = ligand_emb_list * 1000
            #pocket_emb_list = pocket_emb_list * 1000
            
        #rmsd，，
        rmsd_list = []
        for pre_pos, holo_pos in zip(coords_predict_list, holo_coords_list):
            assert pre_pos.shape == holo_pos.shape   #"Coordinate matrices must have the same shape"
            rmsd = np.sqrt(np.mean(np.sum((pre_pos - holo_pos) ** 2, axis=1)))
            rmsd_list.append(rmsd)
        
        sorted_indices = np.argsort(rmsd_list)
        best_index = sorted_indices[0]
        #, ，n
        #best_index  = random.choice(list(range(len(holo_coords_list))))
        
        #1，，N，1，

            
        if GP.cross_distance_num == 'best' or GP.cross_distance_num == None:
            holo_coords     = holo_coords_list[best_index]
            coords_predict  = coords_predict_list[best_index]
            pocket_coords   = pocket_coords_list[best_index]
            cross_distance  = cross_distance_list[best_index]
            
            #ligand_emb = ligand_emb_list[best_index]
            #pocket_emb = pocket_emb_list[best_index]
        else:
            #，
            holo_coords_list    = [holo_coords_list[k] for k in sorted_indices] 
            coords_predict_list = [coords_predict_list[k] for k in sorted_indices] 
            pocket_coords_list  = [pocket_coords_list[k] for k in sorted_indices] 
            cross_distance_list = [cross_distance_list[k] for k in sorted_indices] 
            #ligand_emb_list     = [ligand_emb_list[k] for k in sorted_indices]
            #pocket_emb_list     = [pocket_emb_list[k] for k in sorted_indices]
            

            #
            try:
                holo_coords     = holo_coords_list[self.cross_distance_num]
            except Exception as e:
                print('self.cross_distance_num:', self.cross_distance_num)
                print('len(holo_coords_list):', len(holo_coords_list))

                #self.cross_distance_num: 2
                #len(holo_coords_list): 2

                raise Exception(e)
            
            coords_predict  = coords_predict_list[self.cross_distance_num]
            pocket_coords   = pocket_coords_list[self.cross_distance_num]
            cross_distance  = cross_distance_list[self.cross_distance_num]
            
            #ligand_emb = ligand_emb_list[self.cross_distance_num]
            #pocket_emb = pocket_emb_list[self.cross_distance_num]

        '''
        if self.cross_distance_num == None or self.cross_distance_num == 'best':
            holo_coords = holo_coords_list[best_index]
            coords_predict = coords_predict_list[best_index]
            pocket_coords = pocket_coords_list[best_index]
            cross_distance = cross_distance_list[best_index]
        else:            
            holo_coords = holo_coords_list[self.cross_distance_num]
            coords_predict = coords_predict_list[self.cross_distance_num]
            pocket_coords = pocket_coords_list[self.cross_distance_num]
            cross_distance = cross_distance_list[self.cross_distance_num]
        '''



        #unimol，，
        unique_index_dict       = {}
        for j, ps in enumerate(pocket_coords):
            unique_index_dict[tuple(ps)] = j
            #，
        
        unique_index_list   = list(unique_index_dict.values())
        pocket_coords       = pocket_coords[unique_index_list]
        cross_distance      = cross_distance[:, unique_index_list]
        
        #print('ligand_emb.shape:', ligand_emb.shape)
        #print('pocket_emb.shape:', pocket_emb.shape)
        #print('unique_index_list num:', len(unique_index_list))
        
        #ligand_emb = ligand_emb
        #pocket_emb = pocket_emb[unique_index_list]



        #，unimolrdkit，，
        #cross_distanceO,N,
        #，
        pro_isring_flag = {}
        pro_isO_flag = {}
        pro_isN_flag = {}

        '''
        pro_mol=Chem.MolFromPDBFile(self.protein_file,removeHs=False,sanitize=False) #，, 
        #print('self.protein_file:', self.protein_file)
        #print('ok1')
        atom_isring=np.array([atom.IsInRing() for atom in pro_mol.GetAtoms()])    #
        atom_isO=np.array([atom.GetSymbol() == 'O'  for atom in pro_mol.GetAtoms()])    #O
        atom_isN=np.array([atom.GetSymbol() == 'N'  for atom in pro_mol.GetAtoms()])    #N
        #coords=np.array(pro_mol.GetConformer(0).GetPositions(), dtype=np.float32) #,float32，float64
        coords=np.array(pro_mol.GetConformer(0).GetPositions())
        '''
        atom_isO, atom_isN, atom_isring, atom_isspecial, atom_id, coords = self._read_pdb_ONRing_biopandas(self.protein_file)
        count = 0
        coords_atom_dict = defaultdict(list)
        for r, o, n, c in zip(atom_isring, atom_isO, atom_isN, coords):
            count += 1
            k = torch.FloatTensor(c)
            tg = ''
            for ii in k:
                #tg += str(round(i.item(), 4)) + '_'
                tg += str(self.truncate(ii.item(), 3)) + '_' #，2，
            c_sum = str(tg)

            coords_atom_dict[c_sum].append(c)

            pro_isring_flag[c_sum] = r
            pro_isO_flag[c_sum]    = o
            pro_isN_flag[c_sum]    = n
        
        assert len(pro_isring_flag) == len(atom_isring)
        
        print('：', count)
        if len(pro_isring_flag) != len(coords) or len(pro_isO_flag) != len(coords) or len(pro_isN_flag) != len(coords):
            print(f'{len(pro_isring_flag)} != {len(coords)} or {len(pro_isO_flag)} != {len(coords)} or {len(pro_isN_flag)} != {len(coords)}')
            #979 != 990 or 979 != 990 or 979 != 990 #？
            raise Exception("pro atom num is error")


        cross_pro_isring_flag = copy.deepcopy(atom_isring)
        cross_pro_isO_flag = copy.deepcopy(atom_isO)
        cross_pro_isN_flag = copy.deepcopy(atom_isN)



        #，,
        #，unimolrdkit，unimolrdkit，sdf，rdkit mol，rdkit
        #，unimol，rdkit
        lig_isring_flag = {}
        lig_isO_flag = {}
        lig_isN_flag = {}


        #，rdkit3d，，
        if self.data_flag == 'new_test':
            
            lig_mol = copy.deepcopy(self.ligand_dict['mol']) #
            '''
            #，？cross，，
            lig_rdkit_mol = copy.deepcopy(self.ligand_dict['rd_mol'])

            unimol_pos  = torch.FloatTensor(holo_coords)
            rdkit_pos   = torch.FloatTensor(lig_rdkit_mol.GetConformer(0).GetPositions())
            ground_pos  = torch.FloatTensor(lig_mol.GetConformer(0).GetPositions())

            #
            try:
                assert len(ground_pos) == len(unimol_pos) #，rkdit，，，
            except AssertionError:
                raise SystemExit

            #，rdkit, rdkitsmiles3dground，3d，
            if self.compare_atom_order(lig_mol, lig_rdkit_mol) and torch.allclose(ground_pos, unimol_pos, atol=0.02):
                lig_mol     = copy.deepcopy(lig_rdkit_mol) #，org_liandlig_rdkit_mol, 
                holo_coords = np.array(lig_rdkit_mol.GetConformer(0).GetPositions()) 
            
            #lig_mol = copy.deepcopy(self.ligand_dict['mol']) 
            '''
        else:
            lig_mol = copy.deepcopy(self.ligand_dict['mol']) #





        #print('ok1')
        atom_isring=np.array([atom.IsInRing() for atom in lig_mol.GetAtoms()])    #
        atom_isO=np.array([atom.GetSymbol() == 'O'  for atom in lig_mol.GetAtoms()])    #O
        atom_isN=np.array([atom.GetSymbol() == 'N'  for atom in lig_mol.GetAtoms()])    #N
        #coords=np.array(lig_mol.GetConformer(0).GetPositions(), dtype=np.float32) #,float32，float64
        coords=np.array(lig_mol.GetConformer(0).GetPositions())

        for r, o, n, c in zip(atom_isring, atom_isO, atom_isN, coords):
            k = torch.FloatTensor(c)
            tg = ''
            for ii in k:
                #tg += str(round(i.item(), 4)) + '_'
                tg += str(self.truncate(ii.item(), 3)) + '_' #，2，
            c_sum = str(tg)

            lig_isring_flag[c_sum] = r
            lig_isO_flag[c_sum]    = o
            lig_isN_flag[c_sum]    = n
        
        
        try:
            assert len(coords) == len(holo_coords) #，rkdit，，，
            assert len(lig_isring_flag) == len(atom_isring)
        except Exception as e:
            print(e)
            raise SystemExit

        if len(lig_isring_flag) != len(coords) or len(lig_isO_flag) != len(coords) or len(lig_isN_flag) != len(coords) or len(coords) != len(holo_coords):
            print(f'{len(coords)} != {len(holo_coords)}') #19 != 18, rdkit，crosss_liand，
            raise Exception("lig atom num is error") #，

    
        #unimolrdkit, 0.02
        if GP.glide_vina:
            #glidevina，ground truth
            pass
        else:
            assert torch.allclose(torch.FloatTensor(coords), torch.FloatTensor(holo_coords), atol=0.02)

        holo_coords = copy.deepcopy(coords)

        #，rdkitsdf，，，，，，holo_coords = coords
        #cross_distance, cross_ligand, cross_protein, cross_ligand_atom_flag, cross_protein_atom_flag

        cross_lig_isring_flag = copy.deepcopy(atom_isring)
        cross_lig_isO_flag = copy.deepcopy(atom_isO)
        cross_lig_isN_flag = copy.deepcopy(atom_isN)

        
        if old_data['pos'].shape[0] <= 0:
            print("old_data['pos'].size(0):", old_data['pos'].shape[0])
            print("old_data['pos']:", old_data['pos'].shape)
            return old_data
        else:
            data = {}
            #self.ligand_centor
            dis   = np.linalg.norm(old_data['pos'] - self.ligand_centor, axis = 1) #

        #try:
            #，12ai，400，400

            if pocket_coords.shape[0] < 300:
                cutoff_num = 2000
            else:
                cutoff_num = 400

            #，，rdkit
            #print('self.protein_file:', self.protein_file)
            '''
            pro_mol=Chem.MolFromPDBFile(self.protein_file,removeHs=False,sanitize=False)
            atom_isspecial = np.array([atom.IsInRing() or atom.GetSymbol() == 'O' or atom.GetSymbol() == 'N' for atom in pro_mol.GetAtoms()]) #
            atom_id = np.array([atom.GetIdx() for atom in pro_mol.GetAtoms()])
            #coords=np.array(pro_mol.GetConformer(0).GetPositions(), dtype=np.float32) #,float32，float64
            all_coords=np.array(pro_mol.GetConformer(0).GetPositions())
            '''
            atom_isO, atom_isN, atom_isring, atom_isspecial, atom_id, all_coords = self._read_pdb_ONRing_biopandas(self.protein_file)
            #
            #cutoff_indices = np.argsort(dis)[:] #,cutoff_num，，
            #cutoff_indices = np.array(list(range(len(all_coords))))
            cutoff_indices = atom_id
            coords = all_coords[cutoff_indices] #，

            assert old_data['pos'].shape == coords.shape

            assert cutoff_indices.shape == np.unique(cutoff_indices).shape

            #，pocket_coords，400
            new_indices = []
            pro_flag_dict = {} #value
            pro_flag_dict2 = {}

            for j, c in zip(cutoff_indices, coords): #j，，
                k = torch.FloatTensor(c)
                tg = ''
                for ii in k:
                    #tg += str(round(i.item(), 4)) + '_'
                    tg += str(self.truncate(ii.item(), 3)) + '_' #，2，
                k = str(tg)

                pro_flag_dict[k] = j

                k2 = torch.FloatTensor(c)
                tg2 = ''
                for ii in k2:
                    #tg += str(round(i.item(), 4)) + '_'
                    tg2 += str(self.truncate2(ii.item(), 3)) + '_' #，2，
                k2 = str(tg2)

                pro_flag_dict2[k2] = j

            

            if len(pro_flag_dict) != len(coords):
                raise Exception(f"{len(pro_flag_dict)} != {len(coords)}")


            if len(pro_flag_dict2) != len(coords):
                raise Exception(f"{len(pro_flag_dict2)} != {len(coords)}")
            
            #cutoff_protein_ids = torch.ones(pocket_coords.shape[0], dtype = torch.bool)
            cutoff_protein_ids = torch.zeros(pocket_coords.shape[0], dtype = torch.bool)

            assert pocket_coords.shape == np.unique(pocket_coords, axis = 0).shape

            for ids, c in enumerate(pocket_coords): #pocket_coords
                k = torch.FloatTensor(c)
                tg = ''
                for ii in k:
                    #tg += str(round(i.item(), 4)) + '_'
                    tg += str(self.truncate(ii.item(), 3)) + '_' #，2，
                k = str(tg)

                #pocket_coords，
                #if cross_pro_flag_dict.get(c_sum):

                
                try:
                    new_indices.append(pro_flag_dict[k]) #，unimol，，, 
                    cutoff_protein_ids[ids] = True
                except Exception as e:
                    print(e)
                    #cutoff_protein_ids[ids] = False
                    #print('key error, skip:', k)
                    #print('key all:', list(pro_flag_dict.keys()))
                    #print('-----------------------------------------')
                    continue
                    
            #，，
            pocket_coords  = pocket_coords[cutoff_protein_ids]
            cross_distance = cross_distance[:, cutoff_protein_ids]

            cross_pro_isring_flag = cross_pro_isring_flag[new_indices]
            cross_pro_isO_flag    = cross_pro_isO_flag[new_indices]
            cross_pro_isN_flag    = cross_pro_isN_flag[new_indices]

            #print('cutoff_protein_ids num:', torch.sum(cutoff_protein_ids)) # tensor(252)
            #print('len(new_indices):', np.sum(np.array(new_indices))) #len(new_indices): 251

            #print('new_indices1:', sorted(np.nonzero(np.array(new_indices))[0]))
            #print('new_indices2:', sorted(np.array(new_indices2))) #id，
            
            if len(cross_pro_isring_flag) != len(pocket_coords):
                raise Exception(f'{len(cross_pro_isring_flag)} != {len(pocket_coords)}') #Exception: 251 != 252
            
            #if len(new_indices) != len(self.unimol_pcoords[0]):
                #raise Exception(f'{len(new_indices)} <= 0')

            #print('len(new_indices):', len(new_indices)) #
            #print('new_indices:', new_indices)

            new_atom_isspecial = atom_isspecial[new_indices] #atom_isspecial
            new_atom_id = atom_id[new_indices] #atom_id
            nonzero_indices = new_indices

            #，
            atom_isspecial_index = np.nonzero(new_atom_isspecial == True)[0]
            atom_isgeneral_index = np.nonzero(new_atom_isspecial == False)[0]

            def extract_atoms_by_ids(input_file, atom_id_list, coords, output_file):

                # PDB
                pdb = PandasPdb().read_pdb(input_file)


                # 
                #print(pdb.df['ATOM']['atom_number'].isin(atom_id_list))#，atom_id_list
                #filtered_atoms = pdb.df['ATOM'][pdb.df['ATOM']['atom_number'].isin(atom_id_list)].copy()
                #  iloc 
                #print("pdb.df['ATOM']:", type(pdb.df['ATOM'])) # <class 'pandas.core.frame.DataFrame'>
                #print('atom_id_list.shape:', atom_id_list.shape) #atom_id_list.shape: (125, 3)
                #，atom_id_listpdb？，
                filtered_atoms = pdb.df['ATOM'].iloc[atom_id_list] #pandans
                #print(filtered_atoms.columns.tolist())
                '''
                ['record_name', 'atom_number', 'blank_1', 'atom_name', 'alt_loc', 'residue_name', 'blank_2', 'chain_id', 'residue_number', 'insertion', 
                'blank_3', 'x_coord', 'y_coord', 'z_coord', 'occupancy', 'b_factor', 'blank_4', 'segment_id', 'element_symbol', 'charge', 'line_idx']
                '''
                #print('print(filtered_atoms)1:', filtered_atoms)
                #
                #print(filtered_atoms[['line_idx']])
                #print(type(filtered_atoms[['line_idx']])) #<class 'pandas.core.frame.DataFrame'>
                #print(filtered_atoms[['line_idx']].shape)
                filtered_atoms  = pd.DataFrame(filtered_atoms)
                new_ids         = np.array([list(range(copy.deepcopy(filtered_atoms).shape[0]))]).reshape(-1, 1) #pd.DataFrame
                new_atom_number = np.array([list(range(copy.deepcopy(filtered_atoms).shape[0]))]).reshape(-1, 1) + 1 #pd.DataFrame
                
                #new_atom_number = pd.DataFrame(new_atom_number) #pd.DataFrame,
                #new_ids = pd.DataFrame(new_ids)

                filtered_atoms['line_idx'] = new_ids #2 dim
                filtered_atoms['atom_number'] = new_atom_number# 2 dim
                #filtered_atoms['x_coord'] = np.array([list(range(copy.deepcopy(filtered_atoms).shape[0]))])# 2 dim, ，nan

                #print('print(filtered_atoms)2:', filtered_atoms)
                
                new_coords = filtered_atoms[['x_coord', 'y_coord', 'z_coord']]

                #，
                if not np.allclose(new_coords, coords, atol=0.02):
                    raise SystemExit

                # PandasPdb
                new_pdb = PandasPdb()
                new_pdb.df['ATOM'] = filtered_atoms


                # PDB
                new_pdb.to_pdb(path=output_file, records=['ATOM'], gz=False, append_newline=True) 
                #pdb，id，，，

            sub_protein_file = os.path.join(os.path.dirname(self.protein_file), os.path.splitext(os.path.basename(self.protein_file))[0] + '_400.pdb')
            #extract_atoms_by_ids(self.protein_file, np.array(new_indices), all_coords[new_indices], sub_protein_file)

    
            


            #，
            #400pdb，,# 5l8c/5l8c_pocket10.pdb，5l8c/5l8c_pocket10_400.pdb
            #nonzero_indices
            #: '../CrossDocked2020/data/pdbbind2020_r10/v2020-other-PL/5qb1/5qb1_pocket10_400.pdb'

            '''
            sub_protein_file = os.path.join(os.path.dirname(self.protein_file), os.path.splitext(os.path.basename(self.protein_file))[0] + '_400.pdb')
            sub_protein_w = open(sub_protein_file, 'w')

            org_pro_list = []
            count = 0
            with open(self.protein_file, 'r')as f:
                for i, line in enumerate(f):
                    if line[0:6].strip() == 'ATOM' and i in nonzero_indices:
                        sub_protein_w.write(line)
                

                sub_protein_w.write('END')
            sub_protein_w.close()

            '''
        
            #print('ok3')

            #O,N, ，rdkitpdb，，rdkit，
            #。，，rdkit，
            #，，，rdkit
            '''
            pro_mol=Chem.MolFromPDBFile(self.protein_file,removeHs=False,sanitize=False)
            #print('ok1')
            atom_isring=np.array([atom.IsInRing() for atom in pro_mol.GetAtoms()])    #
            atom_isO=np.array([atom.GetSymbol() == 'O'  for atom in pro_mol.GetAtoms()])    #O
            atom_isN=np.array([atom.GetSymbol() == 'N'  for atom in pro_mol.GetAtoms()])    #N
            coords=np.array(pro_mol.GetConformer(0).GetPositions(), dtype=np.float32) #,float32，float64
            #coords=np.array(pro_mol.GetConformer(0).GetPositions())
            '''
            atom_isO, atom_isN, atom_isring, atom_isspecial, atom_id, coords = self._read_pdb_ONRing_biopandas(self.protein_file)
            #，，-，
            #indexs = []  #, 
            target_xyz =  np.array(self.pos) #xyz
            #target_xyz_sum =  np.round(np.array(self.pos, dtype=np.float32).sum(axis = -1), 2)

            #assert np.array_equal(target_xyz, coords) #，, postbus, assert，try, except

            if not np.array_equal(target_xyz, coords):
                print('target_xyz.shape:', target_xyz.shape)
                print('coords.shape:', coords.shape)
                raise Exception('np.array_equal(target_xyz, coords):', np.array_equal(target_xyz, coords))

            new_atom_isring = atom_isring
            new_atom_isO = atom_isO
            new_atom_isN = atom_isN

            assert len(new_atom_isN[nonzero_indices]) == len(np.array(self.element, dtype=np.int64)[nonzero_indices])
            #print('ok4')



            #

            l2p = [] #
            p2l = [] #
            l2p_type = [] #
            p2l_type = [] #

            #
            ligand_atom_isring = copy.deepcopy(cross_lig_isring_flag)
            ligand_atom_isO    = copy.deepcopy(cross_lig_isO_flag)
            ligand_atom_isN    = copy.deepcopy(cross_lig_isN_flag)

            protein_atom_isring = copy.deepcopy(cross_pro_isring_flag)
            protein_atom_isO    = copy.deepcopy(cross_pro_isO_flag)
            protein_atom_isN    = copy.deepcopy(cross_pro_isN_flag)


            ligand_cross_isring_flag = copy.deepcopy(cross_lig_isring_flag)
            ligand_cross_isO_flag    = copy.deepcopy(cross_lig_isO_flag)
            ligand_cross_isN_flag    = copy.deepcopy(cross_lig_isN_flag)
            ligand_cross_lp_pos      = copy.deepcopy(holo_coords)

            protein_cross_isring_flag = copy.deepcopy(cross_pro_isring_flag)
            protein_cross_isO_flag    = copy.deepcopy(cross_pro_isO_flag) 
            protein_cross_isN_flag    = copy.deepcopy(cross_pro_isN_flag) 
            protein_cross_lp_pos      = copy.deepcopy(pocket_coords)

            cross_distance_matrix = copy.deepcopy(cross_distance) #cross_distancelist，cross_distance

            #[2, n*m,], 4.5O,N,，，3.5，2
            #
            centor = holo_coords.mean(axis = 0)
            l_combination_less3_5, l_combination_greater3_5 = self.combinations_optim_split_3_5(torch.from_numpy(holo_coords), torch.from_numpy(pocket_coords), 
                    torch.from_numpy(np.array(list(range(len(holo_coords))))), 
                    torch.from_numpy(np.array(list(range(len(pocket_coords))))), None, None, 
                    None, 
                    None, None, torch.from_numpy(cross_distance_matrix), 
                    torch.from_numpy(ligand_cross_lp_pos), torch.from_numpy(protein_cross_lp_pos), flag = 'ligand')

            #, ，
            p_combination_less3_5, p_combination_greater3_5 = copy.deepcopy(l_combination_less3_5[[1, 0], :]), copy.deepcopy(l_combination_greater3_5[[1, 0], :])
            assert p_combination_less3_5.shape[0] == 2

            '''
            p_combination_less3_5, p_combination_greater3_5 = self.combinations_optim_split_3_5(torch.from_numpy(pocket_coords), torch.from_numpy(holo_coords), 
                    torch.from_numpy(np.array(list(range(len(pocket_coords))))), 
                    torch.from_numpy(np.array(list(range(len(holo_coords))))), None, None, 
                    None, 
                    None, None, torch.from_numpy(cross_distance_matrix), 
                    torch.from_numpy(ligand_cross_lp_pos), torch.from_numpy(protein_cross_lp_pos), flag = 'protein')
            '''


            if l_combination_less3_5.shape[0] != 0:
                l2p.append(l_combination_less3_5)
                l2p_type.append(np.full(l_combination_less3_5.shape[1], 12))  #，，2*N,N*2？cat
            if l_combination_greater3_5.shape[0] != 0:
                l2p.append(l_combination_greater3_5)
                l2p_type.append(np.full(l_combination_greater3_5.shape[1], 13))


            if p_combination_less3_5.shape[0] != 0:
                p2l.append(p_combination_less3_5)
                p2l_type.append(np.full(p_combination_less3_5.shape[1], 14))  #，，2*N,N*2？cat
            if p_combination_greater3_5.shape[0] != 0:
                p2l.append(p_combination_greater3_5)
                p2l_type.append(np.full(p_combination_greater3_5.shape[1], 15))



            #ON,NO,，，，，knn
            if len(l2p) != 0:
                l2p_edge_index = np.concatenate(l2p, axis = -1, dtype = int)
                l2p_type = np.concatenate(l2p_type, axis = -1, dtype = int)
            else:
                print('')
                exit()
                l2p_edge_index = np.empty((0, 0))
                l2p_type = np.empty((0, 0))
            
            if len(p2l) != 0:
                p2l_edge_index = np.concatenate(p2l, axis = -1, dtype = int)
                p2l_type = np.concatenate(p2l_type, axis = -1, dtype = int)
            else:
                print('')
                exit()
                p2l_edge_index = np.empty((0, 0))
                p2l_type = np.empty((0, 0))

            
            #id，id，cross_bond_index, cross_bond_type， cross_bond_index_reverse, cross_bond_type_reverse
            #cross_bond_index_reverse, cross_bond_type_reverse, [5,6,7],[8,9,10]

            #，12,13,14,15，2，1617，
            #，
            protein_index = torch.from_numpy(np.array(list(range(len(pocket_coords)))))
            protein_pos   = torch.from_numpy(pocket_coords)

            
            #new_cross_bond_index，3.5~4.5，<4.5，。2ai，
            #，，id
            #
            ligand_to_protein_index_dict = defaultdict(set)
            for ids_i, ids_j in torch.LongTensor(l2p_edge_index).T: #N*2
                ligand_to_protein_index_dict[ids_i].add(ids_j)

            extend_cross_bond_index = []
            # , ，
            for l_atom_index in ligand_to_protein_index_dict:
                p_atom_set = ligand_to_protein_index_dict[l_atom_index]

                #
                exit_protein_index = p_atom_set #
                #exit_protein_pos   = x[exit_protein_index]
                
                #，source_protein_index
                target_protein_index = set(protein_index) - set(exit_protein_index)
                #target_protein_pos   = x[target_protein_index]

                #exit_protein_indextarget_protein_index，2ai

                #，
                vec1 = torch.LongTensor(list(exit_protein_index))
                vec2 = torch.LongTensor(list(target_protein_index))
                #  torch.meshgrid 
                grid_x, grid_y = torch.meshgrid(vec1, vec2, indexing='ij')
                # 
                combination = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=0) #shape = [2, n*m]
                
                #6ai, ，,
                dis_limit = 2.0

                dis = torch.norm(protein_pos[combination[0]] - protein_pos[combination[1]], p = 2, dim = -1) #
                dis_index = dis <= dis_limit #

                combination = combination.t()[dis_index] #k * 2
                combination = combination.t() # 2 * k
                extend_pro_atom_index = torch.unique(combination[1])


                #[l_atom_index]extend_pro_atom_index
                vec1 = torch.LongTensor([l_atom_index])
                vec2 = extend_pro_atom_index
                #  torch.meshgrid 
                grid_x, grid_y = torch.meshgrid(vec1, vec2, indexing='ij')
                # 
                combination = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=0) #shape = [2, n*m]
                extend_cross_bond_index.append(combination)

            #12,13,14,15，2，1617，
            try:
                extend_cross_bond_index = torch.cat(extend_cross_bond_index, dim=-1).numpy() # 2 * N
            except Exception as e:
                raise SystemExit
            extend_cross_bond_type  = torch.full([extend_cross_bond_index.shape[1]], 16, dtype = torch.int64).numpy()

            # 
            extend_cross_bond_index_reverse = extend_cross_bond_index[[1, 0], :]
            extend_cross_bond_type_reverse  = torch.full([extend_cross_bond_index_reverse.shape[1]], 17, dtype = torch.int64).numpy()
        
            l2p_edge_index = np.concatenate([l2p_edge_index, extend_cross_bond_index], axis = -1, dtype = int) # 2 * N
            l2p_type       = np.concatenate([l2p_type, extend_cross_bond_type], axis = -1, dtype = int)
            p2l_edge_index = np.concatenate([p2l_edge_index, extend_cross_bond_index_reverse], axis = -1, dtype = int) # 2 * N
            p2l_type       = np.concatenate([p2l_type, extend_cross_bond_type_reverse], axis = -1, dtype = int)
            
            
            mask_protein_pos = np.zeros([GP.max_protein_atoms], dtype=bool)
            mask_protein_pos[:protein_pos.shape[0]] = True

            fill_protein_pos = np.zeros([GP.max_protein_atoms, 3])
            fill_protein_pos[mask_protein_pos] = protein_pos

            data = {
                'element': np.array(self.element, dtype=np.int64)[nonzero_indices], #
                'molecule_name': self.title, # pocket
                #'pos': np.array(self.pos, dtype=np.float32)[nonzero_indices], #
                'pos': pocket_coords,
                'is_backbone': np.array(self.is_backbone, dtype=np.bool_)[nonzero_indices], #Boolean，
                'atom_name': [self.atom_name[i] for i in nonzero_indices], #
                'atom_to_aa_type': np.array(self.atom_to_aa_type, dtype=np.int64)[nonzero_indices], ##
                'atom_isring': new_atom_isring[nonzero_indices],
                'atom_isO': new_atom_isO[nonzero_indices],
                'atom_isN': new_atom_isN[nonzero_indices],

                'cross_lig_isring_flag': cross_lig_isring_flag,
                'cross_lig_isO_flag': cross_lig_isO_flag,
                'cross_lig_isN_flag': cross_lig_isN_flag,

                'cross_pro_isring_flag': cross_pro_isring_flag,
                'cross_pro_isO_flag': cross_pro_isO_flag,
                'cross_pro_isN_flag': cross_pro_isN_flag,

                'cross_ligand': holo_coords,
                'cross_protein': pocket_coords,

                #'cross_distance': set(torch.from_numpy(cross_distance)), #，，？，
                #'cross_distance': list(torch.from_numpy(cross_distance)), #pyglist，
                #'cross_distance': tuple(torch.from_numpy(cross_distance)), #tuple，pyg
                #'cross_distance': OrderedSet(torch.from_numpy(cross_distance)), #，pyg，numpy

                'cross_distance': cross_distance,

                #'cross_bond_index': np.ones_like(l2p_edge_index),
                #'cross_distance2': np.ones_like(l2p_edge_index), #？？，，
                #Batch.from_data_list([data.clone() for _ in range(n_data)], follow_batch=FOLLOW_BATCH, exclude_keys = collate_exclude_keys).to(device)，
                #bond？pyg？?

                #'cross_distance1': l2p_edge_index,
                #'cross_distance2': l2p_type,
                #'cross_distance3': p2l_edge_index,
                #'cross_distance4': p2l_type,
                

                'link_e': l2p_edge_index.T, # N * 2
                'link_t': l2p_type,
                'link_e_reverse': p2l_edge_index.T, # N * 2
                'link_t_reverse': p2l_type,

                #coords_predict, unimol，ground true，
                'coords_predict':coords_predict,

                'mask_protein_pos': mask_protein_pos,
                'fill_protein_pos': fill_protein_pos,
                #'ligand_emb': ligand_emb,
                #'pocket_emb': pocket_emb[cutoff_protein_ids],


                #def torchify_dict(data):
            }

            #print('ok5')
            return data
        
        
        #except Exception as e:
            #print('protein error:', e)
            #print('self.protein_file:', self.protein_file)
            #raise Exception('protein error, stop')
            #exit()
            #return None



    def to_dict_atom_not_interaction_gen_split3_5_extend(self):
        #<3.53.5~4.5，，4，，，，20，2，
        # 4index：12,13,14,15

        old_data = {
            'element': np.array(self.element, dtype=np.int64), #
            'molecule_name': self.title, # pocket
            'pos': np.array(self.pos, dtype=np.float32), #
            'is_backbone': np.array(self.is_backbone, dtype=np.bool_), #Boolean，
            'atom_name': self.atom_name, #
            'atom_to_aa_type': np.array(self.atom_to_aa_type, dtype=np.int64) ##
        }

        #，400
        if 'pdbbind2020' in self.protein_file:
            file_path = os.path.join(os.path.dirname(self.protein_file), 'interaction_' + os.path.basename(self.protein_file).split('_')[0] + '_v2.pkl')   #5S8I_protein.pdb
        else:
            #file_path = os.path.join(os.path.dirname(self.protein_file), 'interaction_' + os.path.basename(self.protein_file).split('_')[0] + '_v2.pkl')   #5S8I_protein.pdb
            file_path = os.path.join(os.path.dirname(self.protein_file), 'interaction_' + os.path.basename(self.protein_file).rsplit("_protein.pdb", 1)[0] + '_v2.pkl')
        with open(file_path, 'rb') as file:
            interaction_data = dill.load(file)

        if GP.glide_vina:
            #glidevina
            tmp             = os.path.basename(self.protein_file).split('_')[0]
            predict_file    = os.path.join(os.path.dirname(self.protein_file), f'{tmp}_ligand.sdf')
            predict_sup     = list(Chem.rdmolfiles.SDMolSupplier(predict_file)) #，

            if len(predict_sup) < 40:
                predict_sup.extend([predict_sup[0]] * (40 - len(predict_sup)))

            conf_num = len(predict_sup)
            coords_predict_list = []
            cross_distance_list = []

            #40, 40
            if conf_num >= 41:
                conf_num = 40


            #2, 
            holo_coords_list    = interaction_data['holo_coords_list'][:conf_num]
            pocket_coords_list  = interaction_data['pocket_coords_list'][:conf_num]

            for k in range(conf_num):
                coords_predict  = np.array(Chem.RemoveHs(predict_sup[k]).GetConformer(0).GetPositions(), dtype = np.float32)
                coords_predict_list.append(coords_predict)

                #, 40
                if k >= 40:
                    k = 39

                try:
                    cross_distance = calculate_distance_matrix_numpy(coords_predict, pocket_coords_list[k]) #
                except Exception as e:
                    print("interaction_data['pocket_coords_list'] num:", len(interaction_data['pocket_coords_list']))
                    print('pocket_coords_list:', len(pocket_coords_list))
                    print('conf_num:', conf_num)
                    print('k:', k)
                    raise Exception(e) #40
                cross_distance_list.append(cross_distance)




        else:
            holo_coords_list = interaction_data['holo_coords_list']
            coords_predict_list = interaction_data['coords_predict_list']
            pocket_coords_list = interaction_data['pocket_coords_list']
            cross_distance_list = interaction_data['cross_distance_list']

        assert np.allclose(holo_coords_list[0], holo_coords_list[-1], atol=0.02)

        #print('pocket_coords_list[0]:', pocket_coords_list[0][:])
        #np.set_printoptions(suppress=True, precision=4)
        #print('pocket_coords_list[-1]:', pocket_coords_list[-1]) #，？？，
        #print('pocket_coords_list[0].shape:', pocket_coords_list[0].shape)
        #print('pocket_coords_list[-1].shape:', pocket_coords_list[-1].shape) #，？？
        #assert np.allclose(pocket_coords_list[0], pocket_coords_list[-1], atol=0.02)

        assert len(holo_coords_list) == len(coords_predict_list) and len(holo_coords_list) == len(pocket_coords_list) and len(holo_coords_list) == len(cross_distance_list)

        if len(holo_coords_list) == 1:
            holo_coords_list     = holo_coords_list * 100
            coords_predict_list  = coords_predict_list * 100
            pocket_coords_list   = pocket_coords_list * 100
            cross_distance_list  = cross_distance_list * 100
            
        #rmsd，，
        rmsd_list = []
        for pre_pos, holo_pos in zip(coords_predict_list, holo_coords_list):
            assert pre_pos.shape == holo_pos.shape   #"Coordinate matrices must have the same shape"
            rmsd = np.sqrt(np.mean(np.sum((pre_pos - holo_pos) ** 2, axis=1)))
            rmsd_list.append(rmsd)
        
        sorted_indices = np.argsort(rmsd_list)
        best_index = sorted_indices[0]
        #, ，n
        #best_index  = random.choice(list(range(len(holo_coords_list))))
        
        #1，，N，1，

            
        if GP.cross_distance_num == 'best' or GP.cross_distance_num == None:
            holo_coords     = holo_coords_list[best_index]
            coords_predict  = coords_predict_list[best_index]
            pocket_coords   = pocket_coords_list[best_index]
            cross_distance  = cross_distance_list[best_index]
        else:
            #，
            holo_coords_list    = [holo_coords_list[k] for k in sorted_indices] 
            coords_predict_list = [coords_predict_list[k] for k in sorted_indices] 
            pocket_coords_list  = [pocket_coords_list[k] for k in sorted_indices] 
            cross_distance_list = [cross_distance_list[k] for k in sorted_indices] 

            #
            try:
                holo_coords     = holo_coords_list[self.cross_distance_num]
            except Exception as e:
                print('self.cross_distance_num:', self.cross_distance_num)
                print('len(holo_coords_list):', len(holo_coords_list))

                #self.cross_distance_num: 2
                #len(holo_coords_list): 2

                raise Exception(e)
            
            coords_predict  = coords_predict_list[self.cross_distance_num]
            pocket_coords   = pocket_coords_list[self.cross_distance_num]
            cross_distance  = cross_distance_list[self.cross_distance_num]

        '''
        if self.cross_distance_num == None or self.cross_distance_num == 'best':
            holo_coords = holo_coords_list[best_index]
            coords_predict = coords_predict_list[best_index]
            pocket_coords = pocket_coords_list[best_index]
            cross_distance = cross_distance_list[best_index]
        else:            
            holo_coords = holo_coords_list[self.cross_distance_num]
            coords_predict = coords_predict_list[self.cross_distance_num]
            pocket_coords = pocket_coords_list[self.cross_distance_num]
            cross_distance = cross_distance_list[self.cross_distance_num]
        '''



        #unimol，，
        unique_index_dict       = {}
        for j, ps in enumerate(pocket_coords):
            unique_index_dict[tuple(ps)] = j
            #，
        
        unique_index_list   = list(unique_index_dict.values())
        pocket_coords       = pocket_coords[unique_index_list]
        cross_distance      = cross_distance[:, unique_index_list]



        #，unimolrdkit，，
        #cross_distanceO,N,
        #，
        pro_isring_flag = {}
        pro_isO_flag = {}
        pro_isN_flag = {}

        '''
        pro_mol=Chem.MolFromPDBFile(self.protein_file,removeHs=False,sanitize=False) #，, 
        #print('self.protein_file:', self.protein_file)
        #print('ok1')
        atom_isring=np.array([atom.IsInRing() for atom in pro_mol.GetAtoms()])    #
        atom_isO=np.array([atom.GetSymbol() == 'O'  for atom in pro_mol.GetAtoms()])    #O
        atom_isN=np.array([atom.GetSymbol() == 'N'  for atom in pro_mol.GetAtoms()])    #N
        #coords=np.array(pro_mol.GetConformer(0).GetPositions(), dtype=np.float32) #,float32，float64
        coords=np.array(pro_mol.GetConformer(0).GetPositions())
        '''
        atom_isO, atom_isN, atom_isring, atom_isspecial, atom_id, coords = self._read_pdb_ONRing_biopandas(self.protein_file)
        count = 0
        coords_atom_dict = defaultdict(list)
        for r, o, n, c in zip(atom_isring, atom_isO, atom_isN, coords):
            count += 1
            k = torch.FloatTensor(c)
            tg = ''
            for ii in k:
                #tg += str(round(i.item(), 4)) + '_'
                tg += str(self.truncate(ii.item(), 3)) + '_' #，2，
            c_sum = str(tg)

            coords_atom_dict[c_sum].append(c)

            pro_isring_flag[c_sum] = r
            pro_isO_flag[c_sum]    = o
            pro_isN_flag[c_sum]    = n
        
        assert len(pro_isring_flag) == len(atom_isring)
        
        print('：', count)
        if len(pro_isring_flag) != len(coords) or len(pro_isO_flag) != len(coords) or len(pro_isN_flag) != len(coords):
            print(f'{len(pro_isring_flag)} != {len(coords)} or {len(pro_isO_flag)} != {len(coords)} or {len(pro_isN_flag)} != {len(coords)}')
            #979 != 990 or 979 != 990 or 979 != 990 #？
            raise Exception("pro atom num is error")


        cross_pro_isring_flag = copy.deepcopy(atom_isring)
        cross_pro_isO_flag = copy.deepcopy(atom_isO)
        cross_pro_isN_flag = copy.deepcopy(atom_isN)



        #，,
        #，unimolrdkit，unimolrdkit，sdf，rdkit mol，rdkit
        #，unimol，rdkit
        lig_isring_flag = {}
        lig_isO_flag = {}
        lig_isN_flag = {}


        #，rdkit3d，，
        if self.data_flag == 'new_test':
            
            lig_mol = copy.deepcopy(self.ligand_dict['mol']) #
            '''
            #，？cross，，
            lig_rdkit_mol = copy.deepcopy(self.ligand_dict['rd_mol'])

            unimol_pos  = torch.FloatTensor(holo_coords)
            rdkit_pos   = torch.FloatTensor(lig_rdkit_mol.GetConformer(0).GetPositions())
            ground_pos  = torch.FloatTensor(lig_mol.GetConformer(0).GetPositions())

            #
            try:
                assert len(ground_pos) == len(unimol_pos) #，rkdit，，，
            except AssertionError:
                raise SystemExit

            #，rdkit, rdkitsmiles3dground，3d，
            if self.compare_atom_order(lig_mol, lig_rdkit_mol) and torch.allclose(ground_pos, unimol_pos, atol=0.02):
                lig_mol     = copy.deepcopy(lig_rdkit_mol) #，org_liandlig_rdkit_mol, 
                holo_coords = np.array(lig_rdkit_mol.GetConformer(0).GetPositions()) 
            
            #lig_mol = copy.deepcopy(self.ligand_dict['mol']) 
            '''
        else:
            lig_mol = copy.deepcopy(self.ligand_dict['mol']) #





        #print('ok1')
        atom_isring=np.array([atom.IsInRing() for atom in lig_mol.GetAtoms()])    #
        atom_isO=np.array([atom.GetSymbol() == 'O'  for atom in lig_mol.GetAtoms()])    #O
        atom_isN=np.array([atom.GetSymbol() == 'N'  for atom in lig_mol.GetAtoms()])    #N
        #coords=np.array(lig_mol.GetConformer(0).GetPositions(), dtype=np.float32) #,float32，float64
        coords=np.array(lig_mol.GetConformer(0).GetPositions())

        for r, o, n, c in zip(atom_isring, atom_isO, atom_isN, coords):
            k = torch.FloatTensor(c)
            tg = ''
            for ii in k:
                #tg += str(round(i.item(), 4)) + '_'
                tg += str(self.truncate(ii.item(), 3)) + '_' #，2，
            c_sum = str(tg)

            lig_isring_flag[c_sum] = r
            lig_isO_flag[c_sum]    = o
            lig_isN_flag[c_sum]    = n
        
        
        try:
            assert len(coords) == len(holo_coords) #，rkdit，，，
            assert len(lig_isring_flag) == len(atom_isring)
        except Exception as e:
            print(e)
            raise SystemExit

        if len(lig_isring_flag) != len(coords) or len(lig_isO_flag) != len(coords) or len(lig_isN_flag) != len(coords) or len(coords) != len(holo_coords):
            print(f'{len(coords)} != {len(holo_coords)}') #19 != 18, rdkit，crosss_liand，
            raise Exception("lig atom num is error") #，

    
        #unimolrdkit, 0.02
        if GP.glide_vina:
            #glidevina，ground truth
            pass
        else:
            assert torch.allclose(torch.FloatTensor(coords), torch.FloatTensor(holo_coords), atol=0.02)

        holo_coords = copy.deepcopy(coords)

        #，rdkitsdf，，，，，，holo_coords = coords
        #cross_distance, cross_ligand, cross_protein, cross_ligand_atom_flag, cross_protein_atom_flag

        cross_lig_isring_flag = copy.deepcopy(atom_isring)
        cross_lig_isO_flag = copy.deepcopy(atom_isO)
        cross_lig_isN_flag = copy.deepcopy(atom_isN)

        
        if old_data['pos'].shape[0] <= 0:
            print("old_data['pos'].size(0):", old_data['pos'].shape[0])
            print("old_data['pos']:", old_data['pos'].shape)
            return old_data
        else:
            data = {}
            #self.ligand_centor
            dis   = np.linalg.norm(old_data['pos'] - self.ligand_centor, axis = 1) #

        #try:
            #，12ai，400，400

            if pocket_coords.shape[0] < 300:
                cutoff_num = 2000
            else:
                cutoff_num = 400

            #，，rdkit
            #print('self.protein_file:', self.protein_file)
            '''
            pro_mol=Chem.MolFromPDBFile(self.protein_file,removeHs=False,sanitize=False)
            atom_isspecial = np.array([atom.IsInRing() or atom.GetSymbol() == 'O' or atom.GetSymbol() == 'N' for atom in pro_mol.GetAtoms()]) #
            atom_id = np.array([atom.GetIdx() for atom in pro_mol.GetAtoms()])
            #coords=np.array(pro_mol.GetConformer(0).GetPositions(), dtype=np.float32) #,float32，float64
            all_coords=np.array(pro_mol.GetConformer(0).GetPositions())
            '''
            atom_isO, atom_isN, atom_isring, atom_isspecial, atom_id, all_coords = self._read_pdb_ONRing_biopandas(self.protein_file)
            #
            #cutoff_indices = np.argsort(dis)[:] #,cutoff_num，，
            #cutoff_indices = np.array(list(range(len(all_coords))))
            cutoff_indices = atom_id
            coords = all_coords[cutoff_indices] #，

            assert old_data['pos'].shape == coords.shape

            assert cutoff_indices.shape == np.unique(cutoff_indices).shape

            #，pocket_coords，400
            new_indices = []
            pro_flag_dict = {} #value
            pro_flag_dict2 = {}

            for j, c in zip(cutoff_indices, coords): #j，，
                k = torch.FloatTensor(c)
                tg = ''
                for ii in k:
                    #tg += str(round(i.item(), 4)) + '_'
                    tg += str(self.truncate(ii.item(), 3)) + '_' #，2，
                k = str(tg)

                pro_flag_dict[k] = j

                k2 = torch.FloatTensor(c)
                tg2 = ''
                for ii in k2:
                    #tg += str(round(i.item(), 4)) + '_'
                    tg2 += str(self.truncate2(ii.item(), 3)) + '_' #，2，
                k2 = str(tg2)

                pro_flag_dict2[k2] = j

            

            if len(pro_flag_dict) != len(coords):
                raise Exception(f"{len(pro_flag_dict)} != {len(coords)}")


            if len(pro_flag_dict2) != len(coords):
                raise Exception(f"{len(pro_flag_dict2)} != {len(coords)}")
            
            #cutoff_protein_ids = torch.ones(pocket_coords.shape[0], dtype = torch.bool)
            cutoff_protein_ids = torch.zeros(pocket_coords.shape[0], dtype = torch.bool)

            assert pocket_coords.shape == np.unique(pocket_coords, axis = 0).shape

            for ids, c in enumerate(pocket_coords): #pocket_coords
                k = torch.FloatTensor(c)
                tg = ''
                for ii in k:
                    #tg += str(round(i.item(), 4)) + '_'
                    tg += str(self.truncate(ii.item(), 3)) + '_' #，2，
                k = str(tg)

                #pocket_coords，
                #if cross_pro_flag_dict.get(c_sum):

                
                try:
                    new_indices.append(pro_flag_dict[k]) #，unimol，，, 
                    cutoff_protein_ids[ids] = True
                except Exception as e:
                    print(e)
                    #cutoff_protein_ids[ids] = False
                    #print('key error, skip:', k)
                    #print('key all:', list(pro_flag_dict.keys()))
                    #print('-----------------------------------------')
                    continue
                    
            #，，
            pocket_coords  = pocket_coords[cutoff_protein_ids]
            cross_distance = cross_distance[:, cutoff_protein_ids]

            cross_pro_isring_flag = cross_pro_isring_flag[new_indices]
            cross_pro_isO_flag    = cross_pro_isO_flag[new_indices]
            cross_pro_isN_flag    = cross_pro_isN_flag[new_indices]

            #print('cutoff_protein_ids num:', torch.sum(cutoff_protein_ids)) # tensor(252)
            #print('len(new_indices):', np.sum(np.array(new_indices))) #len(new_indices): 251

            #print('new_indices1:', sorted(np.nonzero(np.array(new_indices))[0]))
            #print('new_indices2:', sorted(np.array(new_indices2))) #id，
            
            if len(cross_pro_isring_flag) != len(pocket_coords):
                raise Exception(f'{len(cross_pro_isring_flag)} != {len(pocket_coords)}') #Exception: 251 != 252
            
            #if len(new_indices) != len(self.unimol_pcoords[0]):
                #raise Exception(f'{len(new_indices)} <= 0')

            #print('len(new_indices):', len(new_indices)) #
            #print('new_indices:', new_indices)

            new_atom_isspecial = atom_isspecial[new_indices] #atom_isspecial
            new_atom_id = atom_id[new_indices] #atom_id
            nonzero_indices = new_indices

            #，
            atom_isspecial_index = np.nonzero(new_atom_isspecial == True)[0]
            atom_isgeneral_index = np.nonzero(new_atom_isspecial == False)[0]

            def extract_atoms_by_ids(input_file, atom_id_list, coords, output_file):

                # PDB
                pdb = PandasPdb().read_pdb(input_file)


                # 
                #print(pdb.df['ATOM']['atom_number'].isin(atom_id_list))#，atom_id_list
                #filtered_atoms = pdb.df['ATOM'][pdb.df['ATOM']['atom_number'].isin(atom_id_list)].copy()
                #  iloc 
                #print("pdb.df['ATOM']:", type(pdb.df['ATOM'])) # <class 'pandas.core.frame.DataFrame'>
                #print('atom_id_list.shape:', atom_id_list.shape) #atom_id_list.shape: (125, 3)
                #，atom_id_listpdb？，
                
                print('atom_id_list.shape:', atom_id_list.shape) #(699,)
                print("pdb.df['ATOM'].shape:", pdb.df['ATOM'].shape) #pdb.df['ATOM'].shape: (9596, 21)
    
                filtered_atoms = pdb.df['ATOM'].iloc[atom_id_list] #pandans
                #print(filtered_atoms.columns.tolist())
                '''
                ['record_name', 'atom_number', 'blank_1', 'atom_name', 'alt_loc', 'residue_name', 'blank_2', 'chain_id', 'residue_number', 'insertion', 
                'blank_3', 'x_coord', 'y_coord', 'z_coord', 'occupancy', 'b_factor', 'blank_4', 'segment_id', 'element_symbol', 'charge', 'line_idx']
                '''
                #print('print(filtered_atoms)1:', filtered_atoms)
                #
                #print(filtered_atoms[['line_idx']])
                #print(type(filtered_atoms[['line_idx']])) #<class 'pandas.core.frame.DataFrame'>
                #print(filtered_atoms[['line_idx']].shape)
                filtered_atoms  = pd.DataFrame(filtered_atoms)
                new_ids         = np.array([list(range(copy.deepcopy(filtered_atoms).shape[0]))]).reshape(-1, 1) #pd.DataFrame
                new_atom_number = np.array([list(range(copy.deepcopy(filtered_atoms).shape[0]))]).reshape(-1, 1) + 1 #pd.DataFrame
                
                #new_atom_number = pd.DataFrame(new_atom_number) #pd.DataFrame,
                #new_ids = pd.DataFrame(new_ids)

                filtered_atoms['line_idx'] = new_ids #2 dim
                filtered_atoms['atom_number'] = new_atom_number# 2 dim
                #filtered_atoms['x_coord'] = np.array([list(range(copy.deepcopy(filtered_atoms).shape[0]))])# 2 dim, ，nan

                #print('print(filtered_atoms)2:', filtered_atoms)
                
                new_coords = filtered_atoms[['x_coord', 'y_coord', 'z_coord']]

                #，
                if not np.allclose(new_coords, coords, atol=0.02):
                    raise SystemExit

                # PandasPdb
                new_pdb = PandasPdb()
                new_pdb.df['ATOM'] = filtered_atoms


                # PDB
                new_pdb.to_pdb(path=output_file, records=['ATOM'], gz=False, append_newline=True) 
                #pdb，id，，，

            sub_protein_file = os.path.join(os.path.dirname(self.protein_file), os.path.splitext(os.path.basename(self.protein_file))[0] + '_400.pdb')
            #extract_atoms_by_ids(self.protein_file, np.array(new_indices), all_coords[new_indices], sub_protein_file)

    
            


            #，
            #400pdb，,# 5l8c/5l8c_pocket10.pdb，5l8c/5l8c_pocket10_400.pdb
            #nonzero_indices
            #: '../CrossDocked2020/data/pdbbind2020_r10/v2020-other-PL/5qb1/5qb1_pocket10_400.pdb'

            '''
            sub_protein_file = os.path.join(os.path.dirname(self.protein_file), os.path.splitext(os.path.basename(self.protein_file))[0] + '_400.pdb')
            sub_protein_w = open(sub_protein_file, 'w')

            org_pro_list = []
            count = 0
            with open(self.protein_file, 'r')as f:
                for i, line in enumerate(f):
                    if line[0:6].strip() == 'ATOM' and i in nonzero_indices:
                        sub_protein_w.write(line)
                

                sub_protein_w.write('END')
            sub_protein_w.close()

            '''
        
            #print('ok3')

            #O,N, ，rdkitpdb，，rdkit，
            #。，，rdkit，
            #，，，rdkit
            '''
            pro_mol=Chem.MolFromPDBFile(self.protein_file,removeHs=False,sanitize=False)
            #print('ok1')
            atom_isring=np.array([atom.IsInRing() for atom in pro_mol.GetAtoms()])    #
            atom_isO=np.array([atom.GetSymbol() == 'O'  for atom in pro_mol.GetAtoms()])    #O
            atom_isN=np.array([atom.GetSymbol() == 'N'  for atom in pro_mol.GetAtoms()])    #N
            coords=np.array(pro_mol.GetConformer(0).GetPositions(), dtype=np.float32) #,float32，float64
            #coords=np.array(pro_mol.GetConformer(0).GetPositions())
            '''
            atom_isO, atom_isN, atom_isring, atom_isspecial, atom_id, coords = self._read_pdb_ONRing_biopandas(self.protein_file)
            #，，-，
            #indexs = []  #, 
            target_xyz =  np.array(self.pos) #xyz
            #target_xyz_sum =  np.round(np.array(self.pos, dtype=np.float32).sum(axis = -1), 2)

            #assert np.array_equal(target_xyz, coords) #，, postbus, assert，try, except

            if not np.array_equal(target_xyz, coords):
                print('target_xyz.shape:', target_xyz.shape)
                print('coords.shape:', coords.shape)
                raise Exception('np.array_equal(target_xyz, coords):', np.array_equal(target_xyz, coords))

            new_atom_isring = atom_isring
            new_atom_isO = atom_isO
            new_atom_isN = atom_isN

            assert len(new_atom_isN[nonzero_indices]) == len(np.array(self.element, dtype=np.int64)[nonzero_indices])
            #print('ok4')



            #

            l2p = [] #
            p2l = [] #
            l2p_type = [] #
            p2l_type = [] #

            #
            ligand_atom_isring = copy.deepcopy(cross_lig_isring_flag)
            ligand_atom_isO    = copy.deepcopy(cross_lig_isO_flag)
            ligand_atom_isN    = copy.deepcopy(cross_lig_isN_flag)

            protein_atom_isring = copy.deepcopy(cross_pro_isring_flag)
            protein_atom_isO    = copy.deepcopy(cross_pro_isO_flag)
            protein_atom_isN    = copy.deepcopy(cross_pro_isN_flag)


            ligand_cross_isring_flag = copy.deepcopy(cross_lig_isring_flag)
            ligand_cross_isO_flag    = copy.deepcopy(cross_lig_isO_flag)
            ligand_cross_isN_flag    = copy.deepcopy(cross_lig_isN_flag)
            ligand_cross_lp_pos      = copy.deepcopy(holo_coords)

            protein_cross_isring_flag = copy.deepcopy(cross_pro_isring_flag)
            protein_cross_isO_flag    = copy.deepcopy(cross_pro_isO_flag) 
            protein_cross_isN_flag    = copy.deepcopy(cross_pro_isN_flag) 
            protein_cross_lp_pos      = copy.deepcopy(pocket_coords)

            cross_distance_matrix = copy.deepcopy(cross_distance) #cross_distancelist，cross_distance

            #[2, n*m,], 4.5O,N,，，3.5，2
            #
            centor = holo_coords.mean(axis = 0)
            l_combination_less3_5, l_combination_greater3_5 = self.combinations_optim_split_3_5(torch.from_numpy(holo_coords), torch.from_numpy(pocket_coords), 
                    torch.from_numpy(np.array(list(range(len(holo_coords))))), 
                    torch.from_numpy(np.array(list(range(len(pocket_coords))))), None, None, 
                    None, 
                    None, None, torch.from_numpy(cross_distance_matrix), 
                    torch.from_numpy(ligand_cross_lp_pos), torch.from_numpy(protein_cross_lp_pos), flag = 'ligand')

            #, ，
            p_combination_less3_5, p_combination_greater3_5 = copy.deepcopy(l_combination_less3_5[[1, 0], :]), copy.deepcopy(l_combination_greater3_5[[1, 0], :])
            assert p_combination_less3_5.shape[0] == 2

            '''
            p_combination_less3_5, p_combination_greater3_5 = self.combinations_optim_split_3_5(torch.from_numpy(pocket_coords), torch.from_numpy(holo_coords), 
                    torch.from_numpy(np.array(list(range(len(pocket_coords))))), 
                    torch.from_numpy(np.array(list(range(len(holo_coords))))), None, None, 
                    None, 
                    None, None, torch.from_numpy(cross_distance_matrix), 
                    torch.from_numpy(ligand_cross_lp_pos), torch.from_numpy(protein_cross_lp_pos), flag = 'protein')
            '''


            if l_combination_less3_5.shape[0] != 0:
                l2p.append(l_combination_less3_5)
                l2p_type.append(np.full(l_combination_less3_5.shape[1], 12))  #，，2*N,N*2？cat
            if l_combination_greater3_5.shape[0] != 0:
                l2p.append(l_combination_greater3_5)
                l2p_type.append(np.full(l_combination_greater3_5.shape[1], 13))


            if p_combination_less3_5.shape[0] != 0:
                p2l.append(p_combination_less3_5)
                p2l_type.append(np.full(p_combination_less3_5.shape[1], 14))  #，，2*N,N*2？cat
            if p_combination_greater3_5.shape[0] != 0:
                p2l.append(p_combination_greater3_5)
                p2l_type.append(np.full(p_combination_greater3_5.shape[1], 15))



            #ON,NO,，，，，knn
            if len(l2p) != 0:
                l2p_edge_index = np.concatenate(l2p, axis = -1, dtype = int)
                l2p_type = np.concatenate(l2p_type, axis = -1, dtype = int)
            else:
                print('')
                exit()
                l2p_edge_index = np.empty((0, 0))
                l2p_type = np.empty((0, 0))
            
            if len(p2l) != 0:
                p2l_edge_index = np.concatenate(p2l, axis = -1, dtype = int)
                p2l_type = np.concatenate(p2l_type, axis = -1, dtype = int)
            else:
                print('')
                exit()
                p2l_edge_index = np.empty((0, 0))
                p2l_type = np.empty((0, 0))

            #id，id，cross_bond_index, cross_bond_type， cross_bond_index_reverse, cross_bond_type_reverse
            #cross_bond_index_reverse, cross_bond_type_reverse, [5,6,7],[8,9,10]

            #，12,13,14,15，2，1617，
            #，
            protein_index = torch.from_numpy(np.array(list(range(len(pocket_coords)))))
            protein_pos   = torch.from_numpy(pocket_coords)

            #new_cross_bond_index，3.5~4.5，<4.5，。2ai，
            #，，id
            #
            ligand_to_protein_index_dict = defaultdict(set)
            for ids_i, ids_j in torch.LongTensor(l2p_edge_index).T: #N*2
                ligand_to_protein_index_dict[ids_i].add(ids_j)

            extend_cross_bond_index = []
            # , ，
            for l_atom_index in ligand_to_protein_index_dict:
                p_atom_set = ligand_to_protein_index_dict[l_atom_index]

                #
                exit_protein_index = p_atom_set #
                #exit_protein_pos   = x[exit_protein_index]
                
                #，source_protein_index
                target_protein_index = set(protein_index) - set(exit_protein_index)
                #target_protein_pos   = x[target_protein_index]

                #exit_protein_indextarget_protein_index，2ai

                #，
                vec1 = torch.LongTensor(list(exit_protein_index))
                vec2 = torch.LongTensor(list(target_protein_index))
                #  torch.meshgrid 
                grid_x, grid_y = torch.meshgrid(vec1, vec2, indexing='ij')
                # 
                combination = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=0) #shape = [2, n*m]
                
                #6ai, ，,
                dis_limit = 2.0

                dis = torch.norm(protein_pos[combination[0]] - protein_pos[combination[1]], p = 2, dim = -1) #
                dis_index = dis <= dis_limit #

                combination = combination.t()[dis_index] #k * 2
                combination = combination.t() # 2 * k
                extend_pro_atom_index = torch.unique(combination[1])


                #[l_atom_index]extend_pro_atom_index
                vec1 = torch.LongTensor([l_atom_index])
                vec2 = extend_pro_atom_index
                #  torch.meshgrid 
                grid_x, grid_y = torch.meshgrid(vec1, vec2, indexing='ij')
                # 
                combination = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=0) #shape = [2, n*m]
                extend_cross_bond_index.append(combination)

            #12,13,14,15，2，1617，
            try:
                extend_cross_bond_index = torch.cat(extend_cross_bond_index, dim=-1).numpy() # 2 * N
            except Exception as e:
                raise SystemExit
            extend_cross_bond_type  = torch.full([extend_cross_bond_index.shape[1]], 16, dtype = torch.int64).numpy()

            # 
            extend_cross_bond_index_reverse = extend_cross_bond_index[[1, 0], :]
            extend_cross_bond_type_reverse  = torch.full([extend_cross_bond_index_reverse.shape[1]], 17, dtype = torch.int64).numpy()
        
            l2p_edge_index = np.concatenate([l2p_edge_index, extend_cross_bond_index], axis = -1, dtype = int) # 2 * N
            l2p_type       = np.concatenate([l2p_type, extend_cross_bond_type], axis = -1, dtype = int)
            p2l_edge_index = np.concatenate([p2l_edge_index, extend_cross_bond_index_reverse], axis = -1, dtype = int) # 2 * N
            p2l_type       = np.concatenate([p2l_type, extend_cross_bond_type_reverse], axis = -1, dtype = int)

            mask_protein_pos = np.zeros([GP.max_protein_atoms], dtype=bool)
            mask_protein_pos[:protein_pos.shape[0]] = True

            fill_protein_pos = np.zeros([GP.max_protein_atoms, 3])
            fill_protein_pos[mask_protein_pos] = protein_pos

            data = {
                'element': np.array(self.element, dtype=np.int64)[nonzero_indices], #
                'molecule_name': self.title, # pocket
                #'pos': np.array(self.pos, dtype=np.float32)[nonzero_indices], #
                'pos': pocket_coords,
                'is_backbone': np.array(self.is_backbone, dtype=np.bool_)[nonzero_indices], #Boolean，
                'atom_name': [self.atom_name[i] for i in nonzero_indices], #
                'atom_to_aa_type': np.array(self.atom_to_aa_type, dtype=np.int64)[nonzero_indices], ##
                'atom_isring': new_atom_isring[nonzero_indices],
                'atom_isO': new_atom_isO[nonzero_indices],
                'atom_isN': new_atom_isN[nonzero_indices],

                'cross_lig_isring_flag': cross_lig_isring_flag,
                'cross_lig_isO_flag': cross_lig_isO_flag,
                'cross_lig_isN_flag': cross_lig_isN_flag,

                'cross_pro_isring_flag': cross_pro_isring_flag,
                'cross_pro_isO_flag': cross_pro_isO_flag,
                'cross_pro_isN_flag': cross_pro_isN_flag,

                'cross_ligand': holo_coords,
                'cross_protein': pocket_coords,

                #'cross_distance': set(torch.from_numpy(cross_distance)), #，，？，
                #'cross_distance': list(torch.from_numpy(cross_distance)), #pyglist，
                #'cross_distance': tuple(torch.from_numpy(cross_distance)), #tuple，pyg
                #'cross_distance': OrderedSet(torch.from_numpy(cross_distance)), #，pyg，numpy

                'cross_distance': cross_distance,

                #'cross_bond_index': np.ones_like(l2p_edge_index),
                #'cross_distance2': np.ones_like(l2p_edge_index), #？？，，
                #Batch.from_data_list([data.clone() for _ in range(n_data)], follow_batch=FOLLOW_BATCH, exclude_keys = collate_exclude_keys).to(device)，
                #bond？pyg？?

                #'cross_distance1': l2p_edge_index,
                #'cross_distance2': l2p_type,
                #'cross_distance3': p2l_edge_index,
                #'cross_distance4': p2l_type,
                

                'link_e': l2p_edge_index.T, # N * 2
                'link_t': l2p_type,
                'link_e_reverse': p2l_edge_index.T, # N * 2
                'link_t_reverse': p2l_type,

                #coords_predict, unimol，ground true，
                'coords_predict':coords_predict,

                'mask_protein_pos': mask_protein_pos,
                'fill_protein_pos': fill_protein_pos,


                #def torchify_dict(data):
            }

            #print('ok5')
            return data
        
        
        #except Exception as e:
            #print('protein error:', e)
            #print('self.protein_file:', self.protein_file)
            #raise Exception('protein error, stop')
            #exit()
            #return None





    def to_dict_atom_interaction_gen_split3_5_coords_connection(self):
        #<3.53.5~4.5，，4，，，，20，2，
        # 4index：12,13,14,15

        old_data = {
            'element': np.array(self.element, dtype=np.int64), #
            'molecule_name': self.title, # pocket
            'pos': np.array(self.pos, dtype=np.float32), #
            'is_backbone': np.array(self.is_backbone, dtype=np.bool_), #Boolean，
            'atom_name': self.atom_name, #
            'atom_to_aa_type': np.array(self.atom_to_aa_type, dtype=np.int64) ##
        }

        #，400
        if 'pdbbind2020' in self.protein_file:
            file_path = os.path.join(os.path.dirname(self.protein_file), 'interaction_' + os.path.basename(self.protein_file).split('_')[0] + '_v2.pkl')   #5S8I_protein.pdb
        else:
            file_path = os.path.join(os.path.dirname(self.protein_file), 'interaction_' + os.path.basename(self.protein_file).split('_')[0] + '_v2.pkl')   #5S8I_protein.pdb
        with open(file_path, 'rb') as file:
            interaction_data = dill.load(file)
        holo_coords_list = interaction_data['holo_coords_list']
        coords_predict_list = interaction_data['coords_predict_list']
        pocket_coords_list = interaction_data['pocket_coords_list']
        cross_distance_list = interaction_data['cross_distance_list']

        assert np.allclose(holo_coords_list[0], holo_coords_list[-1], atol=0.02)

        #print('pocket_coords_list[0]:', pocket_coords_list[0][:])
        #np.set_printoptions(suppress=True, precision=4)
        #print('pocket_coords_list[-1]:', pocket_coords_list[-1]) #，？？，
        #print('pocket_coords_list[0].shape:', pocket_coords_list[0].shape)
        #print('pocket_coords_list[-1].shape:', pocket_coords_list[-1].shape) #，？？
        #assert np.allclose(pocket_coords_list[0], pocket_coords_list[-1], atol=0.02)

        assert len(holo_coords_list) == len(coords_predict_list) and len(holo_coords_list) == len(pocket_coords_list) and len(holo_coords_list) == len(cross_distance_list)

        
        #rmsd，，
        rmsd_list = []
        for pre_pos, holo_pos in zip(coords_predict_list, holo_coords_list):
            assert pre_pos.shape == holo_pos.shape   #"Coordinate matrices must have the same shape"
            rmsd = np.sqrt(np.mean(np.sum((pre_pos - holo_pos) ** 2, axis=1)))
            rmsd_list.append(rmsd)
        
        sorted_indices = np.argsort(rmsd_list)
        best_index = sorted_indices[0]
        
        #
        #best_index  = random.choice(list(range(len(holo_coords_list))))


        if self.cross_distance_num == None or self.cross_distance_num == 'best':
            holo_coords = holo_coords_list[best_index]
            coords_predict = coords_predict_list[best_index]
            pocket_coords = pocket_coords_list[best_index]
            cross_distance = cross_distance_list[best_index]
        else:            
            holo_coords = holo_coords_list[self.cross_distance_num]
            coords_predict = coords_predict_list[self.cross_distance_num]
            pocket_coords = pocket_coords_list[self.cross_distance_num]
            cross_distance = cross_distance_list[self.cross_distance_num]



        #unimol，，
        unique_index_dict       = {}
        for j, ps in enumerate(pocket_coords):
            unique_index_dict[tuple(ps)] = j
            #，
        
        unique_index_list   = list(unique_index_dict.values())
        pocket_coords       = pocket_coords[unique_index_list]
        cross_distance      = cross_distance[:, unique_index_list]



        #，unimolrdkit，，
        #cross_distanceO,N,
        #，
        pro_isring_flag = {}
        pro_isO_flag = {}
        pro_isN_flag = {}

        '''
        pro_mol=Chem.MolFromPDBFile(self.protein_file,removeHs=False,sanitize=False) #，, 
        #print('self.protein_file:', self.protein_file)
        #print('ok1')
        atom_isring=np.array([atom.IsInRing() for atom in pro_mol.GetAtoms()])    #
        atom_isO=np.array([atom.GetSymbol() == 'O'  for atom in pro_mol.GetAtoms()])    #O
        atom_isN=np.array([atom.GetSymbol() == 'N'  for atom in pro_mol.GetAtoms()])    #N
        #coords=np.array(pro_mol.GetConformer(0).GetPositions(), dtype=np.float32) #,float32，float64
        coords=np.array(pro_mol.GetConformer(0).GetPositions())
        '''
        atom_isO, atom_isN, atom_isring, atom_isspecial, atom_id, coords = self._read_pdb_ONRing_biopandas(self.protein_file)
        count = 0
        coords_atom_dict = defaultdict(list)
        for r, o, n, c in zip(atom_isring, atom_isO, atom_isN, coords):
            count += 1
            k = torch.FloatTensor(c)
            tg = ''
            for ii in k:
                #tg += str(round(i.item(), 4)) + '_'
                tg += str(self.truncate(ii.item(), 3)) + '_' #，2，
            c_sum = str(tg)

            coords_atom_dict[c_sum].append(c)

            pro_isring_flag[c_sum] = r
            pro_isO_flag[c_sum]    = o
            pro_isN_flag[c_sum]    = n
        
        assert len(pro_isring_flag) == len(atom_isring)
        
        print('：', count)
        if len(pro_isring_flag) != len(coords) or len(pro_isO_flag) != len(coords) or len(pro_isN_flag) != len(coords):
            print(f'{len(pro_isring_flag)} != {len(coords)} or {len(pro_isO_flag)} != {len(coords)} or {len(pro_isN_flag)} != {len(coords)}')
            #979 != 990 or 979 != 990 or 979 != 990 #？
            raise Exception("pro atom num is error")


        cross_pro_isring_flag = copy.deepcopy(atom_isring)
        cross_pro_isO_flag = copy.deepcopy(atom_isO)
        cross_pro_isN_flag = copy.deepcopy(atom_isN)



        #，,
        #，unimolrdkit，unimolrdkit，sdf，rdkit mol，rdkit
        #，unimol，rdkit
        lig_isring_flag = {}
        lig_isO_flag = {}
        lig_isN_flag = {}


        #，rdkit3d，，
        if self.data_flag == 'new_test':
            
            lig_mol = copy.deepcopy(self.ligand_dict['mol']) #
            #，？cross，，
            lig_rdkit_mol = copy.deepcopy(self.ligand_dict['rd_mol'])

            unimol_pos  = torch.FloatTensor(holo_coords)
            rdkit_pos   = torch.FloatTensor(lig_rdkit_mol.GetConformer(0).GetPositions())
            ground_pos  = torch.FloatTensor(lig_mol.GetConformer(0).GetPositions())

            #
            assert len(ground_pos) == len(unimol_pos) #，rkdit，，，

            #，rdkit, rdkitsmiles3dground，3d，
            if self.compare_atom_order(lig_mol, lig_rdkit_mol) and torch.allclose(ground_pos, unimol_pos, atol=0.02):
                lig_mol     = copy.deepcopy(lig_rdkit_mol) #，org_liandlig_rdkit_mol, 
                holo_coords = np.array(lig_rdkit_mol.GetConformer(0).GetPositions()) 
            
            #lig_mol = copy.deepcopy(self.ligand_dict['mol']) 
        else:
            lig_mol = copy.deepcopy(self.ligand_dict['mol']) #





        #print('ok1')
        atom_isring=np.array([atom.IsInRing() for atom in lig_mol.GetAtoms()])    #
        atom_isO=np.array([atom.GetSymbol() == 'O'  for atom in lig_mol.GetAtoms()])    #O
        atom_isN=np.array([atom.GetSymbol() == 'N'  for atom in lig_mol.GetAtoms()])    #N
        #coords=np.array(lig_mol.GetConformer(0).GetPositions(), dtype=np.float32) #,float32，float64
        coords=np.array(lig_mol.GetConformer(0).GetPositions())

        for r, o, n, c in zip(atom_isring, atom_isO, atom_isN, coords):
            k = torch.FloatTensor(c)
            tg = ''
            for ii in k:
                #tg += str(round(i.item(), 4)) + '_'
                tg += str(self.truncate(ii.item(), 3)) + '_' #，2，
            c_sum = str(tg)

            lig_isring_flag[c_sum] = r
            lig_isO_flag[c_sum]    = o
            lig_isN_flag[c_sum]    = n
        
        assert len(lig_isring_flag) == len(atom_isring)
        assert len(coords) == len(holo_coords) #，rkdit，，，

        if len(lig_isring_flag) != len(coords) or len(lig_isO_flag) != len(coords) or len(lig_isN_flag) != len(coords) or len(coords) != len(holo_coords):
            print(f'{len(coords)} != {len(holo_coords)}') #19 != 18, rdkit，crosss_liand，
            raise Exception("lig atom num is error") #，

    
        #unimolrdkit, 0.02
        assert torch.allclose(torch.FloatTensor(coords), torch.FloatTensor(holo_coords), atol=0.02)

        holo_coords = copy.deepcopy(coords)

        #，rdkitsdf，，，，，，holo_coords = coords
        #cross_distance, cross_ligand, cross_protein, cross_ligand_atom_flag, cross_protein_atom_flag

        cross_lig_isring_flag = copy.deepcopy(atom_isring)
        cross_lig_isO_flag = copy.deepcopy(atom_isO)
        cross_lig_isN_flag = copy.deepcopy(atom_isN)

        
        if old_data['pos'].shape[0] <= 0:
            print("old_data['pos'].size(0):", old_data['pos'].shape[0])
            print("old_data['pos']:", old_data['pos'].shape)
            return old_data
        else:
            data = {}
            #self.ligand_centor
            dis   = np.linalg.norm(old_data['pos'] - self.ligand_centor, axis = 1) #

        #try:
            #，12ai，400，400

            if pocket_coords.shape[0] < 300:
                cutoff_num = 2000
            else:
                cutoff_num = 400

            #，，rdkit
            #print('self.protein_file:', self.protein_file)
            '''
            pro_mol=Chem.MolFromPDBFile(self.protein_file,removeHs=False,sanitize=False)
            atom_isspecial = np.array([atom.IsInRing() or atom.GetSymbol() == 'O' or atom.GetSymbol() == 'N' for atom in pro_mol.GetAtoms()]) #
            atom_id = np.array([atom.GetIdx() for atom in pro_mol.GetAtoms()])
            #coords=np.array(pro_mol.GetConformer(0).GetPositions(), dtype=np.float32) #,float32，float64
            all_coords=np.array(pro_mol.GetConformer(0).GetPositions())
            '''
            atom_isO, atom_isN, atom_isring, atom_isspecial, atom_id, all_coords = self._read_pdb_ONRing_biopandas(self.protein_file)
            cutoff_indices = np.argsort(dis)[:] #,cutoff_num，，
            coords = all_coords[cutoff_indices] #，

            assert old_data['pos'].shape == coords.shape

            assert cutoff_indices.shape == np.unique(cutoff_indices).shape

            #，pocket_coords，400
            new_indices = []
            pro_flag_dict = {} #value
            pro_flag_dict2 = {}

            for j, c in zip(cutoff_indices, coords): #j，，
                k = torch.FloatTensor(c)
                tg = ''
                for ii in k:
                    #tg += str(round(i.item(), 4)) + '_'
                    tg += str(self.truncate(ii.item(), 3)) + '_' #，2，
                k = str(tg)

                pro_flag_dict[k] = j

                k2 = torch.FloatTensor(c)
                tg2 = ''
                for ii in k2:
                    #tg += str(round(i.item(), 4)) + '_'
                    tg2 += str(self.truncate2(ii.item(), 3)) + '_' #，2，
                k2 = str(tg2)

                pro_flag_dict2[k2] = j

            

            if len(pro_flag_dict) != len(coords):
                raise Exception(f"{len(pro_flag_dict)} != {len(coords)}")


            if len(pro_flag_dict2) != len(coords):
                raise Exception(f"{len(pro_flag_dict2)} != {len(coords)}")
            
            #cutoff_protein_ids = torch.ones(pocket_coords.shape[0], dtype = torch.bool)
            cutoff_protein_ids = torch.zeros(pocket_coords.shape[0], dtype = torch.bool)

            assert pocket_coords.shape == np.unique(pocket_coords, axis = 0).shape

            for ids, c in enumerate(pocket_coords): #pocket_coords
                k = torch.FloatTensor(c)
                tg = ''
                for ii in k:
                    #tg += str(round(i.item(), 4)) + '_'
                    tg += str(self.truncate(ii.item(), 3)) + '_' #，2，
                k = str(tg)

                #pocket_coords，
                #if cross_pro_flag_dict.get(c_sum):

                
                try:
                    new_indices.append(pro_flag_dict[k]) #，unimol，，, 
                    cutoff_protein_ids[ids] = True
                except Exception as e:
                    print(e)
                    #cutoff_protein_ids[ids] = False
                    #print('key error, skip:', k)
                    #print('key all:', list(pro_flag_dict.keys()))
                    #print('-----------------------------------------')
                    continue
                    
            #，，
            pocket_coords  = pocket_coords[cutoff_protein_ids]
            cross_distance = cross_distance[:, cutoff_protein_ids]

            cross_pro_isring_flag = cross_pro_isring_flag[new_indices]
            cross_pro_isO_flag    = cross_pro_isO_flag[new_indices]
            cross_pro_isN_flag    = cross_pro_isN_flag[new_indices]

            #print('cutoff_protein_ids num:', torch.sum(cutoff_protein_ids)) # tensor(252)
            #print('len(new_indices):', np.sum(np.array(new_indices))) #len(new_indices): 251

            #print('new_indices1:', sorted(np.nonzero(np.array(new_indices))[0]))
            #print('new_indices2:', sorted(np.array(new_indices2))) #id，
            
            if len(cross_pro_isring_flag) != len(pocket_coords):
                raise Exception(f'{len(cross_pro_isring_flag)} != {len(pocket_coords)}') #Exception: 251 != 252
            
            #if len(new_indices) != len(self.unimol_pcoords[0]):
                #raise Exception(f'{len(new_indices)} <= 0')

            #print('len(new_indices):', len(new_indices)) #
            #print('new_indices:', new_indices)

            new_atom_isspecial = atom_isspecial[new_indices] #atom_isspecial
            new_atom_id = atom_id[new_indices] #atom_id
            nonzero_indices = new_indices

            #，
            atom_isspecial_index = np.nonzero(new_atom_isspecial == True)[0]
            atom_isgeneral_index = np.nonzero(new_atom_isspecial == False)[0]
            

            #rdkit，，
            #new_atom_id
            sub_protein_file = os.path.join(os.path.dirname(self.protein_file), os.path.splitext(os.path.basename(self.protein_file))[0] + '_400.pdb')

            def keep_atoms_by_id(mol, atom_ids):
                """ID"""
                editable_mol = Chem.EditableMol(mol)
                all_atoms = list(editable_mol.GetMol().GetAtoms())
                atoms_to_keep = {atom.GetIdx() for atom in all_atoms if atom.GetIdx() in atom_ids}
                
                atoms_to_remove = [atom.GetIdx() for atom in all_atoms if atom.GetIdx() not in atom_ids]
                atoms_to_remove.sort(reverse=True)  # ，

                for atom_id in atoms_to_remove:
                    editable_mol.RemoveAtom(atom_id)

                return editable_mol.GetMol()
            

            # ID（0）
            atom_ids_to_keep = new_atom_id  # ID，
            pro_mol=Chem.MolFromPDBFile(self.protein_file,removeHs=False,sanitize=False)
            # 
            new_pro_mol = keep_atoms_by_id(pro_mol, atom_ids_to_keep)
            Chem.MolToPDBFile(new_pro_mol, sub_protein_file)



            #，
            #400pdb，,# 5l8c/5l8c_pocket10.pdb，5l8c/5l8c_pocket10_400.pdb
            #nonzero_indices
            #: '../CrossDocked2020/data/pdbbind2020_r10/v2020-other-PL/5qb1/5qb1_pocket10_400.pdb'

            '''
            sub_protein_file = os.path.join(os.path.dirname(self.protein_file), os.path.splitext(os.path.basename(self.protein_file))[0] + '_400.pdb')
            sub_protein_w = open(sub_protein_file, 'w')

            org_pro_list = []
            count = 0
            with open(self.protein_file, 'r')as f:
                for i, line in enumerate(f):
                    if line[0:6].strip() == 'ATOM' and i in nonzero_indices:
                        sub_protein_w.write(line)
                

                sub_protein_w.write('END')
            sub_protein_w.close()

            '''
        
            #print('ok3')

            #O,N, ，rdkitpdb，，rdkit，
            #。，，rdkit，
            #，，，rdkit
            '''
            pro_mol=Chem.MolFromPDBFile(self.protein_file,removeHs=False,sanitize=False)
            #print('ok1')
            atom_isring=np.array([atom.IsInRing() for atom in pro_mol.GetAtoms()])    #
            atom_isO=np.array([atom.GetSymbol() == 'O'  for atom in pro_mol.GetAtoms()])    #O
            atom_isN=np.array([atom.GetSymbol() == 'N'  for atom in pro_mol.GetAtoms()])    #N
            coords=np.array(pro_mol.GetConformer(0).GetPositions(), dtype=np.float32) #,float32，float64
            #coords=np.array(pro_mol.GetConformer(0).GetPositions())
            '''
            atom_isO, atom_isN, atom_isring, atom_isspecial, atom_id, coords = self._read_pdb_ONRing_biopandas(self.protein_file)
            #，，-，
            #indexs = []  #, 
            target_xyz =  np.array(self.pos) #xyz
            #target_xyz_sum =  np.round(np.array(self.pos, dtype=np.float32).sum(axis = -1), 2)

            #assert np.array_equal(target_xyz, coords) #，, postbus, assert，try, except

            if not np.array_equal(target_xyz, coords):
                print('target_xyz.shape:', target_xyz.shape)
                print('coords.shape:', coords.shape)
                raise Exception('np.array_equal(target_xyz, coords):', np.array_equal(target_xyz, coords))

            new_atom_isring = atom_isring
            new_atom_isO = atom_isO
            new_atom_isN = atom_isN

            assert len(new_atom_isN[nonzero_indices]) == len(np.array(self.element, dtype=np.int64)[nonzero_indices])
            #print('ok4')



            #

            l2p = [] #
            p2l = [] #
            l2p_type = [] #
            p2l_type = [] #

            #
            ligand_atom_isring = copy.deepcopy(cross_lig_isring_flag)
            ligand_atom_isO    = copy.deepcopy(cross_lig_isO_flag)
            ligand_atom_isN    = copy.deepcopy(cross_lig_isN_flag)

            protein_atom_isring = copy.deepcopy(cross_pro_isring_flag)
            protein_atom_isO    = copy.deepcopy(cross_pro_isO_flag)
            protein_atom_isN    = copy.deepcopy(cross_pro_isN_flag)


            ligand_cross_isring_flag = copy.deepcopy(cross_lig_isring_flag)
            ligand_cross_isO_flag    = copy.deepcopy(cross_lig_isO_flag)
            ligand_cross_isN_flag    = copy.deepcopy(cross_lig_isN_flag)
            ligand_cross_lp_pos      = copy.deepcopy(holo_coords)

            protein_cross_isring_flag = copy.deepcopy(cross_pro_isring_flag)
            protein_cross_isO_flag    = copy.deepcopy(cross_pro_isO_flag) 
            protein_cross_isN_flag    = copy.deepcopy(cross_pro_isN_flag) 
            protein_cross_lp_pos      = copy.deepcopy(pocket_coords)

            cross_distance_matrix = copy.deepcopy(cross_distance) #cross_distancelist，cross_distance



            #[2, n*m,]
            #
            centor = holo_coords.mean(axis = 0)
            l_combinations_isring = self.combinations(torch.from_numpy(coords_predict), torch.from_numpy(pocket_coords), torch.from_numpy(np.array(list(range(len(ligand_atom_isring))))), 
                    torch.from_numpy(np.array(list(range(len(protein_atom_isring))))), torch.from_numpy(ligand_atom_isring), torch.from_numpy(protein_atom_isring), 
                    )
            
            l_combinations_isO    = self.combinations(torch.from_numpy(coords_predict), torch.from_numpy(pocket_coords), torch.from_numpy(np.array(list(range(len(ligand_atom_isring))))), 
                    torch.from_numpy(np.array(list(range(len(protein_atom_isring))))), torch.from_numpy(ligand_atom_isO), torch.from_numpy(protein_atom_isN), 
                    )
            
            l_combinations_isN    = self.combinations(torch.from_numpy(coords_predict), torch.from_numpy(pocket_coords), torch.from_numpy(np.array(list(range(len(ligand_atom_isring))))), 
                    torch.from_numpy(np.array(list(range(len(protein_atom_isring))))), torch.from_numpy(ligand_atom_isN), torch.from_numpy(protein_atom_isO), 
                    )

            if l_combinations_isring.shape[0] != 0:
                l2p.append(l_combinations_isring.numpy()) #2 * N
                l2p_type.append(np.full(l_combinations_isring.shape[1], 5))  #，，2*N,N*2？cat
            if l_combinations_isO.shape[0] != 0:
                l2p.append(l_combinations_isO.numpy())
                l2p_type.append(np.full(l_combinations_isO.shape[1], 6))
            if l_combinations_isN.shape[0] != 0:
                l2p.append(l_combinations_isN.numpy())
                l2p_type.append(np.full(l_combinations_isN.shape[1], 7))

            #print('l_combinations_isring:', l_combinations_isring.shape)
            #print('l_combinations_isO:', l_combinations_isO.shape)
            #print('l_combinations_isN:', l_combinations_isN.shape)

            #
            p_combinations_isring = self.combinations(torch.from_numpy(pocket_coords), torch.from_numpy(coords_predict), torch.from_numpy(np.array(list(range(len(protein_atom_isring))))), 
                    torch.from_numpy(np.array(list(range(len(ligand_atom_isring))))), torch.from_numpy(protein_atom_isring), torch.from_numpy(ligand_atom_isring), 
                    )
            
            p_combinations_isO    = self.combinations(torch.from_numpy(pocket_coords), torch.from_numpy(coords_predict), torch.from_numpy(np.array(list(range(len(protein_atom_isring))))), 
                    torch.from_numpy(np.array(list(range(len(ligand_atom_isring))))), torch.from_numpy(protein_atom_isN), torch.from_numpy(ligand_atom_isO), 
                    )
            
            p_combinations_isN    = self.combinations(torch.from_numpy(pocket_coords), torch.from_numpy(coords_predict), torch.from_numpy(np.array(list(range(len(protein_atom_isring))))), 
                    torch.from_numpy(np.array(list(range(len(ligand_atom_isring))))), torch.from_numpy(protein_atom_isO), torch.from_numpy(ligand_atom_isN), 
                    )
            
            if p_combinations_isring.shape[0] != 0:
                p2l.append(p_combinations_isring)
                p2l_type.append(np.full(p_combinations_isring.shape[1], 8))  #，，2*N,N*2？cat
            if p_combinations_isO.shape[0] != 0:
                p2l.append(p_combinations_isO)
                p2l_type.append(np.full(p_combinations_isO.shape[1], 9))
            if p_combinations_isN.shape[0] != 0:
                p2l.append(p_combinations_isN)
                p2l_type.append(np.full(p_combinations_isN.shape[1], 10))

            #print('p_combinations_isring:', p_combinations_isring.shape)
            #print('p_combinations_isO:', p_combinations_isO.shape)
            #print('p_combinations_isN:', p_combinations_isN.shape)

            #ON,NO,，，，，knn
            if len(l2p) != 0:
                l2p_edge_index = np.concatenate(l2p, axis = -1, dtype = int)
                l2p_type = np.concatenate(l2p_type, axis = -1, dtype = int)
            else:
                print('')
                exit()
                l2p_edge_index = np.empty((0, 0))
                l2p_type = np.empty((0, 0))
            
            if len(p2l) != 0:
                p2l_edge_index = np.concatenate(p2l, axis = -1, dtype = int)
                p2l_type = np.concatenate(p2l_type, axis = -1, dtype = int)
            else:
                print('')
                exit()
                p2l_edge_index = np.empty((0, 0))
                p2l_type = np.empty((0, 0))


            data = {
                'element': np.array(self.element, dtype=np.int64)[nonzero_indices], #
                'molecule_name': self.title, # pocket
                'pos': np.array(self.pos, dtype=np.float32)[nonzero_indices], #
                'is_backbone': np.array(self.is_backbone, dtype=np.bool_)[nonzero_indices], #Boolean，
                'atom_name': [self.atom_name[i] for i in nonzero_indices], #
                'atom_to_aa_type': np.array(self.atom_to_aa_type, dtype=np.int64)[nonzero_indices], ##
                'atom_isring': new_atom_isring[nonzero_indices],
                'atom_isO': new_atom_isO[nonzero_indices],
                'atom_isN': new_atom_isN[nonzero_indices],

                'cross_lig_isring_flag': cross_lig_isring_flag,
                'cross_lig_isO_flag': cross_lig_isO_flag,
                'cross_lig_isN_flag': cross_lig_isN_flag,

                'cross_pro_isring_flag': cross_pro_isring_flag,
                'cross_pro_isO_flag': cross_pro_isO_flag,
                'cross_pro_isN_flag': cross_pro_isN_flag,

                'cross_ligand': holo_coords,
                'cross_protein': pocket_coords,

                #'cross_distance': set(torch.from_numpy(cross_distance)), #，，？，
                #'cross_distance': list(torch.from_numpy(cross_distance)), #pyglist，
                #'cross_distance': tuple(torch.from_numpy(cross_distance)), #tuple，pyg
                #'cross_distance': OrderedSet(torch.from_numpy(cross_distance)), #，pyg，numpy

                'cross_distance': cross_distance,

                #'cross_bond_index': np.ones_like(l2p_edge_index),
                #'cross_distance2': np.ones_like(l2p_edge_index), #？？，，
                #Batch.from_data_list([data.clone() for _ in range(n_data)], follow_batch=FOLLOW_BATCH, exclude_keys = collate_exclude_keys).to(device)，
                #bond？pyg？?

                #'cross_distance1': l2p_edge_index,
                #'cross_distance2': l2p_type,
                #'cross_distance3': p2l_edge_index,
                #'cross_distance4': p2l_type,
                

                'link_e': l2p_edge_index.T, # N * 2
                'link_t': l2p_type,
                'link_e_reverse': p2l_edge_index.T, # N * 2
                'link_t_reverse': p2l_type,

                #coords_predict, unimol，ground true，
                'coords_predict':coords_predict,


                #def torchify_dict(data):
            }

            #print('ok5')
            return data
        
        
        #except Exception as e:
            #print('protein error:', e)
            #print('self.protein_file:', self.protein_file)
            #raise Exception('protein error, stop')
            #exit()
            #return None







    def to_dict_atom_unimol(self):
        #unimol
        #，，6ai，

        old_data = {
            'element': np.array(self.element, dtype=np.int64), #
            'molecule_name': self.title, # pocket
            'pos': np.array(self.pos, dtype=np.float32), #
            'is_backbone': np.array(self.is_backbone, dtype=np.bool_), #Boolean，
            'atom_name': self.atom_name, #
            'atom_to_aa_type': np.array(self.atom_to_aa_type, dtype=np.int64) ##
        }


        
        if old_data['pos'].shape[0] <= 0:
            print("old_data['pos'].size(0):", old_data['pos'].shape[0])
            print("old_data['pos']:", old_data['pos'].shape)
            return old_data
        else:
            data = {}
            #self.ligand_centor
            dis   = np.linalg.norm(old_data['pos'] - self.ligand_centor, axis = 1) #

            try:
                #，12ai，400，400
                cutoff_num = 400
                #，，rdkit
                #print('self.protein_file:', self.protein_file)
                '''
                pro_mol=Chem.MolFromPDBFile(self.protein_file,removeHs=False,sanitize=False)
                atom_isspecial = np.array([atom.IsInRing() or atom.GetSymbol() == 'O' or atom.GetSymbol() == 'N' for atom in pro_mol.GetAtoms()]) #
                atom_id = np.array([atom.GetIdx() for atom in pro_mol.GetAtoms()])
                #coords=np.array(pro_mol.GetConformer(0).GetPositions(), dtype=np.float32) #,float32，float64
                coords=np.array(pro_mol.GetConformer(0).GetPositions())
                '''
                atom_isO, atom_isN, atom_isring, atom_isspecial, atom_id, coords = self._read_pdb_ONRing_biopandas(self.protein_file)
                #new_indices = np.argsort(dis)[:cutoff_num] #,cutoff_num，，
                #，pocket_coords，400
                new_indices = []
                cross_pro_flag_dict = {}

                for j, c in enumerate(coords):
                    k = torch.FloatTensor(c)
                    #print('k1:', k)
                    tg = ''
                    for ii in k:
                        #tg += str(round(i.item(), 4)) + '_'
                        tg += str(self.truncate(ii.item(), 3)) + '_' #，2，
                    k = str(tg)
                    
                    cross_pro_flag_dict[k] = j
                
                if len(cross_pro_flag_dict) != len(coords):
                    raise Exception(f"{len(cross_pro_flag_dict)} != {len(coords)}")


                for c in self.unimol_pcoords[0]:
                    #print(type(self.unimol_pcoords[0]))
                    k = torch.FloatTensor(c)
                    #print('k2:', k)
                    tg = ''
                    for ii in k:
                        #tg += str(round(i.item(), 4)) + '_'
                        tg += str(self.truncate(ii.item(), 3)) + '_' #，2，
                    k = str(tg)

                    #pocket_coords，
                    #if cross_pro_flag_dict.get(c_sum):
                    new_indices.append(cross_pro_flag_dict[k])

                if len(new_indices) != len(self.unimol_pcoords[0]):
                    raise Exception(f'{len(new_indices)} <= 0')
                print('len(new_indices):', len(new_indices))
                #print('new_indices:', new_indices)

                new_atom_isspecial = atom_isspecial[new_indices] #atom_isspecial
                new_atom_id = atom_id[new_indices] #atom_id
                nonzero_indices = new_indices

                #，
                atom_isspecial_index = np.nonzero(new_atom_isspecial == True)[0]
                atom_isgeneral_index = np.nonzero(new_atom_isspecial == False)[0]
                

                #rdkit，，
                #new_atom_id
                sub_protein_file = os.path.join(os.path.dirname(self.protein_file), os.path.splitext(os.path.basename(self.protein_file))[0] + '_400.pdb')

                def keep_atoms_by_id(mol, atom_ids):
                    """ID"""
                    editable_mol = Chem.EditableMol(mol)
                    all_atoms = list(editable_mol.GetMol().GetAtoms())
                    atoms_to_keep = {atom.GetIdx() for atom in all_atoms if atom.GetIdx() in atom_ids}
                    
                    atoms_to_remove = [atom.GetIdx() for atom in all_atoms if atom.GetIdx() not in atom_ids]
                    atoms_to_remove.sort(reverse=True)  # ，

                    for atom_id in atoms_to_remove:
                        editable_mol.RemoveAtom(atom_id)

                    return editable_mol.GetMol()
                

                # ID（0）
                atom_ids_to_keep = new_atom_id  # ID，
                pro_mol=Chem.MolFromPDBFile(self.protein_file,removeHs=False,sanitize=False)
                # 
                new_pro_mol = keep_atoms_by_id(pro_mol, atom_ids_to_keep)
                Chem.MolToPDBFile(new_pro_mol, sub_protein_file)



                #，
                #400pdb，,# 5l8c/5l8c_pocket10.pdb，5l8c/5l8c_pocket10_400.pdb
                #nonzero_indices
                #: '../CrossDocked2020/data/pdbbind2020_r10/v2020-other-PL/5qb1/5qb1_pocket10_400.pdb'

                '''
                sub_protein_file = os.path.join(os.path.dirname(self.protein_file), os.path.splitext(os.path.basename(self.protein_file))[0] + '_400.pdb')
                sub_protein_w = open(sub_protein_file, 'w')

                org_pro_list = []
                count = 0
                with open(self.protein_file, 'r')as f:
                    for i, line in enumerate(f):
                        if line[0:6].strip() == 'ATOM' and i in nonzero_indices:
                            sub_protein_w.write(line)
                    

                    sub_protein_w.write('END')
                sub_protein_w.close()

                '''
            
                #print('ok3')

                #O,N, ，rdkitpdb，，rdkit，
                #。，，rdkit，
                #，，，rdkit
                '''
                pro_mol=Chem.MolFromPDBFile(self.protein_file,removeHs=False,sanitize=False)
                #print('ok1')
                atom_isring=np.array([atom.IsInRing() for atom in pro_mol.GetAtoms()])    #
                atom_isO=np.array([atom.GetSymbol() == 'O'  for atom in pro_mol.GetAtoms()])    #O
                atom_isN=np.array([atom.GetSymbol() == 'N'  for atom in pro_mol.GetAtoms()])    #N
                coords=np.array(pro_mol.GetConformer(0).GetPositions(), dtype=np.float32) #,float32，float64
                #coords=np.array(pro_mol.GetConformer(0).GetPositions())
                '''
                atom_isO, atom_isN, atom_isring, atom_isspecial, atom_id, coords = self._read_pdb_ONRing_biopandas(self.protein_file)
                #，，-，
                #indexs = []  #, 
                target_xyz =  np.array(self.pos) #xyz
                #target_xyz_sum =  np.round(np.array(self.pos, dtype=np.float32).sum(axis = -1), 2)

                #assert np.array_equal(target_xyz, coords) #，, postbus, assert，try, except

                if not np.array_equal(target_xyz, coords):
                    print('target_xyz.shape:', target_xyz.shape)
                    print('coords.shape:', coords.shape)
                    raise Exception('np.array_equal(target_xyz, coords):', np.array_equal(target_xyz, coords))

                new_atom_isring = atom_isring
                new_atom_isO = atom_isO
                new_atom_isN = atom_isN

                assert len(new_atom_isN[nonzero_indices]) == len(np.array(self.element, dtype=np.int64)[nonzero_indices])
                #print('ok4')

                data = {
                    'element': np.array(self.element, dtype=np.int64)[nonzero_indices], #
                    'molecule_name': self.title, # pocket
                    'pos': np.array(self.pos, dtype=np.float32)[nonzero_indices], #
                    'is_backbone': np.array(self.is_backbone, dtype=np.bool_)[nonzero_indices], #Boolean，
                    'atom_name': [self.atom_name[i] for i in nonzero_indices], #
                    'atom_to_aa_type': np.array(self.atom_to_aa_type, dtype=np.int64)[nonzero_indices], ##
                    'atom_isring': new_atom_isring[nonzero_indices],
                    'atom_isO': new_atom_isO[nonzero_indices],
                    'atom_isN': new_atom_isN[nonzero_indices],

                    #'cross_lig_isring_flag': cross_lig_isring_flag,
                    #'cross_lig_isO_flag': cross_lig_isO_flag,
                    #'cross_lig_isN_flag': cross_lig_isN_flag,

                    #'cross_pro_isring_flag': cross_pro_isring_flag,
                    #'cross_pro_isO_flag': cross_pro_isO_flag,
                    #'cross_pro_isN_flag': cross_pro_isN_flag,

                    #'cross_ligand': holo_coords,
                    #'cross_protein': pocket_coords,
                    #'cross_distance': cross_distance,
                }

                #print('ok5')
                return data
            
            
            except Exception as e:
                print('protein error:', e)
                print('self.protein_file:', self.protein_file)
                #raise Exception('protein error, stop')
                #exit()
                return None



    def to_dict_atom_old(self):
        #，，6ai，

        old_data = {
            'element': np.array(self.element, dtype=np.int64), #
            'molecule_name': self.title, # pocket
            'pos': np.array(self.pos, dtype=np.float32), #
            'is_backbone': np.array(self.is_backbone, dtype=np.bool_), #Boolean，
            'atom_name': self.atom_name, #
            'atom_to_aa_type': np.array(self.atom_to_aa_type, dtype=np.int64) ##
        }

        
        if old_data['pos'].shape[0] <= 0:
            print("old_data['pos'].size(0):", old_data['pos'].shape[0])
            print("old_data['pos']:", old_data['pos'].shape)
            return old_data
        else:
            data = {}
            #self.ligand_centor
            dis   = np.linalg.norm(old_data['pos'] - self.ligand_centor, axis = 1) #
            try:
                #，12ai，400，400
                cutoff_num = 400
                #，，rdkit
                #print('self.protein_file:', self.protein_file)
                '''
                pro_mol=Chem.MolFromPDBFile(self.protein_file,removeHs=False,sanitize=False)
                atom_isspecial = np.array([atom.IsInRing() or atom.GetSymbol() == 'O' or atom.GetSymbol() == 'N' for atom in pro_mol.GetAtoms()]) #
                atom_id = np.array([atom.GetIdx() for atom in pro_mol.GetAtoms()])
                '''
                atom_isO, atom_isN, atom_isring, atom_isspecial, atom_id, coords = self._read_pdb_ONRing_biopandas(self.protein_file)
                
                new_indices = np.argsort(dis)[:cutoff_num] #,cutoff_num，，
                new_atom_isspecial = atom_isspecial[new_indices] #atom_isspecial
                new_atom_id = atom_id[new_indices] #atom_id
                nonzero_indices = new_indices

                #，
                atom_isspecial_index = np.nonzero(new_atom_isspecial == True)[0]
                atom_isgeneral_index = np.nonzero(new_atom_isspecial == False)[0]
                

                #rdkit，，
                #new_atom_id
                sub_protein_file = os.path.join(os.path.dirname(self.protein_file), os.path.splitext(os.path.basename(self.protein_file))[0] + '_400.pdb')

                def keep_atoms_by_id(mol, atom_ids):
                    """ID"""
                    editable_mol = Chem.EditableMol(mol)
                    all_atoms = list(editable_mol.GetMol().GetAtoms())
                    atoms_to_keep = {atom.GetIdx() for atom in all_atoms if atom.GetIdx() in atom_ids}
                    
                    atoms_to_remove = [atom.GetIdx() for atom in all_atoms if atom.GetIdx() not in atom_ids]
                    atoms_to_remove.sort(reverse=True)  # ，

                    for atom_id in atoms_to_remove:
                        editable_mol.RemoveAtom(atom_id)

                    return editable_mol.GetMol()
                

                # ID（0）
                atom_ids_to_keep = new_atom_id  # ID，
                pro_mol=Chem.MolFromPDBFile(self.protein_file,removeHs=False,sanitize=False)
                # 
                new_pro_mol = keep_atoms_by_id(pro_mol, atom_ids_to_keep)
                Chem.MolToPDBFile(new_pro_mol, sub_protein_file)



                #，
                #400pdb，,# 5l8c/5l8c_pocket10.pdb，5l8c/5l8c_pocket10_400.pdb
                #nonzero_indices
                #: '../CrossDocked2020/data/pdbbind2020_r10/v2020-other-PL/5qb1/5qb1_pocket10_400.pdb'

                '''
                sub_protein_file = os.path.join(os.path.dirname(self.protein_file), os.path.splitext(os.path.basename(self.protein_file))[0] + '_400.pdb')
                sub_protein_w = open(sub_protein_file, 'w')

                org_pro_list = []
                count = 0
                with open(self.protein_file, 'r')as f:
                    for i, line in enumerate(f):
                        if line[0:6].strip() == 'ATOM' and i in nonzero_indices:
                            sub_protein_w.write(line)
                    

                    sub_protein_w.write('END')
                sub_protein_w.close()

                '''
            
                #print('ok3')

                #O,N, ，rdkitpdb，，rdkit，
                #。，，rdkit，
                #，，，rdkit
                '''
                pro_mol=Chem.MolFromPDBFile(self.protein_file,removeHs=False,sanitize=False)
                #print('ok1')
                atom_isring=np.array([atom.IsInRing() for atom in pro_mol.GetAtoms()])    #
                atom_isO=np.array([atom.GetSymbol() == 'O'  for atom in pro_mol.GetAtoms()])    #O
                atom_isN=np.array([atom.GetSymbol() == 'N'  for atom in pro_mol.GetAtoms()])    #N
                coords=np.array(pro_mol.GetConformer(0).GetPositions(), dtype=np.float32) #,float32，float64
                #，，-，
                #indexs = []  #, 
                '''
                atom_isO, atom_isN, atom_isring, atom_isspecial, atom_id, coords = self._read_pdb_ONRing_biopandas(self.protein_file)
                target_xyz =  np.array(self.pos) #xyz
                #target_xyz_sum =  np.round(np.array(self.pos, dtype=np.float32).sum(axis = -1), 2)

                #assert np.array_equal(target_xyz, coords) #，, postbus, assert，try, except

                if not np.array_equal(target_xyz, coords):
                    print('target_xyz.shape:', target_xyz.shape)
                    print('coords.shape:', coords.shape)
                    raise Exception('np.array_equal(target_xyz, coords):', np.array_equal(target_xyz, coords))

                new_atom_isring = atom_isring
                new_atom_isO = atom_isO
                new_atom_isN = atom_isN

                assert len(new_atom_isN[nonzero_indices]) == len(np.array(self.element, dtype=np.int64)[nonzero_indices])
                #print('ok4')

                data = {
                    'element': np.array(self.element, dtype=np.int64)[nonzero_indices], #
                    'molecule_name': self.title, # pocket
                    'pos': np.array(self.pos, dtype=np.float32)[nonzero_indices], #
                    'is_backbone': np.array(self.is_backbone, dtype=np.bool_)[nonzero_indices], #Boolean，
                    'atom_name': [self.atom_name[i] for i in nonzero_indices], #
                    'atom_to_aa_type': np.array(self.atom_to_aa_type, dtype=np.int64)[nonzero_indices], ##
                    'atom_isring': new_atom_isring[nonzero_indices],
                    'atom_isO': new_atom_isO[nonzero_indices],
                    'atom_isN': new_atom_isN[nonzero_indices],
                }

                #print('ok5')
                return data
            
            except Exception as e:
                print('protein error:', e)
                print('self.protein_file:', self.protein_file)
                #raise Exception('protein error, stop')
                #exit()
                return None
        

        

    def truncate2(self, arr, decimals):
        factor = 10.0 ** decimals
        #return np.floor(arr * factor) / factor
        #return int(arr * factor) #
        #return math.ceil(arr * factor) #
        return np.round(arr, decimals)
        

    def truncate(self, arr, decimals = 2):
        factor = 10.0 ** decimals
        #return np.floor(arr * factor) / factor
        #return int(arr * factor) #
        #return math.ceil(arr * factor) #
        return np.round(arr, decimals)



    def combinations(self, x1, x2, atom1, atom2, atom_index1, atom_index2):
            #，
            vec1 = atom1[atom_index1]
            vec2 = atom2[atom_index2]
            #  torch.meshgrid 
            grid_x, grid_y = torch.meshgrid(vec1, vec2, indexing='ij')
            # 
            combination = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=0) #shape = [2, n*m]
            copy_combination = combination.clone()
            
            #6ai, ，,
            dis_limit = GP.atom2atom_distance

            dis = torch.norm(x1[combination[0]] - x2[combination[1]], p = 2, dim = -1) #
            dis_index = dis <= dis_limit #

            combination = combination.t()[dis_index]

            '''
            if len(combination) == 0:
                #print('12')
                dis_limit = 12.0 #，
                combination = copy_combination.clone()
                dis = torch.norm(x[combination[0]] - x[combination[1]], p = 2, dim = -1) #
                dis_index = dis <= dis_limit #

                combination = combination.t()[dis_index]
            '''
            
            if len(combination) == 0:
                #print('')
                dis_limit = 100000000.0 #，
                combination = copy_combination.clone()
                dis = torch.norm(x1[combination[0]] - x2[combination[1]], p = 2, dim = -1) #
                dis_index = dis <= dis_limit #

                combination = combination.t()[dis_index]

            '''
            #6ai, ，。60，，，
            nun_limit = 60

            dis = torch.norm(combination.to(torch.float32), p = 2, dim = 1) #
            dis_index = dis <= dis_limit #

            combination = combination[dis_index].t()
            '''

            return combination.t()








    def combinations_optim(self, pos1, pos2, index1, index2, atom_index1, atom_index2, centor, 
            cross_atom_flag1, cross_atom_flag2, cross_distance, cross_ligand, cross_protein, 
            flag
            ):
            #cross_ligand_atom_flag，cross_protein_atom_flag，，O，N，, 1O, 2N, 3, 0
            #，。atom1, atom2id，atom_index1, atom_index2，bool
            #cross_ligand, cross_protein，x

            if GP.interaction_stype == 'interaction':
                #，
                x1 = pos1[atom_index1]
                x2 = pos2[atom_index2]
                vec1 = index1[atom_index1]
                vec2 = index2[atom_index2]
                if flag == 'ligand':
                    l_x = x1
                    p_x = x2
                    l_index = vec1
                    p_index = vec2
                    cross_ligand_atom_flag  = cross_atom_flag1
                    cross_protein_atom_flag = cross_atom_flag2
                elif flag == 'protein':
                    l_x = x2
                    p_x = x1
                    l_index = vec2
                    p_index = vec1
                    cross_ligand_atom_flag  = cross_atom_flag2
                    cross_protein_atom_flag = cross_atom_flag1

                #key，indexvalue
                l_x_index_dict = {}
                p_x_index_dict = {}
                
                l_x_index_dict2 = {}
                p_x_index_dict2 = {}

                for coord, index in zip(l_x, l_index):
                    #k = coord.sum()
                    #k = torch.round(k * 10000) / 10000 #3，torch.round，

                    tg = ''
                    for i in coord:
                        #tg += str(round(i.item(), 3)) + '_'
                        tg += str(self.truncate(i.item(), 3)) + '_' #，2，
                    k = str(tg)


                    v = index
                    l_x_index_dict[k] = v
                    
                    
                    tg = ''
                    for i in coord:
                        #tg += str(round(i.item(), 3)) + '_'
                        tg += str(self.truncate2(i.item(), 3)) + '_' #，2，
                    k = str(tg)


                    v = index
                    l_x_index_dict2[k] = v

                
                assert len(l_x_index_dict) == len(l_x)
                assert len(l_x_index_dict2) == len(l_x)

                for coord, index in zip(p_x, p_index):
                    #k = coord.sum()
                    #k = torch.round(k * 10000) / 10000 #3，torch.round，

                    tg = ''
                    for i in coord:
                        #tg += str(round(i.item(), 3)) + '_'
                        tg += str(self.truncate(i.item(), 3)) + '_' #，2，
                    k = str(tg)

                    v = index
                    p_x_index_dict[k] = v
                    
                    
                    tg = ''
                    for i in coord:
                        #tg += str(round(i.item(), 3)) + '_'
                        tg += str(self.truncate2(i.item(), 3)) + '_' #，2，
                    k = str(tg)

                    v = index
                    p_x_index_dict2[k] = v

                assert len(p_x_index_dict) == len(p_x)
                assert len(p_x_index_dict2) == len(p_x)

                #，，small_cross_distance, 
                #cross_ligand_atom_flag, cross_protein_atom_flagO，N，，，O~N，
                # cross_ligand_atom_flag = cross_ligand_atom_flag[cross_ligand_atom_flag == 1], cross_protein_atom_flag = cross_protein_atom_flag[cross_protein_atom_flag == 2]
                #print('cross_distance:', cross_distance.shape) #cross_distance: torch.Size([13, 125])
                #print('cross_ligand_atom_flag, cross_protein_atom_flag:', cross_ligand_atom_flag.shape, cross_protein_atom_flag.shape) #torch.Size([13]) torch.Size([125])
                small_cross_distance = cross_distance[cross_ligand_atom_flag][:,cross_protein_atom_flag]
                small_cross_ligand   = cross_ligand[cross_ligand_atom_flag]
                small_cross_protein  = cross_protein[cross_protein_atom_flag]


                #，small_cross_ligandsmall_cross_proteinx
                ligand_index  = []
                protein_index = []

                for coord in small_cross_ligand:
                    #k = coord.sum()
                    #k = torch.round(k * 10000) / 10000 #3，torch.round，

                    tg = ''
                    for i in coord:
                        #tg += str(round(i.item(), 3)) + '_'
                        tg += str(self.truncate(i.item(), 3)) + '_' #，2，
                    k = str(tg)

                    try:
                        v = l_x_index_dict[k] #，，，，，，
                    except KeyError as e:
                        try:
                            tg = ''
                            for i in coord:
                                #tg += str(round(i.item(), 3)) + '_'
                                tg += str(self.truncate2(i.item(), 3)) + '_' #，2，
                            k = str(tg)
                            v = l_x_index_dict2[k]
                        except KeyError as e:
                            print('error:', e)
                            print('l_x_index_dict.keys:', list(l_x_index_dict.keys()))
                            raise Exception('error')

                    ligand_index.append(v)

                for coord in small_cross_protein:
                    #k = coord.sum()
                    #k = torch.round(k * 10000) / 10000 #3，torch.round，

                    tg = ''
                    for i in coord:
                        #tg += str(round(i.item(), 3)) + '_'
                        tg += str(self.truncate(i.item(), 3)) + '_' #，2，
                    k = str(tg)
                    try:
                        v = p_x_index_dict[k] #，，，，，，
                    except KeyError as e:
                        try:
                            tg = ''
                            for i in coord:
                                #tg += str(round(i.item(), 3)) + '_'
                                tg += str(self.truncate2(i.item(), 3)) + '_' #，2，
                            k = str(tg)
                            v = p_x_index_dict2[k]
                        except KeyError as e:
                            print('error:', e)
                            print('p_x_index_dict.keys:', list(p_x_index_dict.keys()))
                            raise Exception('error')
                
                    protein_index.append(v)

                #print('ligand_index:', ligand_index)  #list
                #print('protein_index:', protein_index)#list

                #small_cross_distance_flag = (2.0 < small_cross_distance) & (small_cross_distance < GP.cross_distance_cutoff)  #，，8ai？shape = [n, m]
                small_cross_distance_flag = small_cross_distance < GP.cross_distance_cutoff
                #print('small_cross_distance_flag.shape[0]*[1]:', small_cross_distance_flag.shape[0] * small_cross_distance_flag.shape[1]) #torch.Size([9, 37]), 37*9 = 333, 
                #print('small_cross_distance_flag.sum():', small_cross_distance_flag.sum()) # tensor(327, device='cuda:0') 8ai，6ai，tensor(99, device='cuda:0')

                new_protein_index = []
                new_ligand_index  = []
                print('small_cross_distance_flag:', small_cross_distance_flag.shape) #(9, 37)
                for k in range(small_cross_distance_flag.shape[0]):
                    #print('protein_index:', protein_index)
                    if protein_index:
                        tg = torch.stack(protein_index, dim = 0)[small_cross_distance_flag[k]]
                        new_protein_index.append(tg) #tg
                        new_ligand_index.append(ligand_index[k].view(-1)) #，
                    else:
                        #print('protein_index is [] ?:', protein_index)
                        pass
                
                #print('new_ligand_index:', new_ligand_index)  #list
                #print('new_protein_index:', new_protein_index)#list

                assert len(new_ligand_index) == len(new_protein_index)

                #raise Exception('test')

                #，O,N，。
                #new_protein_index = protein_index
                #new_ligand_index  = ligand_index
                
                
                if flag == 'ligand':
                    combination_list = []
                    for l_i, p_i in zip(new_ligand_index, new_protein_index):
                        grid_x, grid_y = torch.meshgrid(l_i, p_i, indexing='ij') #grid_x, grid_y
                        # 
                        combination = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=0) #shape = [2, n*m]
                        combination_list.append(combination)

                elif flag == 'protein':
                    combination_list = []
                    for l_i, p_i in zip(new_ligand_index, new_protein_index):
                        grid_x, grid_y = torch.meshgrid(p_i, l_i, indexing='ij') #grid_x, grid_y
                        # 
                        combination = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=0) #shape = [2, n*m]
                        combination_list.append(combination)

                #print('combination_list:', combination_list)
                if combination_list:
                    combination = torch.cat(combination_list, dim = 1)
                else:
                    combination = torch.empty(0, 0) #，
                    #print('None') #None？
                    #combination = []


            elif GP.interaction_stype == 'interaction_all':
                #，，
                #4.5，4.5，o,n,，，，
                #vec1 = atom1[atom_index1]
                #vec2 = atom2[atom_index2]
                x1 = pos1[atom_index1]
                x2 = pos2[atom_index2]
                vec1 = index1[atom_index1]
                vec2 = index2[atom_index2]
                if flag == 'ligand':
                    l_x = x1
                    p_x = x2
                    l_index = vec1
                    p_index = vec2
                    cross_ligand_atom_flag  = cross_atom_flag1
                    cross_protein_atom_flag = cross_atom_flag2
                elif flag == 'protein':
                    l_x = x2
                    p_x = x1
                    l_index = vec2
                    p_index = vec1
                    cross_ligand_atom_flag  = cross_atom_flag2
                    cross_protein_atom_flag = cross_atom_flag1

                #key，indexvalue
                l_x_index_dict = {}
                p_x_index_dict = {}
                
                l_x_index_dict2 = {}
                p_x_index_dict2 = {}

                for coord, index in zip(l_x, l_index):
                    #k = coord.sum()
                    #k = torch.round(k * 10000) / 10000 #3，torch.round，

                    tg = ''
                    for i in coord:
                        #tg += str(round(i.item(), 3)) + '_'
                        tg += str(self.truncate(i.item(), 3)) + '_' #，2，
                    k = str(tg)


                    v = index
                    l_x_index_dict[k] = v
                    
                    
                    tg = ''
                    for i in coord:
                        #tg += str(round(i.item(), 3)) + '_'
                        tg += str(self.truncate2(i.item(), 3)) + '_' #，2，
                    k = str(tg)


                    v = index
                    l_x_index_dict2[k] = v

                
                assert len(l_x_index_dict) == len(l_x)
                assert len(l_x_index_dict2) == len(l_x)

                for coord, index in zip(p_x, p_index):
                    #k = coord.sum()
                    #k = torch.round(k * 10000) / 10000 #3，torch.round，

                    tg = ''
                    for i in coord:
                        #tg += str(round(i.item(), 3)) + '_'
                        tg += str(self.truncate(i.item(), 3)) + '_' #，2，
                    k = str(tg)

                    v = index
                    p_x_index_dict[k] = v
                    
                    
                    tg = ''
                    for i in coord:
                        #tg += str(round(i.item(), 3)) + '_'
                        tg += str(self.truncate2(i.item(), 3)) + '_' #，2，
                    k = str(tg)

                    v = index
                    p_x_index_dict2[k] = v

                assert len(p_x_index_dict) == len(p_x)
                assert len(p_x_index_dict2) == len(p_x)

                #，，small_cross_distance, 
                #cross_ligand_atom_flag, cross_protein_atom_flagO，N，，，O~N，
                # cross_ligand_atom_flag = cross_ligand_atom_flag[cross_ligand_atom_flag == 1], cross_protein_atom_flag = cross_protein_atom_flag[cross_protein_atom_flag == 2]
                #print('cross_distance:', cross_distance.shape) #cross_distance: torch.Size([13, 125])
                #print('cross_ligand_atom_flag, cross_protein_atom_flag:', cross_ligand_atom_flag.shape, cross_protein_atom_flag.shape) #torch.Size([13]) torch.Size([125])
                #small_cross_distance = cross_distance[cross_ligand_atom_flag][:,cross_protein_atom_flag]
                #small_cross_ligand   = cross_ligand[cross_ligand_atom_flag]
                #small_cross_protein  = cross_protein[cross_protein_atom_flag]

                #
                small_cross_distance = cross_distance
                small_cross_ligand   = cross_ligand
                small_cross_protein  = cross_protein


                #，small_cross_ligandsmall_cross_proteinx
                ligand_index  = []
                protein_index = []

                for coord in small_cross_ligand:
                    #k = coord.sum()
                    #k = torch.round(k * 10000) / 10000 #3，torch.round，

                    tg = ''
                    for i in coord:
                        #tg += str(round(i.item(), 3)) + '_'
                        tg += str(self.truncate(i.item(), 3)) + '_' #，2，
                    k = str(tg)

                    try:
                        v = l_x_index_dict[k] #，，，，，，
                    except KeyError as e:
                        try:
                            tg = ''
                            for i in coord:
                                #tg += str(round(i.item(), 3)) + '_'
                                tg += str(self.truncate2(i.item(), 3)) + '_' #，2，
                            k = str(tg)
                            v = l_x_index_dict2[k]
                        except KeyError as e:
                            print('error:', e)
                            print('l_x_index_dict.keys:', list(l_x_index_dict.keys()))
                            raise Exception('error')

                    ligand_index.append(v)

                for coord in small_cross_protein:
                    #k = coord.sum()
                    #k = torch.round(k * 10000) / 10000 #3，torch.round，

                    tg = ''
                    for i in coord:
                        #tg += str(round(i.item(), 3)) + '_'
                        tg += str(self.truncate(i.item(), 3)) + '_' #，2，
                    k = str(tg)
                    try:
                        v = p_x_index_dict[k] #，，，，，，
                    except KeyError as e:
                        try:
                            tg = ''
                            for i in coord:
                                #tg += str(round(i.item(), 3)) + '_'
                                tg += str(self.truncate2(i.item(), 3)) + '_' #，2，
                            k = str(tg)
                            v = p_x_index_dict2[k]
                        except KeyError as e:
                            print('error:', e)
                            print('p_x_index_dict.keys:', list(p_x_index_dict.keys()))
                            raise Exception('error')
                
                    protein_index.append(v)

                #print('ligand_index:', ligand_index)  #list
                #print('protein_index:', protein_index)#list

                #small_cross_distance_flag = (2.0 < small_cross_distance) & (small_cross_distance < GP.cross_distance_cutoff)  #，，8ai？shape = [n, m]
                small_cross_distance_flag = small_cross_distance < GP.cross_distance_cutoff
                #print('small_cross_distance_flag.shape[0]*[1]:', small_cross_distance_flag.shape[0] * small_cross_distance_flag.shape[1]) #torch.Size([9, 37]), 37*9 = 333, 
                #print('small_cross_distance_flag.sum():', small_cross_distance_flag.sum()) # tensor(327, device='cuda:0') 8ai，6ai，tensor(99, device='cuda:0')

                new_protein_index = []
                new_ligand_index  = []
                for k in range(small_cross_distance_flag.shape[0]):
                    #print('protein_index:', protein_index)
                    if protein_index:
                        tg = torch.stack(protein_index, dim = 0)[small_cross_distance_flag[k]]
                        new_protein_index.append(tg) #tg
                        new_ligand_index.append(ligand_index[k].view(-1)) #，
                    else:
                        #print('protein_index is [] ?:', protein_index)
                        pass
                
                #print('new_ligand_index:', new_ligand_index)  #list
                #print('new_protein_index:', new_protein_index)#list

                assert len(new_ligand_index) == len(new_protein_index)

                #raise Exception('test')

                #，O,N，。
                #new_protein_index = protein_index
                #new_ligand_index  = ligand_index
                
                
                if flag == 'ligand':
                    combination_list = []
                    for l_i, p_i in zip(new_ligand_index, new_protein_index):
                        grid_x, grid_y = torch.meshgrid(l_i, p_i, indexing='ij') #grid_x, grid_y
                        # 
                        combination = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=0) #shape = [2, n*m]
                        combination_list.append(combination)

                elif flag == 'protein':
                    combination_list = []
                    for l_i, p_i in zip(new_ligand_index, new_protein_index):
                        grid_x, grid_y = torch.meshgrid(p_i, l_i, indexing='ij') #grid_x, grid_y
                        # 
                        combination = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=0) #shape = [2, n*m]
                        combination_list.append(combination)

                #print('combination_list:', combination_list)
                if combination_list:
                    combination = torch.cat(combination_list, dim = 1)
                else:
                    combination = torch.empty(0, 0) #，
                    #print('None') #None？
                    #combination = []
            
            #，
            #if combination.shape[0] != 0:
                #combination = torch.unique(combination, dim = -1)
            return combination




    def combinations_optim_corrds_connection(self, pos1, pos2, index1, index2, atom_index1, atom_index2):
            #，
            vec1 = atom1[atom_index1]
            vec2 = atom2[atom_index2]
            #  torch.meshgrid 
            grid_x, grid_y = torch.meshgrid(vec1, vec2, indexing='ij')
            # 
            combination = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=0) #shape = [2, n*m]
            copy_combination = combination.clone()
            
            #6ai, ，,
            dis_limit = GP.atom2atom_distance

            dis = torch.norm(x[combination[0]] - x[combination[1]], p = 2, dim = -1) #
            dis_index = dis <= dis_limit #

            combination = combination.t()[dis_index]

            '''
            if len(combination) == 0:
                #print('12')
                dis_limit = 12.0 #，
                combination = copy_combination.clone()
                dis = torch.norm(x[combination[0]] - x[combination[1]], p = 2, dim = -1) #
                dis_index = dis <= dis_limit #

                combination = combination.t()[dis_index]
            '''
            
            if len(combination) == 0:
                #print('')
                dis_limit = 100000000.0 #，
                combination = copy_combination.clone()
                dis = torch.norm(x[combination[0]] - x[combination[1]], p = 2, dim = -1) #
                dis_index = dis <= dis_limit #

                combination = combination.t()[dis_index]

            '''
            #6ai, ，。60，，，
            nun_limit = 60

            dis = torch.norm(combination.to(torch.float32), p = 2, dim = 1) #
            dis_index = dis <= dis_limit #

            combination = combination[dis_index].t()
            '''

            return combination.t()




    def combinations_optim_split_3_5(self, pos1, pos2, index1, index2, atom_index1, atom_index2, centor, 
            cross_atom_flag1, cross_atom_flag2, cross_distance, cross_ligand, cross_protein, 
            flag
            ):
            
            #，3.5

            if GP.interaction_stype == 'interaction':
                #，
                x1 = pos1
                x2 = pos2
                vec1 = index1
                vec2 = index2
                if flag == 'ligand':
                    l_x = x1
                    p_x = x2
                    l_index = vec1
                    p_index = vec2
                    #cross_ligand_atom_flag  = cross_atom_flag1
                    #cross_protein_atom_flag = cross_atom_flag2
                elif flag == 'protein':
                    l_x = x2
                    p_x = x1
                    l_index = vec2
                    p_index = vec1
                    #cross_ligand_atom_flag  = cross_atom_flag2
                    #cross_protein_atom_flag = cross_atom_flag1

                #key，indexvalue
                l_x_index_dict = {}
                p_x_index_dict = {}
                
                l_x_index_dict2 = {}
                p_x_index_dict2 = {}

                for coord, index in zip(l_x, l_index):
                    #k = coord.sum()
                    #k = torch.round(k * 10000) / 10000 #3，torch.round，

                    tg = ''
                    for i in coord:
                        #tg += str(round(i.item(), 3)) + '_'
                        tg += str(self.truncate(i.item(), 3)) + '_' #，2，
                    k = str(tg)


                    v = index
                    l_x_index_dict[k] = v
                    
                    
                    tg = ''
                    for i in coord:
                        #tg += str(round(i.item(), 3)) + '_'
                        tg += str(self.truncate2(i.item(), 3)) + '_' #，2，
                    k = str(tg)


                    v = index
                    l_x_index_dict2[k] = v

                
                assert len(l_x_index_dict) == len(l_x)
                assert len(l_x_index_dict2) == len(l_x)

                for coord, index in zip(p_x, p_index):
                    #k = coord.sum()
                    #k = torch.round(k * 10000) / 10000 #3，torch.round，

                    tg = ''
                    for i in coord:
                        #tg += str(round(i.item(), 3)) + '_'
                        tg += str(self.truncate(i.item(), 3)) + '_' #，2，
                    k = str(tg)

                    v = index
                    p_x_index_dict[k] = v
                    
                    
                    tg = ''
                    for i in coord:
                        #tg += str(round(i.item(), 3)) + '_'
                        tg += str(self.truncate2(i.item(), 3)) + '_' #，2，
                    k = str(tg)

                    v = index
                    p_x_index_dict2[k] = v

                assert len(p_x_index_dict) == len(p_x)
                assert len(p_x_index_dict2) == len(p_x)

                #，，small_cross_distance, 
                #cross_ligand_atom_flag, cross_protein_atom_flagO，N，，，O~N，
                # cross_ligand_atom_flag = cross_ligand_atom_flag[cross_ligand_atom_flag == 1], cross_protein_atom_flag = cross_protein_atom_flag[cross_protein_atom_flag == 2]
                #print('cross_distance:', cross_distance.shape) #cross_distance: torch.Size([13, 125])
                #print('cross_ligand_atom_flag, cross_protein_atom_flag:', cross_ligand_atom_flag.shape, cross_protein_atom_flag.shape) #torch.Size([13]) torch.Size([125])
                #small_cross_distance = cross_distance[cross_ligand_atom_flag][:,cross_protein_atom_flag]
                #small_cross_ligand   = cross_ligand[cross_ligand_atom_flag]
                #small_cross_protein  = cross_protein[cross_protein_atom_flag]

                small_cross_distance = cross_distance
                small_cross_ligand   = cross_ligand
                small_cross_protein  = cross_protein


                #，small_cross_ligandsmall_cross_proteinx
                ligand_index  = []
                protein_index = []

                for coord in small_cross_ligand:
                    #k = coord.sum()
                    #k = torch.round(k * 10000) / 10000 #3，torch.round，

                    tg = ''
                    for i in coord:
                        #tg += str(round(i.item(), 3)) + '_'
                        tg += str(self.truncate(i.item(), 3)) + '_' #，2，
                    k = str(tg)

                    try:
                        v = l_x_index_dict[k] #，，，，，，
                    except KeyError as e:
                        try:
                            tg = ''
                            for i in coord:
                                #tg += str(round(i.item(), 3)) + '_'
                                tg += str(self.truncate2(i.item(), 3)) + '_' #，2，
                            k = str(tg)
                            v = l_x_index_dict2[k]
                        except KeyError as e:
                            print('error:', e)
                            print('l_x_index_dict.keys:', list(l_x_index_dict.keys()))
                            raise Exception('error')

                    ligand_index.append(v)

                for coord in small_cross_protein:
                    #k = coord.sum()
                    #k = torch.round(k * 10000) / 10000 #3，torch.round，

                    tg = ''
                    for i in coord:
                        #tg += str(round(i.item(), 3)) + '_'
                        tg += str(self.truncate(i.item(), 3)) + '_' #，2，
                    k = str(tg)
                    try:
                        v = p_x_index_dict[k] #，，，，，，
                    except KeyError as e:
                        try:
                            tg = ''
                            for i in coord:
                                #tg += str(round(i.item(), 3)) + '_'
                                tg += str(self.truncate2(i.item(), 3)) + '_' #，2，
                            k = str(tg)
                            v = p_x_index_dict2[k]
                        except KeyError as e:
                            print('error:', e)
                            print('p_x_index_dict.keys:', list(p_x_index_dict.keys()))
                            raise Exception('error')
                
                    protein_index.append(v)


                
                #0~3.5
                small_cross_distance_flag = (torch.tensor(0, dtype=torch.float32).cpu() < small_cross_distance) & (small_cross_distance < torch.tensor(3.5, dtype=torch.float32).cpu())

                new_protein_index = []
                new_ligand_index  = []

                for k in range(small_cross_distance_flag.shape[0]):
                    #print('protein_index:', protein_index)
                    if protein_index:
                        tg = torch.stack(protein_index, dim = 0)[small_cross_distance_flag[k]]
                        new_protein_index.append(tg) #tg
                        new_ligand_index.append(ligand_index[k].view(-1)) #，
                    else:
                        #print('protein_index is [] ?:', protein_index)
                        pass
        
                assert len(new_ligand_index) == len(new_protein_index)
                
                if flag == 'ligand':
                    combination_list = []
                    for l_i, p_i in zip(new_ligand_index, new_protein_index):
                        grid_x, grid_y = torch.meshgrid(l_i, p_i, indexing='ij') #grid_x, grid_y
                        # 
                        combination = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=0) #shape = [2, n*m]
                        combination_list.append(combination)

                elif flag == 'protein':
                    combination_list = []
                    for l_i, p_i in zip(new_ligand_index, new_protein_index):
                        grid_x, grid_y = torch.meshgrid(p_i, l_i, indexing='ij') #grid_x, grid_y
                        # 
                        combination = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=0) #shape = [2, n*m]
                        combination_list.append(combination)

                #print('combination_list:', combination_list)
                if combination_list:
                    combination_less3_5 = torch.cat(combination_list, dim = 1)
                else:
                    combination_less3_5 = torch.empty(0, 0) #，




                #3.4~4.5
                small_cross_distance_flag = (torch.tensor(3.5, dtype=torch.float32) <= small_cross_distance) & (small_cross_distance <= torch.tensor(4.5, dtype=torch.float32).cpu())

                new_protein_index = []
                new_ligand_index  = []

                for k in range(small_cross_distance_flag.shape[0]):
                    #print('protein_index:', protein_index)
                    if protein_index:
                        tg = torch.stack(protein_index, dim = 0)[small_cross_distance_flag[k]]
                        new_protein_index.append(tg) #tg
                        new_ligand_index.append(ligand_index[k].view(-1)) #，
                    else:
                        #print('protein_index is [] ?:', protein_index)
                        pass
        
                assert len(new_ligand_index) == len(new_protein_index)
                
                if flag == 'ligand':
                    combination_list = []
                    for l_i, p_i in zip(new_ligand_index, new_protein_index):
                        grid_x, grid_y = torch.meshgrid(l_i, p_i, indexing='ij') #grid_x, grid_y
                        # 
                        combination = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=0) #shape = [2, n*m]
                        combination_list.append(combination)

                elif flag == 'protein':
                    combination_list = []
                    for l_i, p_i in zip(new_ligand_index, new_protein_index):
                        grid_x, grid_y = torch.meshgrid(p_i, l_i, indexing='ij') #grid_x, grid_y
                        # 
                        combination = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=0) #shape = [2, n*m]
                        combination_list.append(combination)

                #print('combination_list:', combination_list)
                if combination_list:
                    combination_greater3_5 = torch.cat(combination_list, dim = 1)
                else:
                    combination_greater3_5 = torch.empty(0, 0) #，




            
            #，
            #if combination.shape[0] != 0:
                #combination = torch.unique(combination, dim = -1)
            #assert combination_less3_5.shape != combination_greater3_5.shape

            return combination_less3_5, combination_greater3_5





    def to_dict_residue(self):
        return {
            'amino_acid': np.array(self.amino_acid, dtype=np.int64),
            'center_of_mass': np.array(self.center_of_mass, dtype=np.float32),
            'pos_CA': np.array(self.pos_CA, dtype=np.float32),
            'pos_C': np.array(self.pos_C, dtype=np.float32),
            'pos_N': np.array(self.pos_N, dtype=np.float32),
            'pos_O': np.array(self.pos_O, dtype=np.float32),
        }

    def query_residues_radius(self, center, radius, criterion='center_of_mass'):
        center = np.array(center).reshape(3)
        selected = []
        for residue in self.residues:
            distance = np.linalg.norm(residue[criterion] - center, ord=2)
            #print(residue[criterion], distance)
            if distance < radius:
                selected.append(residue)
        return selected

    def query_residues_ligand(self, ligand, radius, criterion='center_of_mass'):
        selected = []
        sel_idx = set()
        # The time-complexity is O(mn).
        for center in ligand['pos']:
            for i, residue in enumerate(self.residues):
                distance = np.linalg.norm(residue[criterion] - center, ord=2)
                if distance < radius and i not in sel_idx:
                    selected.append(residue)
                    sel_idx.add(i)
        return selected

    def residues_to_pdb_block(self, residues, name='POCKET'):
        block = "HEADER    %s\n" % name
        block += "COMPND    %s\n" % name
        for residue in residues:
            for atom_idx in residue['atoms']:
                block += self.atoms[atom_idx]['line'] + "\n"
        block += "END\n"
        return block


def parse_pdbbind_index_file(path):
    pdb_id = []
    with open(path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        if line.startswith('#'): continue
        pdb_id.append(line.split()[0])
    return pdb_id




def single_conf_gen(tgt_mol, num_confs=1, seed=42, removeHs=True):
    mol = copy.deepcopy(tgt_mol)
    mol = Chem.AddHs(mol)
    allconformers = AllChem.EmbedMultipleConfs(
        mol, numConfs=num_confs, randomSeed=seed, clearConfs=True
    )
    sz = len(allconformers)
    for i in range(sz):
        try:
            AllChem.MMFFOptimizeMolecule(mol, confId=i)
        except:
            continue
    if removeHs:
        mol = Chem.RemoveHs(mol)
    return mol

def single_conf_gen_no_MMFF(tgt_mol, num_confs=1, seed=42, removeHs=True):
    mol = copy.deepcopy(tgt_mol)
    mol = Chem.AddHs(mol)
    allconformers = AllChem.EmbedMultipleConfs(
        mol, numConfs=num_confs, randomSeed=seed, clearConfs=True
    )
    if removeHs:
        mol = Chem.RemoveHs(mol)
    return mol





def parse_sdf_file(path, data_flag = 'new_test', protein_centor = None):
    #，
    '''
    #
    ATOM_FAMILIES = ['Acceptor', 'Donor', 'Aromatic', 'Hydrophobe', 'LumpedHydrophobe', 'NegIonizable', 'PosIonizable',
                'ZnBinder']
    ATOM_FAMILIES_ID = {s: i for i, s in enumerate(ATOM_FAMILIES)}
    BOND_TYPES = {
        BondType.UNSPECIFIED: 0,
        BondType.SINGLE: 1,
        BondType.DOUBLE: 2,
        BondType.TRIPLE: 3,
        BondType.AROMATIC: 4,
    }
    BOND_NAMES = {v: str(k) for k, v in BOND_TYPES.items()}
    HYBRIDIZATION_TYPE = ['S', 'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2'] #
    HYBRIDIZATION_TYPE_ID = {s: i for i, s in enumerate(HYBRIDIZATION_TYPE)}
    #

    '''
    fdefName = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
    factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
    # read mol
    name = path.split('/')[-2]
    base_name = os.path.dirname(path)
    #print('ok01?')

    try:
        rdmol  = Chem.SDMolSupplier(path)[0] 
        rdmol  = Chem.RemoveHs(rdmol)
    except Exception as e:
        try:
            file = os.path.join(f'{base_name}', f'{name}_ligand.sdf')
            #，，
            rdmol  = Chem.SDMolSupplier(file, sanitize=False)[0]
            rdmol  = Chem.RemoveHs(rdmol)

            #
            #mol = standardize(mol)
            #mol = Neutralize_atoms(mol)
        except Exception as e:
            try:
                rdmol = Chem.MolFromMol2File(os.path.join(f'{base_name}', f'{name}_ligand.mol2'), sanitize=False)
                rdmol = Chem.RemoveHs(rdmol)
                #mol = standardize(mol)
                #mol = Neutralize_atoms(mol)
            except Exception as e:
                print(f'Error occurred when parsing sdf file: {path}, {e}')
                raise SystemExit


    #print('ok02?')
    org_rdmol = copy.deepcopy(rdmol)
    #print('rdmol:', rdmol)


    # Remove Hydrogens.
    # rdmol = next(iter(Chem.SDMolSupplier(path, removeHs=True)))

    #，
    rd_num_atoms = rdmol.GetNumAtoms()
    feat_mat = np.zeros([rd_num_atoms, len(ATOM_FAMILIES)], dtype=np.int64)
    for feat in factory.GetFeaturesForMol(rdmol): #1，one-hot
        feat_mat[feat.GetAtomIds(), ATOM_FAMILIES_ID[feat.GetFamily()]] = 1

    # Get hybridization() in the order of atom idx.
    hybridization = []
    for atom in rdmol.GetAtoms():
        hybr = str(atom.GetHybridization()) #
        idx = atom.GetIdx() #ID
        hybridization.append((idx, hybr))
    hybridization = sorted(hybridization)
    hybridization = [v[1] for v in hybridization]

    ptable = Chem.GetPeriodicTable() #

    pos = np.array(rdmol.GetConformers()[0].GetPositions(), dtype=np.float32) #rdkit mol GetPositions，GetPosition
    element = []   #
    accum_pos = 0  #
    accum_mass = 0 #

    atom_isring=np.array([atom.IsInRing() for atom in rdmol.GetAtoms()])    #
    atom_isO=np.array([atom.GetSymbol() == 'O'  for atom in rdmol.GetAtoms()])    #O
    atom_isN=np.array([atom.GetSymbol() == 'N'  for atom in rdmol.GetAtoms()])    #N
    
    for atom_idx in range(rd_num_atoms): #
        atom = rdmol.GetAtomWithIdx(atom_idx) #id 
        atom_num = atom.GetAtomicNum() #，
        element.append(atom_num)
        atom_weight = ptable.GetAtomicWeight(atom_num) #
        accum_pos += pos[atom_idx] * atom_weight
        accum_mass += atom_weight
    center_of_mass = accum_pos / accum_mass #，
    element = np.array(element, dtype=int)

    #，
    # in edge_type, we have 1 for single bond, 2 for double bond, 3 for triple bond, and 4 for aromatic bond.
    row, col, edge_type = [], [], []
    for bond in rdmol.GetBonds():
        start = bond.GetBeginAtomIdx() #id
        end = bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]

        try:
            edge_type += 2 * [BOND_TYPES[bond.GetBondType()]] #
        except KeyError:
            raise SystemExit('ligand band type Key error')

    edge_index = np.array([row, col], dtype=int) #2 * 2E
    edge_type = np.array(edge_type, dtype=int)   #2E

    perm = (edge_index[0] * rd_num_atoms + edge_index[1]).argsort() #。，，
    edge_index = edge_index[:, perm]
    edge_type = edge_type[perm]

    #print('ok020?')
    try:
        #，，，，。，？
        #zamts
        #，rdmol。mol，，，EcDock，，
        gmol = Molgraph(rdkitmol = rdmol, smiles = Chem.MolToSmiles(rdmol)) #，zmats，，
        #print('gmol:', gmol)
        atoms,chiraltags,adjs,coords,zmats,masks,atom_order=gmol.Get_3D_Graph_Tensor(max_atoms=GP.max_atoms) #，，
        #print('coords:', coords)
        
    except Exception as e:
        print(e)
        return None
    


    #print('ok03?')
    #rdkit, new_test
    if data_flag == 'new_test':
        num_confs = 40
    else:
        num_confs = 40 #40
        
    #new_ligand_mol = None
    #rd_pos = None
    #3d，rdkit 3d mol，，rdkit，groudn。rdkit，
    try:
        #3d
        #ligand_mol = generate_3d_molecule(Chem.MolToSmiles(rdmol)) #smilesrdkit，，
        mol = copy.deepcopy(Chem.RemoveHs(rdmol))
        try:
            ligand_mol = single_conf_gen(mol, num_confs=num_confs, seed=42, removeHs=True)
        except Exception as e:
            ligand_mol = single_conf_gen_no_MMFF(mol, num_confs=num_confs, seed=42, removeHs=True) #，

    
        origin_centor = np.mean(np.array(rdmol.GetConformer(0).GetPositions()), axis=0)
        
        
        conf_coords = []
        for conf_id in range(ligand_mol.GetNumConformers()):
            lig_pos = np.array(ligand_mol.GetConformer(conf_id).GetPositions())
            self_centor = np.mean(lig_pos, axis=0)

            # 
            new_rd_pos = move_to_pocket_only_pos(lig_pos, self_centor, origin_centor)
            conf_coords.append(new_rd_pos)

        #  (num_confs, N_i, 3)
        conf_coords = np.stack(conf_coords, axis=0)
        
        
        
        m = ligand_mol.GetNumConformers()   # conformer 
        n = ligand_mol.GetNumAtoms()        # 
        assert conf_coords.shape == (m, n, 3), f"coords shape  ({m}, {n}, 3)， {coords.shape}"

        #  conformer 
        for conf_id, conformer in enumerate(ligand_mol.GetConformers()):
            for atom_idx in range(n):
                x, y, z = conf_coords[conf_id, atom_idx]
                conformer.SetAtomPosition(atom_idx, (float(x), float(y), float(z)))

        if m < num_confs:
            mn = num_confs - m
            mutiple_rd_pos = np.stack([conf_coords[-1]] * mn, axis=0)
            mutiple_rd_pos = np.concatenate((conf_coords, mutiple_rd_pos), axis=0)
        else:
            mutiple_rd_pos = conf_coords
            
        new_ligand_mol = ligand_mol

        #self_centor = np.mean(np.array(ligand_mol.GetConformer(0).GetPositions()), axis=0)

        #, 2，rdkit mol，，
        #new_ligand_mol = move_to_pocket(ligand_mol, self_centor, origin_centor)
        rd_pos = np.array(ligand_mol.GetConformer(0).GetPositions())

        #sdf
        out_file = os.path.join(os.path.dirname(path), os.path.splitext(os.path.basename(path))[0] + '-rdkit.sdf')
        save_molecule(ligand_mol, out_file)
    
    except Exception as e:
        print('rdkit error:', e)    
        new_ligand_mol = copy.deepcopy(rdmol)
        rd_pos         = copy.deepcopy(pos)
        mutiple_rd_pos = np.stack([rd_pos] * num_confs, axis=0)
    
    #print('ok04?')
    
    #align_pos, align_mol = pos, rdmol
    # ，rdkit, 
    #align_pos, align_mol = get_lig_graph_with_matching(copy.deepcopy(rdmol), popsize = 20, maxiter = 20, matching  = True, num_conformers = CFMGP.rdkit_align_num, remove_hs = True)
    #align_pos = align_pos.detach().cpu().numpy()
    #assert align_pos.shape == pos.shape
    align_pos, align_mol = None, None
    
    
    
    
    #，zmats，，
    data = {
        'smiles': Chem.MolToSmiles(rdmol),
        'element': element, #
        'pos': pos, #，rdkit mol 
        'mol': rdmol,  
        'rd_mol': new_ligand_mol, #
        'rd_pos': rd_pos, #
        'mutiple_rd_pos': mutiple_rd_pos, #list，
        'align_pos': align_pos,
        'align_mol': align_mol,
        'bond_index': edge_index,
        'bond_type': edge_type,
        'center_of_mass': center_of_mass,
        'atom_feature': feat_mat,
        'hybridization': hybridization, #
        #'fill_adjs': adjs,
        'fill_coords': coords,
        'fill_zmats': zmats,
        'fill_masks': masks,
        'fill_atom_order':atom_order,
        'atom_isring': atom_isring,
        'atom_isO': atom_isO,
        'atom_isN': atom_isN,
        
    }

    return data






#  SMILES  3D 
def generate_3d_molecule(smiles):
    mol = Chem.MolFromSmiles(smiles) #, "sanitize = False"
    #print('mol:', mol)
    #mol = Chem.RemoveHs(mol)
    #，,，
    
    try:
        mol = Chem.AddHs(mol)  # 
        # 
        AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        #AllChem.UFFOptimizeMolecule(mol)
        AllChem.MMFFOptimizeMolecule(mol)
        #Chem.SanitizeMol(mol) #
        mol = Chem.RemoveHs(mol) #

        #AllChem.UFFOptimizeMolecule(mol)  # 3D
        #confids=AllChem.EmbedMultipleConfs(mol,1) # mol，，，
        pos = mol.GetConformer(0).GetPositions()
    except Exception as e:
        mol = Chem.AddHs(mol)  # 
        success = rdDistGeom.EmbedMolecule(mol, rdDistGeom.ETKDGv3())  # 3D
        #AllChem.UFFOptimizeMolecule(mol)  # 3D
        AllChem.MMFFOptimizeMolecule(mol)
        #confids=AllChem.EmbedMultipleConfs(mol,1) # mol，，，
        mol = Chem.RemoveHs(mol) #
        pos = mol.GetConformer(0).GetPositions()
    
    #print('mol:', mol)

    '''
    mol_for_sdf=Chem.MolFromSmiles(smi)
    writer=Chem.SDWriter(ligname[:-4]+'.sdf')
    writer.write(mol_for_sdf)
    writer.close()
    mol_for_sdf_h=Chem.AddHs(mol_for_sdf)
    confids=AllChem.EmbedMultipleConfs(mol_for_sdf_h,1)
    writer=Chem.SDWriter(ligname[:-4]+'_rdkit.sdf')
    writer.write(mol_for_sdf_h)
    writer.close()
    '''
    return Chem.RemoveHs(mol)

#  SDF 
def save_molecule(mol, filename):
    writer = Chem.SDWriter(filename)
    writer.write(mol)
    writer.close()


def move_to_pocket(ligand_mol, self_centor, centor):
    xyz = np.array(ligand_mol.GetConformer(0).GetPositions()) - self_centor + centor

    #mol
    new_mol = Change_Mol_D3coord(ligand_mol, xyz)
    return new_mol

def move_to_pocket_only_pos(ligand_mol_pos, self_centor, centor):
    xyz = ligand_mol_pos - self_centor + centor

    return xyz



class Molgraph:
    def __init__(self,rdkitmol,boltzmannweight = None, totalenergy = None, atom_type = None, edge_index = None, edge_type = None, idx = None, smiles='', RemoveHs=True):
        self.smiles=smiles
        self.atoms=[atom.GetAtomicNum() for atom in rdkitmol.GetAtoms()] #
        self.atoms=np.array(self.atoms)
        self.chiraltags=[GP.chiral_types.index(atom.GetChiralTag()) for atom in rdkitmol.GetAtoms()]
        self.chiraltags=np.array(self.chiraltags)
        self.natoms=len(self.atoms)
        self.adjs=np.zeros((self.natoms,self.natoms))
        

        '''
        #boltzmannweight，totalenergy，
        self.boltzmannweight = boltzmannweight
        self.totalenergy     = totalenergy
        self.atom_type       = atom_type
        self.edge_index      = edge_index #，0|all_atoms| - 1, ，，
        self.edge_type       = edge_type
        self.idx             = idx
        self.rdkitmol        = rdkitmol  #，，
        '''
        bond_type = set()
        for bond in rdkitmol.GetBonds():
            a1=bond.GetBeginAtom().GetIdx()
            a2=bond.GetEndAtom().GetIdx()
            bt=bond.GetBondType() 
            ch=GP.bond_types.index(bt)
            self.adjs[a1,a2]=ch+1  #1，1
            self.adjs[a2,a1]=ch+1
            bond_type.add(ch+1)
        #self.zmats=np_adjs_to_zmat(adjs_onek)
        self.coords=np.array(rdkitmol.GetConformer(0).GetPositions())
        #print('bond_type:', sorted(bond_type)) #, [1, 2, 4]

        self.Standardrize()
        return 
    
    def RemoveHs(self):
        noH_idx=np.array([i for i in range(len(self.atoms)) if self.atoms[i]!=1])
        n_heavy_atoms=len(noH_idx)
        coords=self.coords[noH_idx]
        adjs=self.adjs[np.ix_(noH_idx,noH_idx)]
        self.atoms=self.atoms[noH_idx]

        self.adjs=adjs
        self.chiraltags=self.chiraltags[noH_idx]        
        self.coords=coords
        self.natoms=n_heavy_atoms
        return
    
    def PermIndex(self,mode='random'):
        #，，EcDock，zamts，rdkit
        #，，
        if mode=='random':
            start_id=random.choice(np.arange(self.natoms).astype(int))
        else:
            start_id=0
        graph=nx.Graph()
        bonds=[]
        for i in range(self.natoms):
            for j in range(i+1,self.natoms):
                if self.adjs[i,j]!=0:
                    bonds.append((i,j))
        graph.add_edges_from(bonds)
        #print('self.natoms:', self.natoms) #81,，81，bfs36，？
        atom_order=bfs_seq(graph,start_id) #rdkit，bfs？zmatsbfs。atom_order，rdkit
        #print('atom_order:', atom_order) # [25, 23, 22, 24, 21, 26, 19, 27, 18, 20, 28, 32, 14, 29, 31, 13, 15, 30, 12, 16, 17, 11, 10, 9, 7, 6, 8, 0, 5, 1, 4, 2, 3]
        self.atoms=self.atoms[np.ix_(atom_order)]
        self.chiraltags=self.chiraltags[np.ix_(atom_order)] #atom_order，：pred_pos[atom_order]
        #np.ix_(atom_order)，
        self.coords=self.coords[np.ix_(atom_order)] #，，

        #print('atom_order:', len(atom_order))
        self.adjs=self.adjs[np.ix_(atom_order,atom_order)]
        self.atom_order = atom_order
        return
    
    def Generate_Zmats(self):
        adjs_onek=Adjs_to_Onek(self.adjs)
        zmats=np_adjs_to_zmat(adjs_onek)[:,:4]
        return zmats
    
    def Standardrize(self):
        self.RemoveHs() #,
        self.PermIndex(mode='random') #，，
        self.zmats=self.Generate_Zmats() #zmats
        return 

    def Get_3D_Graph_Tensor(self,max_atoms= None):
        #print('self.adjs:', self.adjs)
        #raise Exception('test')
        atom_order = self.atom_order
        if max_atoms: 
            #，，，，masks，True
            adjs=torch.zeros((max_atoms,max_atoms)).long()
            zmats=torch.zeros((max_atoms,4)).long()
            coords=torch.zeros((max_atoms,3))
            masks=torch.zeros(max_atoms).bool()
            masks[:self.natoms]=True

            #print (self.natoms,self.atoms,self.zmats,self.adjs)
            #print('self.zmats:', self.zmats.shape)
            #print('zmats:', zmats.shape)
            #print('self.natoms:', self.natoms)
            '''
            self.zmats: (36, 4)                                                                                    
            zmats: torch.Size([250, 4])         
            self.natoms: 81
            #， self.natoms = self.zmats.shape[0]
            '''
            zmats[:self.natoms]=torch.Tensor(self.zmats).long()  #Target sizes: [81, 4].  Tensor sizes: [36, 4]
            adjs[:self.natoms,:self.natoms]=torch.Tensor(self.adjs).long()
            coords[:self.natoms]=torch.Tensor(self.coords)
            #return None,None,adjs,coords,zmats,masks #，numpy
            return None,None,adjs.numpy(),coords.numpy(),zmats.numpy(),masks.numpy(),np.array(atom_order)
        else:
            #print (self.natoms)
            atom_idx_=Atoms_to_Idx(self.atoms,GP.atom_types)
            return torch.Tensor(atom_idx_).long(),torch.Tensor(self.chiraltags),torch.Tensor(self.adjs).long(),torch.Tensor(self.coords),torch.Tensor(self.zmats).long(),torch.ones(self.natoms).bool()    

    
    def Trans_to_Rdkitmol(self):
        molecule=Chem.RWMol()
        for j in range(self.natoms):
            new_atom=Chem.Atom(int(self.atoms[j]))
            molecule_idx=molecule.AddAtom(new_atom)
        adjs=copy.deepcopy(self.adjs)
        row,col=np.diag_indices_from(adjs)
        adjs[row,col]=0
        idx1,idx2=np.where(adjs!=0)
        for id1,id2 in zip(idx1,idx2):
            if id1<id2:
                molecule.AddBond(int(id1),int(id2),GP.bond_types[int(adjs[id1,id2])-1])
        mol=molecule.GetMol()
        Chem.SanitizeMol(mol)
        AllChem.Compute2DCoords(mol)
        mol=Change_mol_xyz(mol,self.coords)
        return mol


    def update_mol(self):
        #rdkitmol
        mol=Change_mol_xyz(self.rdkitmol, self.coords) #，，zamt，
        return mol

    def Update_Coords(self,coords):
        self.coords=coords

def Adjs_to_Onek(adjs,nchannels=3):
    #nchannels=np.max(adjs)-1 #pdbbind20204，5, ，，，，，，1
    #，4，，34
    #，4
    nchannels = 4
    adjs_onek=np.zeros((adjs.shape[0],adjs.shape[0],nchannels))
    idx1,idx2=np.where(adjs)
    channel_idx=adjs[idx1,idx2].astype(int)-1
    for id1,id2,cid in zip(idx1,idx2,channel_idx):
        #print('id1,id2,cid:', id1,id2,cid) #cid3，。？
        adjs_onek[id1,id2,cid]=1
    return adjs_onek.astype(int)

def Atoms_to_Idx(atoms,possible_atom_types=[1,6,7,8,9,15,16,17,35,53]): #，
    atom_idx_=[possible_atom_types.index(int(a))+1 for a in atoms] 
    return atom_idx_

def Atoms_to_Onek(atoms,possible_atom_types=[1,6,7,8,9,15,16,17,35,53]): #，
    atoms_onek=np.zeros((len(atoms),len(possible_atom_types)))
    for i in range(len(atoms)):
        atoms_onek[i][possible_atom_types.index(atoms[i])]=1
    return atoms_onek.astype(int)
def Chiraltag_to_Onek(chiraltags,chiral_types=[ ChiralType.CHI_UNSPECIFIED, ChiralType.CHI_TETRAHEDRAL_CW, ChiralType.CHI_TETRAHEDRAL_CCW, ChiralType.CHI_OTHER, ChiralType.CHI_TETRAHEDRAL, ChiralType.CHI_ALLENE, ChiralType.CHI_SQUAREPLANAR, ChiralType.CHI_TRIGONALBIPYRAMIDAL,ChiralType.CHI_OCTAHEDRAL]):
    ntags=len(chiral_types)
    chiral_onek=np.zeros((len(chiraltags),ntags))
    for i in range(len(chiraltags)):
        chiral_onek[i][chiraltags[i]]=1
    return chiral_onek.astype(int)