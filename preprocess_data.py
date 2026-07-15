import argparse
import os
import shutil
import time
import sys
sys.path.append(os.path.abspath('./'))\


import os
from collections import defaultdict
from ordered_set import OrderedSet


import numpy as np 
from rdkit import Chem

import numpy as np

from tqdm.auto import tqdm


import copy
from rdkit import Chem
from rdkit.Chem import AllChem
import copy
from tqdm import tqdm
from rdkit.Geometry.rdGeometry import Point3D
from collections import Counter
import matplotlib.pyplot as plt
import random 
import dill
import json

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import defaultdict

import torch



import numpy as np
import shutil
import os

import oddt
from oddt.toolkits.extras import rdkit as o_rdkit
from rdkit import Chem
from oddt.docking.AutodockVina import write_vina_pdbqt
import subprocess
import time
import json


import glob
from multiprocessing import Pool
import multiprocessing as mp
from functools import partial

import gzip
import shutil

from Bio.PDB import PDBParser, PDBIO, Select
from Bio import PDB

import pdbreader
from rdkit.Chem import PDBWriter

from pathlib import Path


from multiprocessing import Pool
from typing import List
from sklearn.cluster import KMeans
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolAlign import AlignMolConformers
from biopandas.pdb import PandasPdb
import dill


#import pymol
#from pymol import cmd

from openbabel import pybel

#from openbabel import openbabel as pybel

#from pymol import cmd






def read_pdb_ligand_list(posebuster_file):
    """
    Reads the PDB codes and ligand codes from the input file.
    """
    # Read input file
    with open(posebuster_file, 'r') as f:
        content = f.read()

    # Replace newlines with spaces and split on commas
    content = content.replace('\n', ' ')
    entries = content.split(',')

    pdb_ligand_list = []
    for entry in entries:
        entry = entry.strip()
        if entry:
            parts = entry.split()
            if len(parts) == 2:
                pdb_code, ligand_code = parts
                pdb_code = pdb_code.strip().upper()
                ligand_code = ligand_code.strip().upper()
                pdb_ligand_list.append((pdb_code, ligand_code))
            else:
                print(f"Invalid entry: {entry}")

    return pdb_ligand_list




def save_protein_and_ligand(input_pdb, protein_code, ligand_code, output_protein_pdb, output_ligand_sdf):
    """
    PDB。

    Args:
        input_pdb (str): PDB。
        protein_code (str): （ "chain A"）。
        ligand_code (str): （ "resn LIG"）。
        output_protein_pdb (str): PDB。
        output_ligand_sdf (str): SDF。
    """
    # PDB
    cmd.reinitialize()
    try:
        cmd.load(input_pdb) #
    except Exception as e:
        print(f'local loading error, downloading input {protein_code}')
        a = cmd.fetch(protein_code) #
    
    # 
    #，
    #cmd.select("protein", f'{protein_code}')       #  "chain A"
    cmd.select("ligand", f"resn {ligand_code}")    #  "resn LIG"
    
    # 
    mol2 = os.path.join(os.path.dirname(output_protein_pdb), os.path.basename(output_protein_pdb).split('_')[0] + '_ligand.mol2')
    cmd.save(mol2, "ligand")  # MOL2

    # SDF
    #
    #import os
    #conversion_command = f"obabel temp_ligand.mol2 -O {output_ligand_sdf}"
    #os.system(conversion_command)

    #python
    try:
        #  MOL2 
        mol = next(pybel.readfile("mol2", mol2))
        #  SDF 
        mol.write("sdf", output_ligand_sdf, overwrite=True)
        #print(f"Conversion successful! SDF file saved to: {output_ligand_sdf}")
    except Exception as e:
        print(f"Error during conversion: {e}")



    #，
    # remove solvent
    cmd.remove("solvent")
    # remove non-ligand, inorganic atoms
    cmd.select(f"inorganic and (not (byres first resn {ligand_code}))")
    cmd.remove("sele")
    # remove ligand
    cmd.select(f"resn {ligand_code}")
    cmd.remove("sele")
    # remove nonpolymer
    cmd.select(f"not polymer")
    cmd.remove("sele")
    cmd.save(output_protein_pdb)
    
    #print(f"Protein saved as: {output_protein_pdb}")
    #print(f"Ligand saved as: {output_ligand_sdf}")






def data_move(exclude):
    '''，['index', 'readme']'''

    s_dir1 = 'v2020-other-PL'
    s_dir2 = 'refined-set'
    t_dir  = 'new_pdbbind2020'

    count = 0
    for name in os.listdir(s_dir1):
        if name not in exclude:
            s = os.path.join(s_dir1, name)
            t = os.path.join(t_dir, name)
            shutil.copytree(s, t, dirs_exist_ok = True)
            count += 1

    for name in os.listdir(s_dir2):
        if name not in exclude:
            s = os.path.join(s_dir2, name)
            t = os.path.join(t_dir, name)
            shutil.copytree(s, t, dirs_exist_ok = True)
            count += 1

    
    with open('new_pdbbind2020_name.txt', 'w') as f:
        for name in tqdm(os.listdir(t_dir)):
            f.write(name + '\n')

    print('succeeded to move num:', count) #19443



def pymol_deal_protein(pdb_file):
    # PyMOL
    #pymol.finish_launching()

    # PyMOL
    cmd.reinitialize()

    # PDB
    cmd.load(pdb_file)

    # （）
    cmd.remove("solvent")


    # ，
    cmd.remove("inorganic")

    # （、）
    cmd.select("not polymer")
    if cmd.count_atoms("sele") > 0:
        cmd.remove("sele")

    # PDB
    cmd.save(f"{pdb_file}")






from biopandas.pdb import PandasPdb
from pdbfixer import PDBFixer
from openmm.app import PDBFile
import pandas as pd
import numpy as np

def clean_protein2(
    path, name,
    merge_chain_id="A",    #  A
    round_decimal=3,       # 
    remove_water=True,
    remove_metals=True,
    remove_nonstandard=True
):
    
    input_pdb = output_pdb = os.path.join(path, name, f'{name}_protein.pdb')
    # -------------------------------
    # 1.  PDBFixer , #，，
    # -------------------------------
    '''
    print(">>> Step 1: Running PDBFixer to repair missing residues...")
    fixer = PDBFixer(filename=input_pdb)
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(pH=7.4)  # ，
    PDBFile.writeFile(fixer.topology, fixer.positions, open("tmp_fixed.pdb", "w"))
    '''

    # -------------------------------
    # 2. Biopandas 
    # -------------------------------
    print(">>> Step 2: Load repaired PDB with Biopandas")
    #ppdb = PandasPdb().read_pdb("tmp_fixed.pdb")
    ppdb = PandasPdb().read_pdb(input_pdb)

    atom_df = ppdb.df["ATOM"]
    het_df = ppdb.df["HETATM"]

    # -------------------------------
    # 3.  /  / 
    # -------------------------------
    if remove_water:
        water_names = {"HOH", "H2O", "WAT"}
        het_df = het_df[~het_df.residue_name.isin(water_names)]

    if remove_metals:
        metal_names = {"NA", "K", "MG", "CA", "ZN", "FE", "CU", "MN", "CO", "NI", "CD"}
        het_df = het_df[~het_df.residue_name.isin(metal_names)]

    if remove_nonstandard:
        std_aa = {
            "ALA","ARG","ASN","ASP","CYS","GLU","GLN","GLY","HIS","ILE",
            "LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL"
        }
        atom_df = atom_df[atom_df.residue_name.isin(std_aa)]

    # -------------------------------
    # 4. 
    # -------------------------------
    print(">>> Step 4: Removing hydrogens")
    atom_df = atom_df[~atom_df.atom_name.str.startswith("H")]

    # -------------------------------
    # 5. （ xyz）
    # -------------------------------
    print(">>> Step 5: Removing duplicate atoms by coordinates")
    atom_df["coord_key"] = (
        atom_df["x_coord"].round(round_decimal).astype(str) + "_" +
        atom_df["y_coord"].round(round_decimal).astype(str) + "_" +
        atom_df["z_coord"].round(round_decimal).astype(str)
    )
    atom_df = atom_df.drop_duplicates(subset="coord_key", keep="first")
    atom_df = atom_df.drop(columns=["coord_key"])

    # -------------------------------
    # 6. （ A）
    # -------------------------------
    #print(">>> Step 6: Merging chains")
    #atom_df["chain_id"] = merge_chain_id 

    # -------------------------------
    # 7.  PandasPdb 
    # -------------------------------
    ppdb.df["ATOM"] = atom_df
    ppdb.df["HETATM"] = het_df.iloc[0:0]  # 

    print(">>> Step 7: Saving cleaned protein PDB")
    ppdb.to_pdb(path=output_pdb, records=["ATOM"], gz=False)

    print(f"\n✨ Clean protein saved to: {output_pdb}")






def pdb2020_filter_pdb(path, data_name_file):
    ''' ，，, ，‘origin_protein_*.pdb'''

    name_set = OrderedSet()
    with open(f'{data_name_file}') as f:
        for i in f:
            tg = i.strip('\n')
            name_set.add(tg)
        
    print('len(name_set):', len(name_set))

    
    ''''''
    count = 0
    for name in tqdm(name_set):
        try:
            pdb_file = os.path.join(path, name, f'{name}_protein.pdb')
            shutil.copy2(pdb_file, os.path.join(path, name, f'origin_{name}_protein.pdb')) #
            output_pdb_file = os.path.join(path, name, f'{name}_protein.pdb') #


            parser = PDB.PDBParser(QUIET=True)
            structure = parser.get_structure('protein', pdb_file)
            
            io = PDB.PDBIO()
            io.set_structure(structure)
            
            class NonStandardResidueSelect(PDB.Select):
                def accept_residue(self, residue):
                    return is_standard_amino_acid(residue) and not is_metal_ion(residue) and not is_water(residue)
            
            io.save(output_pdb_file, NonStandardResidueSelect())

            pymol_deal_protein(output_pdb_file)
        
        except FileNotFoundError as e:
            print(f'error in {name}: {e}')
            count += 1
            continue
    
    print('error num:', count)
    

    

    ''''''
    count = 0
    for name in tqdm(name_set):
        try:
            pdb_file = os.path.join(path, name, f'{name}_pocket.pdb')
            shutil.copy2(pdb_file, os.path.join(path, name, f'origin_{name}_pocket.pdb')) #
            output_pdb_file = os.path.join(path, name, f'{name}_pocket.pdb') #


            parser = PDB.PDBParser(QUIET=True)
            structure = parser.get_structure('protein', pdb_file)
            
            io = PDB.PDBIO()
            io.set_structure(structure)
            
            class NonStandardResidueSelect(PDB.Select):
                def accept_residue(self, residue):
                    return is_standard_amino_acid(residue) and not is_metal_ion(residue) and not is_water(residue)
            
            io.save(output_pdb_file, NonStandardResidueSelect())

            pymol_deal_protein(output_pdb_file)
        
        except FileNotFoundError as e:
            print(f'error in {name}: {e}')
            count += 1
            continue
    
    print('error num:', count)
    



from biopandas.pdb import PandasPdb

def clean_protein_pdb(path, name):
    
    input_pdb = output_pdb = os.path.join(path, name, f'{name}_protein.pdb')
    ppdb = PandasPdb().read_pdb(input_pdb)

    df = ppdb.df['ATOM']    #  ATOM （）
    het = ppdb.df['HETATM'] # HETATM 、、

    # --- 1.  ---
    water_resnames = {"HOH", "WAT", "H2O"}
    het = het[~het['residue_name'].isin(water_resnames)]

    # --- 2.  ---
    metal_list = {
        "NA", "K", "CA", "MG", "ZN", "FE", "CU", "MN", "CO",
        "NI", "CD", "HG", "SR", "CS", "BA", "PB"
    }
    het = het[~het['residue_name'].isin(metal_list)]

    # --- 3. ， 20  ---
    standard_aa = {
        "ALA","ARG","ASN","ASP","CYS",
        "GLU","GLN","GLY","HIS","ILE",
        "LEU","LYS","MET","PHE","PRO",
        "SER","THR","TRP","TYR","VAL"
    }

    #  DF['ATOM'] 
    df = df[df['residue_name'].isin(standard_aa)]

    # --- 4.  ATOM ， HETATM ---
    ppdb.df['ATOM'] = df
    ppdb.df['HETATM'] = het.iloc[0:0]  #  HETATM
    
    # 
    df_no_h = ppdb.df['ATOM'][ppdb.df['ATOM']['element_symbol'] != 'H']
    ppdb.df['ATOM'] = df_no_h
    if 'HETATM' in ppdb.df:
        ppdb.df['HETATM'] =ppdb.df['HETATM'][ppdb.df['HETATM']['element_symbol'] != 'H']
    

    # --- 5.  ---
    ppdb.to_pdb(path=output_pdb, records=['ATOM'], gz=False)

    print(f"Clean protein saved to {output_pdb}")






def pdb2020_filter_pdb_parallel(path, name):
    ''' ，，, ，‘origin_protein_*.pdb,，'''

    ''''''
    try:
        pdb_file = os.path.join(path, name, f'{name}_protein.pdb')
        #shutil.copy2(pdb_file, os.path.join(path, name, f'origin_{name}_protein.pdb')) #
        output_pdb_file = os.path.join(path, name, f'{name}_protein.pdb') #``


        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure('protein', pdb_file)
        
        io = PDB.PDBIO()
        io.set_structure(structure)
        
        class NonStandardResidueSelect(PDB.Select):
            def accept_residue(self, residue):
                return is_standard_amino_acid(residue) and not is_metal_ion(residue) and not is_water(residue)
            
            def accept_atom(self, atom):
                """Filter atoms: remove hydrogen atoms"""
                return atom.element != 'H'
        
        io.save(output_pdb_file, NonStandardResidueSelect())

        #pymol_deal_protein(output_pdb_file) #，pymol，。。，，
    
    except Exception as e:
        print(f'error in {name}: {e}')
        exit()

    """

    ''''''
    try:
        pdb_file = os.path.join(path, name, f'{name}_pocket.pdb')
        shutil.copy2(pdb_file, os.path.join(path, name, f'origin_{name}_pocket.pdb')) #
        output_pdb_file = os.path.join(path, name, f'{name}_pocket.pdb') #


        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure('protein', pdb_file)
        
        io = PDB.PDBIO()
        io.set_structure(structure)
        
        class NonStandardResidueSelect(PDB.Select):
            def accept_residue(self, residue):
                return is_standard_amino_acid(residue) and not is_metal_ion(residue) and not is_water(residue)
        
        io.save(output_pdb_file, NonStandardResidueSelect())

        pymol_deal_protein(output_pdb_file) #，pymol，
    
    except Exception as e:
        print(f'error in {name}: {e}')
    """



def is_standard_amino_acid(residue):
    standard_aa = {
        'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
        'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'
    }
    return residue.get_resname() in standard_aa

def is_metal_ion(residue):
    # Common metal ions in PDB files
    metal_ions = {'NA', 'K', 'MG', 'CA', 'MN', 'FE', 'CO', 'NI', 'CU', 'ZN'}
    return residue.get_resname() in metal_ions

def is_water(residue):
    return residue.get_resname() == 'HOH'




def pdbbind2020_gen_name(path, data_name):
    '''，，，idnex. ，，'''


    ''''''
    protein_set = OrderedSet()
    with open(f'{data_name}/{data_name}_name.txt', 'r') as f:
        for i in f:
            tg = i.strip()
            protein_set.add(tg)
    print('len(protein_set):', len(protein_set))





    ''''''
    data_index = []
    for pn in protein_set:
        #('new_pdbbind2020/5uxf/5uxf_pocket10.pdb', new_pdbbind2020/5uxf/5uxf_protein.pdb', [7.06, 2017, '1.50'], 'new_pdbbind2020/5uxf/5uxf_ligand.sdf', '5uxf')
        tmp = []

        if data_name == 'posebusters_glide':
            tmp.append(f'{path}/{path}/{pn}/{pn}_protein_256.pdb')
            tmp.append(f'{path}/{path}/{pn}/{pn}_protein_processed_glide.pdb')
            tmp.append([0.00, 2024, '0.00'])
            tmp.append(f'{path}/{path}/{pn}/{pn}_ligand-rdkit-glide.sdf')
        else:
            tmp.append(f'{path}/{path}/{pn}/{pn}_protein_256.pdb')
            tmp.append(f'{path}/{path}/{pn}/{pn}_protein.pdb')
            tmp.append([0.00, 2024, '0.00']) #，
            tmp.append(f'{path}/{path}/{pn}/{pn}_ligand.sdf')


        tmp.append(pn)
        data_index.append(tuple(tmp))

    with open(f'{data_name}/index.pkl', 'wb') as f:
        dill.dump(data_index, f)

    with open(f'{data_name}/index.pkl', 'rb') as f:
        data_index = dill.load(f)
    
    print('data_index:', data_index[:1])  #
    # [('new_pdbbind2020/3fee/3fee_pocket10.pdb', 'new_pdbbind2020/3fee/3fee_protein.pdb', [0.0, 2024, '0.00'], 'new_pdbbind2020/3fee/3fee_ligand.sdf', '3fee')]
    print('le(data_index):', len(data_index)) #19443





    '''、、. ，'''
    train = []
    valid = []
    test  = []

    with open(f'{data_name}/{data_name}_name.txt', 'r') as f:
        for i in f:
            test.append(i.strip('\n'))

    data_dict = {}
    data_dict['train'] = train
    data_dict['valid'] = valid
    data_dict['test']  = test

    torch.save(data_dict, f'{data_name}/{data_name}_split.pt')




def ligand2mae(path, name):
    '''，pdb，sdf，mae，maegz'''

    current_pwd = os.path.join(path, name)
    print('current_pwd:', current_pwd)
    if os.path.exists(f'{current_pwd}') and os.listdir(f'{current_pwd}'):
        #
        os.chdir(f'{current_pwd}')

        # 
        all_files_and_dirs = os.listdir(current_pwd)

        # 
        sdf_list = [f for f in all_files_and_dirs if os.path.isfile(os.path.join(current_pwd, f)) and f.endswith(".sdf") and 'origin' in f]
        for file in sdf_list: #sdf
            # 
            base_name = os.path.basename(file)

            # 
            file_name, _ = os.path.splitext(base_name)

            new_file_path = f'{name}_ligand.sdf' 

            # 
            command = f"ligprep -isd {file} -epik -ph 7.0 -pht 0.2 -s 1 -NJOBS 12 -osd {new_file_path}"

            # prepwizard
            process = subprocess.run(command, shell=True, text=True) #,
            #process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, text=True)



def pdbbind2020_gen_centor(data_name, name_file, add_size = 10):
    '''
    #box, unimolbox,
    data_name：
    file：
    add_size：xyz，
    '''

    name_list  = []
    with open(f'{name_file}') as f:
        for i in f:
            name_list.append(i.strip('\n'))
    print('name_list:', len(name_list))

    #name_list = ['3acl']
    error_count = 0
    ligand_pos = []
    error_ligand_list = []
    for name in name_list:
        dt_dict = {}

        #glide，，，origin_name_ligand.sdf
        #sdf，，
        file = os.path.join(f'{data_name}/{data_name}', name, f'{name}_ligand.sdf')
        try:
            mol  = Chem.SDMolSupplier(file)[0] 
            mol  = Chem.RemoveHs(mol)
        except Exception as e:
            try:
                file = os.path.join(f'{data_name}/{data_name}', name, f'{name}_ligand.sdf')
                #，，
                mol  = Chem.SDMolSupplier(file, sanitize=False)[0]
                mol  = Chem.RemoveHs(mol)

                #
                #mol = standardize(mol)
                #mol = Neutralize_atoms(mol)
            except Exception as e:
                try:
                    mol = Chem.MolFromMol2File(os.path.join(os.path.dirname(file), file.split('/')[-2] + '_ligand.mol2'), sanitize=False)
                    mol = Chem.RemoveHs(mol)
                    #mol = standardize(mol)
                    #mol = Neutralize_atoms(mol)
                except Exception as e:
                    error_ligand_list.append(file.split('/')[-2])
                    print('error:', e)
                    error_count += 1
                    continue


        ligand_pos = np.array(mol.GetConformer(0).GetPositions())
        centor = np.mean(ligand_pos, axis = 0)
        assert len(centor) == 3

        add_size = add_size
        min_xyz = [min(coord[i] for coord in ligand_pos) for i in range(3)] #3
        max_xyz = [max(coord[i] for coord in ligand_pos) for i in range(3)]
        center = np.mean(ligand_pos, axis=0)
        size = [abs(max_xyz[i] - min_xyz[i]) for i in range(3)]
        center_x, center_y, center_z = center
        size_x, size_y, size_z = size
        size_x = size_x + add_size
        size_y = size_y + add_size
        size_z = size_z + add_size
        dt_dict = {
            "center_x": float(center_x),
            "center_y": float(center_y),
            "center_z": float(center_z),
            "size_x": float(size_x),
            "size_y": float(size_y),
            "size_z": float(size_z)
        }

        new_file = os.path.join(f'{data_name}/{data_name}', name, f'{name}_ligand_docking_grid_boxsize10.json')

        with open(new_file, 'w', encoding='utf-8') as json_file:
            json.dump(dt_dict, json_file, ensure_ascii=False, indent=4)

    print('error_count == 22 ？:', error_count)

    with open(f'{data_name}/error_ligand_list.txt', 'w') as f:
        for name in error_ligand_list:
            f.write(name + '\n')



def Neutralize_atoms(mol):
    pattern = Chem.MolFromSmarts("[+1!h0!$([*]~[-1,-2,-3,-4]),-1!$([*]~[+1,+2,+3,+4])]")
    at_matches = mol.GetSubstructMatches(pattern)
    at_matches_list = [y[0] for y in at_matches]
    if len(at_matches_list) > 0:
        for at_idx in at_matches_list:
            atom = mol.GetAtomWithIdx(at_idx)
            chg = atom.GetFormalCharge()
            hcount = atom.GetTotalNumHs()
            atom.SetFormalCharge(0)
            atom.SetNumExplicitHs(hcount - chg)
            atom.UpdatePropertyCache()
    return mol

def standardize(mol):
    # follows the steps in
    # https://github.com/greglandrum/RSC_OpenScience_Standardization_202104/blob/main/MolStandardize%20pieces.ipynb
     # as described **excellently** (by Greg) in
    # https://www.youtube.com/watch?v=eWTApNX8dJQ

    # removeHs, disconnect metal atoms, normalize the molecule, reionize the molecule
    clean_mol = Chem.rdMolStandardize.Cleanup(mol) 

    # if many fragments, get the "parent" (the actual mol we are interested in) 
    parent_clean_mol = Chem.rdMolStandardize.FragmentParent(clean_mol)

    # try to neutralize molecule
    uncharger = Chem.rdMolStandardize.Uncharger() # annoying, but necessary as no convenience method exists
    uncharged_parent_clean_mol = uncharger.uncharge(parent_clean_mol)

    # note that no attempt is made at reionization at this step
    # nor at ionization at some pH (rdkit has no pKa caculator)
    # the main aim to to represent all molecules from different sources
    # in a (single) standard way, for use in ML, catalogue, etc.

    te = Chem.rdMolStandardize.TautomerEnumerator() # idem
    taut_uncharged_parent_clean_mol = te.Canonicalize(uncharged_parent_clean_mol)

    return taut_uncharged_parent_clean_mol




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
    #print('atom_id_list.shape:', atom_id_list.shape) #(699,)
    #print("pdb.df['ATOM'].shape:", pdb.df['ATOM'].shape) #pdb.df['ATOM'].shape: (9596, 21)
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

    new_coords = np.array(new_coords)
    #，
    #print('new_coords:', new_coords) # new_coords
    #print('coords:', coords) # object
    
    #print('new_coords.shape:', new_coords.shape) # (699, 3)
    #print('coords.shape:', coords.shape) # (699, 3)
    
    #print('type(new_coords):', new_coords.dtype) # new_coords
    #print('type(coords):', coords.dtype) # object
    if not np.allclose(new_coords, coords, atol=0.02):
        #print('not same')
        raise SystemExit

    # PandasPdb
    new_pdb = PandasPdb()
    new_pdb.df['ATOM'] = filtered_atoms


    # PDB
    new_pdb.to_pdb(path=output_file, records=['ATOM'], gz=False, append_newline=True) 
    #pdb，id，，，




def find_residues_in_pocket(pocket: dict, pdf):
    """
    Given a pocket config and a residue df, 
    return a list of residues that are in the pocket
    """
    def _get_vertex(pocket: dict, axis: str) -> tuple:
        """
        Return the minimum and maximum values of the given axis

        Args:
        pocket (dict): pocket config
        axis (str): ["x", "y", "z"]

        Returns:
        A tuple of floats.
        """
        return (
            pocket["center_{}".format(axis)] \
                - pocket["size_{}".format(axis)] / 2,
            pocket["center_{}".format(axis)] \
                + pocket["size_{}".format(axis)] / 2
            )
    min_x, max_x = _get_vertex(pocket, "x")
    min_y, max_y = _get_vertex(pocket, "y")
    min_z, max_z = _get_vertex(pocket, "z")
    min_array = np.array([min_x, min_y, min_z]).reshape(1,3)
    max_array = np.array([max_x, max_y, max_z]).reshape(1,3)
    patoms, pcoords, residues = [], np.empty((0,3)), []
    global_int_index = []
    for i in range(len(pdf)):
        atom_info = pdf.iloc[i]
        _rescoor = np.array(atom_info[['x_coord','y_coord','z_coord']].values).reshape(-1,3)
        mapping = (_rescoor > min_array) & (_rescoor < max_array)
        if (mapping.sum(-1) == 3).sum() > 0:
            patoms += [atom_info['atom_name']]
            pcoords = np.concatenate((pcoords, _rescoor), axis=0)
            residues += [str(atom_info['chain_id'])+str(atom_info['residue_number'])]
            global_int_index.append(i) #
    return patoms, pcoords, residues, global_int_index

def extract_pocket(input_protein, input_docking_grid, pocket_file):
    main_atoms=["N", "CA", "C", "O", "H"]
    allow_pocket_atoms=[['C', 'H', 'N', 'O', 'S']]
    try:
        pmol = PandasPdb().read_pdb(input_protein)
    except Exception as e:
        print('error:', e)
        #
        with open('failed_pocket.txt', 'a') as f:
            f.write(input_protein+'\n')
        
        raise Exception('extract_pocket error')
        return None
    
    
    df_no_h = pmol.df['ATOM'][pmol.df['ATOM']['element_symbol'] != 'H']
    pmol.df['ATOM'] = df_no_h
    if 'HETATM' in pmol.df:
        pmol.df['HETATM'] = pmol.df['HETATM'][pmol.df['HETATM']['element_symbol'] != 'H']
    
            
    with open(input_docking_grid, "r") as file:
        box_dict = json.load(file)

    pdf = pmol.df['ATOM']
    patoms, pcoords, residues, global_int_index = find_residues_in_pocket(box_dict, pdf) #
    def _filter_pocketatoms(atom):
        if atom[:2] in ['Cd','Cs', 'Cn', 'Ce', 'Cm', 'Cf', 'Cl', 'Ca', 'Cr', 'Co', 'Cu', 'Nh', 'Nd', 'Np', 'No', 'Ne', 'Na', 'Ni', \
            'Nb', 'Os', 'Og', 'Hf', 'Hg', 'Hs', 'Ho', 'He', 'Sr', 'Sn', 'Sb', 'Sg', 'Sm', 'Si', 'Sc', 'Se']:
            return None
        if atom[0] >= '0' and atom[0] <= '9':
            return _filter_pocketatoms(atom[1:])
        if atom[0] in ['Z','M','P','D','F','K','I','B']:
            return None
        if atom[0] in allow_pocket_atoms:
            return atom
        return atom

    atoms, index, residues_tmp = [], [], []
    int_index = []
    for i,(a, g_index) in enumerate(zip(patoms, global_int_index)): 
        output = _filter_pocketatoms(a)
        if output is not None:
            index.append(True)
            int_index.append(g_index)
            atoms.append(output)
            residues_tmp.append(residues[i])
        else:
            index.append(False)
            
    #
    
    extract_atoms_by_ids(input_protein, np.array(int_index), np.array(pcoords[index], dtype=np.float64), pocket_file)
    
    
    coordinates = pcoords[index].astype(np.float32)
    residues = residues_tmp
    patoms = atoms
    pcoords = [coordinates]
    side = [0 if a in main_atoms else 1 for a in patoms]
    return patoms, pcoords, residues, side, box_dict



def calculate_distance_matrix_numpy(A, B):
    """
    Calculate Euclidean distance matrix between two coordinate matrices using NumPy

    Parameters:
    A (np.ndarray): Coordinate matrix of shape (n, 3)
    B (np.ndarray): Coordinate matrix of shape (m, 3)

    Returns:
    np.ndarray: Distance matrix of shape (n, m)
    """
    # A shape (n, 3), B shape (m, 3)
    # Calculate pairwise differences using broadcasting
    diff = A[:, np.newaxis, :] - B[np.newaxis, :, :]  # shape (n, m, 3)
    
    # Square differences, sum along last axis, then sqrt
    dist_matrix = np.sqrt(np.sum(diff**2, axis=2))  # shape (n, m)
    
    return dist_matrix


import numpy as np

def select_closest_256(pcoords: np.ndarray):
    """
     N×3 ，，
     300 。
    
    :
        pcoords: N×3  numpy 
    
    :
        centroid:  (3,)
        distances:  (N,)
        selected_coords:  300  (300×3)
        selected_indices: 300 (300,)
    """
    if pcoords.ndim != 2 or pcoords.shape[1] != 3:
        raise ValueError("pcoords  N×3 ")
    
    # Step 1: 
    centroid = np.mean(pcoords, axis=0)
    
    # Step 2: 
    distances = np.linalg.norm(pcoords - centroid, axis=1)
    
    # Step 3:  300 
    top_k = min(256, len(pcoords))  # 
    sorted_indices = np.argsort(distances)[:top_k]
    selected_coords = pcoords[sorted_indices]
    
    return centroid, distances, selected_coords, sorted_indices

def gen_distance_matrix(path, name):
    '''，rdkit'''

    try:
        #
        #print('path, name:', path, name)
        sdf_file = os.path.join(path, path, name, f'{name}_ligand.sdf')
        #print('sdf_file:', sdf_file)
        ligand_mol = Chem.SDMolSupplier(sdf_file)[0] 
        ligand_mol = Chem.RemoveHs(ligand_mol)
        
        #，
        input_protein       = os.path.join(path, path, name, f'{name}_protein.pdb') 
        input_docking_grid  = os.path.join(path, path, name, f'{name}_ligand_docking_grid_boxsize10.json')
        pocket_file         = os.path.join(path, path, name, f'{name}_protein_400.pdb')
        patoms, pcoords, residues, side, box_dict = extract_pocket(input_protein, input_docking_grid, pocket_file)
        
        #，300
        centroid, distances, selected_coords, sorted_indices = select_closest_256(np.array(pcoords[0], dtype=np.float32))
        
                    
        #
        pocket_file_256 = os.path.join(path, path, name, f'{name}_protein_256.pdb')
        extract_atoms_by_ids(pocket_file, np.array(sorted_indices), np.array(selected_coords), pocket_file_256)
        
        ligand_pos  = np.array(ligand_mol.GetConformer(0).GetPositions(), dtype=np.float32)
        protein_pos = np.array(selected_coords, dtype=np.float32) 
        interaction_data = {}
        interaction_data['holo_coords_list']    = [np.array(ligand_mol.GetConformers()[0].GetPositions(), dtype=np.float32)]
        interaction_data['coords_predict_list'] = [np.array(ligand_mol.GetConformers()[0].GetPositions(), dtype=np.float32)]
        interaction_data['pocket_coords_list']  = [np.array(selected_coords, dtype=np.float32)]
        
        '''
        exit()
        #
        print('ligand_pos, protein_pos:', ligand_pos.shape, protein_pos.shape)
        cross_distance = calculate_distance_matrix_numpy(ligand_pos, protein_pos)
        print('cross_distance,shape:', cross_distance.shape)
        interaction_data['cross_distance_list'] = [cross_distance]
        #interaction_5SAK_v2.pkl
        with open(os.path.join(path, name, f'interaction_{name}_v2.pkl'), 'wb') as f:
            dill.dump(interaction_data, f)
        
        

        file_path = os.path.join(path, name, f'interaction_{name}_v2.pkl')

        # （）
        if os.path.isfile(file_path):
            try:
                os.remove(file_path)
                print(f": {file_path}")
            except OSError as e:
                print(f": {e}")
        else:
            print(f": {file_path}")
        '''
    except Exception as e:
        print(e)
        print('path, name:', path, name)
        


def recreate_dir(dir_path):
    """
    ，；
    ，。
    """
    if os.path.exists(dir_path):
        if os.path.isdir(dir_path):
            shutil.rmtree(dir_path)
        else:
            raise ValueError(f"{dir_path} ")

    os.makedirs(dir_path, exist_ok=True)
        
        
        



def parse_preprocess_args():
    """Parse command-line arguments for preprocessing one user-provided complex."""
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess a user-provided protein-ligand pair and organize it into "
            "store_dir/name/name_ligand.sdf and store_dir/name/name_protein.pdb."
        )
    )
    parser.add_argument(
        "--input_ligand",
        type=str,
        required=True,
        help="Path to the input ligand SDF file, e.g., input_data/5SB2/5SB2_ligand.sdf or 5SB2.sdf.",
    )
    parser.add_argument(
        "--input_protein",
        type=str,
        required=True,
        help="Path to the input protein PDB file, e.g., input_data/5SB2/5SB2_protein.pdb or 5SB2.pdb.",
    )
    parser.add_argument(
        "--store_dir",
        type=str,
        default="user_data",
        help="Directory used to store organized user data. Default: user_data.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help=(
            "Optional complex name. If not provided, it will be inferred from the ligand/protein filename "
            "by removing suffixes such as _ligand and _protein."
        ),
    )
    parser.add_argument(
        "--data_name",
        type=str,
        default="tmpdata",
        help="Temporary dataset directory name. Default: tmpdata.",
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        default="tmpresault",
        help="Directory used to store docking results. Default: tmpresault.",
    )
    parser.add_argument(
        "--add_size",
        type=float,
        default=10.0,
        help="Extra box size added to ligand xyz range when generating docking grid. Default: 10.0.",
    )
    parser.add_argument(
        "--clean_workers",
        type=int,
        default=10,
        help="Number of workers for protein cleaning. Default: 10.",
    )
    parser.add_argument(
        "--pocket_workers",
        type=int,
        default=20,
        help="Number of workers for pocket extraction. Default: 20.",
    )
    return parser.parse_args()


def infer_complex_name(file_path: str) -> str:
    """
    Infer complex name from ligand/protein filename.

    Supported examples:
        5SD5.sdf              -> 5SD5
        5SD5.pdb              -> 5SD5
        5SB2_ligand.sdf       -> 5SB2
        5SB2_protein.pdb      -> 5SB2
        5SB2_ligand_rdkit.sdf -> 5SB2_ligand_rdkit  # unchanged unless suffix is exactly _ligand/_protein
    """
    stem = Path(file_path).stem
    for suffix in ("_ligand", "-ligand", "_protein", "-protein"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def prepare_user_data(input_ligand: str, input_protein: str, store_dir: str, name: str = None) -> str:
    """
    Organize a user-provided ligand/protein pair into:
        store_dir/name/name_ligand.sdf
        store_dir/name/name_protein.pdb

    Returns:
        name: the final complex name.
    """
    input_ligand = os.path.abspath(input_ligand)
    input_protein = os.path.abspath(input_protein)
    store_dir = os.path.abspath(store_dir)

    if not os.path.isfile(input_ligand):
        raise FileNotFoundError(f"Input ligand file does not exist: {input_ligand}")
    if not os.path.isfile(input_protein):
        raise FileNotFoundError(f"Input protein file does not exist: {input_protein}")

    ligand_suffix = Path(input_ligand).suffix.lower()
    protein_suffix = Path(input_protein).suffix.lower()
    if ligand_suffix != ".sdf":
        raise ValueError(f"Input ligand must be an .sdf file, but got: {input_ligand}")
    if protein_suffix != ".pdb":
        raise ValueError(f"Input protein must be a .pdb file, but got: {input_protein}")

    ligand_name = infer_complex_name(input_ligand)
    protein_name = infer_complex_name(input_protein)

    if name is None:
        if ligand_name != protein_name:
            raise ValueError(
                "Cannot infer a unique complex name because ligand and protein names are different: "
                f"ligand={ligand_name}, protein={protein_name}. Please pass --name explicitly."
            )
        name = ligand_name

    target_dir = os.path.join(store_dir, name)
    os.makedirs(target_dir, exist_ok=True)

    target_ligand = os.path.join(target_dir, f"{name}_ligand.sdf")
    target_protein = os.path.join(target_dir, f"{name}_protein.pdb")

    shutil.copy2(input_ligand, target_ligand)
    shutil.copy2(input_protein, target_protein)

    print("User input data has been organized:")
    print(f"  ligand : {target_ligand}")
    print(f"  protein: {target_protein}")

    return name

if __name__ == '__main__':

    args = parse_preprocess_args()

    # ：
    # input_ligand/input_protein -> store_dir/name/name_ligand.sdf and name_protein.pdb
    prepared_name = prepare_user_data(
        input_ligand=args.input_ligand,
        input_protein=args.input_protein,
        store_dir=args.store_dir,
        name=args.name,
    )

    # 
    data_name = args.data_name
    base_path = os.path.join(data_name, data_name)
    s_path    = base_path
    t_path    = base_path

    # 
    resault_dir = args.result_dir

    # ，
    recreate_dir(data_name)
    recreate_dir(resault_dir)

    # 
    '''
    

     args.store_dir/name ：
        args.store_dir/name/name_ligand.sdf
        args.store_dir/name/name_protein.pdb

    ：
        tmpdata/tmpdata/name/name_ligand.sdf
        tmpdata/tmpdata/name/name_protein.pdb

    ：tmpresault。 5SB2，：
        tmpresault/5SB2/gen_5SB2_ligand.sdf      # 
        tmpresault/5SB2/origin_5SB2_ligand.sdf   # 
        tmpresault/5SB2/5SB2_protein.pdb         # 
    '''

    user_dir = os.path.abspath(args.store_dir)
    prepared_src_dir = os.path.join(user_dir, prepared_name)
    prepared_dst_dir = os.path.join(base_path, prepared_name)

    os.makedirs(base_path, exist_ok=True)
    shutil.copytree(prepared_src_dir, prepared_dst_dir, dirs_exist_ok=True)

    name_set = set(list(os.listdir(base_path)))

    with open(f'{data_name}/{data_name}_name.txt', 'w') as f:
        for name in name_set:
            f.write(name + '\n')

    path = base_path
    # 
    for name in tqdm(name_set):
        sdf_file = os.path.join(path, name, f'{name}_ligand.sdf')
        shutil.copy2(sdf_file, os.path.join(path, name, f'origin_{name}_ligand.sdf'))

        pdb_file = os.path.join(path, name, f'{name}_protein.pdb')
        shutil.copy2(pdb_file, os.path.join(path, name, f'origin_{name}_protein.pdb'))

    ''' ，，， '''
    path = base_path
    data_name_file = f'{data_name}/{data_name}_name.txt'

    # ，，
    name_set = OrderedSet()
    with open(f'{data_name_file}') as f:
        for i in f:
            tg = i.strip('\n')
            name_set.add(tg)
    print('len(name_set):', len(name_set))

    path_list = [path] * len(name_set)
    parameters = [(i, j) for i, j in zip(path_list, name_set)]
    with Pool(processes=args.clean_workers) as pool:
        pool.starmap(clean_protein2, parameters)

    print('，，， is ending')

    '''，unimol，xyz'''
    name_file = f'{data_name}/{data_name}_name.txt'
    pdbbind2020_gen_centor(data_name, name_file, add_size=args.add_size)

    '''index'''
    path = data_name
    pdbbind2020_gen_name(path, data_name)

    '''，，pdb'''
    path = data_name
    data_name_file = f'{data_name}/{data_name}_name.txt'

    # ，，
    name_set = OrderedSet()
    with open(f'{data_name_file}') as f:
        for i in f:
            tg = i.strip('\n')
            name_set.add(tg)
    print('len(name_set):', len(name_set))

    path_list = [path] * len(name_set)
    parameters = [(i, j) for i, j in zip(path_list, name_set)]

    with Pool(processes=args.pocket_workers) as pool:
        pool.starmap(gen_distance_matrix, parameters)
