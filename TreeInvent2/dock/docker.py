from collections import namedtuple
import torch,pickle 
from rdkit import DataStructs,Chem
from rdkit.Chem import QED, AllChem

import numpy as np

from ..comparm import DP 
from ..utils.utils_rdkit import *
from .utils_dock import *

def sigmoid_transformation(scores,_low=0,_high=1.0,_k=0.25):
    def _exp(pred_val, low, high, k):
        try:
            return math.pow(10, (10 * k * (pred_val - (low + high) * 0.5) / (low - high)))
        except:
            return 0
    transformed = [1 / (1 + _exp(pred_val, _low, _high, _k)) for pred_val in scores]
    return np.array(transformed, dtype=np.float32)

def reverse_sigmoid_transformation(scores,_low, _high, _k):
    def _reverse_sigmoid_formula(value, low, high, k):
        try:
            return 1 / (1 + 10 ** (k * (value - (high + low) / 2) * 10 / (high - low)))
        except:
            return 0
    transformed = [_reverse_sigmoid_formula(pred_val, _low, _high, _k) for pred_val in scores]
    return np.array(transformed, dtype=np.float32)

class Dockstream_Docker():
    def __init__(self):
        self.input_path=DP.dock_input_path
        self.center=[None,None,None]
        pass
    def prepare_target(self):
        if DP.backend=='AutoDock-Vina':
            try:
                prepare_target_in_vina_format(  
                                                input_path=DP.dock_input_path,
                                                target_pdb_path=DP.target_pdb,
                                                reflig_pdb_path=DP.reflig_pdb,
                                                out_path=DP.dock_input_path,
                                                log_path='target_prep.log',
                                                dockstream_root_path=DP.dockstream_root_path,
                                                bin_path=DP.vina_bin_path
                                                )
                self.receptor_pdbqt=f"{DP.target_pdb.strip('pdb')}fix.pdbqt"
                with open (f'{DP.dock_input_path}/target_prep.log','r') as f:
                    for line in f.readlines():
                        if 'X coordinates' in line:
                            self.center[0]=float(line.strip().split()[-1][5:])
                        if 'Y coordinates' in line:
                            self.center[1]=float(line.strip().split()[-1][5:])
                        if 'Z coordinates' in line:
                            self.center[2]=float(line.strip().split()[-1][5:])
            except Exception as e:
                print (f'Dockstream prepare target failed due to {e}, center is {self.center}')
            else:
                print ('Dockstream prepared target into Vina format PDBQT!')
                print (self.center)
        elif DP.backend=="Glide":
            pass
        return 

    def dock(self,mols=None,SDF=None,output_path=None):
        smiles=[]
        if mols is None:
            assert SDF is not None, "Please provide the input mols or SDF file!"
            mols=[mol for mol in Chem.SDMolSupplier(SDF)]
        if output_path is None:
            print (f"Output path is not provided, docking results will be saved in {DP.dock_input_path}")
            output_path=DP.dock_input_path
            
        for mol in mols:
            if mol:
                smi=Chem.MolToSmiles(mol)
                smiles.append(smi)
            else:
                smiles.append('None')
        if not os.path.exists(output_path):
            os.system(f'mkdir -p {output_path}')
        with open(f'{output_path}/ligand.smi','w') as f:
            for smi in smiles:
                f.write(smi+'\n') 
        if  DP.backend=='AutoDock-Vina': 
            prepare_docking_in_vina_format( 
                                            input_path=DP.dock_input_path,
                                            center=self.center,
                                            box_size=DP.box_size,
                                            receptor_pdbqt_path=f'{DP.dock_input_path}/{self.receptor_pdbqt}',
                                            lig_smiles_csv_path=f'{output_path}/ligand.smi',
                                            lig_conformers_path=f'{output_path}/ligand.sdf',
                                            vina_binary_location=DP.vina_bin_path,
                                            out_path=output_path,
                                            log_path='dock.log',
                                            pose_path='pose.sdf',
                                            score_path='score.log',
                                            ncores=DP.ncores,
                                            nposes=DP.nposes
                                        )

        elif DP.backend=="Glide":
            #print (output_path)
            prepare_docking_in_glide_format(
                                            input_path=DP.dock_input_path,
                                            grid_path=DP.grid_path,
                                            smiles_path=f'{output_path}/ligand.smi',
                                            glide_flags=DP.glide_flags,
                                            glide_keywords=DP.glide_keywords,
                                            ncores=DP.ncores,
                                            out_path=output_path,
                                            log_path='dock.log',
                                            pose_path='pose.sdf',
                                            score_path='score.log',
                                            glide_ver=DP.glide_ver
                                            )
            
        docking_scores=dockstream_dock(conf=f'{output_path}/dock.json') 
        return docking_scores
        
    def compute_scores(self,mols,output_path='./'):
        dock_scores=self.dock(mols,output_path)
        transformed_scores=reverse_sigmoid_transformation(dock_scores,_low=DP.low_threshold,
                                                _high=DP.high_threshold,
                                                _k=DP.k) 
        #print (scores)
        return dock_scores,transformed_scores