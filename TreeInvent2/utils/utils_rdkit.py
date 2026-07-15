#from IPython.display import Image
from rdkit.Geometry.rdGeometry import Point3D
from rdkit import Chem 
#from rdkit.Chem import AllChem,rdmolfiles,rdFMCS,Draw
#from rdkit.Chem.Draw import rdMolDraw2D
import copy 
import numpy as np 
from ..comparm import Table_Element
from rdkit.Chem import MolStandardize
from rdkit.Chem.MolStandardize import rdMolStandardize

def Check_warning_structures(rdkitmol):
    warning_smarts=["[*;r8]",
                "[*;r9]",
                "[*;r10]",
                "[*;r11]",
                "[*;r12]",
                "[*;r13]",
                "[*;r14]",
                "[*;r15]",
                "[*;r16]",
                "[*;r17]",
                "[#8][#8]",
                "[#6;+]",
                "[#16][#16]",
                "[#7;!n][S;!$(S(=O)=O)]",
                "[#7;!n][#7;!n]",
                "C#C",
                "C(=[O,S])[O,S]",
                "[#7;!n][C;!$(C(=[O,N])[N,O])][#16;!s]",
                "[#7;!n][C;!$(C(=[O,N])[N,O])][#7;!n]",
                "[#7;!n][C;!$(C(=[O,N])[N,O])][#8;!o]",
                "[#8;!o][C;!$(C(=[O,N])[N,O])][#16;!s]",
                "[#8;!o][C;!$(C(=[O,N])[N,O])][#8;!o]",
                "[#16;!s][C;!$(C(=[O,N])[N,O])][#16;!s]"]
    for smarts in warning_smarts:
        patt=Chem.MolFromSmarts(smarts)
        if rdkitmol.HasSubstructMatch(patt):
            print (f'Find warning structure {smarts} in {Chem.MolToSmiles(rdkitmol)}')
            return True
    return False

    

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
    clean_mol = rdMolStandardize.Cleanup(mol) 
     
    # if many fragments, get the "parent" (the actual mol we are interested in) 
    parent_clean_mol = rdMolStandardize.FragmentParent(clean_mol)
         
    # try to neutralize molecule
    uncharger = rdMolStandardize.Uncharger() # annoying, but necessary as no convenience method exists
    uncharged_parent_clean_mol = uncharger.uncharge(parent_clean_mol)
     
    # note that no attempt is made at reionization at this step
    # nor at ionization at some pH (rdkit has no pKa caculator)
    # the main aim to to represent all molecules from different sources
    # in a (single) standard way, for use in ML, catalogue, etc.
     
    te = rdMolStandardize.TautomerEnumerator() # idem
    taut_uncharged_parent_clean_mol = te.Canonicalize(uncharged_parent_clean_mol)
     
    return taut_uncharged_parent_clean_mol


def Prepare_mols_from_smi(smiles,savepath='./datasets/mols.smi'):
    with open(savepath,'w') as f:
        mols=[]
        for smi in tqdm(smiles):
            if smi:
                try:
                    mol=Chem.MolFromSmiles(smi)
                    mol=Neutralize_atoms(mol)
                    Chem.Kekulize(mol)
                    if mol:
                        flag=molfielter(mol)
                        if flag:
                            mols.append(mol)
                            f.write(Chem.MolToSmiles(mol)+'\n')
                except Exception as e:
                    print (smi,e)
    return mols 

def Prepare_mols_from_sdf(sdfname,smi_path):
    mols=[]
    supp=Chem.rdmolfiles.SDMolSupplier(sdfname)
    with open(f'{smi_path}','w') as f:
        print (f'Prepare mols from {sdfname}')
        for mol in tqdm(supp):
            try:
                mol=Neutralize_atoms(mol)
                Chem.Kekulize(mol)
                if mol:
                    flag=molfielter(mol)
                    if flag:
                        mols.append(mol)
                        f.write(Chem.MolToSmiles(mol)+'\n')
                    #else:
                        #print (Chem.MolToSmiles(mol)+' is not allowed')
            except Exception as e:
                print (e)
                pass
    return mols 

def Find_similar_molecules(target_smi,smis):
    target_mol=Chem.MolFromSmiles(target_smi)
    target_mol=Neutralize_atoms(target_mol)
    Chem.Kekulize(target_mol)
    collect_mols=[target_mol]
    for smi in tqdm(smis):
        mol=Chem.MolFromSmiles(smi)
        if mol:
            mol=Neutralize_atoms(mol)
            Chem.Kekulize(mol)
            simi=tanimoto_similarities(target_mol,mol)
            if simi>0.7:
                collect_mols.append(mol)
    return collect_mols

def Change_mol_xyz(rdkitmol,coords):
    molobj=copy.deepcopy(rdkitmol)
    conformer=molobj.GetConformer()
    id=conformer.GetId()
    for cid,xyz in enumerate(coords):
        conformer.SetAtomPosition(cid,Point3D(float(xyz[0]),float(xyz[1]),float(xyz[2])))
    conf_id=molobj.AddConformer(conformer)
    molobj.RemoveConformer(id)
    return molobj

def MolToXYZ(rdkitmol,xyzfile,mode='w'):
    atoms=[atom.GetAtomicNum() for atom in rdkitmol.GetAtoms()]
    coords=np.array(rdkitmol.GetConformer(0).GetPositions())
    natoms=len(atoms)
    with open(xyzfile,mode) as f:
        f.write(f'{natoms}\n')
        f.write('\n')
        for i in range(len(atoms)):
            f.write(f'{Table_Element[atoms[i]]}\t{coords[i][0]:.3F}\t{coords[i][1]:.3F}\t{coords[i][2]:.3F}\n')
    return 

def Drawmols(rdkitmol,filename='Mol.png',permindex=[],cliques=[]):
    reindex=np.zeros(len(permindex))
    #print (len(reindex),len(permindex))
    for pid,p in enumerate(permindex):
        reindex[p]=pid 
    #print (len(reindex),len(permindex))
    mol=copy.deepcopy(rdkitmol)
    
    Chem.rdDepictor.Compute2DCoords(mol)
    hatomlist=[]
    hbondlist=[]
    colors=[(1,1,0),(1,0,1),(1,0,0),(0,1,1),(0,1,0),(0,0,1)]
    if len(cliques)>0:
        for clique in cliques:
            if len(clique)>1:
                clique_bonds=[]
                hatomlist.append([int(a) for a in clique])
                for bond in mol.GetBonds():
                    a1=bond.GetBeginAtom().GetIdx()
                    a2=bond.GetEndAtom().GetIdx()
                    if a1 in clique and a2 in clique:
                        clique_bonds.append(bond.GetIdx())
                hbondlist.append(clique_bonds)
        atom_colors={}
        bond_colors={}
        atomlist=[]
        bondlist=[]
        for i,(hl_atom,hl_bond) in enumerate(zip(hatomlist,hbondlist)):
            #print (hl_atom,hl_bond)
            hl_atom=list(hl_atom)
            for at in hl_atom:
                atom_colors[at]=colors[i%6]
                atomlist.append(at)
            for bt in hl_bond:
                bond_colors[bt]=colors[i%6]
                bondlist.append(bt)

    options=rdMolDraw2D.MolDrawOptions()
    options.addAtomIndices=True
    draw=rdMolDraw2D.MolDraw2DCairo(500,500)
    for i in range(len(reindex)):
        #print (i,len(reindex),len(mol.GetAtoms()))
        mol.GetAtomWithIdx(i).SetProp("atomNote",':'+str(int(reindex[i])))
    draw.SetDrawOptions(options)
    #print (type(atomlist[0]),type(atom_colors),type(bondlist[0]),type(bond_colors))
    rdMolDraw2D.PrepareAndDrawMolecule(draw,mol,highlightAtoms=atomlist,
                                                highlightAtomColors=atom_colors,
                                                highlightBonds=bondlist,
                                                highlightBondColors=bond_colors)
    draw.FinishDrawing()
    draw.WriteDrawingText(filename)


def Gen_ETKDG_structures(rdkitmol,nums=1,basenum=50,mode='opt+lowest',withh=False,ifwrite=False,path='./mol'):
    mol=copy.deepcopy(rdkitmol)
    mol_h=Chem.AddHs(mol)
    confids=AllChem.EmbedMultipleConfs(mol_h,basenum)
    confs=[]
    energies=[]
    mollist=[]
    for cid,c in enumerate(confids):
        conformer=mol_h.GetConformer(c)
        tmpmol=Chem.Mol(mol_h)
        ff=AllChem.UFFGetMoleculeForceField(tmpmol)
        if 'opt' in mode:
            ff.Minimize()
        uffenergy=ff.CalcEnergy()
        energies.append(uffenergy)
        if not withh:
            tmpmol=Chem.RemoveHs(tmpmol) 
        optconf=tmpmol.GetConformer(0).GetPositions()
        confs.append(optconf)
        mollist.append(tmpmol)
    lowest_ids=np.argsort(energies)
    lowest_confs=[confs[i] for i in lowest_ids[:nums]]
    if ifwrite:
        for i in lowest_ids[:nums]:
            rdmolfiles.MolToMolFile(mollist[i],f'{path}.mol2')
    return lowest_confs

def SmilesToSVG(smiles,legends=None,fname='mol.svg'):
    mols=[]
    vlegends=[]
    for sid,smi in enumerate(smiles):
        mol=Chem.MolFromSmiles(smi)
        if mol:
            Chem.AllChem.Compute2DCoords(mol)
            mols.append(mol)
            if legends:
                vlegends.append(legends[sid])
    img=Draw.MolsToGridImage(mols,legends=vlegends,molsPerRow=5,subImgSize=(250,250),useSVG=True)
    with open (fname,'w') as f:
        f.write(img)
    return 

def Analysis_molecules_properties(smis):
    molwts=[]
    qeds=[]
    tpsas=[]
    logps=[]
    hbas=[]
    hbds=[]
    for smi in smis:
        if smi:
            mol=Chem.MolFromSmiles(smi)
            qed=QED.qed(mol)
            logp  = Descriptors.MolLogP(mol)
            tpsa  = Descriptors.TPSA(mol)
            molwt = Descriptors.ExactMolWt(mol)
            hba   = rdMolDescriptors.CalcNumHBA(mol)
            hbd   = rdMolDescriptors.CalcNumHBD(mol)
            molwts.append(molwts)
            qeds.append(qed)
            logps.append(logp)
            tpsas.append(tpsa)
            hbas.append(hba)
            hbds.append(hbd)
    return molwts,qeds,tpsas,logps,hbas,hbds

def group_descriptor_to_feats(group_descriptor):
    desvars=group_descriptor.split('-')
    Rnum=desvars[0][1:]
    C_num=desvars[1][1:]
    N_num=desvars[2][1:]
    O_num=desvars[3][1:]
    F_num=desvars[4][1:]
    P_num=desvars[5][1:]
    S_num=desvars[6][1:]
    Cl_num=desvars[7][2:]
    Br_num=desvars[8][2:]
    I_num=desvars[9][1:]
    Besnum=desvars[10][3:]
    ARnum=desvars[11][2:]
    HD_flag=desvars[12][2:]
    HA_flag=desvars[13][2:]
    Neg_flag=desvars[14][3:]
    Pos_flag=desvars[15][3:]
    Aro_flag=desvars[16][3:]
    Hyd_flag=desvars[17][3:]
    LHyd_flag=desvars[18][4:]
    element_num_dict={'C':C_num,'N':N_num,'O':O_num,'F':F_num,'P':P_num,'S':S_num,'Cl':Cl_num,'Br':Br_num,'I':I_num}
    pharm_flag_dict={'HD':HD_flag,'HA':HA_flag,'Neg':Neg_flag,'Pos':Pos_flag,'Aro':Aro_flag,'Hyd':Hyd_flag,'LHyd':LHyd_flag}
    return Rnum,Besnum,ARnum,element_num_dict,pharm_flag_dict

    
def mol_with_atom_index_and_formal_charge(mol):
    for atom in mol.GetAtoms():
        #atom.SetAtomMapNum(atom.GetIdx())
        atom.SetProp("atomNote",str(atom.GetFormalCharge()))
    return mol
    