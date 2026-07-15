#from plip.structure.preparation import PDBComplex
from pprint import pprint 
import pickle 
import numpy as np 
def get_atom_id_in_coords(xyz,coords):
    dismat=np.sqrt(np.sum((coords-xyz.reshape(-1,3))**2,axis=-1))
    min_dis=np.min(dismat)
    xyz_id=np.argmin(dismat)
    #print (min_dis,xyz_id)
    if min_dis>0.1:
        print ('Atom Mismatch in Interaction detections')
    return xyz_id,min_dis
def detect_interactions(Complex_PDBFile,coords):
    my_mol = PDBComplex()
    my_mol.load_pdb(Complex_PDBFile) 
    my_bsid=(':').join([str(i) for i in my_mol.ligands[-1].members[-1]])

    my_mol.analyze()
    my_interactions = my_mol.interaction_sets[my_bsid]

    Interaction={}
    Interaction["name"]=Complex_PDBFile 
    Interaction["pistacking"]=[]
    Interaction["hbond_ldon"]=[]
    Interaction["hbond_lacc"]=[]
    Interaction["hydrophobic"]=[]
    Interaction["pication_lpi"]=[]
    Interaction["pication_lcation"]=[]
    Interaction["halogen_bonds_lacc"]=[]
    Interaction["halogen_bonds_ldon"]=[]
    Interaction["saltbridge_lpos"]=[]
    Interaction["saltbridge_lneg"]=[]
    Interaction["waterbridge_lacc"]=[]
    Interaction["waterbridge_ldon"]=[]
    Interaction["unpaired_halogen"]=[(my_interactions.unpaired_hal_orig_idx,[],10)]
    Interaction["unpaired_hbond_acc"]=[(my_interactions.unpaired_hba_orig_idx,[],10)]
    Interaction["unpaired_hbond_don"]=[(my_interactions.unpaired_hbd_orig_idx,[],10)]
    
    for pistack in my_interactions.pistacking:
        #Interaction['pistacking'].append((pistack.proteinring.atoms_orig_idx,pistack.ligandring.atoms_orig_idx,pistack.distance))
        ring1_atom_coords=[np.array(atom.coords) for atom in pistack.proteinring.atoms]
        r1atom_ids=[]
        for rcrd in ring1_atom_coords:
            rid,_=get_atom_id_in_coords(rcrd,coords)
            r1atom_ids.append(rid)
        ring2_atom_coords=[np.array(atom.coords) for atom in pistack.ligandring.atoms]
        r2atom_ids=[]
        for rcrd in ring2_atom_coords:
            rid,_=get_atom_id_in_coords(rcrd,coords)
            r2atom_ids.append(rid)
        Interaction['pistacking'].append((r1atom_ids,r2atom_ids,pistack.distance))
        
        
    for hbond in my_interactions.all_hbonds_ldon:
        acrd=np.array(hbond.a.coords)
        dcrd=np.array(hbond.d.coords)
        aid,err_a=get_atom_id_in_coords(acrd,coords)
        did,err_d=get_atom_id_in_coords(dcrd,coords)
        if err_a>0.1 or err_d>0.1:
            print ('Atom Mismatch in HBOND Interaction detections')
        else:
            Interaction['hbond_ldon'].append(([aid],[did],hbond.distance_ad))

    for hbond in my_interactions.all_hbonds_pdon:
        acrd=np.array(hbond.a.coords)
        dcrd=np.array(hbond.d.coords)
        aid,err_a=get_atom_id_in_coords(acrd,coords)
        did,err_d=get_atom_id_in_coords(dcrd,coords)
        Interaction['hbond_lacc'].append(([aid],[did],hbond.distance_ad))
    
    for hydrophobic in my_interactions.all_hydrophobic_contacts:
        bsatom_crd=np.array(hydrophobic.bsatom.coords)
        ligatom_crd=np.array(hydrophobic.ligatom.coords)
        bsatom_id,err_bsatom=get_atom_id_in_coords(bsatom_crd,coords)
        ligatom_id,err_ligatom=get_atom_id_in_coords(ligatom_crd,coords)

        Interaction["hydrophobic"].append(([bsatom_id],[ligatom_id],
                                            hydrophobic.distance))
    
    for pication in my_interactions.all_pi_cation_laro:
        ring_atom_coords=[np.array(atom.coords) for atom in pication.ring.atoms]
        ratom_ids=[]
        for rcrd in ring_atom_coords:
            rid,_=get_atom_id_in_coords(rcrd,coords)
            ratom_ids.append(rid)

        charge_atom_coords=[np.array(atom.coords) for atom in pication.charge.atoms]
        catom_ids=[]
        for ccrd in charge_atom_coords:
            cid,_=get_atom_id_in_coords(ccrd,coords)
            catom_ids.append(cid) 

        if pication.ring.atoms_orig_idx[0]<pication.charge.atoms_orig_idx[0]:
            Interaction["pication_lcation"].append((ratom_ids,catom_ids,pication.distance))
        else:
            Interaction["pication_lpi"].append((catom_ids,ratom_ids,pication.distance))


    for halogen in my_interactions.halogen_bonds:
        #print (halogen)
        a_o_crd=np.array(halogen.acc.o.coords)
        a_y_crd=np.array(halogen.acc.y.coords)
        d_x_crd=np.array(halogen.don.x.coords)
        d_c_crd=np.array(halogen.don.c.coords)
        a_o_id,err_=get_atom_id_in_coords(a_o_crd,coords)
        a_y_id,err_=get_atom_id_in_coords(a_y_crd,coords)

        d_x_id,err_=get_atom_id_in_coords(d_x_crd,coords)
        d_c_id,err_=get_atom_id_in_coords(d_c_crd,coords)
        if a_o_id<d_x_id:
            Interaction["halogen_bonds_ldon"].append(([a_o_id,a_y_id],[d_x_id,d_c_id],halogen.distance))
        else:
            Interaction["halogen_bonds_lacc"].append(([d_x_id,d_c_id],[a_o_id,a_y_id],halogen.distance))

    for saltbrige in my_interactions.saltbridge_lneg:

        patom_coords=[np.array(atom.coords) for atom in saltbrige.positive.atoms]
        patom_ids=[]
        for pcrd in patom_coords:
            pid,_=get_atom_id_in_coords(pcrd,coords)
            patom_ids.append(pid)

        natom_coords=[np.array(atom.coords) for atom in saltbrige.negative.atoms]
        natom_ids=[]
        for ncrd in natom_coords:
            nid,_=get_atom_id_in_coords(ncrd,coords)
            natom_ids.append(nid) 

        Interaction["saltbridge_lneg"].append((patom_ids,natom_ids,saltbrige.distance))

    for saltbrige in my_interactions.saltbridge_pneg:

        patom_coords=[np.array(atom.coords) for atom in saltbrige.positive.atoms]
        patom_ids=[]
        for pcrd in patom_coords:
            pid,_=get_atom_id_in_coords(pcrd,coords)
            patom_ids.append(pid)

        natom_coords=[np.array(atom.coords) for atom in saltbrige.negative.atoms]
        natom_ids=[]
        for ncrd in natom_coords:
            nid,_=get_atom_id_in_coords(ncrd,coords)
            natom_ids.append(nid) 

        Interaction["saltbridge_lpos"].append((natom_ids,patom_ids,saltbrige.distance))


    for waterbridge in my_interactions.water_bridges:
        #print (waterbridge)
        acrd=np.array(waterbridge.a.coords)
        dcrd=np.array(waterbridge.d.coords)
        aid,err_a=get_atom_id_in_coords(acrd,coords)
        did,err_d=get_atom_id_in_coords(dcrd,coords)
        
        if aid<did:
            Interaction["waterbridge_ldon"].append(([aid],[did],[waterbrige.water_orig_idx],waterbrige.distance_aw,waterbrige.distance_dw))
        else :
            Interaction["waterbridge_lacc"].append(([aid],[did],[waterbrige.water_orig_idx],waterbrige.distance_dw,waterbrige.distance_aw))

    return Interaction
    

