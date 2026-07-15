import numpy as np 
#import tensorflow as tf 
from scipy  import spatial 
from multiprocessing import Pool 
from itertools import product 
import os 
import rdkit
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem, QED
from rdkit import Chem, DataStructs

import math 

fragmentdict={'COO':'C(O)=O','NHCO':'O=CN','ARG':'NC(N)=N','Methoxy':'C1=CN=CN1',\
              'ILE':'[C@H](CC)C[H]','PRO':'[C@@H]1CCCN1',\
              'TRP':'C1=CNC2=C1C=CC=C2','TYR':'C1=CC=C(O)C=C1','PHE':'C1=CC=CC=C1',\
              'LYS':'CN','MET':'CSC','CYS':'CS','OH':'CO'}
fragidxdict={}
for fid,i in enumerate(fragmentdict.keys()):
    fragidxdict[i]=(fid+2)
Massdict={1:1.0079,2:4.0026,3:6.941,4:9.012,5:10.811,6:12.011,7:14.007,8:15.999,9:18.998,10:20.17,\
        11:22.989,12:24.305,13:26.982,14:28.085,15:30.974,16:32.06,17:35.453,18:39.94,19:39.098,20:24.305,\
        21:44.956,22:47.9,23:50.9415,24:51.996,25:54.938,26:55.84,27:58.9332,28:58.69,29:63.54,30:65.38,31:69.72,32:72.5,33:74.922,34:78.9,35:79.904,36:83.8,53:126.905,}
PeriodTable={'H':1,'He':2,'Li':3,'Be':4,'B':5,'C':6,'N':7,'O':8,'F':9,'Ne':10,'Na':11,'Mg':12,'Al':13,'Si':14,'P':15,'S':16,'Cl':17,'Ar':18,'K':19,'Ca':20,'Sc':21,'Mn':25,'Fe':26\
        ,'Co':27,'Ni':28,'Cu':29,'Zn':30,'Ga':31,'Ge':32,'As':33,'Se':34,'Br':35,'Kr':36,'I':53,'CL':17,'Au':79,'Ag':47,'Cd':48,'Pt':78,'Hg':80}
keylist=list(PeriodTable.keys())
for key in keylist:
    PeriodTable[key.upper()]=PeriodTable[key]
    PeriodTable[key.capitalize()]=PeriodTable[key]

class molecule():
    def __init__(self,**kwargs):
        self.atoms=[]
        self.coord=[]
        self.atomnamelist=[]
        self.atomidlist=[]
        self.atomresid=[]
        self.atomresname=[]
        self.atomtype=[]
        self.name='mol'
        self.pdbmarkstring=' '
        self.connectstring=[]
        if "pdbname" in kwargs:
            self.name=kwargs.get("pdbname").strip('.pdb')
            self.parse_pdb(kwargs.get("pdbname"))
        if "name" in kwargs:
            self.name=kwargs.get("name")
        if "atomlist" in kwargs:
            self.atoms=kwargs.get("atomlist")
        if "coord" in kwargs:
            self.coord=kwargs.get("coord")
        if "pdbstr" in kwargs:
            self.parse_pdbstr(kwargs.get("pdbstr"))

    def parse_pdb(self,filename):
        self.pdbstring=[]
        residlist=[]
        resnamelist=[]
        altloclist=[]
        atomlist=[]
        #print ("----For initialize molecules!")
        with open(filename,'r') as pdbfile:
            for eachline in pdbfile:
                if ('ATOM' in eachline or "HETATM" in eachline) and self.pdbmarkstring in eachline:
                    atom=parse_pdb_atomstring(eachline)
                    atomlist.append(atom)
                    residlist.append(atom["resseq"])
                    resnamelist.append(atom["resname"])
                    altloclist.append(atom["altloc"])
                    self.pdbstring.append(eachline)
                elif 'CONECT' in eachline:
                    self.connectstring.append(eachline)
            if len(set(altloclist))>1:
                saveid=[i for i in range(len(altloclist)) if altloclist[i]=='' or altloclist[i]=='A']
            else:
                saveid=[i for i in range(len(altloclist))]
            self.atoms=[atomlist[i]["element"] for i in range(len(atomlist)) if i in saveid]
            self.coord=[atomlist[i]["xyz"] for i in range(len(atomlist)) if i in saveid]
            self.atomresid=[residlist[i] for i in range(len(atomlist)) if i in saveid]
            self.atomresname=[resnamelist[i] for i in range(len(resnamelist)) if i in saveid]
            self.atomnamelist=[atomlist[i]["name"] for i in range(len(atomlist)) if i in saveid]
            self.atomidlist=[atomlist[i]["serial"] for i in range(len(atomlist)) if i in saveid]
            self.pdbstring=[self.pdbstring[i] for i in range(len(self.pdbstring)) if i in saveid] 
            
    def parse_pdbstr(self,pdbstr):
        self.pdbstring=[]
        atomlist=[]
        altloclist=[]
        residlist=[]
        resnamelist=[]
        strlist=pdbstr.split('\n')[:-1]
        for eachline in strlist:
            if 'ATOM' in eachline or "HETATM" in eachline and self.pdbmarkstring in eachline:
                atom=parse_pdb_atomstring(eachline)
                altloclist.append(atom["altloc"])
                atomlist.append(atom)
                residlist.append(atom["resseq"])
                resnamelist.append(atom["resname"])
                self.pdbstring.append(eachline+'\n')
            elif 'CONECT' in eachline:
                self.connectstring.append(eachline+'\n')
        if len(set(altloclist))>1:
            saveid=[i for i in range(len(altloclist)) if altloclist[i]=='' or altloclist[i]=='A']
        else:
            saveid=[i for i in range(len(altloclist))]
        self.atoms=[atomlist[i]["element"] for i in range(len(atomlist)) if i in saveid]
        self.coord=[atomlist[i]["xyz"] for i in range(len(atomlist)) if i in saveid]
        self.pdbstring=[self.pdbstring[i] for i in range(len(self.pdbstring)) if i in saveid]
        self.atomresid=[residlist[i] for i in range(len(atomlist)) if i in saveid]
        self.atomresname=[resnamelist[i] for i in range(len(resnamelist)) if i in saveid]
        self.atomnamelist=[atomlist[i]["name"] for i in range(len(atomlist)) if i in saveid]
        self.atomidlist=[atomlist[i]["serial"] for i in range(len(atomlist)) if i in saveid]

    def distance(self,mol):
        coords1=np.array(self.coord)
        coords2=np.array(mol.coord)
        dismat=spatial.distance_matrix(coords1,coords2)
        id=np.unravel_index(np.argmin(dismat),dismat.shape)
        dis=dismat[id]
        return dis,id

    def internal_distance_matrix(self):
        coords1=np.array(self.coord)
        coords2=np.array(self.coord)
        self.inter_dismat=spatial.distance_matrix(coords1,coords2)
        return 

    def get_atomtype(self,tmppath=''):
        if tmppath!='':
            if tmppath[-1]!='/':
                tmppath+='/'
        if len(self.connectstring)==0:
            print ("Warning: transform the PDB file without connect information to Mol2file may cause some problems!")
        else:
            with open(tmppath+'%s.pdb'%self.name,'w') as f:
                f.write(''.join(self.pdbstring))
                f.write(''.join(self.connectstring))
                
        os.system('structconvert -ipdb %s%s.pdb -omol2 %s%s.mol2>/dev/null'%(tmppath,self.name,tmppath,self.name))
        with open(tmppath+'%s.mol2'%self.name,'r') as f:
            line=f.readline()
            while 'ATOM' not in line and line:
                line=f.readline()
            line=f.readline()
            while not '<TRIPOS>BOND' in line and line:
                self.atomtype.append(line[47:53].strip())
                line=f.readline()
        """
        rdkitmol=Chem.MolFromMol2File(tmppath+'%s.mol2'%self.name)
        self.rdkitsmiles=Chem.MolToSmiles(rdkitmol) 
        print (self.name,self.rdkitsmiles)
        """
        return 
            
    def create_channel(self):
        elementchannel=list(set(self.atoms))
        atomtypechannel=list(set(self.atomtype))
        channel=elementchannel+atomtypechannel
        self.channel={}
        self.channel["T"]=[i for i in range(len(self.atoms)) if self.atoms[i]!='H'] 
        for element in elementchannel:
            self.channel[element]=[i for i in range(len(self.atoms)) if self.atoms[i] == element]
        for atomtype in atomtypechannel:
            self.channel[atomtype]=np.array([i for i in range(len(self.atomtype)) if self.atomtype[i] == atomtype])
        
    def topdbstr(self):
        pdbstr=''.join(self.pdbstring)
        pdbstr+=''.join(self.connectstring)
        return pdbstr

    def writepdb(self,pdbname):
        with open(pdbname,'w') as pdbfile:
            pdbfile.write(self.topdbstr())
        return     

    def minimize(self):
        self.pdbstring=[]

    def cal_masscenter(self):
        self.masslist=[Massdict[PeriodTable[i]] for i in self.atoms]
        self.masscenter=np.sum(np.array(self.coord)*np.array(self.masslist).reshape(-1,1),axis=0)/np.sum(self.masslist)
        return 
        

class ligand(molecule):
    def __init__(self,**kwargs):
        super(ligand,self).__init__()
        self.atomtype=[]
        self.charge=[]
        self.name="mol"
        if "pdbmarkstring" in kwargs:
            self.pdbmarkstring=kwargs.get("pdbmarkstring")
        if "pdbname" in kwargs:
            self.name=kwargs.get("pdbname").strip('pdb')[:-1]
            self.parse_pdb(kwargs.get("pdbname"))
        if "name" in kwargs:
            self.name=kwargs.get("name")
        if "pdbstr" in kwargs:
            self.parse_pdbstr(kwargs.get("pdbstr"))
            
class residue(molecule):
    def __init__(self,**kwargs):
        super(residue,self).__init__()
        self.resid=0
        self.resname='RES'
        self.Nid=0
        self.Cid=0
        self.atomidlist=[]
        self.atomnamelist=[]
        self.pt=[]
        self.name="mol"
        if "pdbstr" in kwargs:
            self.parse_pdbstr(kwargs.get("pdbstr"))
    def parse_pdbstr(self,pdbstr):
        self.pdbstring=[]
        residlist=[];resnamelist=[];altloclist=[]
        atomlist=[]
        for id,eachline in enumerate(pdbstr.split('\n')[:-1]):
            atom=parse_pdb_atomstring(eachline)
            atomlist.append(atom)
            #self.atoms.append(atom["element"])
            #self.coord.append(atom["xyz"])
            altloclist.append(atom["altloc"])
            residlist.append(atom["resseq"])
            resnamelist.append(atom["resname"])
        #    if atom["name"]=='  N ':
        #        self.Nid==id
        #    elif atom["name"]=='  C ':
        #        self.Cid==id
        #    if id==0:
        #        self.pt.append(atom["serial"])
            #self.atomnamelist.append(atom["name"])
            #self.atomidlist.append(atom["serial"])
            self.pdbstring.append(eachline+'\n')
        if len(set(altloclist))>1:
            print ("NMR")
            saveid=[i for i in range(len(altloclist)) if altloclist[i]=='' or altloclist[i]=='A']
        else:
            saveid=[i for i in range(len(altloclist)) ]

        self.atoms=[atomlist[i]["element"] for i in range(len(atomlist)) if i in saveid]
        self.coord=[atomlist[i]["xyz"] for i in range(len(atomlist)) if i in saveid]
        residlist=[residlist[i] for i in range(len(atomlist)) if i in saveid]
        resnamelist=[resnamelist[i] for i in range(len(resnamelist)) if i in saveid]
        self.atomnamelist=[atomlist[i]["name"] for i in range(len(atomlist)) if i in saveid]
        self.atomidlist=[atomlist[i]["serial"] for i in range(len(atomlist)) if i in saveid]
        self.pdbstring=[self.pdbstring[i] for i in range(len(self.pdbstring)) if i in saveid]
            
        for aid,atomname in enumerate(self.atomnamelist):
            if atomname=='  N ':
                self.Nid=aid
            elif atomname=='  C ':
                self.Cid==aid
        
        self.pt=[atomlist[0]["serial"],atomlist[-1]["serial"]]
        self.atomlist=[]
        self.resid=list(set(residlist))[0]
        self.resname=list(set(resnamelist))[0]
        self.atomresid=[self.resid for atom in self.atomidlist]
        self.atomresname=[self.resname for atom in self.atomidlist]

Standardresidue=['ALA','GLY','PRO','GLN','ASP','ARG','LYS','ILE','VAL','PHE','MET','CYS','HIE','LEU','TRP','TYR','SER','ASN','GLU','THR','HIS','HID']
class protein(molecule):
    def __init__(self,**kwargs) :
        super(protein,self).__init__() 
        self.reslist=[]
        self.respt=[]
        self.chainpt=[]
        self.pdbstring=[]
        self.atomnamelist=[]
        self.atomidlist=[]
        self.atomresname=[]
        self.atomresid=[]
        self.atomtype=[]
        self.name="mol"
        if "pdbname" in kwargs:
            self.name=kwargs.get("pdbname").strip('.pdb')
            self.parse_pdb(kwargs.get("pdbname"))
        if "name" in kwargs:
            self.name=kwargs.get("name")
        if "pdbstr" in kwargs:
            self.parse_pdbstr(kwargs.get("pdbstr"))

    def parse_pdb(self,filename):
        self.pdbstring=[]
        tmpresiduestr='';residlist=[]
        with open(filename,'r') as pdbfile:
            for id,eachline in enumerate(pdbfile):

                if ('ATOM' in eachline or "HETATM" in eachline): #and eachline[17:20] in Standardresidue:
                    if eachline[17:26] not in residlist and tmpresiduestr!='' :
                        res=residue(pdbstr=tmpresiduestr)
                        tmpresiduestr=''
                        self.reslist.append(res)
                    tmpresiduestr+=eachline
                    residlist.append(eachline[17:26])
                elif 'CONECT' in eachline:
                    self.connectstring.append(eachline+'\n')
                        
            res=residue(pdbstr=tmpresiduestr)
            tmpresiduestr=''
            self.reslist.append(res)
        for i in range(len(self.reslist)):
            self.atoms+=self.reslist[i].atoms
            self.coord+=self.reslist[i].coord
            self.atomidlist+=self.reslist[i].atomidlist
            self.atomnamelist+=self.reslist[i].atomnamelist
            self.atomresname+=self.reslist[i].atomresname
            self.atomresid+=self.reslist[i].atomresid
            self.respt.append(self.reslist[i].pt)
    def parse_pdbstr(self,pdbstr):
        self.pdbstring=[]
        tmpresiduestr='';residlist=[]

        for id,eachline in enumerate(pdbstr.split('\n')[:-1]):
            if ('ATOM' in eachline or "HETATM" in eachline):# and eachline[17:20] in Standardresidue:
                eachline+='\n'
                if eachline[17:26] not in residlist and tmpresiduestr!='' :
                    res=residue(pdbstr=tmpresiduestr)
                    tmpresiduestr=''
                    self.reslist.append(res)
                tmpresiduestr+=eachline
                residlist.append(eachline[17:26])
                self.pdbstring.append(eachline)
            elif 'CONECT' in eachline:
                self.connectstring.append(eachline+'\n')
                    
        res=residue(pdbstr=tmpresiduestr)
        tmpresiduestr=''
        self.reslist.append(res)
        for i in range(len(self.reslist)):
            self.atoms+=self.reslist[i].atoms
            self.coord+=self.reslist[i].coord
            self.atomidlist+=self.reslist[i].atomidlist
            self.atomnamelist+=self.reslist[i].atomnamelist
            self.respt.append(self.reslist[i].pt)
            self.atomresname+=self.reslist[i].atomresname
            self.atomresid+=self.reslist[i].atomresid 

    def distance(self,mol,nproc=2,info=0):
        #p=Pool(nproc)
        #disinfo=p.map(mol_distance,list(product([mol],self.reslist)))           
        disinfo=[]
        for res in self.reslist:
            disinfo.append(mol_distance((mol,res)))
        disarray=np.array([dis[0] for dis in disinfo])
        disatompair=np.array([dis[1] for dis in disinfo])
        p2ldis=np.min(disarray)
        p2ldisid=np.argmin(disarray)         
        if info==0:
            return p2ldis,p2ldisid
        elif info==1:
            return p2ldis,p2ldisid,disinfo

    def calculate_surface_atoms(self,ifH=False,path='.'):
        if ifH:
            pstr=''.join(self.pdbstring)
        else:
            pstr=''.join([pdbstr for pdbstr in self.pdbstring if '    H ' not in pdbstr])
        with open(path+'/%s_surface.pdb'%self.name,'w')  as f:
             f.write(pstr)
        os.system('pdb_to_xyzrn %s/%s_surface.pdb > %s/%s_surface.xyzrn'%(path,self.name,path,self.name))
        os.system('msms -if %s/prt_surface.xyzrn -of %s/prt_surface > /dev/null'%(path,path))
        #print (self.atomidlist[0],self.atomnamelist[0],self.atomresid[0],self.atomresname[0])
        atommarks=[self.atomnamelist[i]+'_'+self.atomresname[i]+'_'+str(self.atomresid[i]) for i in range(len(self.atoms))]
        surfaceatoms=[]
        with open (path+'/prt_surface.vert','r') as f:
            for line in f.readlines():
                var=line.split()[-1]
                surfaceatoms.append(var)
        surfaceatoms=list(set(surfaceatoms))
        
        #print (surfaceatoms,len(surfaceatoms))
        self.surfaceatoms=[]
        for aid,atommark in enumerate(atommarks):
            if atommark in surfaceatoms:
                self.surfaceatoms.append(1)
            else:
                self.surfaceatoms.append(0)
        return 
    
    def grep_neighbor_residue(self,lig,cutoff=6.5):
        #disarray=np.min(disarray[:][0])
        p2ldis,p2lid,p2ldisinfo=self.distance(lig,nproc=4,info=1)
        disarray=np.array([dis[0] for dis in p2ldisinfo])
        greplist=[]
        grepidlist=[]
        for id,res in enumerate(self.reslist):
            if disarray[id]<=cutoff:
                greplist.append(res)
                grepidlist.append(id)    
        return greplist,grepidlist
        
    def writepdb(self,**kwargs):
        if "reslist" in kwargs:
            reslist=kwargs.get("reslist")
        elif "residlist" in kwargs:
            reslist=self.reslist[kwargs.get("residlist")]
        else:
            reslist=self.reslist
        if "pdbname" in kwargs:
            pdbname=kwargs.get("pdbname") 
        with open(pdbname,'w') as pdbf:
            for res in reslist:
                respdb=res.topdbstr()
                pdbf.write(respdb)
            pdbf.write(''.join(self.connectstring))
        return             

    def minimize(self):
        for res in self.reslist:
            res.minimize()
            
class complexpdb(dict):
    def __init__(self,**kwargs):
        super(complexpdb,self).__init__()
        if "fname"  in kwargs:
            self["name"]=kwargs.get("fname").strip('.pdb')
            self["pdbstring"]=[]
            self["pstr"]=[]
            self["lstr"]=[]
            self["conect"]=[]
            self.parse_pdb(kwargs.get("fname"))
    def parse_pdb(self,filename):
        with open(filename,'r') as pdbfile:
            for eachline in pdbfile.readlines():
                if "REMARK <Ligand SMILES>" in eachline:
                    var=eachline.split()
                    self["ligsmiles"]=var[3]
                    self["ligname"]=var[-1]
                if "REMARK eThread-Template:" in eachline:
                    var=eachline.split()
                    self["template"]=var[2]
                    self["seqidentity"]=var[4]
                if "REMARK Holo" in eachline:
                    var=eachline.split()
                    self["Holoptemplate"]=var[2]
                    self["tmscore"]=var[-1]
                if "ATOM" in eachline or "HETATM" in eachline:
                    self["pdbstring"].append(eachline)
                if "CONECT" in eachline:
                    self["conect"].append(eachline)
        self['lstr']=[];ligindexlist=[]

        for line in self["pdbstring"]:
            if line[:6]=="ATOM  ":
                self["pstr"].append(line)
            elif line[:6]=="HETATM":
                if line[22:26] not in ligindexlist:
                    ligindexlist.append(line[22:26])
                    self['lstr'].append([])
                self['lstr'][-1].append(line)
        
    def split_protein_ligand(self,**kwargs):
        if 'path' in kwargs:
            path=kwargs.get('path')
            if path[-1]!='/':
                path+='/'
        else:
            path='./'
        if 'ligmark' in kwargs:
            ligmark=kwargs.get('ligmark')
            ligids=[]
            for lid,lstr in enumerate(self['lstr']):
                if ligmark in lstr[0]:
                    ligids.append(lid)
        
        if 'ligids' in kwargs:
            ligids=[]
            tmpligids=kwargs.get('ligids')
            for lid in tmpligids:
                if lid <0:
                    ligids.append(len(self["lstr"])+lid)
        lstr=''
        for lid in ligids:
            lstr+=(''.join(self['lstr'][lid]))
            #print (lstr)
        for lid in ligids:
            pstr=''.join(self["pstr"])
            for id,tmplstr in enumerate(self["lstr"]):
                if id not in ligids:
                    pstr+=''.join(tmplstr)
        #print (lstr) 
        latomid=[]
        for lid in ligids:
            latomid+=[int(line[6:11].strip()) for line in (self['lstr'][lid])]
        ligconnectstr=''
        prtconnectstr=''
        for connectstr in self["conect"]:
            idlist=[]
            idlist.append(connectstr[6:11].strip())
            idlist.append(connectstr[11:16].strip())
            idlist.append(connectstr[16:21].strip())
            idlist.append(connectstr[21:26].strip())
            idlist.append(connectstr[26:31].strip())
            idlist=[int(atomid) for atomid in idlist if atomid !=''] 
            flag=False 
            for atomid in idlist:
                if atomid in latomid:
                    flag=True
            if flag:
                ligconnectstr+=connectstr
            else:
                prtconnectstr+=connectstr
        self["ligand"]=ligand(pdbstr=lstr+ligconnectstr,name='lig')
        self["protein"]=protein(pdbstr=pstr+prtconnectstr,name='prt')
        #with open(path+'lig.pdb','w') as f:
        #    f.write(lstr)
        #    f.write(ligconnectstr)
        #with open(path+'prt.pdb','w') as f:
        #    f.write(pstr)
        #    f.write(prtconnectstr)
        self["prtconnectstr"]=prtconnectstr
        self["ligconnectstr"]=ligconnectstr
        return 
    def grep_cativity(self,**kwargs): 
        if "path" in kwargs:
            path=kwargs.get("path")
            if path[-1]!='/':
                path+='/'
        else: 
            path='./'
        if "protein" not in self.keys() or "ligand" not in self.keys():
            print ("Please perform split_protein_ligand firstly!")
        else:
            self["cativity"],_=self["protein"].grep_neighbor_residue(self["ligand"])
            os.system('mkdir -p %s'%path)
            if path!='./':
                self["protein"].writepdb(reslist=self["cativity"],pdbname=self["name"]+'/_site.pdb')

    def minimize(self):
        self["ligand"].minimize()
        #self["protein"].minimize()
        self["protein"]=None
        self["ligconnectstr"]=''
        self["prtconnectstr"]=''
        self["conect"]=''
        for res in self["cativity"]:
            res.minimize()
        self["lstr"]=''
        self["pstr"]=''
        self["cativity"]=None
    def transtordkitmol(self):
        try:
            obabelcmd=os.popen("cd %s && obabel -i pdb _ligand.pdb -o mol2 -O _ligand.mol2 && obabel -i pdb _site.pdb -o mol2 -O _site.mol2 &&cd -"%(self["name"].strip('.pdb')))
            #print (obabelcmd)
            self.obabelflag=True
            self["binarylig"]=rdmolfiles.MolFromMol2File(self["name"].strip('.pdb')+'_ligand.mol2')
            self["binarysite"]=rdmolfiles.MolFromMol2File(self["name"].strip('.pdb')+'_site.mol2')
            self["rdkitsmiles"]=Chem.MoltoSmiles(self["binarylig"])
            
        except Exception:
            self.obabelflag=False
            print ('{} transfer from pdb to rdkit mol2 failed!'.format(self["name"]))
      
    def pcb_properties(self,qsar_model=None,show_actives=False,active_thresh=0.5,qed_thresh=0.5):
        mol =self["binarylig"]
        if mol:
            try:
                logp  = Descriptors.MolLogP(mol)
                tpsa  = Descriptors.TPSA(mol)
                molwt = Descriptors.ExactMolWt(mol)
                hba   = rdMolDescriptors.CalcNumHBA(mol)
                hbd   = rdMolDescriptors.CalcNumHBD(mol)
                qed   = QED.qed(mol)
                # Calculate fingerprints
                fp = AllChem.GetMorganFingerprintAsBitVect(mol,2, nBits=2048)
                ecfp4 = np.zeros((2048,))
                DataStructs.ConvertToNumpyArray(fp, ecfp4)
                # Predict activity and pick only the second component
                active = qsar_model.predict_proba([ecfp4])[0][1]
                descriptors=np.array([logp, tpsa, molwt, qed, hba, hbd, active])
                self["PCBdescriptor"]=descriptors
            except Exception as e:
                print (e)
                self["PCBdescriptor"] =None
        else:
            print ("Invalid generation for complex :%s"%(self["name"].strip('.pdb')))
        return       
    
def mol_distance(moltruple):
    mol1,mol2=moltruple
    distance=mol1.distance(mol2)
    return distance
def parse_pdb_atomstring(atomstring):
    strlen=len(atomstring)
    pdbatom={}
    if strlen >0:
        pdbatom["serial"]=int(atomstring[6:11])
        pdbatom["name"]=atomstring[12:16].strip()
        pdbatom["altloc"]=atomstring[16].strip()
        pdbatom["resname"]=atomstring[17:20].strip()
        pdbatom["chainid"]=atomstring[21].strip()
        pdbatom["resseq"]=int(atomstring[22:26])
        pdbatom["xyz"]=np.array([float(atomstring[30:38]),float(atomstring[38:46]),float(atomstring[46:54])])
        pdbatom["occupancy"]=float(atomstring[54:60])
        pdbatom["tempfactor"]=float(atomstring[60:66])
        pdbatom["element"]=atomstring[76:78].strip()
        pdbatom["charge"]=atomstring[78:80].strip()
        return  pdbatom
    else:
        print ("Error: Its not a valid pdb atom string!")
        return None

def get_pcb_properties(smiles, qsar_model=None):
    from tqdm import tqdm_notebook as tqdm
    import rdkit
    from rdkit import Chem, DataStructs
    from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem, QED
    descriptors = []
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        try:
            logp  = Descriptors.MolLogP(mol)
            tpsa  = Descriptors.TPSA(mol)
            molwt = Descriptors.ExactMolWt(mol)
            hba   = rdMolDescriptors.CalcNumHBA(mol)
            hbd   = rdMolDescriptors.CalcNumHBD(mol)
            qed   = QED.qed(mol)
            # Calculate fingerprints
            fp = AllChem.GetMorganFingerprintAsBitVect(mol,2, nBits=2048)
            ecfp4 = np.zeros((2048,))
            DataStructs.ConvertToNumpyArray(fp, ecfp4)
            # Predict activity and pick only the second component
            if qsar_model:
                active = qsar_model.predict_proba([ecfp4])[0][1]
                descriptors=[logp, tpsa, molwt, qed, hba, hbd, active]
            else:
                descriptors=[logp, tpsa, molwt, qed, hba, hbd]
        except Exception as e:
            print (e)
    else:
        print("Invalid smiles")
    return np.asarray(descriptors)

def write_xyz(filename,atoms,coords):
    with open(filename,'w') as f:
        f.write(f'{len(atoms)}\n')
        f.write('\n')
        for i in range(len(atoms)):
            f.write(f'{atoms[i]} {coords[i][0]} {coords[i][1]} {coords[i][2]}\n')

