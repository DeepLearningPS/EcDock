import re
import gzip 
from pprint import PrettyPrinter 
pp=PrettyPrinter(depth=5,compact=True,indent=4)
def parse_dictstring(strings):
    slist=strings.split(';')
    sdict={}
    for s in slist:
        keyvalue=re.split(r': ',s.strip())
        key=keyvalue[0]
        value=re.split(r', ',keyvalue[1].strip()+' ')
        if key in sdict.keys():
            sdict[key].append([i.strip() for i in value])
        else:
            sdict[key]=[[i.strip() for i in value]]
    return sdict

class rscbpdb(dict):
    def __init__(self,filepath):
        super(rscbpdb,self).__init__()
        self["SITE"]={}
        self["DBREF"]=[]
        self['SEQRES']={}
        self['HET']={}
        self['NUMMDL']=1
        self['MODELS']=[]
        self["HETNAM"]={}
        self["HEADER CLASS"]=''
        self['DATE']=''
        self['PDB ID']=''
        self['CAVEAT']=False
        self['COMPND']={}
        self["LINK"]=[]
        self["SPLIT"]=[]
        self["KEYWDS"]=[]
        self["EXPDTA"]=[]
        self["RESOLUTION"]=0
        self["MISSING RESIDUES"]=[]
        self["MISSING ATOMS"]=[]
        self["CLOSE CONTACTS"]=[]
        self["METAL COORDINATION"]=[]
        self["HELIX"]=[]
        self["SHEET"]={}
        self["MODRES"]={}
        self["MODRES LIST"]=[]
        self["METAL LIST"]=[]
        self["NAME"]='PDB'
        self.parse_pdb(filepath)
    def parse_pdb(self,fpath):
        if "gzip" in fpath:
            ftype = "gzip"
        else:
            ftype="txt"

        if ftype=="gzip":
            with gzip.open(fpath,'rt',encoding='utf-8') as f:
                self.parse_pdb_file(f)
        else:
            with open(fpath,'r') as f:
                self.parse_pdb_file(f)
        return 
    def parse_pdb_file(self,file_handle):
        splitpdbids=[];compnd='';keywds='';expdat='';
        f=file_handle
        line=f.readline()
        while line:
            if line[:6]=='HEADER':
                self['HEADER CLASS']=line[10:50]
                self['DATE']=line[50:59]
                self['PDB ID']=line[62:66]
                self["NAME"]=self['PDB ID'].strip()
            if line[:6] =="SPLIT ":
                pdbids=line[6:].split()
                splitpdbids.append(pdbids)
            if line[:6]=='CAVEAT':
                self['CAVEAT']=True
            if line[:6]=='COMPND':
                tmpline=line[10:].strip()
                tmpstring=''
                for cid,char in enumerate(tmpline):
                    if char==';' and cid!=len(tmpline):
                        tmpstring+=','
                    else:
                        tmpstring+=char 
                compnd+=tmpstring
            if line[:6]=="KEYWDS":
                keywds+=line[10:].strip()+' '
            if line[:6]=="EXPDTA":
                expdat+=line[10:].strip('\n')
            if line[:6]=="NUMMDL":
                self["NUMMDL"]=int(line[10:].strip())
                #self["MODELS"]=[[] for i in range(self["NUMMDL"])]
            if line[:22]=="REMARK   2 RESOLUTION.":
                if 'NOT' not in  line:
                    self["RESOLUTION"]=float(line[23:30])
                else:
                    self["RESOLUTION"]=0
            if line[:10]=="REMARK 465" and "M RES C SSSEQI" in line:
                self["MISSING RESIDUES"]=[]
                line=f.readline()
                while line[:10]=="REMARK 465":
                    modelindex=line[11:14].strip()
                    residuename=line[14:18].strip()
                    chainindex=line[18:20].strip()
                    resindex=line[20:26].strip()
                    self["MISSING RESIDUES"].append([modelindex,residuename,chainindex,resindex])
                    line=f.readline()
            if line[:10]=="REMARK 470" and "M RES CSSEQI  ATOMS" in line:
                self["MISSING ATOMS"]=[]
                line=f.readline()
                while line[:10]=="REMARK 470":
                    modelindex=line[11:14].strip()
                    resname=line[14:18].strip()
                    chainindex=line[18:20].strip()
                    resindex=line[20:24].strip()
                    atomnamelist=line[24:].split()
                    self["MISSING ATOMS"].append([modelindex,resname,chainindex,resindex,atomnamelist])
                    line=f.readline()
            if line[:20]=="REMARK 500 SUBTOPIC" and "CLOSE CONTACTS IN SAME ASYMMETRIC UNIT" in line:
                self["CLOSE CONTACTS"]=[]
                line=f.readline()
                while "ATM1 RES C SSEQI" not in line:
                    line=f.readline()
                line=f.readline()
                while not line[10:].strip()=='':
                    closecontact={}
                    closecontact["ATM1 NAME"]=line[10:17]
                    closecontact["ATM1 RES"]=line[17:21]
                    closecontact["ATM1 CHAIN"]=line[21:23]
                    closecontact["ATM1 RESINDEX"]=line[23:30]
                    closecontact["ATM2 NAME"]=line[30:38]
                    closecontact["ATM2 RESINDEX"]=line[38:41]
                    closecontact["ATM2 CHAIN"]=line[42:44]
                    closecontact["ATM2 RESINDEX"]=line[44:51]
                    closecontact["DISTANCE"]=line[51:70].strip()
                    self["CLOSE CONTACTS"].append(closecontact)
                    line=f.readline()
            if line[:10]=="REMARK 620" and "METAL COORDINATION" in line:
                line=f.readline()
                self["METAL COORDINATION"]=[] 
                self["METAL LIST"]=[]
                while "REMARK 620" in line:
                    if "COORDINATION ANGLES FOR" in line:
                        line=f.readline()
                        coordination={}
                        coordination["METAL"]={}
                        coordination["METAL"]["MODEL INDEX"]=line[35:38].strip()
                        coordination["METAL"]["RESNAME"]=line[38:42].strip()
                        coordination["METAL"]["CHAIN"]=line[42:44].strip()
                        coordination["METAL"]["RESINDEX"]=int(line[44:48])
                        coordination["METAL"]["INSERTION CODE"]=line[48]
                        coordination["METAL"]["ATOMNAME"]=line[49:55].strip()
                        self["METAL LIST"].append((line[42:44].strip(),int(line[44:48])))
                    if "N RES CSSEQI ATOM" in line:
                        coordination_structure=[]
                        anglematrix=[]
                        line=f.readline()
                        while not line[11]=='N':
                            index=int(line[10:12])
                            resname=line[12:16]
                            chainindex=line[16:18]
                            resindex=line[18:23]
                            atomname=line[23:28]
                            coordination_structure.append([index,resname,chainindex,resindex,atomname])
                            anglearray=line[28:].split()
                            for angleindex,angle in enumerate(anglearray):
                                anglematrix.append((index,angleindex+1,float(angle)))
                            line=f.readline()
                        coordination["ATOMS"]=coordination_structure
                        coordination["ANGLES"]=anglematrix
                        self["METAL COORDINATION"].append(coordination)
                    line=f.readline()
            if line[:6]=="DBREF ":
                chainid=line[11:13].strip();idcode=line[7:11].strip();seqbegin=int(line[14:18]);
                insertbegin=line[18].strip();seqend=int(line[20:24]);inserend=line[24].strip();
                database=line[26:32].strip();dbaccession=line[33:41].strip();dbidcode=line[42:54].strip();
                dbseqbegin=line[55:60].strip();idbnsbeg=line[60].strip();dbseqend=line[62:67].strip();
                dbinsend=line[67].strip()
                self["DBREF"].append([chainid,idcode,seqbegin,insertbegin,seqend,inserend,
                database,dbaccession,dbidcode,dbseqbegin,idbnsbeg,dbseqend,dbinsend])
            if line[:6]=="SEQRES":
                chainid=line[10:12].strip()
                if chainid not in self["SEQRES"].keys():
                    self["SEQRES"][chainid]=[]
                reslist=line[17:].split()
                self["SEQRES"][chainid]+=reslist
            if line[:6]=="MODRES":
                idcode=line[7:11].strip()
                resname=line[12:15].strip()
                chainid=line[16].strip()
                seqnum=int(line[18:22])
                icode=line[22].strip()
                stdres=line[24:27].strip()
                comment=line[29:70].strip()
                if chainid not in self["MODRES"].keys():
                    self["MODRES"][chainid]=[]
                self["MODRES"][chainid].append([idcode,resname,chainid,seqnum,icode,stdres,comment])
                self["MODRES LIST"].append((chainid,seqnum)) 
                
            if line[:6]=="HET   ":
                    chainid=line[12].strip();hetid=line[7:10].strip();seqnum=int(line[13:17]);icode=line[17].strip();
                    numhetatoms=int(line[20:25]);link=0;
                    if chainid not in self["HET"].keys():
                        self["HET"][chainid]=[]
                    if chainid in self["MODRES"].keys(): 
                        if seqnum not in [modres[3] for modres in self["MODRES"][chainid]] and hetid not in ["ACE","NH2","NME"]:
                            self["HET"][chainid].append([chainid,hetid,seqnum,icode,numhetatoms,link])
                        if seqnum in [modres[3] for modres in self["MODRES"][chainid]] and hetid not in self["SEQRES"][chainid] and hetid not in ["ACE","NH2","NME"]:
                            self["HET"][chainid].append([chainid,hetid,seqnum,icode,numhetatoms,link])
                    else:
                        if hetid not in ["ACE","NH2","NME"]:
                            self["HET"][chainid].append([chainid,hetid,seqnum,icode,numhetatoms,link])
            
            if line[:6]=="HETNAM":
                hetid=line[11:14].strip()
                if hetid not in self["HETNAM"].keys():
                    self["HETNAM"][hetid]={}
                    self["HETNAM"][hetid]["CHEMICAL NAME"]=''
                self["HETNAM"][hetid]["CHEMICAL NAME"]+=line[15:70].strip()
            
            if line[:6]=="FORMUL":
                hetid=line[12:15].strip()
                if hetid not in self["HETNAM"]:
                    self["HETNAM"][hetid]={}
                try:
                    self["HETNAM"][hetid]["CHEMICAL FORMULA"]+=line[19:70].strip()
                except:
                    self["HETNAM"][hetid]["CHEMICAL FORMULA"]=line[19:70].strip()
                self["HETNAM"][hetid]["COMPONENT NUMBER"]=int(line[8:10])

            if line[:6]=="HELIX ":
                if "HELIX" not in self.keys():
                    self["HELIX"]=[]
                index=int(line[7:10]);initresname=line[15:18].strip();initchainid=line[19].strip();initseqnum=int(line[21:25].strip());
                helixid=line[11:14];initicode=line[25].strip();endresname=line[27:30].strip();
                endchainid=line[31].strip();endseqnum=int(line[33:37].strip());endicode=line[37].strip();helixtype=line[38:40].strip();helixlen=int(line[71:76])
                self["HELIX"].append({"INDEX":index,"INITRESNAME":initresname,"INITCHAINID":initchainid,
                "INITSEQNUM":initseqnum,"HELIXID":helixid,"INITICODE":initicode,"ENDRESNAME":endresname,
                "ENDCHAINID":endchainid,"ENDSEQNUM":endseqnum,"ENDICODE":endicode,"HELIXTYPE":helixtype,"LENGTH":helixlen})
            if line[:6]=="SHEET ":
                if "SHEET" not in self.keys():
                    self["SHEET"]={}
                index=int(line[7:10]);sheetid=line[11:14].strip()
                if sheetid not in self["SHEET"].keys():
                    self["SHEET"][sheetid]={"STRANDS":[],"REGISTRATION":[]}
                initresname=line[17:20].strip()
                initchainid=line[21].strip();initseqnum=int(line[22:26]);initicode=line[26].strip();
                endresname=line[28:31];endchainid=line[32];endseqnum=int(line[33:37]);endicode=line[37].strip();
                sense=line[38:40]
                self["SHEET"][sheetid]["STRANDS"].append([index,initresname,initchainid,initseqnum,initicode,
                endresname,endchainid,endseqnum,endicode,sense])
                if index!=1:
                    curatom=line[41:45].strip()
                    curres=line[45:48].strip()
                    curchainid=line[49].strip()
                    curresseq=line[50:54].strip()
                    curicode=line[54].strip()
                    prevatom=line[56:60].strip()
                    prevres=line[60:63].strip()
                    prevchainid=line[64].strip()
                    prevresseq=line[65:69].strip()
                    previcode=line[69].strip()
                    self["SHEET"][sheetid]["REGISTRATION"].append([curatom,curres,curchainid,curresseq,curicode,
                    prevatom,prevres,prevchainid,prevresseq,previcode])

            if line[:6]=="LINK  ":
                atomname1=line[12:16].strip();altloc1=line[16].strip();resname1=line[17:20].strip();chainid1=line[21].strip();resseq1=int(line[22:26]);icode1=line[26]
                atomname2=line[42:46].strip();altloc2=line[46].strip();resname2=line[47:50].strip();chainid2=line[51].strip();resseq2=int(line[52:56]);icode2=line[56]
                try:
                    distance=float(line[74:78])
                except:
                    distance=line[74:78].strip()
                if "LINK" not in self.keys():
                    self["LINK"]=[]
                link={"ATOM1":[atomname1,altloc1,resname1,chainid1,resseq1,icode1],
                      "ATOM2":[atomname2,altloc2,resname2,chainid2,resseq2,icode2],
                      "DISTANCE":distance}
                self["LINK"].append(link)
                if (chainid1,resseq1) in self["MODRES LIST"] and chainid2 in self["HET"].keys():
                    for HET in self["HET"][chainid2]:
                        if HET[2]==resseq2:
                            HET[-1]=1
                elif (chainid2,resseq2) in self["MODRES LIST"] and chainid1 in self["HET"].keys():
                    for HET in self["HET"][chainid1]:
                        if HET[2]==resseq1:
                            HET[-1]=1
                #self["LINK RES LIST"].append((chainid1,resseq1))
                #self["LINK RES LIST"].append((chainid2,resseq2))
            if line[:6]=="SITE  ":
                siteid=line[11:14].strip();numres=int(line[15:17]);
                resname1=line[18:21].strip();chainid1=line[22];seqnum1=line[23:27].strip();icode1=line[27]
                resname2=line[29:32].strip();chainid2=line[33];seqnum2=line[34:38].strip();icode2=line[38]
                resname3=line[40:43].strip();chainid3=line[44];seqnum3=line[45:49].strip();icode3=line[49]
                resname4=line[51:54].strip();chainid4=line[55];seqnum4=line[56:60].strip();icode4=line[60]
                if siteid not in self["SITE"].keys():
                    self["SITE"][siteid]={"ID":siteid,"NUMRES":numres,"RESLIST":[]}
                self["SITE"][siteid]["RESLIST"]+=[(resname1,chainid1,seqnum1,icode1),(resname2,chainid2,seqnum2,icode2),(resname3,chainid3,seqnum3,icode3),(resname4,chainid4,seqnum4,icode4)]

            if line[:6]=="MODEL ":
                self["MODELS"].append({"CHAINS":{}})

            if line[:6]=="ATOM  ":
                chainid=line[21]
                if len(self["MODELS"])==0:
                    self["MODELS"].append({"CHAINS":{}})
                if chainid not in self["MODELS"][-1]["CHAINS"].keys():
                    self["MODELS"][-1]["CHAINS"][chainid]={"PROTEINSTR":[],"HETSTR":[],"PDB ID":self["PDB ID"],"CHAIN ID":chainid,"METAL COORDINATION":[],"HET":[]}
                if line[16]=='A' or line[16]==' ':
                    self["MODELS"][-1]["CHAINS"][chainid]["PROTEINSTR"].append(line)


            if line[:6]=="HETATM":
                chainid=line[21]
                seqnum=int(line[22:26])
                resname=line[17:20].strip()
                if len(self["MODELS"])==0:
                    self["MODELS"].append({"CHAINS":{}})
                if chainid not in self["MODELS"][-1]["CHAINS"].keys():
                    self["MODELS"][-1]["CHAINS"][chainid]={"PROTEINSTR":[],"HETSTR":[],"PDB ID":self["PDB ID"],"CHAIN ID":chainid,"METAL COORDINATION":[],"HET":[]}
                if chainid in self["MODRES"].keys():
                    modreslist=[modres[1] for modres in self["MODRES"][chainid]]
                    modidlist=[modres[3] for modres in self["MODRES"][chainid]]
                    hetreslist=[het[2] for het in self['HET'][chainid]]
                    if seqnum not in hetreslist:
                        if resname in modreslist and seqnum in modidlist:
                            self["MODELS"][-1]["CHAINS"][chainid]["PROTEINSTR"].append(line)
                        else:
                            if resname in ["ACE","NH2","NME"]:
                                self["MODELS"][-1]["CHAINS"][chainid]["PROTEINSTR"].append(line)
                            else:
                                self["MODELS"][-1]["CHAINS"][chainid]["HETSTR"].append(line) 
                    else:
                        if resname in ["ACE","NH2","NME"]:
                            self["MODELS"][-1]["CHAINS"][chainid]["PROTEINSTR"].append(line)
                        else:
                            self["MODELS"][-1]["CHAINS"][chainid]["HETSTR"].append(line)                        
                    
                else:
                    if resname in ["ACE","NH2","NME"]:
                        self["MODELS"][-1]["CHAINS"][chainid]["PROTEINSTR"].append(line)
                    else:
                        self["MODELS"][-1]["CHAINS"][chainid]["HETSTR"].append(line)                        
            line=f.readline()
        for siteid in self["SITE"].keys():
            self["SITE"][siteid]["RESLIST"]=self["SITE"][siteid]["RESLIST"][:self["SITE"][siteid]["NUMRES"]]
        if compnd!='':
            self["COMPND"]=parse_dictstring(compnd)
        self["KEYWDS"]=[i.strip() for i in keywds.split(';')]
        self["EXPDTA"]=[i.strip() for i in expdat.split(';')]        
        for i in range(len(self["DBREF"])):
            for j in range(len(self["MODELS"])):
                self["MODELS"][j]["CHAINS"][self["DBREF"][i][0]]["DBREF"]=self["DBREF"][i]
              
        hetkeys=list(self["HET"].keys())
        for i in range(len(hetkeys)):
            for j in range(len(self["MODELS"])):
                try:
                    self["MODELS"][j]["CHAINS"][hetkeys[i]]["HET"]=self["HET"][hetkeys[i]]
                except Exception as e:
                    print ('hetkeys',self["PDB ID"],e)

        for i in range(len(self["METAL COORDINATION"])):
            if self["METAL COORDINATION"][i]["METAL"]["MODEL INDEX"]=='':
                for j in range(len(self["MODELS"])):
                    try:
                        self["MODELS"][j]["CHAINS"][self["METAL COORDINATION"][i]["METAL"]["CHAIN"]]["METAL COORDINATION"].append(self["METAL COORDINATION"][i])
                    except Exception as e:
                        print ('Metal',self["PDB ID"],e)
            else:
                modelindex=int(self["METAL COORDINATION"][i]["METAL"]["MODEL INDEX"])
                try:
                    self["MODELS"][modelindex]["CHAINS"][self["METAL COORDINATION"][i]["METAL"]["CHAIN"]]["METAL COORDINATION"].append(self["METAL COORDINATION"][i])
                except Exception as e:
                    print ('Metal',self["PDB ID"],e)
            
        for i in range(len(self["MODELS"])):
            chainkeys=list(self["MODELS"][i]["CHAINS"].keys())
            for j in range(len(chainkeys)):
                hetatmlist=self["MODELS"][i]["CHAINS"][chainkeys[j]]["HETSTR"]
                hetstrlist=[[] for m in range(len(self["MODELS"][i]["CHAINS"][chainkeys[j]]["HET"]))]
                for id,hetinfo in enumerate(self["MODELS"][i]["CHAINS"][chainkeys[j]]["HET"]):
                    resname=hetinfo[1];resseqnum=hetinfo[2]
                    hetstrlist[id]=[hetatmstr for hetatmstr in hetatmlist if int(hetatmstr[22:26])==resseqnum]
                hetstrlist.append([hetatmstr for hetatmstr in hetatmlist if hetatmstr[17:20].strip()=="HOH" or hetatmstr[17:20].strip()=="WAT"])
                self["MODELS"][i]["CHAINS"][chainkeys[j]]["HETSTR"]=hetstrlist
        return 


        
