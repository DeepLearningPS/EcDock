import copy
import numpy as np
from rdkit import Chem 
from rdkit.Chem import AllChem,rdmolfiles
import copy
import networkx as nx 
from .utils_os import Flatten

def bfs_seq(G, start_id):
    '''
    get a bfs node sequence
    :param G:
    :param start_id:
    :return:
    '''
    dictionary = dict(nx.bfs_successors(G, start_id))
    start = [start_id]
    output = [start_id]
    while len(start) > 0:
        next = []
        while len(start) > 0:
            current = start.pop(0)
            neighbor = dictionary.get(current)
            if neighbor is not None:
                next = next + neighbor
        output = output + next
        start = next
    return output

def dfs_seq(G,start_id):
    #dictionary=dict(nx.dfs_successors(G,start_id))
    dictionary={i:[n for n in G.neighbors(i)] for i in range(G.number_of_nodes())}
    #print (dictionary)
    start=[start_id]
    output=[]
    while len(start)>0:
        v=start.pop()
        output.append(v)
        for w in dictionary[v]:
            if w not in output and w not in start:
            #if w not in output:
                start.append(w)
                #output.append(w)
    return output

def rdkit_bfs_seq_mol(molobj,start_id=0):
    natoms=len(molobj.GetAtoms())
    bonds=[]
    for i in range(natoms):
        for j in range(natoms):
            bond=molobj.GetBondBetweenAtoms(i,j)
            if bond:
                bonds.append((i,j))
    G=nx.Graph()
    G.add_edges_from(bonds)
    seq=bfs_seq(G,start_id=start_id)
    #print (seq)
    #print (seq)
    reseq=np.argsort(seq)
    molobj=Chem.rdmolops.RenumberAtoms(molobj,seq)
    return molobj

def rdkit_dfs_seq_mol(molobj,start_id=0):
    natoms=len(molobj.GetAtoms())
    bonds=[]
    for i in range(natoms):
        for j in range(natoms):
            bond=molobj.GetBondBetweenAtoms(i,j)
            if bond:
                bonds.append((i,j))
    G=nx.Graph()
    G.add_edges_from(bonds)
    seq=dfs_seq(G,start_id=start_id)
    #print (seq)
    reseq=np.argsort(seq)
    molobj=Chem.rdmolops.RenumberAtoms(molobj,seq)
    return molobj

def Merge_single_rings_to_nodes(adjs,rings,atomics,pharm_groups=None):
    natoms=adjs.shape[0]
    nrings=len(rings)
    ring_flags=np.zeros(natoms)
    
    ring_atoms=np.array(list(set(Flatten(rings))))
    
        
    if pharm_groups is None:
        pharm_groups=[]
        
    pharm_atoms=np.array(list(set(Flatten(pharm_groups))))
    npharms=len(pharm_groups)
         
    if len(ring_atoms)>0 :
        ring_flags[ring_atoms]=1
    if len(pharm_atoms)>0:
        ring_flags[pharm_atoms]=1    
    #print ('rings',rings)
    unmerged_nodes=rings+pharm_groups+[[i] for i in range(natoms) if not ring_flags[i]]
    #print ('unmerged_nodes',unmerged_nodes) 
    n_unmerged_nodes=len(unmerged_nodes)
    unmerged_node_adjs=np.zeros((n_unmerged_nodes,n_unmerged_nodes))
    for i,node_i in enumerate(unmerged_nodes[:nrings+npharms]):
        for j,node_j in enumerate(unmerged_nodes[:nrings+npharms]):
            if len(set(node_i+node_j))<len(node_i)+len(node_j):
                unmerged_node_adjs[i,j]=1
                unmerged_node_adjs[j,i]=1
            else:
                for ai,atomi in enumerate(node_i):
                    for aj,atomj in enumerate(node_j):
                        if adjs[atomi,atomj]>1 and atomics[atomj] in [7,8,16]:
                            unmerged_node_adjs[i,j]=1
                            unmerged_node_adjs[j,i]=1
        for j,node_j in enumerate(unmerged_nodes[nrings+npharms:]):
            for ai,atomi in enumerate(node_i):
                atomj=node_j[0]
                if adjs[atomi,atomj]>1 and atomics[node_j[0]] in [7,8,16]:
                    unmerged_node_adjs[i,j+nrings+npharms]=1
                    unmerged_node_adjs[j+nrings+npharms,i]=1
    
    unmerged_node_graph=nx.Graph(unmerged_node_adjs)
    
    merged_nodes=[]
    connected_nodes=[]
    for c in nx.connected_components(unmerged_node_graph):
        nodeset=unmerged_node_graph.subgraph(c).nodes()
        connected_nodes+=nodeset
        merged_nodes.append([])
        for i in nodeset:
            merged_nodes[-1]+=unmerged_nodes[i]

    for i in range(n_unmerged_nodes):
        if i not in connected_nodes:
            merged_nodes.append(unmerged_nodes[i])
            
    merged_nodes=[np.sort(list(set(node))) for node in merged_nodes]
    return merged_nodes