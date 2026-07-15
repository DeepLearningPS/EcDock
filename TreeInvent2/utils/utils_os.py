import os 

def Find_Files(start_path, name):
    flist=[]
    for relpath, dirs, files in os.walk(start_path):
        for fname in files:
            #print (fname)
            if name in fname:
                full_path = os.path.join(start_path, relpath, fname)
                flist.append(os.path.normpath(os.path.abspath(full_path)))
    return flist

def Flatten(lst, ltypes=(list, tuple)):
    if not isinstance(lst, ltypes):
        return [lst]
    result = []
    for item in lst:
        result.extend(Flatten(item, ltypes))
    return result


