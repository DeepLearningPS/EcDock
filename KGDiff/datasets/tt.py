    def _process_single(self):
        #
        db = lmdb.open(
            self.processed_path,
            map_size=300*(1024*1024*1024),   # 500GB
            create=True,
            subdir=False,
            readonly=False,  # Writable
        )
        with open(self.index_path, 'rb') as f:
            index = pickle.load(f)

        error_file = open(os.path.join(os.path.dirname(self.processed_path), 'error_file.txt'), 'w')
        num_skipped = 0
        num_pocket  = 0
        num_zmats   = 0
        num_success = 0
        
        count = 0
        nameid2id_dict = {}


        with db.begin(write=True, buffers=True) as txn:
            for i, (pocket_fn, protein_fn, (pka, year, resl), ligand_fn, pdbid) in enumerate(tqdm(index[:])): #100
                if not os.path.exists(os.path.join(self.raw_path, pocket_fn)):
                    pocket_fn = protein_fn

                print('pocket_fn:', pocket_fn) #data/O00329-6PYR/O00329-6PYR/active0/active0_protein.pdb
                print('ligand_fn:', ligand_fn) #data/O00329-6PYR/O00329-6PYR/active0/active0_ligand.sdf
            

                if pocket_fn is None:
                    continue
                

                    
                #，
                if self.data_name == 'pdbbind2020_r10' and not os.path.isfile(os.path.join(self.raw_path, pocket_fn)):
                    s_file = os.path.join('../CrossDocked2020/data/pdbbind2020/pdbbind2020',  '/'.join(pocket_fn.split('/')[1:]))
                    t_file = os.path.join(self.raw_path, pocket_fn)
                    try:
                        shutil.copy(s_file, t_file)
                    except Exception as e:
                        print(e)
                        exit()



                try:
                    data_prefix = self.raw_path
                    ligand_dict = parse_sdf_file(os.path.join(data_prefix, ligand_fn), self.data_flag)
                    try:
                        ligand_centor = np.mean(ligand_dict['pos'], axis=0)
                    except TypeError as e:
                        raise SystemExit
                    
                    pocket_dict = PDBProtein(os.path.join(data_prefix, pocket_fn), ligand_centor, ligand_dict, self.data_flag, self.cross_distance_num, unimol_pcoords = 
                                            None).to_dict_atom_interaction_gen_split3_5_extend()
                

                    data = ProteinLigandData.from_protein_ligand_dicts(
                        protein_dict=torchify_dict(pocket_dict),
                        ligand_dict=torchify_dict(ligand_dict),
                    )
                    protein_file = os.path.join(data_prefix, pocket_fn)
                    sub_protein_file = os.path.join(os.path.dirname(protein_file), os.path.splitext(os.path.basename(protein_file))[0] + '.pdb') 
                    l_mol = ligand_dict['mol']
                    p_mol = Chem.MolFromPDBFile(sub_protein_file,removeHs=True,sanitize=False)
                    complex_mol = Chem.CombineMols(p_mol, l_mol)
                    data.complex_mol = complex_mol

                    output_file = os.path.join(os.path.dirname(protein_file), os.path.splitext(os.path.basename(protein_file))[0] + '_complex.pdb')
                    with Chem.PDBWriter(output_file) as writer:
                        writer.write(complex_mol)

                    data.protein_filename = pocket_fn
                    data.ligand_filename = ligand_fn
                    data.affinity = pka
                    
                    n = os.path.basename(pocket_fn).rsplit("_protein_256.pdb", 1)[0] #os.path.basename(self.protein_file).rsplit("_protein.pdb", 1)[0]
                    data.name = n #
                    data = data.to_dict()  # avoid torch_geometric version issue
                    assert data['protein_pos'].size(0) > 0 #zmats，


                    txn.put(
                        key=(n + '_' + str(i)).encode(), #lmdbint. ，+id，id
                        value=pickle.dumps(data)
                    )
                    num_success += 1
                    nameid2id_dict[n + '_' + str(i)] = str(i)
                
                
                #except (Exception, AssertionError, ValueError, TypeError, OSError, SystemExit) as e:
                #except Exception as e: #SystemExitException
                except (FileNotFoundError, Exception, SystemExit) as e:
                    self.exclude[count] = count
                    self.exclude_name.add(protein_fn)
                    data = None
                    n = os.path.basename(pocket_fn).rsplit("_protein_256.pdb", 1)[0]
                    txn.put(
                    key=(n + '_' + str(i)).encode(), #lmdbint
                    value=pickle.dumps(data)
                    )
                    count += 1
                    nameid2id_dict[n + '_' + str(i)] = str(i)


                    self.exclude_name.add(protein_fn)
                    error_file.write(f'error: {e}\n')
                    error_file.write(f'type(e): {type(e)}\n')   
                    print('error:', e)
                    print(f": {type(e)}")
                    num_skipped += 1
                    error_file.write('Skipping ligand_fn (%d) %s \n' % (num_skipped, ligand_fn))
                    error_file.write('Skipping pocket_fn (%d) %s \n' % (num_skipped, pocket_fn))
                    print('Skipping (%d) %s' % (num_skipped, ligand_fn, ))
                    print('complex name:', n)
                    #exit()
                    continue
                    
                    
                    
                

        error_file.write(f'num_skipped: {num_skipped}\n') 
        error_file.write(f'num_pocket: {num_pocket}\n')
        error_file.write(f'num_zmats: {num_zmats}\n')
        error_file.write(f'num_success: {num_success}\n')
        error_file.write(f'num_error: {count}\n')
        error_file.close()
        print('num_skipped:', num_skipped)
        print(f'num_pocket: {num_pocket}')
        print(f'num_zmats: {num_zmats}')
        print(f'num_success: {num_success}')
        print(f'num_error: {count}')

        self.nameid2id_dict = nameid2id_dict


        file_name = os.path.join(os.path.dirname(self.processed_path), f'{self.data_name}_name2id_dict.txt')


        with open(file_name, 'w')as f:
            for k, v in nameid2id_dict.items():
                f.write(k + '\t' + v + '\n')

        db.close()

        with open(os.path.join(os.path.dirname(self.processed_path), 'exclude_index.txt'), 'w') as f:
            for i in self.exclude.keys():
                f.write(str(i) + '\n')

        with open(os.path.join(os.path.dirname(self.processed_path), 'exclude_name.txt'), 'w') as f:
            for i in self.exclude_name:
                f.write(str(i) + '\n')