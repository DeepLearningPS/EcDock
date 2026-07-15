CUDA_VISIBLE_DEVICES=0 python demo.py --mode batch_one2one --batch-size 1 --conf-size 1 --cluster \
        --input-batch-file tmpdata_input_batch_one2one_boxsize10.csv \
        --output-ligand-dir tmpdata_predict_sdf_random_protein_cutoff \
        --model-dir ../premodel/best.pt \
        --steric-clash-fix \
        --start_idx 0 \
        --end_idx 1000000000 \

        #，？  #posebusters428
        #--model-dir ../../model/unimol_docking_v2_240517.pt \
        #posebusters_input_batch_one2one_boxsize10.csv
        #posebusters_predict_sdf_interaction

        #pdb2020_input_batch_one2one_boxsize10.csv
        #pdb2020_predict_sdf_ecdock_train
        #40，12G，
        #40rdkit，，，，，
        #，
        
