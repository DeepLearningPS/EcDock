#!/usr/bin/env python
# run_sample.py
import os
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Run KGDiff sample_diffusion.py with custom args")
    parser.add_argument("--gpu", type=str, default="7", help="CUDA_VISIBLE_DEVICES")
    parser.add_argument("--conf_num", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--si", type=int, default=0, help="start index")
    parser.add_argument("--ei", type=int, default=100000000, help="end index")
    parser.add_argument("--test_name", type=str, default="ecdock_sample")
    parser.add_argument("--step", type=int, default=15)
    parser.add_argument('--user_test_data_dir', type=str, default='tmpdata')
    parser.add_argument('--out_dir', type=str, default='tmpresault')


    args = parser.parse_args()

    # 
    cmd = (
        f"CUDA_VISIBLE_DEVICES={args.gpu} torchrun --rdzv_backend c10d --rdzv_endpoint localhost:0 "
        f"--nnodes 1 --nproc_per_node 1 --rdzv_id 1 "
        f"KGDiff/scripts/sample_diffusion.py --config ./configs/sampling.yml "
        f"-i 0 --guide_mode pdbbind_random --type_grad_weight 100 "
        f"--pos_grad_weight 25 --result_path new_ecdock_step5 --data_flag new_test "
        f"--data_name tmpdata --diffusion cm --gnn equiformer --sample_num 500 "
        f"--conf_num {args.conf_num} --batch_size {args.batch_size} "
        f"--si {args.si} --ei {args.ei} --test_name {args.test_name} --step {args.step} --user_test_data_dir {args.user_test_data_dir} --out_dir {args.out_dir} "
        f"--ckpt test_premodel/all_atom_ecdock/rate_188.pt"
    )

    #  GPU
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu

    print("Running command:", cmd)
    os.system(cmd)

if __name__ == "__main__":
    main()