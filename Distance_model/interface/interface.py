#!/usr/bin/env python
# demo_run.py
import argparse
import subprocess
import os

def main():
    parser = argparse.ArgumentParser(description="Run demo.py with specified GPU")
    parser.add_argument("--gpu", type=str, default="0", help="CUDA_VISIBLE_DEVICES")
    parser.add_argument("--mode", type=str, default="batch_one2one")
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--conf_size", type=int, default=5)
    parser.add_argument("--cluster", action="store_true")
    parser.add_argument("--input_batch_file", type=str, required=True)
    parser.add_argument("--output_ligand_dir", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--steric_clash_fix", action="store_true")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=1000000000)
    args = parser.parse_args()

    cmd = (
        f"CUDA_VISIBLE_DEVICES={args.gpu} "
        f"python demo.py "
        f"--mode {args.mode} "
        f"--batch-size {args.batch_size} "
        f"--conf-size {args.conf_size} "
        f"--input-batch-file {args.input_batch_file} "
        f"--output-ligand-dir {args.output_ligand_dir} "
        f"--model-dir {args.model_dir} "
        f"--start_idx {args.start_idx} "
        f"--end_idx {args.end_idx} "
        f"--gpu {args.gpu} "
    )

    if args.cluster:
        cmd += "--cluster "

    if args.steric_clash_fix:
        cmd += "--steric-clash-fix "

    print("Running command:", cmd)
    os.system(cmd)

if __name__ == "__main__":
    '''
    from pathlib import Path
    # 
    CURRENT_DIR = Path(__file__).resolve().parent

    # 
    os.chdir(CURRENT_DIR)

    # 
    print(":", os.getcwd())
    '''
    main()