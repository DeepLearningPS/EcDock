import os
import subprocess

# 设置参数
data_path = "./train_data/casf2016_predict_sdf_train_interaction_random_protein_cutoff"
save_dir = "./save_pose"
n_gpu = 1 #设置GPU数量
MASTER_PORT = 10086
finetune_mol_model = "./weights/mol_pre_no_h_220816.pt"
finetune_pocket_model = "./weights/pocket_pre_220816.pt"

finetune_mol_model = None
finetune_pocket_model = None
#lr = 3e-4
lr = 3e-2
batch_size = 1
epoch = 100
dropout = 0.2
warmup = 0.06
update_freq = 1
dist_threshold = 8.0
recycling = 4
conf_size = 1

# 设置环境变量
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "7" #设置GPU编号，可以多个，与n_gpu值对应

# 构建命令
command = [
    "torchrun",
    "--rdzv_backend", "c10d",
    "--rdzv_endpoint", "localhost:0",
    "--nnodes", "1",
    "--nproc_per_node", str(n_gpu),
    "--rdzv_id", str(n_gpu),
    "unicore_cli/train.py",
    "--user-dir", "./unimol",
    data_path,
    "--train-subset", "train",
    "--valid-subset", "valid",
    "--train_path", "./premodel/unimol_docking_v2_240517.pt",
    "--num-workers", "8",
    "--ddp-backend", "c10d",
    "--task", "docking_pose_v2",
    "--loss", "docking_pose_v2",
    "--arch", "docking_pose_v2",
    "--optimizer", "adam",
    "--adam-betas", "(0.9, 0.99)",
    "--adam-eps", "1e-6",
    "--clip-norm", "1.0",
    "--lr-scheduler", "polynomial_decay",
    "--lr", str(lr),
    "--warmup-ratio", str(warmup),
    "--max-epoch", str(epoch),
    "--batch-size", str(batch_size),
    "--mol-pooler-dropout", str(dropout),
    "--pocket-pooler-dropout", str(dropout),
    "--update-freq", str(update_freq),
    "--seed", "42",
    "--fp16",
    "--fp16-init-scale", "4",
    "--fp16-scale-window", "256",
    "--tensorboard-logdir", f"{save_dir}/tsb",
    "--log-interval", "100",
    "--log-format", "simple",
    "--validate-interval", "1",
    "--keep-last-epochs", "10",
    "--best-checkpoint-metric", "valid_loss",
    "--patience", "2000",
    "--all-gather-list-size", "1024000",
    #"--finetune-mol-model", finetune_mol_model,
    #"--finetune-pocket-model", finetune_pocket_model,
    "--dist-threshold", str(dist_threshold),
    "--recycling", str(recycling),
    "--save-dir", save_dir,
    "--find-unused-parameters",
    "--required-batch-size-multiple", "1",
    "--conf-size", str(conf_size)
]



if finetune_mol_model is not None:
    command += ["--finetune-mol-model", finetune_mol_model]
if finetune_pocket_model is not None:
    command += ["--finetune-pocket-model", finetune_pocket_model]


# 运行命令
subprocess.run(command, check=True)
exit()
try:
    subprocess.run(command, check=True)
except subprocess.CalledProcessError as e:
    print(f"命令执行失败，返回码: {e.returncode}")
except Exception as e:
    print(f"发生错误: {str(e)}")