#!/usr/bin/env bash
set -Eeuo pipefail

# ==============================================================================
# EC-Dock one-click pipeline
# ==============================================================================
# Example:
#   bash run_ecdock_pipeline.sh \
#     --main_dir /mnt_191/zg_fan/new_Github-ECDock \
#     --input_ligand /mnt_191/zg_fan/new_Github-ECDock/input_data/5SB2/5SB2_ligand.sdf \
#     --input_protein /mnt_191/zg_fan/new_Github-ECDock/input_data/5SB2/5SB2_protein.pdb \
#     --gpu 0 \
#     --step 5 \
#     --conf_num 5
#
# Optional install:
#   bash run_ecdock_pipeline.sh ... --install
#
# Keep tmpdata/tmpresault/user_data:
#   bash run_ecdock_pipeline.sh ... --keep_tmp
# ===============================================================================

usage() {
  cat <<'EOF'
Usage:
  bash run_ecdock_pipeline.sh --input_ligand LIGAND.sdf --input_protein PROTEIN.pdb [options]

Required:
  --input_ligand PATH        Input ligand SDF file.
  --input_protein PATH       Input protein PDB file.

Common options:
  --main_dir PATH            Project root directory. Default: /mnt_191/zg_fan/new_Github-ECDock
  --env_name NAME            Conda environment name. Default: ecdock
  --gpu ID                   GPU id. Default: 0
  --name NAME                Complex name. Default: inferred from ligand file name.
  --step N                   Diffusion / sampling steps. Default: 5
  --conf_num N               Number of conformations. Default: 5
  --batch_size N             Docking batch size. Default: 1
  --distance_batch_size N    Distance-model batch size. Default: 5
  --distance_conf_size N     Distance-model conformer size. Default: same as --conf_num
  --start_idx N              Start index for distance model and docking. Default: 0
  --end_idx N                End index for distance model. Default: 100
  --dock_end_idx N           End index for docking. Default: 1000000000

Directories / files:
  --store_dir DIR            Temporary user data dir relative to main_dir. Default: user_data
  --tmp_data_dir DIR         Temporary processed data dir. Default: tmpdata
  --result_dir DIR           Temporary docking result dir. Default: tmpresault
  --distance_csv FILE        Distance input CSV. Default: tmpdata_input_batch_one2one_boxsize10.csv
  --distance_output_dir DIR  Distance model output dir. Default: tmpdata_predict_sdf_random_protein_cutoff
  --distance_model PATH      Distance model checkpoint. Default: ../premodel/best.pt
  --test_name NAME           Docking test name. Default: ecdock_sample
  --output_root DIR          Final result root. Default: inferred from input ligand path.

Installation options:
  --install                  Run environment installation and editable installs before pipeline.
  --install_script FILE      Install script. Default: install_ECDock_merged_fixed_v4.sh
  --base_yaml FILE           Conda base yaml. Default: ECDock_merged_base.yaml
  --pip_requirements FILE    Pip requirements txt. Default: ECDock_merged_pip_requirements.txt

Skip options:
  --skip_preprocess          Skip preprocess_data_modified.py.
  --skip_distance_process    Skip Distance_model/interface/process_data.py.
  --skip_distance_model      Skip Distance_model/interface/interface.py.
  --skip_distance_move       Skip Distance_model/interface/data_move.py.
  --skip_docking             Skip KGDiff/scripts/multiple_data_run.py.
  --skip_force_optim         Skip Diffusion_force_optim/force_optim.py.
  --skip_score               Skip ECDockScore/posebusters_batch_deal.py.
  --keep_tmp                 Do not delete tmpdata, tmpresault, and user_data after copying results.
  --overwrite_output         Remove existing final output directory before copying.

Other:
  -h, --help                 Show this help message.
EOF
}

log() {
  echo -e "\n[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

abs_path() {
  local p="$1"
  if [[ "$p" = /* ]]; then
    echo "$p"
  else
    echo "$(pwd)/$p"
  fi
}

infer_name_from_ligand() {
  local ligand_file
  ligand_file="$(basename "$1")"
  ligand_file="${ligand_file%.*}"
  ligand_file="${ligand_file%_ligand}"
  echo "$ligand_file"
}

infer_output_root() {
  local ligand_abs sample_name ligand_dir parent_dir
  ligand_abs="$1"
  sample_name="$2"
  ligand_dir="$(dirname "$ligand_abs")"
  parent_dir="$(dirname "$ligand_dir")"

  # If ligand is stored as input_data/5SB2/5SB2_ligand.sdf, output root is input_data.
  # Otherwise, use the ligand file directory.
  if [[ "$(basename "$ligand_dir")" == "$sample_name" ]]; then
    echo "$parent_dir"
  else
    echo "$ligand_dir"
  fi
}

activate_conda_env() {
  local env_name="$1"

  if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$env_name"
  elif [[ -n "${CONDA_EXE:-}" ]]; then
    # shellcheck disable=SC1091
    source "$(dirname "$(dirname "$CONDA_EXE")")/etc/profile.d/conda.sh"
    conda activate "$env_name"
  else
    fail "conda command not found. Please initialize conda first."
  fi
}

# -------------------------------
# Default arguments
# -------------------------------
MAIN_DIR="/mnt_191/zg_fan/new_Github-ECDock"
ENV_NAME="ecdock"
INPUT_LIGAND=""
INPUT_PROTEIN=""
SAMPLE_NAME=""
GPU="0"
STEP="5"
CONF_NUM="5"
BATCH_SIZE="1"
DISTANCE_BATCH_SIZE="5"
DISTANCE_CONF_SIZE=""
START_IDX="0"
END_IDX="100"
DOCK_END_IDX="1000000000"
STORE_DIR="user_data"
TMP_DATA_DIR="tmpdata"
RESULT_DIR="tmpresault"
DISTANCE_CSV="tmpdata_input_batch_one2one_boxsize10.csv"
DISTANCE_OUTPUT_DIR="tmpdata_predict_sdf_random_protein_cutoff"
DISTANCE_MODEL="../premodel/best.pt"
TEST_NAME="ecdock_sample"
OUTPUT_ROOT=""
INSTALL_SCRIPT="install_ECDock_merged_fixed_v4.sh"
BASE_YAML="ECDock_merged_base.yaml"
PIP_REQUIREMENTS="ECDock_merged_pip_requirements.txt"

RUN_INSTALL=0
SKIP_PREPROCESS=0
SKIP_DISTANCE_PROCESS=0
SKIP_DISTANCE_MODEL=0
SKIP_DISTANCE_MOVE=0
SKIP_DOCKING=0
SKIP_FORCE_OPTIM=0
SKIP_SCORE=0
KEEP_TMP=0
OVERWRITE_OUTPUT=0

# -------------------------------
# Parse arguments
# -------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --main_dir) MAIN_DIR="$2"; shift 2 ;;
    --env_name) ENV_NAME="$2"; shift 2 ;;
    --input_ligand) INPUT_LIGAND="$2"; shift 2 ;;
    --input_protein) INPUT_PROTEIN="$2"; shift 2 ;;
    --name) SAMPLE_NAME="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --step) STEP="$2"; shift 2 ;;
    --conf_num) CONF_NUM="$2"; shift 2 ;;
    --batch_size) BATCH_SIZE="$2"; shift 2 ;;
    --distance_batch_size) DISTANCE_BATCH_SIZE="$2"; shift 2 ;;
    --distance_conf_size) DISTANCE_CONF_SIZE="$2"; shift 2 ;;
    --start_idx) START_IDX="$2"; shift 2 ;;
    --end_idx) END_IDX="$2"; shift 2 ;;
    --dock_end_idx) DOCK_END_IDX="$2"; shift 2 ;;
    --store_dir) STORE_DIR="$2"; shift 2 ;;
    --tmp_data_dir) TMP_DATA_DIR="$2"; shift 2 ;;
    --result_dir) RESULT_DIR="$2"; shift 2 ;;
    --distance_csv) DISTANCE_CSV="$2"; shift 2 ;;
    --distance_output_dir) DISTANCE_OUTPUT_DIR="$2"; shift 2 ;;
    --distance_model) DISTANCE_MODEL="$2"; shift 2 ;;
    --test_name) TEST_NAME="$2"; shift 2 ;;
    --output_root) OUTPUT_ROOT="$2"; shift 2 ;;
    --install_script) INSTALL_SCRIPT="$2"; shift 2 ;;
    --base_yaml) BASE_YAML="$2"; shift 2 ;;
    --pip_requirements) PIP_REQUIREMENTS="$2"; shift 2 ;;
    --install) RUN_INSTALL=1; shift ;;
    --skip_preprocess) SKIP_PREPROCESS=1; shift ;;
    --skip_distance_process) SKIP_DISTANCE_PROCESS=1; shift ;;
    --skip_distance_model) SKIP_DISTANCE_MODEL=1; shift ;;
    --skip_distance_move) SKIP_DISTANCE_MOVE=1; shift ;;
    --skip_docking) SKIP_DOCKING=1; shift ;;
    --skip_force_optim) SKIP_FORCE_OPTIM=1; shift ;;
    --skip_score) SKIP_SCORE=1; shift ;;
    --keep_tmp) KEEP_TMP=1; shift ;;
    --overwrite_output) OVERWRITE_OUTPUT=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) fail "Unknown argument: $1" ;;
  esac
done

# -------------------------------
# Validate and normalize arguments
# -------------------------------
[[ -n "$INPUT_LIGAND" ]] || fail "--input_ligand is required."
[[ -n "$INPUT_PROTEIN" ]] || fail "--input_protein is required."

MAIN_DIR="$(abs_path "$MAIN_DIR")"
INPUT_LIGAND="$(abs_path "$INPUT_LIGAND")"
INPUT_PROTEIN="$(abs_path "$INPUT_PROTEIN")"

[[ -d "$MAIN_DIR" ]] || fail "main_dir not found: $MAIN_DIR"
[[ -f "$INPUT_LIGAND" ]] || fail "input ligand not found: $INPUT_LIGAND"
[[ -f "$INPUT_PROTEIN" ]] || fail "input protein not found: $INPUT_PROTEIN"

if [[ -z "$SAMPLE_NAME" ]]; then
  SAMPLE_NAME="$(infer_name_from_ligand "$INPUT_LIGAND")"
fi

if [[ -z "$DISTANCE_CONF_SIZE" ]]; then
  DISTANCE_CONF_SIZE="$CONF_NUM"
fi

if [[ -z "$OUTPUT_ROOT" ]]; then
  OUTPUT_ROOT="$(infer_output_root "$INPUT_LIGAND" "$SAMPLE_NAME")"
else
  OUTPUT_ROOT="$(abs_path "$OUTPUT_ROOT")"
fi

FINAL_OUTPUT_DIR="$OUTPUT_ROOT/$SAMPLE_NAME"

# Relative paths are resolved from MAIN_DIR because all commands run there.
cd "$MAIN_DIR"

log "Pipeline configuration"
echo "  MAIN_DIR             = $MAIN_DIR"
echo "  ENV_NAME             = $ENV_NAME"
echo "  INPUT_LIGAND         = $INPUT_LIGAND"
echo "  INPUT_PROTEIN        = $INPUT_PROTEIN"
echo "  SAMPLE_NAME          = $SAMPLE_NAME"
echo "  GPU                  = $GPU"
echo "  STEP                 = $STEP"
echo "  CONF_NUM             = $CONF_NUM"
echo "  STORE_DIR            = $STORE_DIR"
echo "  TMP_DATA_DIR         = $TMP_DATA_DIR"
echo "  RESULT_DIR           = $RESULT_DIR"
echo "  OUTPUT_ROOT          = $OUTPUT_ROOT"
echo "  FINAL_OUTPUT_DIR     = $FINAL_OUTPUT_DIR"

# -------------------------------
# 1. Optional environment installation
# -------------------------------
if [[ "$RUN_INSTALL" -eq 1 ]]; then
  log "Installing EC-Dock environment"
  bash "$INSTALL_SCRIPT" "$ENV_NAME" "$BASE_YAML" "$PIP_REQUIREMENTS"

  log "Activating conda environment: $ENV_NAME"
  activate_conda_env "$ENV_NAME"

  log "Editable install: ocp"
  cd "$MAIN_DIR/ocp"
  pip install -e ./

  log "Editable install: Distance_model"
  cd "$MAIN_DIR/Distance_model"
  pip install -e ./

  log "Editable install: EC-Dock root"
  cd "$MAIN_DIR"
  pip install -e ./
else
  log "Activating conda environment: $ENV_NAME"
  activate_conda_env "$ENV_NAME"
fi

cd "$MAIN_DIR"

# -------------------------------
# 2. Data preprocessing
# -------------------------------
if [[ "$SKIP_PREPROCESS" -eq 0 ]]; then
  log "Preprocessing user input data"
  python preprocess_data.py \
    --input_ligand "$INPUT_LIGAND" \
    --input_protein "$INPUT_PROTEIN" \
    --store_dir "$STORE_DIR" \
    --name "$SAMPLE_NAME" \
    --data_name "$TMP_DATA_DIR" \
    --result_dir "$RESULT_DIR"
else
  log "Skipping preprocessing"
fi

# -------------------------------
# 3. Generate CSV for distance model
# -------------------------------
if [[ "$SKIP_DISTANCE_PROCESS" -eq 0 ]]; then
  log "Generating distance-model input CSV"
  python Distance_model/interface/process_data.py
else
  log "Skipping distance CSV generation"
fi

# -------------------------------
# 4. Distance model inference
# -------------------------------
if [[ "$SKIP_DISTANCE_MODEL" -eq 0 ]]; then
  log "Running distance model"
  cd "$MAIN_DIR/Distance_model/interface"
  python interface.py \
    --gpu "$GPU" \
    --input_batch_file "$DISTANCE_CSV" \
    --output_ligand_dir "$DISTANCE_OUTPUT_DIR" \
    --model_dir "$DISTANCE_MODEL" \
    --cluster \
    --steric_clash_fix \
    --start_idx "$START_IDX" \
    --end_idx "$END_IDX" \
    --batch_size "$DISTANCE_BATCH_SIZE" \
    --conf_size "$DISTANCE_CONF_SIZE"
  cd "$MAIN_DIR"
else
  log "Skipping distance model"
fi

# -------------------------------
# 5. Move predicted distance files back to tmpdata/tmpdata
# -------------------------------
if [[ "$SKIP_DISTANCE_MOVE" -eq 0 ]]; then
  log "Moving distance files into temporary data directory"
  cd "$MAIN_DIR"
  python Distance_model/interface/data_move.py
else
  log "Skipping distance file move"
fi

# -------------------------------
# 6. Docking
# -------------------------------
if [[ "$SKIP_DOCKING" -eq 0 ]]; then
  log "Running docking"
  cd "$MAIN_DIR"
  python KGDiff/scripts/multiple_data_run.py \
    --gpu "$GPU" \
    --conf_num "$CONF_NUM" \
    --batch_size "$BATCH_SIZE" \
    --si "$START_IDX" \
    --ei "$DOCK_END_IDX" \
    --test_name "$TEST_NAME" \
    --step "$STEP" \
    --user_test_data_dir "$TMP_DATA_DIR" \
    --out_dir "$RESULT_DIR"
else
  log "Skipping docking"
fi

# -------------------------------
# 7. Force-field optimization
# -------------------------------
if [[ "$SKIP_FORCE_OPTIM" -eq 0 ]]; then
  log "Running force-field optimization"
  cd "$MAIN_DIR"
  python Diffusion_force_optim/force_optim.py --input_dir "$RESULT_DIR"
else
  log "Skipping force-field optimization"
fi

# -------------------------------
# 8. Scoring and final pose selection
# -------------------------------
if [[ "$SKIP_SCORE" -eq 0 ]]; then
  log "Running EC-Dock scoring"
  cd "$MAIN_DIR"
  python ECDockScore/posebusters_batch_deal.py --gpu "$GPU"
else
  log "Skipping EC-Dock scoring"
fi

# -------------------------------
# 9. Copy final result back to user input directory
# -------------------------------
log "Copying final result"
cd "$MAIN_DIR"

SOURCE_RESULT_DIR="$RESULT_DIR/$SAMPLE_NAME"
[[ -d "$SOURCE_RESULT_DIR" ]] || fail "result directory not found: $SOURCE_RESULT_DIR"

mkdir -p "$OUTPUT_ROOT"

# Keep the final output directory if it already exists.
# Copy the contents inside SOURCE_RESULT_DIR instead of copying SOURCE_RESULT_DIR itself.
# Example: tmpresault/5SB2/* -> input_data/5SB2/*
mkdir -p "$FINAL_OUTPUT_DIR"

if [[ -e "$FINAL_OUTPUT_DIR" ]]; then
  log "Final output exists. Copying files into it and overwriting same-name files: $FINAL_OUTPUT_DIR"
fi

cp -a "$SOURCE_RESULT_DIR"/. "$FINAL_OUTPUT_DIR"/
log "Final result saved to: $FINAL_OUTPUT_DIR"

# -------------------------------
# 10. Cleanup
# -------------------------------
if [[ "$KEEP_TMP" -eq 0 ]]; then
  log "Cleaning temporary directories"
  rm -rf "$TMP_DATA_DIR" "$RESULT_DIR" "$STORE_DIR"
else
  log "Keeping temporary directories: $TMP_DATA_DIR, $RESULT_DIR, $STORE_DIR"
fi

log "EC-Dock pipeline finished successfully"
