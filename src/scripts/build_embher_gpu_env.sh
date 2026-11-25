#!/bin/bash
#SBATCH --job-name=build_embher_gpu
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --time=3:00:00

# Builds the embher_gpu conda env (Borzoi capture and attribution, workflows 06 and 15) in two steps:
#   1. envs/embher_gpu.yaml: the torch / CUDA stack and every dependency that resolves together;
#   2. decima, grelu, enformer-pytorch and embher[attribution] at its pinned commit, with `pip install --no-deps`:
#      every grelu release pins captum==0.5.0 and enformer-pytorch pins transformers==4.56.2, while this env runs
#      captum 0.9.0 and transformers 4.57.6 as the paper's runs did (see the yaml header).
# Run from the repository root, on a machine with internet access (no GPU needed):
#   mkdir -p logs && sbatch src/scripts/build_embher_gpu_env.sh      # or: bash src/scripts/build_embher_gpu_env.sh
# Options (environment variables): CONDA_SH (conda.sh of the conda to use; default: the one on PATH),
# ENV_PREFIX (create the env at this path instead of naming it embher_gpu).

set -euo pipefail
echo Starting at $(date) on $HOSTNAME

YAML=envs/embher_gpu.yaml
EMBHER_COMMIT=ebbad746726e250215c5a7ec59b5c0eb4ba0a3bc

source "${CONDA_SH:-$(conda info --base)/etc/profile.d/conda.sh}"
# the yaml pins the CUDA 13.0 build of pytorch, which needs CUDA visible to conda (absent on a machine without a GPU)
export CONDA_OVERRIDE_CUDA="${CONDA_OVERRIDE_CUDA:-13.3}"
if [ -n "${ENV_PREFIX:-}" ]; then
    conda env create -p "$ENV_PREFIX" -f "$YAML"
    conda activate "$ENV_PREFIX"
else
    conda env create -n embher_gpu -f "$YAML"
    conda activate embher_gpu
fi

pip install --no-deps decima==0.7.0 grelu==1.2.0 enformer-pytorch==0.8.12 \
    "embher[attribution] @ git+https://github.com/TheBil99/EMBHER.git@${EMBHER_COMMIT}"

# sanity check: the commands workflows 06 and 15 call, and the imports they make
python -c "
import torch, embher
from embher.embedding.attribution.attributer import BorzoiAttributer
import borzoi_pytorch, captum, tangermeme, pyBigWig, grelu
from grelu.interpret.motifs import scan_sequences
flags = lambda: (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark, torch.get_float32_matmul_precision())
before = flags()
from decima.core.attribution import Attribution
print('torch', torch.__version__, '| CUDA build', torch.version.cuda, '| embher', embher.__version__)
assert torch.version.cuda, 'pytorch is a CPU-only build: the env cannot use the GPU'
print('cudnn (deterministic, benchmark, fp32 matmul) before decima import:', before, '| after:', flags())
"
which embher-capture embher-attribute

echo Finished at $(date)
