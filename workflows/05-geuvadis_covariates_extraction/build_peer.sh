#!/bin/bash
#SBATCH --job-name=build_peer
#SBATCH --output=logs/%x_%A.out
#SBATCH --error=logs/%x_%A.err
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --time=30:00

# One-off build of the PEER standalone CLI (peertool) from a clone of https://github.com/PMBio/peer at
# src/peer/ (git-ignored). Only the standalone C++ binary is built (-DBUILD_PEERTOOL=ON), not the Python/R
# bindings (both OFF); SWIG is only needed to satisfy the unconditional find_package(SWIG REQUIRED) at the top of
# PEER's CMakeLists.txt. Needs a C++ compiler and CMake on PATH (module load / conda), and the `embher_env`
# conda env (it provides swig if you add `swig` to it: `conda install -n embher_env -c conda-forge swig`).
# Run from the project root:  sbatch workflows/05-geuvadis_covariates_extraction/build_peer.sh

echo Starting at $(date) on $HOSTNAME

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate embher_env

[ -d src/peer ] || git clone https://github.com/PMBio/peer.git src/peer
cd src/peer
mkdir -p build
cd build
cmake .. -DBUILD_PEERTOOL=ON -DBUILD_PYTHON_PACKAGE=OFF -DBUILD_R_PACKAGE=OFF \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5
make

echo "peertool binary:"
find . -name peertool

echo Finished at $(date)
