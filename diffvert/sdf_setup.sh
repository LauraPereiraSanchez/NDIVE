export XLA_PYTHON_CLIENT_PREALLOCATE=false
export NDIVE_MODEL_PATH="/sdf/home/l/lapereir/NDIVE/diffvert/models/saved_models"
export CONDA_PREFIX=/gpfs/slac/atlas/fs1/d/recsmith/mambaforge
export PATH=${CONDA_PREFIX}/bin/:$PATH
source ${CONDA_PREFIX}/etc/profile.d/conda.sh
conda env list
conda activate diffvert-js
CUDNN_PATH=$(dirname $(python3 -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib

