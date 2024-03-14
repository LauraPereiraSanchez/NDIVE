export XLA_PYTHON_CLIENT_PREALLOCATE=false
export NDIVE_MODEL_PATH=/fs/ddn/sdf/group/atlas/d/lapereir/Vertexing/saved_models/
conda env list
conda activate vertex
CUDNN_PATH=$(dirname $(python3 -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib

