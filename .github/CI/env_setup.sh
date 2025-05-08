# This file should be "sourced" into your environment

# Show the executed command, but don't affect spawned shells
trap 'echo "# $BASH_COMMAND"' DEBUG # Show commands

echo "Loading environment"
if [[ -z "$SPACK_SETUP" || ! -e "$SPACK_SETUP" ]]; then
   echo Error! Environment variable \$SPACK_SETUP must point
   echo to a valid setup-env.sh Spack setup script.
   exit 1
fi
source $SPACK_SETUP
spack env activate -V parsec
spack load python
spack load cmake
spack load openmpi
spack load ninja

DEBUG=ON
[ "$BUILD_TYPE" = "Release" ] && DEBUG=OFF

CUDA=OFF
HIP=OFF
if [ "$DEVICE" = "gpu_nvidia" ]; then
   spack load cuda
   CUDA=ON
elif [ "$DEVICE" = "gpu_amd" ]; then
   HIP=ON
fi

! read -d '' BUILD_CONFIG << EOF
        -G Ninja
        -DCMAKE_BUILD_TYPE=$BUILD_TYPE
        -DPARSEC_DEBUG_NOISIER=$DEBUG
        -DPARSEC_DEBUG_PARANOID=$DEBUG
        -DBUILD_SHARED_LIBS=$SHARED_TYPE
        -DPARSEC_PROF_TRACE=$PROFILING
        -DMPIEXEC_PREFLAGS='--bind-to;none;--oversubscribe'
        -DCMAKE_INSTALL_PREFIX=$PARSEC_INSTALL_DIR
        -DPARSEC_GPU_WITH_CUDA=$CUDA
        -DPARSEC_GPU_WITH_HIP=$HIP
EOF
export BUILD_CONFIG
