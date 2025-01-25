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
spack env activate parsec

DEBUG=ON
[ $BUILD_TYPE = "Release" ] && DEBUG=OFF

if [ -z "$BUILD_DIRECTORY" ]; then
   echo Error! ENV \$BUILD_DIRECTORY is undefined.
   exit 1
fi

if [ -z "$INSTALL_DIRECTORY" ]; then
   echo Error! ENV \$INSTALL_DIRECTORY is undefined.
   exit 1
fi

! read -d '' BUILD_CONFIG << EOF
        -G Ninja
        -DCMAKE_BUILD_TYPE=$BUILD_TYPE
        -DPARSEC_DEBUG_NOISIER=$DEBUG
        -DPARSEC_DEBUG_PARANOID=$DEBUG
        -DBUILD_SHARED_LIBS=$SHARED_TYPE
        -DPARSEC_PROF_TRACE=$PROFILING
        -DMPIEXEC_PREFLAGS='--bind-to;none;--oversubscribe'
        -DCMAKE_INSTALL_PREFIX=$INSTALL_DIRECTORY
EOF
export BUILD_CONFIG
