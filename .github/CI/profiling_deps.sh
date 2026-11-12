#!/bin/bash
# Fetch the dependencies of the optional profiling substrates (OTF2, NVTX,
# Score-P) into the prefix given as first argument, so that all of them can
# be compiled. Already present dependencies are kept, the prefix is meant to
# be cached between runs.
#
#  - OTF2 is built from its release tarball
#  - NVTX is header-only
#  - Score-P: only the user adapter headers are needed to build PaRSEC, the
#    measurement system is provided by applications linked with Score-P

set -e

PREFIX=${1:?usage: $0 <prefix>}
OTF2_VERSION=3.1.1
SCOREP_VERSION=9.4
NVTX_VERSION=v3.7.0

mkdir -p "$PREFIX"
cd "$PREFIX"

if [ ! -x "$PREFIX/otf2/bin/otf2-config" ]; then
    curl -sSfLO "https://perftools.pages.jsc.fz-juelich.de/cicd/otf2/tags/otf2-${OTF2_VERSION}/otf2-${OTF2_VERSION}.tar.gz"
    tar xzf "otf2-${OTF2_VERSION}.tar.gz"
    (cd "otf2-${OTF2_VERSION}" && ./configure --prefix="$PREFIX/otf2" && make -j"$(nproc)" install)
    rm -rf "otf2-${OTF2_VERSION}" "otf2-${OTF2_VERSION}.tar.gz"
fi

if [ ! -f "$PREFIX/nvtx/c/include/nvtx3/nvToolsExt.h" ]; then
    rm -rf "$PREFIX/nvtx"
    git clone --quiet --depth 1 --branch "$NVTX_VERSION" https://github.com/NVIDIA/NVTX.git "$PREFIX/nvtx"
fi

if [ ! -f "$PREFIX/scorep/include/scorep/SCOREP_User_Functions.h" ]; then
    curl -sSfLO "https://perftools.pages.jsc.fz-juelich.de/cicd/scorep/tags/scorep-${SCOREP_VERSION}/scorep-${SCOREP_VERSION}.tar.gz"
    mkdir -p "$PREFIX/scorep"
    tar xzf "scorep-${SCOREP_VERSION}.tar.gz" -C "$PREFIX/scorep" --strip-components=1 \
        "scorep-${SCOREP_VERSION}/include/scorep/SCOREP_User_Functions.h" \
        "scorep-${SCOREP_VERSION}/include/scorep/SCOREP_User_Types.h" \
        "scorep-${SCOREP_VERSION}/include/scorep/SCOREP_User_Variables.h" \
        "scorep-${SCOREP_VERSION}/include/scorep/SCOREP_PublicTypes.h"
    rm -f "scorep-${SCOREP_VERSION}.tar.gz"
fi
