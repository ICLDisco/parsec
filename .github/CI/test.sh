#!/bin/bash -e

source .github/CI/env_setup.sh

if [ "$BUILD_TYPE" = "Release" ]; then
   # enable devices only in tests that explicitely require them
   PARSEC_MCA_device_cuda_enabled=0
   PARSEC_MCA_device_hip_enabled=0
   # restrict memory use for oversubscribed runners
   PARSEC_MCA_device_cuda_memory_use=10
   PARSEC_MCA_device_hip_memory_use=10

   ctest --output-on-failure
fi
