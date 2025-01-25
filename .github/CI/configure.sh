#!/bin/bash -e

source .github/CI/env_setup.sh

cmake $GITHUB_WORKSPACE -DCMAKE_BUILD_TYPE=$BUILD_TYPE $BUILD_CONFIG


