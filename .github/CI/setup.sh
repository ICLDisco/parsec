#!/bin/bash -e

source .github/CI/env_setup.sh

cmake -E make_directory $BUILD_DIRECTORY
