#!/bin/bash -e

source .github/CI/env_setup.sh

cmake --build . --target install
