#!/bin/bash -e

source .github/CI/env_setup.sh

cmake $GITHUB_WORKSPACE $BUILD_CONFIG


