# This file should be "sourced" into your environment
# to set up the spack repository

SPACK_DIR=${HOME}/spack

sload () {
  # Installs a software package and loads it into the environment
  spack install --reuse --fail-fast $@
  spack load --first $@
}

slocation () {
  # Returns the installation directory of a given software package
  HASH=`spack find --no-groups --loaded -l $@ | head -1 | awk '{print $1}'`
  spack location -i /$HASH
}

# Only do the heavy lifting once per github_action invocation
if [ ${GITHUB_ACTION} != "setup" ]; then
  source $SPACK_DIR/share/spack/setup-env.sh
  spack env activate ${RUNNER_ENV}
  return
fi

save_dir=`pwd`
#
# Only update the spack clone once a day
#
echo "::group::Spack environment setup"
if [ -r ${SPACK_DIR}/.git/FETCH_HEAD ]; then
  last_update=`stat -c %Y ${SPACK_DIR}/.git/FETCH_HEAD`
  current=`date -d now "+%s"`
  if [ $(((current - last_update) / 86400)) -gt 1 ]; then
    echo "Last update ${last_update}, current ${current}: Do git pull spack"
    cd $SPACK_DIR && git pull
  else
    echo "git pull was less than one day ago"
  fi
else
  echo "git clone spack"
  git clone https://github.com/spack/spack $SPACK_DIR || true
  # We do a git pull to create the .git/FETCH_HEAD for the check
  # next time the script is executed.
  cd $SPACK_DIR && git pull && git status
fi

echo "Load spack environment"
source $SPACK_DIR/share/spack/setup-env.sh

spack external find
spack compiler find

# Start with a fresh env every time
spack env remove -y ${RUNNER_ENV}

cd ${save_dir}
mkdir ${RUNNER_ENV} || true

# Show the known envs
if spack env list | grep ${RUNNER_ENV}; then
  if diff ${GITHUB_WORKSPACE}/.github/CI/${RUNNER_ENV}.yaml ${SPACK_DIR}/var/spack/environments/${RUNNER_ENV}/spack.yaml > /dev/null; then
    echo "Update spack env ${RUNNER_ENV}"
    cp ${GITHUB_WORKSPACE}/.github/CI/${RUNNER_ENV}.yaml ${SPACK_DIR}/var/spack/environments/${RUNNER_ENV}/spack.yaml
  fi
  spack env update ${RUNNER_ENV}
else
  echo "Create spack env ${RUNNER_ENV}"
  spack env create ${RUNNER_ENV} ${GITHUB_WORKSPACE}/.github/CI/${RUNNER_ENV}.yaml
fi
spack env activate ${RUNNER_ENV}
spack concretize --force
spack install --reuse

echo "::endgroup::"

