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

get_fresh_copy_of_spack() {
  echo "git clone spack branch ${RUNNER_SPACK_BRANCH} @ ${SPACK_DIR}"
  git clone -b ${RUNNER_SPACK_BRANCH} https://github.com/spack/spack $SPACK_DIR || true
  # We do a git pull to create the .git/FETCH_HEAD for the check
  # next time the script is executed.
  cd $SPACK_DIR && git pull && git status
}

# Only do the heavy lifting once per github_action invocation
if [ ${GITHUB_ACTION} != "setup" ]; then
  echo "::group::Loading spack ${RUNNER_ENV} env"
  source $SPACK_DIR/share/spack/setup-env.sh
  spack env activate ${RUNNER_ENV}
  echo "::endgroup::"
  return
fi

save_dir=`pwd`
#
# Only update the spack clone once a day
#
echo "::group::Spack environment setup"
if [ "x${RUNNER_SPACK_BRANCH}" == "x" ]; then
  RUNNER_SPACK_BRANCH="e4s-23.08"  # fall back to something stable
fi

if [ -r ${SPACK_DIR}/.git/HEAD ]; then
  # We should never allow spack to update it's packages, it increases the
  # opportunity for mishandling of the runner environment. Instead, when
  # there is a need for update, the entire runner should be flushed, such
  # that a fresh spack install is created.
  last_update=`stat -c %Y ${SPACK_DIR}/.git/HEAD`
  current=`date -d now "+%s"`
  echo "spack is $(((current - last_update) / 86400)) days old"
  HEAD=`head -n1 ${SPACK_DIR}/.git/HEAD`
  if [ "$HEAD" != "ref: refs/heads/${RUNNER_SPACK_BRANCH}" ]; then
    echo "Remove current spack install ($HEAD) and install ${RUNNER_SPACK_BRANCH} "
    rm -rf $SPACK_DIR
    get_fresh_copy_of_spack
  fi
  # if [ $(((current - last_update) / 86400)) -gt 1 ]; then
  #   echo "Last update ${last_update}, current ${current}: Do git pull spack"
  #   cd $SPACK_DIR && git pull
  # else
  #   echo "git pull was less than one day ago"
  # fi
else
  get_fresh_copy_of_spack
fi

rm -rf ${HOME}/.spack

echo "Load spack environment"
source $SPACK_DIR/share/spack/setup-env.sh

spack compiler find

# Start with a fresh env every time
spack env remove -y ${RUNNER_ENV} || true

cd ${save_dir}
mkdir ${RUNNER_ENV} || true

# Show the known envs
if spack env activate "${RUNNER_ENV}"; then
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

