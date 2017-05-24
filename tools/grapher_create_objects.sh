#!/bin/sh

FILES="../dplasma/lib/LU.h
../dplasma/lib/LU_sd.h
../dplasma/lib/QR.h
../dplasma/lib/TSQR.h
../dplasma/lib/dpotrf.h
../dplasma/lib/sgeqrt.h
../dplasma/lib/spotrf_ll.h
../dplasma/lib/spotrf_rl.h"

cat<<EOF
int uplo;
EOF

globals() {
    awk -v FS='[ |;]+' 'BEGIN {dump=0} $6=="globals" {dump=1} $6=="data" {dump=0} $2=="int" && dump==1 { print $3}' $1
}

matrices() {
    awk -v FS='[ |;|\*]+' 'BEGIN {dump=0} $6=="data" {dump=1} $2=="parsec_data_collection_t" && dump==1 { print $3}' $1
}

onefile() {
    BASEFILE=$(basename $1)
    BASE=$(basename $1 .h)
    GLOBALS=$(globals $1)
    MAT=$(matrices $1)

    /bin/echo "#include \"dplasma/lib/$BASEFILE\""
    cat<<EOF
static parsec_taskpool_t *${BASE}_create(int argc, char **argv)
{
EOF

    for g in $GLOBALS ; do
        /bin/echo "  int $g = -1; int ${g}_set = 0;"
    done

    cat <<EOF
  parsec_taskpool_t *ret;
  int allset = 1;
  int i;
  for(i = 0; i < argc; i+= 2) {
EOF

    for g in $GLOBALS ; do
        /bin/echo "    TRY_SET($g);"
    done

    /bin/echo "  }"
    /bin/echo ""

    for g in $GLOBALS ; do
        /bin/echo "  TEST_SET(\"$BASE\", $g);"
    done

    cat<<EOF
  if( allset == 0 )
    return NULL;

EOF
    /bin/echo -n "  ret = (parsec_taskpool_t*)parsec_${BASE}_new"

    V="("
    for m in $MAT; do
        /bin/echo -n "$V&pseudo_desc"
        V=", "
    done

    for g in $GLOBALS; do
        /bin/echo -n "$V$g"
    done

    /bin/echo ");"
    /bin/echo "  return ret;"
    /bin/echo "}"
    /bin/echo ""
}

nb=0
for f in $FILES; do
    onefile $f
    nb=$((nb+1))
done

/bin/echo "#define NB_CREATE_FUNCTIONS $nb"
/bin/echo "static create_function_t create_functions[NB_CREATE_FUNCTIONS] = {"
V=""
for f in $FILES; do
    BASE=$(basename $f .h)
    /bin/echo "$V  { .command_name = \"$BASE\", .create_function = ${BASE}_create }"
    V=","
done
/bin/echo "};"
