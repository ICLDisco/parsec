#!/bin/sh

q2j=../q2j

generatejdf()
{
    src=/tmp/$1
    dstcpp=/tmp/`echo $1 | sed 's/pz/z/' | sed 's/\.c/\.cpp/'`
    dstjdf=`echo $1 | sed 's/pz/z/' | sed 's/\.c/\.jdf/'`

    echo $dstjdf
    echo "  Prepare the file"
    cp $1 $src

    # Change include
    sed -i 's/common\.h/core.h/' $src

    # Change BLKADDR
    echo "  Replace BLKADDR macros"
    sed -i 's/BLKADDR(\([A-Z0-9a-z]*\), PLASMA_Complex64_t, \([a-z]*\), \([a-z]*\))/\1[\2][\3]/' $src

    # Remove PLASMA_sequence and request for function call
    sed -i '{N;s/\(.*plasma_pz.*(.*\),\s*\n\s*PLASMA_sequence \*sequence, PLASMA_request \*request)/\1)/}' $src
    sed -i '/Task_Flag/d' $src
    sed -i '/plasma_context_t \*plasma;/d' $src
    sed -i '/ plasma = plasma_context_self();/d' $src
    sed -i '{N;/if (sequence->status != PLASMA_SUCCESS).*\n.*return;/d}' $src

    # Remove zone, mzone, done, mdone
    sed -i '/PLASMA_Complex64_t *zone/d'  $src
    sed -i '/PLASMA_Complex64_t *mzone/d' $src
    sed -i '/PLASMA_Complex64_t *done/d'  $src
    sed -i '/PLASMA_Complex64_t *mdone/d' $src
    sed -i 's/mzone/-1/g'  $src
    sed -i 's/zone/1./g'   $src
    sed -i 's/mdone/-1./g' $src
    sed -i 's/done/1./g'   $src

    echo "  Precompile the file to get the correct input format"
    cpp -I. -P -E $src -o $dstcpp

    sed -i '/#pragma/d' $dstcpp

    echo "  Generate the jdf file"
    $q2j -anti $dstcpp > $dstjdf

    echo "   Postprocessing"
    # Replace PLASMA_Complex64_t by Dague
    sed -i 's/PLASMA_Complex/dague_complex/g' $dstjdf

    # Replace PLASMA_desc by tiled_matrix_desc_t
    #sed -i 's/PLASMA_desc/tiled_matrix_desc_t/g' $dstjdf

    # Convert desc_X to desc
    # sed -i 's/desc_\([A-Z0-9]*[ .,]\)/desc\1/g' $dstjdf

    # Convert data_X to X
    # sed -i 's/data_\([A-Z0-9]*[ (]\)/data\1/g' $dstjdf

    # Remove extra parentheses
    sed -i 's/(\(desc[A-Z0-9]*\.[mn][bt]\))/\1/g' $dstjdf
    sed -i 's/(\(desc[A-Z0-9]*\.[mn]\))/\1/g' $dstjdf

    # Remove #line to avoid confusion during compilation
    sed -i '/#line/d' $dstjdf

    # Remove sequence and request (should be an option)
    sed -i '/PLASMA_sequence/d' $dstjdf
    sed -i '/PLASMA_request/d' $dstjdf

    rm -f $src $dstcpp
}

for i in $*
do
    generatejdf $i
done

