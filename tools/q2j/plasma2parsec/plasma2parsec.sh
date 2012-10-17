#!/bin/sh

echo $1

# Change include
sed -i 's/common\.h/core.h/' $1

# Change BLKADDR
sed -i 's/BLKADDR(\([A-Z0-9a-z]*\), PLASMA_Complex64_t, \([a-z]*\), \([a-z]*\))/\1[\2][\3]/' $1

# Remove PLASMA_sequence and request for function call
sed -i '{N;s/\(.*plasma_pz.*(.*\),\s*\n\s*PLASMA_sequence \*sequence, PLASMA_request \*request)/\1)/}' $1

sed -i '/Task_Flag/d' $1

sed -i '/plasma_context_t \*plasma;/d' $1
sed -i '/ plasma = plasma_context_self();/d' $1
sed -i '{N;/if (sequence->status != PLASMA_SUCCESS).*\n.*return;/d}' $1

destcpp=`echo $1 | sed 's/pz/z/' | sed 's/\.c/\.cpp/'` 
destjdf=`echo $1 | sed 's/pz/z/' | sed 's/\.c/\.jdf/'` 
cpp -P -E $1 -o $destcpp

sed -i '/#pragma/d' $destcpp

../q2j -anti $destcpp > $destjdf

# Replace PLASMA_Complex64_t by Dague
sed -i 's/PLASMA_Complex/dague_complex/g' $destjdf

# Replace PLASMA_desc by tiled_matrix_desc_t
#sed -i 's/PLASMA_desc/tiled_matrix_desc_t/g' $destjdf

# # Convert desc_X to desc 
# sed -i 's/desc_\([A-Z0-9]*[ .,]\)/desc\1/g' $destjdf

# # Convert data_X to X
# sed -i 's/data_\([A-Z0-9]*[ (]\)/data\1/g' $destjdf

# Remove extra parentheses
sed -i 's/(\(desc[A-Z0-9]*\.[mn][bt]\))/\1/g' $destjdf
sed -i 's/(\(desc[A-Z0-9]*\.[mn]\))/\1/g' $destjdf

# Remove #line to avoid confusion during compilation
sed -i '/#line/d' $destjdf

# Remove sequence and request (should be an option)
sed -i '/PLASMA_sequence/d' $destjdf
sed -i '/PLASMA_request/d' $destjdf
