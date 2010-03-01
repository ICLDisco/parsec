#!/bin/bash

file=$1
flags="-4 -s -b -g"

# Explanation for petit flags:
# -g ignore Entry/Exit
# -o calculate killed dependencies
# -s ignore scalar deps
# -b print relations
# -4 Skip "value-based" deps (useful only because value-based dep analysis breaks petit).

${OMEGA_HOME}/petit/obj/petit ${flags} ${file} > /tmp/out.$$

perl prettyPrint.pl ${file} /tmp/out.$$
rm /tmp/out.$$
