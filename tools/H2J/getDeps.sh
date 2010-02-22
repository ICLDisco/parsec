#!/bin/bash
# -g ignore Entry/Exit
# -o calculate killed dependencies
# -s ignore scalar deps
# -b print relations
# -4 Skip "value-based" deps (useful only because value-based dep analysis breaks petit).

flags="-4 -s -b -g"
file=$1

/Users/adanalis/Desktop/Research/PLASMA_Distributed/Omega/petit/obj/petit ${flags} ${file} > /tmp/out.$$

perl prettyPrint.pl < /tmp/out.$$
rm /tmp/out.$$
