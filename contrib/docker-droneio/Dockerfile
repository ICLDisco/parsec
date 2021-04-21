FROM debian:testing
LABEL authors="Thomas Herault <herault@icl.utk.edu>, Aurelien Bouteiller <bouteill@icl.utk.edu>"
RUN apt-get update
RUN apt-get install -y python pkg-config git cmake
RUN apt-get install -y gcc binutils-doc build-essential make autoconf automake libtool flex bison gdb libgcc1-dbg libgomp1-dbg libitm1-dbg libatomic1-dbg libc-dbg libxml2-dev libhwloc-dev libgtg-dev graphviz-dev libopenmpi-dev

RUN git clone https://bitbucket.org/icldistcomp/parsec.git
RUN mkdir parsec/build
RUN cd parsec/build && ../configure --prefix=$PWD --enable-debug=paranoid --enable-prof-trace --enable-prof-grapher --disable-fortran 
RUN cd parsec/build && make && ctest --output-on-failure

