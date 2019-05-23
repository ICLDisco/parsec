FROM debian:testing
MAINTAINER Thomas Herault <herault@icl.utk.edu>
RUN apt-get update
RUN apt-get install -y python pkg-config git cmake
RUN apt-get install -y gcc binutils-doc build-essential make autoconf automake libtool flex bison gdb libgcc1-dbg libgomp1-dbg libitm1-dbg libatomic1-dbg libc-dbg libxml2-dev libhwloc-dev libgtg-dev graphviz-dev libopenmpi-dev

RUN git clone https://bitbucket.org/icldistcomp/parsec.git
RUN mkdir parsec/build
RUN cd parsec/build && cmake .. -DCMAKE_BUILD_TYPE=Debug -DBUILD_DPLASMA=OFF -DPARSEC_CALL_TRACE=ON -DPARSEC_PROF_GRAPHER=ON -DPARSEC_PROF_RUSAGE_EU=ON -DPARSEC_PROF_TRACE=ON -DPARSEC_WITH_DEVEL_HEADERS=ON -DSUPPORT_FORTRAN=OFF
RUN cd parsec/build && make && ctest

