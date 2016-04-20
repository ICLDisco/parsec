FROM debian:8.4
MAINTAINER Thomas Herault <herault@icl.utk.edu>
RUN apt-get update
RUN apt-get install -y gcc binutils-doc gcc-multilib make autoconf automake libtool flex bison gdb gcc-4.9-multilib libgcc1-dbg libgomp1-dbg libitm1-dbg libatomic1-dbg libc-dbg
RUN apt-get install -y python pkg-config cmake git
RUN git clone https://bitbucket.org/icldistcomp/parsec.git
RUN cd parsec && cmake . -DCMAKE_BUILD_TYPE=Debug -DBUILD_DPLASMA=OFF -DDAGUE_CALL_TRACE=ON -DDAGUE_PROF_GRAPHER=ON -DDAGUE_PROF_RUSAGE_EU=ON -DDAGUE_PROF_TRACE=ON -DDAGUE_WITH_DEVEL_HEADERS=ON -DSUPPORT_FORTRAN=OFF -DDAGUE_DIST_WITH_MPI=OFF
RUN cd parsec && make && ctest

