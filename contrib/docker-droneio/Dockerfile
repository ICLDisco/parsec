FROM debian:8
MAINTAINER Thomas Herault <herault@icl.utk.edu>
RUN apt-get update
RUN apt-get install -y python pkg-config git wget
RUN apt-get install -y gcc binutils-doc gcc-multilib make autoconf automake libtool flex bison gdb gcc-4.9-multilib libgcc1-dbg libgomp1-dbg libitm1-dbg libatomic1-dbg libc-dbg libxml2-dev libhwloc-dev libgtg-dev graphviz-dev libopen-trace-format-dev

RUN wget https://github.com/Kitware/CMake/releases/download/v3.14.3/cmake-3.14.3-Linux-x86_64.tar.gz && \
    tar xf cmake-3.14.3-Linux-x86_64.tar.gz && \
    rm cmake-3.14.3-Linux-x86_64.tar.gz && \
    cp -r cmake-3.14.3-Linux-x86_64/* /usr && \
    rm -rf cmake-3.14.3-Linux-x86_64

RUN git clone https://bitbucket.org/icldistcomp/parsec.git
RUN mkdir parsec/build
RUN cd parsec/build && cmake .. -DCMAKE_BUILD_TYPE=Debug -DBUILD_DPLASMA=OFF -DPARSEC_CALL_TRACE=ON -DPARSEC_PROF_GRAPHER=ON -DPARSEC_PROF_RUSAGE_EU=ON -DPARSEC_PROF_TRACE=ON -DPARSEC_WITH_DEVEL_HEADERS=ON -DSUPPORT_FORTRAN=OFF -DPARSEC_DIST_WITH_MPI=OFF
RUN cd parsec/build && make && ctest

