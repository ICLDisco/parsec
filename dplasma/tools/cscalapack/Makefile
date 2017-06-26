#export OMPI_CC=/opt/intel/bin/icc 
#composer_xe_2013.3.163/bin/intel64/icc
#export OMPI_FC=/opt/intel/bin/ifort
#composer_xe_2013.3.163/bin/intel64/ifort
MPICC=mpicc
MPIF77=mpif77
CFLAGS+=-Wall
CC=$(MPICC)
LD=mpif77
#LD=mpif77 -nofor-main
MKLDIR=-L${MKLROOT}/lib/intel64
#MKLDIR=-L/nics/e/sw/local/kfs/opt/intel/composer_xe_2011_sp1.11.339/mkl/lib/intel64
#SMKL=$(MKLDIR)  -lmkl_scalapack_lp64 -lmkl_blacs_openmpi_lp64 -lmkl_sequential -lmkl_intel_lp64 -lmkl_core
#MKL>=2016
SMKL=$(MKLDIR) -Wl,-Bstatic -Wl,--start-group -lmkl_scalapack_lp64 -lmkl_blacs_openmpi_lp64 -lmkl_sequential -lmkl_gf_lp64 -lmkl_core -Wl,--end-group -Wl,-Bdynamic
# Old MKL < 11.xx
#TMKL=-L/opt/mkl/lib/em64t -lmkl_scalapack_lp64 -lmkl_lapack -lmkl_blacs_openmpi_lp64 -lmkl -lmkl_gf_lp64 -lmkl_core -lguide -lpthread
# New MKL > 11.xx
#TMKL=$(MKLDIR)  -lmkl_scalapack_lp64 -lmkl_gf_lp64 -lmkl_core -lmkl_gnu_thread -lmkl_blacs_openmpi_lp64 -lm -fopenmp
#MKL>=2016
TMKL=$(MKLDIR) -Wl,-Bstatic -Wl,--start-group -lmkl_scalapack_lp64 -lmkl_gf_lp64 -lmkl_core -lmkl_gnu_thread -lmkl_blacs_openmpi_lp64 -Wl,--end-group -Wl,-Bdynamic -lm -fopenmp
#TMKL=$(MKLDIR)  -lmkl_scalapack_lp64 -lmkl_intel_lp64 -lmkl_core -lmkl_intel_thread -lmkl_blacs_openmpi_lp64 -lm -openmp

TLIBS=$(TMKL)
SLIBS=$(SMKL)
TARGETS=pdpotrf tpdpotrf \
        pdgetrf tpdgetrf \
        pdgeqrf tpdgeqrf \
        pdsyev  tpdsyev  \
        pdgebrd tpdgebrd \
        pdgemm  tpdgemm  \
        pdpotri tpdpotri

all: $(TARGETS)

t%: %.o common.o
	$(LD) $(LDFLAGS)	-o $@ $^ $(TLIBS)

p%: p%.o common.o
	$(LD) $(LDFLAGS)	-o $@ $^ $(SLIBS)

clean:
	$(RM) *.o $(TARGETS)

