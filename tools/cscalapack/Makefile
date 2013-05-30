MPICC=mpicc
MPIF77=mpif77
CFLAGS="-Wall"
CC=$(MPICC)
LD=mpif77
SMKL=-L/opt/mkl/lib/em64t -lmkl_scalapack_lp64 -lmkl_lapack95_lp64 -lmkl_blacs_openmpi_lp64 -lmkl_sequential -lmkl_intel_lp64 -lmkl_core 
# Old MKL < 11.xx
#TMKL=-L/opt/mkl/lib/em64t -lmkl_scalapack_lp64 -lmkl_lapack -lmkl_blacs_openmpi_lp64 -lmkl -lmkl_gf_lp64 -lmkl_core -lguide -lpthread
# New MKL > 11.xx
TMKL=-L/opt/mkl/lib/em64t -lmkl_scalapack_lp64 -lmkl_lapack95_lp64 -lmkl_blacs_openmpi_lp64 -lmkl_intel_thread -lmkl_intel_lp64 -lmkl_core -liomp5 -lpthread -lm
TLIBS=$(TMKL)
SLIBS=$(SMKL)
TARGETS=pdpotrf tpdpotrf pdgetrf tpdgetrf pdgeqrf tpdgeqrf pdsyev tpdsyev

all: $(TARGETS)

t%: %.o
	$(LD) -o $@ $^ $(TLIBS)

p%: p%.o
	$(LD) -o $@ $^ $(SLIBS)

clean:
	$(RM) *.o $(TARGETS)

