MPICC=mpicc
MPIF77=mpif77
CFLAGS="-Wall"
CC=$(MPICC)
LD=$(MPIF77)
MKL=-L/opt/mkl/lib/em64t -lmkl_scalapack_lp64 -lmkl_lapack -lmkl_blacs_openmpi_lp64 -lmkl_sequential -lmkl_gf_lp64 -lmkl_core 
LIBS=$(MKL)

all: scadpotrf scadgetrf

scadpotrf: scalapackPOTRF.o
	$(LD) -o $@ $^ $(LIBS)

scadgetrf: scalapackLU.o
	$(LD) -o $@ $^ $(LIBS)

clean:
	rm *.o scadpotrf scadgetrf

