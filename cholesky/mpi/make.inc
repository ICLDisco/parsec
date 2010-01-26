# 
# make.inc to be included from general dplasma Makefile
#

CHOLESKY_MPI_DIR = cholesky/mpi
CHOLESKY_MPI_TARGETS = $(CHOLESKY_MPI_DIR)/dist_exec
CHOLESKY_MPI_OBJECTS = $(CHOLESKY_MPI_DIR)/dist_exec.o $(CHOLESKY_MPI_DIR)/data_management.o $(CHOLESKY_MPI_DIR)/cholesky_mpi.o

$(CHOLESKY_MPI_DIR)/dist_exec: $(CHOLESKY_MPI_OBJECTS) dplasma.a
	$(MPILINKER) -o $@ $^ $(LDFLAGS) $(LIB)

TESTING_TARGETS += $(CHOLESKY_MPI_TARGETS)
MPI_OBJECTS += $(CHOLESKY_MPI_OBJECTS)
CLEAN_OBJECTS += $(CHOLESKY_MPI_TARGETS) $(CHOLESKY_MPI_OBJECTS) $(subst .jdf,.c,$(wildcard $(CHOLESKY_MPI_DIR)/*.jdf))
