# PaRSEC Installation Instructions

## Prerequisites
Ensure the following dependencies are installed:
```bash
sudo apt-get update
sudo apt-get install -y libopenmpi-dev openmpi-bin cmake bison libhwloc-dev
```

## Building PaRSEC
1. Create a build directory and enter it:
   ```bash
   mkdir builddir && cd builddir
   ```

2. Run the configure script. Note that we disable the MPI+HWLOC compatibility check to avoid potential conflicts with system-installed libraries:
   ```bash
   ../configure --with-mpi --without-hwloc --disable-debug --prefix=$PWD/install -DMPI_HWLOC_COMPAT_CHECK=OFF
   ```

3. Build and install:
   ```bash
   make install
   ```

The binaries and libraries will be available in `builddir/install`.
