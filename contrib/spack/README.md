PaRSEC Spack integration package
================================

This is a package to automate the installation of [PaRSEC] with [Spack].


## Basic Usage

To use the PaRSEC Spack integration, issue the following commands:

```shell
spack repo add $PARSEC_SRCDIR/contrib/spack
spack install parsec@devel
spack load parsec
```


## Recommended version

Until the next stable release (2.0), it is recommended to use the _devel_ version.


## Variants

  Name [Default] |  Allowed values  | Description
---------------- | ---------------- | ----------------------------------------------
  cuda [on]      |  True, False     | Use CUDA for GPU acceleration
  debug [off]    |  True, False     | Debug version **incurs performance overhead!**
  profile [off]  |  True, False     | Generate profiling data


## Compiling DPlasma

[DPlasma] is a linear algebra package, similar in feature with [ScaLAPACK],
but using task-parallel tile algorithms, like in [Plasma]. Although DPlasma
does not employ the scheduler or runtime of Plasma (employing PaRSEC instead), it
depends on tiled BLAS routines that are not available in standard BLAS, but are
provided from Plasma _CoreBlas_ library. This Spack does not build Plasma. If you
wish to build Dplasma using this Spack:

1. Build and install Plasma (see the main README and INSTALL for further instructions).
   The recommended Plasma version is the [plasma-installer] for version 2.8.0. As an
   alternative you can use the distribution [plasma-tarball].
2. make sure _pkg-config_ lists the needed libraries (including the accelerated BLAS)
   for _coreblas_. For example on our system:

```shell
pkg-config --libs coreblas
-L$PLASMA_DIR/lib -L$MKLROOT/lib/intel64 -lcoreblas -llapacke -lmkl_sequential -lmkl_core -lmkl_gf_lp64 -lpthread -lm -lhwloc
```

3. build the Spack, Coreblas will be autodetected and DPlasma will be build accordingly.




*****************************************************************************************

[PaRSEC]: http://icl.utk.edu/parsec/
[Spack]: https://spack.readthedocs.io/en/latest/index.html
[DPlasma]: http://icl.utk.edu/dplasma/
[ScaLAPACK]: http://www.netlib.org/scalapack/
[Plasma]: https://bitbucket.org/icl/plasma
[plasma-installer]: http://icl.cs.utk.edu/projectsfiles/plasma/pubs/plasma-installer_2.8.0.tar.gz
[plasma-tarball]: https://bitbucket.org/icl/plasma/downloads/plasma-2.8.tar.gz
