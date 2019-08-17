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


*****************************************************************************************

[PaRSEC]: http://icl.utk.edu/parsec/
[Spack]: https://spack.readthedocs.io/en/latest/index.html
[DPlasma]: http://icl.utk.edu/dplasma/
[ScaLAPACK]: http://www.netlib.org/scalapack/
[Plasma]: https://bitbucket.org/icl/plasma
[plasma-installer]: http://icl.cs.utk.edu/projectsfiles/plasma/pubs/plasma-installer_2.8.0.tar.gz
[plasma-tarball]: https://bitbucket.org/icl/plasma/downloads/plasma-2.8.tar.gz
