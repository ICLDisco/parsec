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

The master branch is considered production stable and is the recommended
version.

For users that favor a stable API, dated releases do not
introduce API changes between patchlevels (e.g., from 19.11.0 to 19.11.1).



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
