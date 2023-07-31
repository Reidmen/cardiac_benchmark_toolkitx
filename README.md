# Cardiac Benchmark Toolkit

This repository contains minimal set of scripts that allow you to reproduce the mesh
as well as the fibers in the cardiac mechanics benchmark using [dolfinx](https://github.com/FEniCS/dolfinx).

## Installation

**Docker** (Recommended) Run the following command to start a container with all the required dependencies:

```shell
docker run --name dolfinx-stable -v $(pwd):/home/shared -w /home/shared -ti reidmen/dolfinx-nightly:e561c6c
```

In order to enter the shell, use:

```shell
docker exec -ti dolfin-stable /bin/bash -l
```

## Quickstart
This file requires `PetscBinaryIO`, the module can be loaded using:
```shell
export PYTHONPATH=$PYTHONPATH:/usr/local/petsc/lib/petsc/bin/
```

Given your tagged mesh located in `./meshes/ellipsoid.xdmf`, you can create the fibers as follows:

```shell
cardiac_benchmark_toolkitx/fiber_generation.py ./meshes/ellipsoid.xdmf
```

If succesfull, the script will create fibers in `xdmf`, `vtk` and `PETSc binaries` files in a `./results/` folder.

Further options can be found with:

```shell
cardiac_benchmark_toolkitx/ellipsoid_fiber_generation.py --help
```
