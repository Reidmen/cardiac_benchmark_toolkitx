# Cardiac Benchmark Toolkit

This repository contains minimal set of scripts that allow you to reproduce the mesh
as well as the fibers in the cardiac mechanics benchmark using [dolfinx](https://github.com/FEniCS/dolfinx).

## Installation

This library requires `FEniCS (dolfinx)` already installed in your system.
If `dolfinx` is not installed in your system, use **Docker** with the provided image in folder `docker`.

### Using Pip
Execute the following command in your local session, it will install the required dependencies:

```shell
pip3 install .
```

### Using Docker
**Docker** (Recommended) Run the following command to start a container with all the required dependencies:

```shell
docker run --name dolfinx-stable -v $(pwd):/home/shared -w /home/shared -ti ghcr.io/fenics/dolfinx/dolfinx:nightly
```

In order to enter the shell, use:

```shell
docker exec -ti dolfin-stable /bin/bash -l
```


## Quickstart

Saving binary solutions require `PetscBinaryIO`, the module can be loaded using:
```shell
export PYTHONPATH=$PYTHONPATH:/usr/local/petsc/lib/petsc/bin/
```
if `PETSc` was installed in `/usr/local/` (e.g. in the case of the `docker` image).

Given your tagged mesh located in `./meshes/ellipsoid.xdmf`, you can create the fibers as follows:

```shell
python3 cardiac_benchmark_toolkitx/fiber_generation.py ./meshes/ellipsoid_0.005.xdmf
```

If succesfull, the script will create fibers in `xdmf`, `vtk` and `PETSc binaries` files in a `./results/` folder.

Further options can be found with:

```shell
python3 cardiac_benchmark_toolkitx/fiber_generation.py --help
```
