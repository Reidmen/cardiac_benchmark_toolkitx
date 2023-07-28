# Cardiac Benchmark Toolkit

This repository contains minimal set of scripts that allow you to reproduce the mesh
as well as the fibers in the cardiac mechanics benchmark using dolfinx.

## Installation

*Docker*
Run the following command to start a container with all the required dependencies:

```shell
docker run --name dolfinx-stable -v $(pwd):/home/shared -w /home/shared -ti ghcr.io/fenics/dolfinx:nightly
```

In order to enter the shell, use:

```shell
docker exec -ti dolfin-stable /bin/bash -l
```

## Quickstart
Given your tagged mesh located in `./meshes/ellipsoid.xdmf`, you can create the fibers as follows:

```shell
cardiac_benchmark_toolkitx/ellipsoid_fiber_generation.py ./meshes/ellipsoid.xdmf
```

If succesfull, the script will create fibers in `xdmf` and `vtk` files in a `./results/` folder.

Further options can be found with:

```shell
cardiac_benchmark_toolkitx/ellipsoid_fiber_generation.py --help
```

