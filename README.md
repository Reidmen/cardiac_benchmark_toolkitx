# Cardiac Benchmark Toolkit

This repository contains minimal set of scripts that allow you to reproduce the mesh
as well as the fibers in the cardiac mechanics benchmark using dolfinx.

## Installation

*Docker*
Run the following command to start a container with all the required dependencies:

```shell
docker run --name dolfin-stable -v $(pwd):/home/shared -w /home/shared -ti ghcr.io/scientificcomputing/fenics-gmsh:2023-04-21
```

In order to enter the shell, use:

```shell
docker exec -ti dolfin-stable /bin/bash -l
```

**Note** docker image is cortesy of *Simula Lab*.

## Quickstart
Given your tagged mesh located in `./meshes/ellipsoid.xdmf`, you can create the fibers as follows:

```shell
dolfin/ellipsoid_fiber_generation.py ./meshes/ellipsoid.xdmf
```

If succesfull, the script will create fibers in `xdmf` and `vtk` files in a `./results/` folder.

Further options can be found with:

```shell
dolfin/ellipsoid_fiber_generation.py --help
```

