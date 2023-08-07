# nightly version, stable is too old
FROM ghcr.io/fenics/dolfinx:nightly 

# We set the working directory to install docker dependencies
WORKDIR /cardiac_benchmark_toolkitx

# Dependencies
RUN python3 -m pip install --no-binary=h5py h5py meshio

# Install the Python-module
COPY . /cardiac_benchmark_toolkitx
RUN chmod +x .
RUN python3 -m pip install --no-cache-dir .
RUN export PYTHONPATH=$PYTHONPATH:/usr/local/petsc/lib/petsc/bin/

ENV SHELL /bin/bash
ENTRYPOINT ["python3", "cardiac_benchmark_toolkitx/fiber_generation.py", "./meshes/ellipsoid0.005.xdmf"]
