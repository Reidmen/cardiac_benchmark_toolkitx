# nightly version, stable is too old
FROM ghcr.io/fenics/dolfinx:nightly 

# We set the working directory to install docker dependencies
WORKDIR /tmp/

# Install the Python-module
COPY . /tmp
RUN python3 -m pip install --no-cache-dir .

# We remove the contents of the temporary directory to minimize the size of the image
RUN rm -rf /tmp
