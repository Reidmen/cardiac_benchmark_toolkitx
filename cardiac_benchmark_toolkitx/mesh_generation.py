import argparse
from pathlib import Path
import subprocess
import sys

import dolfinx
import meshio
from mpi4py import MPI
import numpy as np

from cardiac_benchmark_toolkitx.data import DEFAULTS, MARKERS


def create_mesh(
    mesh: meshio.Mesh, cell_type: str, prune_Z: bool = False
) -> meshio.Mesh:
    cells = mesh.get_cells_type(cell_type)
    cell_data = mesh.get_cell_data("gmsh:physical", cell_type)

    if prune_Z:
        # delete third (obj=2) column (axis=1), this strips the z-component
        points = np.delete(arr=mesh.points, obj=2, axis=1)
    else:
        points = mesh.points

    out_mesh = meshio.Mesh(
        points=points,
        cells={cell_type: cells},
        cell_data={"markers": [cell_data]},
    )
    return out_mesh


def ellipsoid_mesh(
    hc: float,
    defaults: DEFAULTS = DEFAULTS(),
    markers: MARKERS = MARKERS(),
    path: str = "./meshes/",
):
    """Create truncated ellipsoid mesh with physiological dimensions.

    Surface IDs:
    1  endocardium
    2  epicardium
    3  base

    Args:
        hc: characteristic element size
        path: write path
    """
    GEO_CODE = f"""
        r_short_endo = {defaults.R_SHORT_ENDO};
        r_short_epi  = {defaults.R_SHORT_EPI};
        r_long_endo  = {defaults.R_LONG_ENDO};
        r_long_epi   = {defaults.R_LONG_EPI};
        quota_base = {defaults.QUOTA_BASE};

        mu_base = Acos(quota_base / r_long_endo);

        psize_ref = {hc};
        axisymmetric = 0;

        Geometry.CopyMeshingMethod = 1;
        Mesh.ElementOrder = 1;
        Mesh.Optimize = 1;
        Mesh.OptimizeNetgen = 1;
        Mesh.HighOrderOptimize = 1;

        Function EllipsoidPoint
            Point(id) = {{ r_long  * Cos(mu),
                           r_short * Sin(mu) * Cos(theta),
                           r_short * Sin(mu) * Sin(theta), psize }};
        Return

        center = newp; Point(center) = {{ 0.0, 0.0, 0.0 }};

        theta = 0.0;

        r_short = r_short_endo; r_long = r_long_endo;
        mu = -Pi;
        psize = psize_ref / 2.0;
        apex_endo = newp; id = apex_endo; Call EllipsoidPoint;
        mu = -1.0 * Acos(5.0 / 17.0);
        psize = psize_ref;
        base_endo = newp; id = base_endo; Call EllipsoidPoint;

        r_short = r_short_epi; r_long = r_long_epi;
        mu = -Pi;
        psize = psize_ref / 2.0;
        apex_epi = newp; id = apex_epi; Call EllipsoidPoint;
        mu = -1.0 * Acos(5.0 / 20.0);
        psize = psize_ref;
        base_epi = newp; id = base_epi; Call EllipsoidPoint;

        apex = newl; Line(apex) = {{ apex_endo, apex_epi }};
        base = newl; Line(base) = {{ base_endo, base_epi }};
        endo = newl;
        Ellipse(endo) = {{ apex_endo, center, apex_endo, base_endo }};
        epi  = newl;
        Ellipse(epi) = {{ apex_epi, center, apex_epi, base_epi }};

        ll1 = newll; Line Loop(ll1) = {{ apex, epi, -base, -endo }};
        s1 = news; Plane Surface(s1) = {{ ll1 }};

        If (axisymmetric == 0)
            sendoringlist[] = {{ }};
            sepiringlist[]  = {{ }};
            sendolist[] = {{ }};
            sepilist[]  = {{ }};
            sbaselist[] = {{ }};
            vlist[] = {{ }};

            sold = s1;
            For i In {{ 0 : 3 }}
                out[] = Extrude {{ {{ 1.0, 0.0, 0.0 }},
                                   {{ 0.0, 0.0, 0.0 }}, Pi/2 }}
                                {{ Surface{{sold}}; }};
                sendolist[i] = out[4];
                sepilist[i]  = out[2];
                sbaselist[i] = out[3];
                vlist[i] = out[1];
                bout[] = Boundary{{ Surface{{ sbaselist[i] }}; }};
                sendoringlist[i] = bout[1];
                sepiringlist[i] = bout[3];
                sold = out[0];
            EndFor

            // MYOCARDIUM
            Physical Volume(0) = {{ vlist[] }};
            // ENDO
            Physical Surface({markers.ENDOCARDIUM}) = {{ sendolist[] }};
            // EPI
            Physical Surface({markers.EPICARDIUM}) = {{ sepilist[] }};
            // BASE
            Physical Surface({markers.BASE}) = {{ sbaselist[] }};
            // ENDORING
            Physical Line(4) = {{ sendoringlist[] }};
            // EPIRING
            Physical Line(5) = {{ sepiringlist[] }};
        EndIf

        Physical Point("ENDOPT") = {{ apex_endo }};
        Physical Point("EPIPT") = {{ apex_epi }};
    """

    geofile = Path(path).joinpath(f"ellipsoid_{hc}.geo")
    outfile = Path(path).joinpath(f"ellipsoid_{hc}.msh")
    with geofile.open("w") as f:
        f.write(GEO_CODE)
    subprocess.run(["gmsh", "-3", str(geofile)])

    # convert to dolfin XDMF
    mesh = meshio.read(str(outfile))

    xdmf_path = Path(path)
    xdmf_path.mkdir(exist_ok=True, parents=True)
    pth_tmp_msh = xdmf_path.joinpath("ellipsoid_meshio.xdmf")
    pth_tmp_bnd = xdmf_path.joinpath("ellipsoid_meshio_bound.xdmf")

    tetra_mesh = create_mesh(mesh, "tetra")
    triangle_mesh = create_mesh(mesh, "triangle")

    meshio.write(str(pth_tmp_msh), tetra_mesh, file_format="xdmf")
    meshio.write(str(pth_tmp_bnd), triangle_mesh, file_format="xdmf")

    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, pth_tmp_msh, "r") as xdmf:
        mesh = xdmf.read_mesh(name="Grid")
        mesh.name = "mesh"

    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, pth_tmp_bnd, "r") as xdmf:
        boundaries = xdmf.read_meshtags(mesh, name="Grid")
        boundaries.name = "mesh_tags"

    filepath = Path(path).joinpath(f"ellipsoid_{hc}.xdmf")

    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, filepath, "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_meshtags(boundaries, mesh.geometry)

    print(f"wrote {filepath} {filepath.with_suffix('.h5')}")
    geofile.unlink()
    outfile.unlink()
    for pth in [pth_tmp_msh, pth_tmp_bnd]:
        Path(pth).unlink()


def get_parser() -> argparse.ArgumentParser:
    """Get arguments parser.

    Returns:
        parser
    """
    parser = argparse.ArgumentParser(
        description="""
        Generate ellipsoid mesh, with optional path and element size.
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-path",
        "--path_to_save",
        type=str,
        default="./meshes/",
        help="Path where meshes are written",
    )
    parser.add_argument(
        "-size",
        "--element_size",
        metavar="hc",
        type=float,
        default=0.005,
        help="Truncated ellipsoid mesh with characteristic mesh size",
    )
    return parser


if __name__ == "__main__":
    args: argparse.Namespace = get_parser().parse_args()

    if len(sys.argv) > 1:
        ellipsoid_mesh(args.element_size, path=args.path_to_save)
