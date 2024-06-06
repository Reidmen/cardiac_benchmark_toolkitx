import argparse
import os
import pathlib
from re import I
from typing import Type, TypeAlias

import PetscBinaryIO
from mpi4py import MPI
from dolfinx import fem, io
import dolfinx.fem.petsc as petsc
from dolfinx.mesh import Mesh, MeshTags
from mpi4py import MPI
import numpy as np
from numpy.typing import NDArray
from petsc4py import PETSc
import ufl

from cardiac_benchmark_toolkitx.data import DEFAULTS, FiberDirections, MARKERS

NDArray64: TypeAlias = NDArray[np.float64]
NDArray32: TypeAlias = NDArray[np.float32]

VectorFunctionType: TypeAlias = list[fem.Function]
MapsType: TypeAlias = list[NDArray64]

r_short_endo: float = DEFAULTS.R_SHORT_ENDO
r_short_epi: float = DEFAULTS.R_SHORT_EPI
r_long_endo: float = DEFAULTS.R_LONG_ENDO
r_long_epi: float = DEFAULTS.R_LONG_EPI
alpha_endo: float = DEFAULTS.FIBER_ALPHA_ENDO
alpha_epi: float = DEFAULTS.FIBER_ALPHA_EPI


def alpha(x: NDArray64) -> NDArray64:
    return (alpha_endo + (alpha_epi - alpha_endo) * x) * np.pi / 180


def r_long(x: NDArray64) -> NDArray64:
    return r_long_endo + (r_long_epi - r_long_endo) * x


def r_short(x: NDArray64) -> NDArray64:
    return r_short_endo + (r_short_epi - r_short_endo) * x


def read_mesh(mesh_file: os.PathLike[str] | str) -> tuple[Mesh, MeshTags]:
    """Read DOLFINX mesh from XDMF file.

    Args:
        mesh_file:  path to mesh file

    Returns:
        tuple of:
            * mesh
            * boundary identifiers
    """
    path = pathlib.Path(mesh_file)

    if not path.suffix == ".xdmf":
        raise Exception(
            f"Mesh file type not recognized: {path.suffix}. Use XDMF format!"
        )

    with io.XDMFFile(MPI.COMM_WORLD, path, "r") as xdmf:
        mesh = xdmf.read_mesh(name="mesh")

    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    with io.XDMFFile(MPI.COMM_WORLD, path, "r") as xdmf:
        facet_tags = xdmf.read_meshtags(mesh, name="mesh_tags")

    return mesh, facet_tags


def transmural_distance_problem(
    mesh: Mesh, bnds: MeshTags, degree: int
) -> fem.Function:
    T = fem.functionspace(mesh, ("CG", degree))
    u, v = ufl.TrialFunction(T), ufl.TestFunction(T)
    lhs = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
    rhs = v * fem.Constant(mesh, PETSc.ScalarType(0)) * ufl.dx

    transmural_distance = fem.Function(T, name="transmural")

    endo_facets = bnds.find(MARKERS.ENDOCARDIUM)
    endo_dofs = fem.locate_dofs_topological(
        T, mesh.topology.dim - 1, endo_facets
    )
    endo_bc = fem.dirichletbc(
        fem.Constant(mesh, PETSc.ScalarType(0)), endo_dofs, T
    )
    epi_facets = bnds.find(MARKERS.EPICARDIUM)
    epi_dofs = fem.locate_dofs_topological(
        T, mesh.topology.dim - 1, epi_facets
    )
    epi_bc = fem.dirichletbc(
        fem.Constant(mesh, PETSc.ScalarType(1)), epi_dofs, T
    )
    bcs = [endo_bc, epi_bc]

    problem = petsc.LinearProblem(
        lhs,
        rhs,
        bcs=bcs,
        petsc_options={
            "ksp_type": "cg",
            "pc_type": "jacobi",
            "ksp_monitor_true_residual": None,
            "ksp_rtol": 1e-10,
        },
    )
    transmural_distance = problem.solve()

    return transmural_distance


def compute_local_coordinate_system(
    distance_array: NDArray64,
    space_coordinates: NDArray64,
) -> tuple[NDArray64, NDArray64]:
    r_long_distance = r_long(distance_array)
    r_short_distance = r_short(distance_array)

    # get coordinates
    coordinates: NDArray64 = space_coordinates
    x, y, z = coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]

    # compute u, v and assign to values
    a = np.sqrt(y * y + z * z) / r_short_distance
    b = np.array(x / r_long_distance)
    u = np.arctan2(a, b)
    v = np.pi - np.arctan2(z, -y)
    v[u < 1e-7] = 0

    # define local base using derivate matrix
    # (e_1, e_2, e_3) = D(e_t, e_u, e_v)
    zeros = np.zeros_like(u)

    e_2 = np.array(
        [
            -r_long_distance * np.sin(u),
            r_short_distance * np.cos(u) * np.cos(v),
            r_short_distance * np.cos(u) * np.sin(v),
        ]
    )

    e_3 = np.array(
        [
            zeros,
            -r_short_distance * np.sin(u) * np.sin(v),
            r_short_distance * np.sin(u) * np.cos(v),
        ]
    )

    # normalize columns
    for vec in [e_2, e_3]:
        vec /= np.linalg.norm(vec, axis=0)

    return (e_2, e_3)


def compute_sheet_direction(
    fibers_function_space: fem.FunctionSpace,
    fiber_array: NDArray64,
    sheet_normal_array: NDArray64,
) -> fem.Function:
    assert fiber_array.shape == sheet_normal_array.shape
    sheet_array = np.cross(fiber_array, sheet_normal_array, axis=0)

    sheet = fem.Function(fibers_function_space, name="sheet")
    set_local_values(sheet, sheet_array)

    return sheet


def compute_sheet_normal_direction(
    distance: fem.Function, fibers_space: fem.FunctionSpace
) -> tuple[fem.Function, NDArray64]:
    (e_2, e_3) = compute_local_coordinate_system(
        distance.x.array, distance.function_space.tabulate_dof_coordinates()
    )

    sheet_normal_array = np.cross(e_2, e_3, axis=0)
    sheet_normal_array /= np.linalg.norm(sheet_normal_array, axis=0)

    sheet_normal = fem.Function(fibers_space, name="sheet_normal")
    set_local_values(sheet_normal, sheet_normal_array)

    return sheet_normal, sheet_normal_array


def compute_fiber_direction(
    distance: fem.Function, fibers_space: fem.FunctionSpace
) -> tuple[fem.Function, NDArray64]:
    distance_array = distance.x.array
    alpha_distance = alpha(distance_array)
    (e_2, e_3) = compute_local_coordinate_system(
        distance_array, distance.function_space.tabulate_dof_coordinates()
    )

    # rotate in alpha (rad) and assign
    fiber_array = np.sin(alpha_distance) * e_2 + np.cos(alpha_distance) * e_3
    fiber = fem.Function(fibers_space, name="fiber")
    set_local_values(fiber, fiber_array)

    return fiber, fiber_array


def create_vector_to_scalar_functions(
    vector_function: fem.Function, num_sub_spaces: int
) -> tuple[VectorFunctionType, MapsType]:
    functions: list[fem.Function] = []
    maps: list[NDArray64] = []

    for i in range(num_sub_spaces):
        Vi, map_i = vector_function.function_space.sub(i).collapse()
        func = fem.Function(Vi, name=f"{i}-component")
        functions.append(func)
        maps.append(map_i)

    return functions, maps


def synchronize_components(
    functions: VectorFunctionType,
    maps: MapsType,
    target_function: fem.Function,
) -> None:
    for ui, idx_map in zip(functions, maps):
        target_function.x.array[idx_map] = ui.x.array

    target_function.x.scatter_forward()


def set_local_values(function: fem.Function, data_array: NDArray64) -> None:

    ndim = function.function_space.mesh.topology.dim
    functions_lst, dof_maps_lst = create_vector_to_scalar_functions(
        function, ndim
    )
    data_list = [data_array[i, :] for i in range(ndim)]

    assert len(functions_lst) == len(data_list)

    for func, dofs_values in zip(functions_lst, data_list):
        func.x.array[:] = dofs_values
        func.x.scatter_forward()

        print(
            f"rank {MPI.COMM_WORLD.rank}: {func.x.array.min()} {func.x.array.max()}"
        )

    synchronize_components(functions_lst, dof_maps_lst, function)


def build_directions(
    mesh: Mesh,
    transmural_distance: fem.Function,
    degree: int,
) -> FiberDirections:

    ndim = mesh.topology.dim
    fibers_space = fem.functionspace(mesh, ("CG", degree, (ndim,)))

    print("computing fiber direction")
    fiber_direction, fiber_array = compute_fiber_direction(
        transmural_distance, fibers_space
    )
    print("computing sheet normal direction")
    (
        sheet_normal_direction,
        sheet_normal_array,
    ) = compute_sheet_normal_direction(transmural_distance, fibers_space)
    print("computing sheet direction")
    sheet_direction = compute_sheet_direction(
        fiber_direction.function_space, fiber_array, sheet_normal_array
    )

    directions = FiberDirections(
        fiber=fiber_direction,
        sheet=sheet_direction,
        sheet_normal=sheet_normal_direction,
    )
    print("built fiber, sheet and sheet_normal directions")

    return directions


def save_fibers_to_files(fibers: FiberDirections, path_to_save: str) -> None:
    """Save fibers to files in H5, VTK and binary formats."""
    mesh = fibers.fiber.function_space.mesh
    path = pathlib.Path(path_to_save)
    if MPI.COMM_WORLD.Get_rank() == 0:
        path.mkdir(parents=True, exist_ok=True)

    # binaries for parallel use
    path_bin = path.joinpath("bin_format")
    path_bin.mkdir(exist_ok=True, parents=True)

    MPI.COMM_WORLD.barrier()

    directions = [fibers.fiber, fibers.sheet, fibers.sheet_normal]
    names = ["fiber", "sheet", "sheet_normal"]

    for name, direction in zip(names, directions):
        path_to_xdmf = path.joinpath(f"xdmf_format/ellipsoid_{name}.xdmf")
        path_to_vtx = path.joinpath(f"vtx_format/ellipsoid_{name}.bp")
        path_to_binaries = path_bin.joinpath(
            f"ellipsoid_{name}_np{mesh.comm.size}_{mesh.comm.rank}.dat"
        )

        with io.XDMFFile(mesh.comm, str(path_to_xdmf), "w") as xdmf:
            xdmf.write_mesh(mesh)
            xdmf.write_function(direction)

        with io.VTXWriter(mesh.comm, str(path_to_vtx), [direction]) as vtx:
            vtx.write(0.0)

        pbio = PetscBinaryIO.PetscBinaryIO()
        file_view = direction.vector.array_w.view(PetscBinaryIO.Vec)
        pbio.writeBinaryFile(str(path_to_binaries), [file_view])

        print(
            f"wrote {name} in path {str(path_to_xdmf)} and {str(path_to_vtx)}"
        )


def main(
    path_to_mesh: str,
    function_space_degree: str = "P1",
    path_to_save: str = "./results/",
) -> None:

    if function_space_degree.lower() not in ("p1", "p2"):
        raise Exception("Only P1 / P2 as function spaces for benchmark")

    deg = int(function_space_degree.lower()[1])
    mesh, bnds = read_mesh(path_to_mesh)
    transmural_distance = transmural_distance_problem(mesh, bnds, deg)
    fibers = build_directions(mesh, transmural_distance, deg)
    save_fibers_to_files(fibers, path_to_save)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="""Compute fiber, sheet and sheet_normal directions
        from input mesh.""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("path_to_mesh", type=str, help="path to mesh file.")
    parser.add_argument(
        "-space",
        "--function_space",
        default="P1",
        help="function space for fibers to be interpolated.",
    )
    parser.add_argument(
        "-save",
        "--path_to_save",
        default="./results/",
        help="path to save fibers, expected as relative path"
        "e.g. './results/' (default).",
    )

    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    # print(args)
    main(args.path_to_mesh, args.function_space, args.path_to_save)
