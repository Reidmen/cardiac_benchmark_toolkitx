from dataclasses import dataclass
import pathlib
from typing import Literal

import argparse

from dolfinx.mesh import Mesh, MeshTags

import numpy as np

from mpi4py import MPI
from petsc4py import PETSc
from numpy.typing import NDArray
from dolfinx import fem, io
import ufl


@dataclass(frozen=True)
class MARKERS:
    ENDOCARDIUM = 1
    EPICARDIUM = 2


@dataclass(frozen=True)
class DEFAULTS:
    R_SHORT_ENDO = 2.5e-2
    R_SHORT_EPI = 3.5e-2
    R_LONG_ENDO = 9.0e-2
    R_LONG_EPI = 9.7e-2
    FIBER_ALPHA_ENDO = -60.0
    FIBER_ALPHA_EPI = +60.0


@dataclass(frozen=True)
class FiberDirections:
    fiber: fem.Function
    sheet: fem.Function
    sheet_normal: fem.Function


class FiberExpression(UserExpression):
    """Computes fiber expression using its analytic description."""

    def __init__(self, td, **kwargs) -> None:
        super().__init__(**kwargs)
        self.td = td  # transmural distance

    def eval(self, values, x):
        r_short_endo = DEFAULTS.R_SHORT_ENDO
        r_short_epi = DEFAULTS.R_SHORT_EPI
        r_long_endo = DEFAULTS.R_LONG_ENDO
        r_long_epi = DEFAULTS.R_LONG_EPI
        alpha_endo = DEFAULTS.FIBER_ALPHA_ENDO
        alpha_epi = DEFAULTS.FIBER_ALPHA_EPI

        td_x = self.td(x[0], x[1], x[2])

        # compute r_short and r_long using t
        r_s = r_short_endo + (r_short_epi - r_short_endo) * td_x
        r_l = r_long_endo + (r_long_epi - r_long_endo) * td_x
        # compute u, v and assign to values
        a = np.sqrt(x[1] * x[1] + x[2] * x[2]) / r_s
        b = np.array(x[0] / r_l)
        u = np.arctan2(a, b)
        v = 0.0 if u < 1e-7 else np.pi - np.arctan2(x[2], -x[1])

        # define local base using derivate matrix
        # (e_1, e_2, e_3) = D(e_t, e_u, e_v)
        dr_s = r_short_epi - r_short_endo
        dr_l = r_long_epi - r_long_endo
        cos_u, sin_v = np.cos(u), np.sin(v)
        sin_u, cos_v = np.sin(u), np.cos(v)
        e_11, e_12, e_13 = dr_l * cos_u, -r_l * sin_u, 0.0
        e_21, e_22, e_23 = (
            dr_l * cos_u * sin_v,
            r_s * cos_u * cos_v,
            -r_s * sin_u * sin_v,
        )
        e_31, e_32, e_33 = (
            dr_s * sin_u * sin_v,
            r_s * cos_u * sin_v,
            r_s * sin_u * cos_v,
        )

        # e_1 = np.array([e_11, e_21, e_31])
        e_2 = np.array([e_12, e_22, e_32])
        e_3 = np.array([e_13, e_23, e_33])

        # normalize columns
        for vec in [e_2, e_3]:
            vec /= np.linalg.norm(vec)

        # rotate in alpha (rad) to obtain the fiber direction
        alpha = (alpha_endo + (alpha_epi - alpha_endo) * td_x) * np.pi / 180
        fiber = np.sin(alpha) * e_2 + np.cos(alpha) * e_3

        # define fibers using the values r, u, v
        values[0] = fiber[0]
        values[1] = fiber[1]
        values[2] = fiber[2]

    def value_shape(self) -> tuple[Literal[3]]:
        return (3,)


class SheetNormalExpression(UserExpression):
    """Computes normal direction from analytic description."""

    def __init__(self, td, **kwargs) -> None:
        super().__init__(**kwargs)
        self.td = td  # transmural distance

    def eval(self, values, x) -> None:
        # constants
        r_short_endo = DEFAULTS.R_SHORT_ENDO
        r_short_epi = DEFAULTS.R_SHORT_EPI
        r_long_endo = DEFAULTS.R_LONG_ENDO
        r_long_epi = DEFAULTS.R_LONG_EPI

        td_x = self.td(x[0], x[1], x[2])

        # compute r_short and r_long using t
        r_s = r_short_endo + (r_short_epi - r_short_endo) * td_x
        r_l = r_long_endo + (r_long_epi - r_long_endo) * td_x
        # compute u, v and assign to values
        a = np.sqrt(x[1] * x[1] + x[2] * x[2]) / r_s
        b = np.array(x[0] / r_l)
        u = np.arctan2(a, b)
        v = 0.0 if u < 1e-7 else np.pi - np.arctan2(x[2], -x[1])

        # define local base using derivate matrix
        # (e_1, e_2, e_3) = D(e_t, e_u, e_v)
        r_s = r_short_endo + (r_short_epi - r_short_endo) * td_x
        r_l = r_long_endo + (r_long_epi - r_long_endo) * td_x
        dr_s = r_short_epi - r_short_endo
        dr_l = r_long_epi - r_long_endo
        cos_u, sin_v = np.cos(u), np.sin(v)
        sin_u, cos_v = np.sin(u), np.cos(v)
        e_11, e_12, e_13 = dr_l * cos_u, -r_l * sin_u, 0.0
        e_21, e_22, e_23 = (
            dr_l * cos_u * sin_v,
            r_s * cos_u * cos_v,
            -r_s * sin_u * sin_v,
        )
        e_31, e_32, e_33 = (
            dr_s * sin_u * sin_v,
            r_s * cos_u * sin_v,
            r_s * sin_u * cos_v,
        )

        # e_1 = np.array([e_11, e_21, e_31])
        e_2 = np.array([e_12, e_22, e_32])
        e_3 = np.array([e_13, e_23, e_33])

        # normalize columns
        for vec in [e_2, e_3]:
            vec /= np.linalg.norm(vec)

        # compute normal direction (unitary)
        normal = np.cross(e_2, e_3)
        normal /= np.linalg.norm(normal)

        values[0] = normal[0]
        values[1] = normal[1]
        values[2] = normal[2]

    def value_shape(self) -> tuple[Literal[3]]:
        return (3,)


class SheetExpression(UserExpression):
    """Computes sheet direction from fiber and normal directions."""

    def __init__(
        self,
        fiber: FiberExpression,
        sheet_normal: SheetNormalExpression,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.f0 = fiber
        self.n0 = sheet_normal

    def eval(self, values, x) -> None:
        f0_x = self.f0(x[0], x[1], x[2])
        n0_x = self.n0(x[0], x[1], x[2])

        # Cross product, normalize and assign
        s0_x = np.cross(f0_x, n0_x)
        s0_x /= np.linalg.norm(s0_x)

        values[0] = s0_x[0]
        values[1] = s0_x[1]
        values[2] = s0_x[2]

    def value_shape(self) -> tuple[Literal[3]]:
        return (3,)


def read_mesh(mesh_file: str) -> tuple[Mesh, MeshFunction]:
    tmp = mesh_file.split(".")  # [-1].split('.')
    file_type = tmp[-1]

    if file_type == "h5":
        mesh = Mesh()

        with HDF5File(mesh.mpi_comm(), mesh_file, "r") as hdf:
            hdf.read(mesh, "/mesh", False)
            boundaries = MeshFunction(
                "size_t", mesh, mesh.topology().dim() - 1
            )

            if hdf.has_dataset("boundaries"):
                hdf.read(boundaries, "/boundaries")
            else:
                if mesh.mpi_comm().Get_rank() == 0:
                    print(
                        "no <boundaries> datasets found in file {}".format(
                            mesh_file
                        )
                    )

    elif file_type == "xdmf":

        mesh = Mesh()

        with XDMFFile(mesh_file) as xf:
            xf.read(mesh)
            boundaries = MeshFunction(
                "size_t", mesh, mesh.topology().dim() - 1, 0
            )

            xf.read(boundaries)

    else:
        raise Exception("Mesh format not recognized. Use XDMF or HDF5.")

    return mesh, boundaries


def transmural_distance_problem(
    mesh: Mesh, mesh_function: MeshFunction, degree: int
) -> fem.Function:

    T = fem.FunctionSpace(mesh, ("CG", degree))
    u, v = ufl.TrialFunction(T), ufl.TestFunction(T)
    lhs = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
    rhs = v * fem.Constant(mesh, PETSc.ScalarType(0)) * ufl.dx

    transmural_distance = fem.Function(T, name="transmural")

    endo_bc = DirichletBC(
        T,
        fem.Constant(mesh, PETSc.ScalarType(0)),
        mesh_function,
        MARKERS.ENDOCARDIUM,
    )
    epi_bc = DirichletBC(
        T,
        fem.Constant(mehs, PETSc.ScalarType(1)),
        mesh_function,
        MARKERS.EPICARDIUM,
    )
    bcs = [endo_bc, epi_bc]

    problem = fem.petsc.LinearProblem(
        lhs,
        rhs,
        bcs=bcs,
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    )
    transmural_distance = problem.solve()

    return transmural_distance


def build_orientations(
    mesh: Mesh,
    transmural_distance: fem.Function,
    degree: int,
) -> FiberDirections:

    fiber_expression = FiberExpression(transmural_distance, degree=degree)
    sheet_normal_expression = SheetNormalExpression(
        transmural_distance, degree=degree
    )
    sheet_expression = SheetExpression(
        fiber=fiber_expression,
        sheet_normal=sheet_normal_expression,
        degree=degree,
    )

    fibers_function_space = VectorFunctionSpace(mesh, "CG", degree)

    f0 = fem.Function(fibers_function_space, name="f0")
    s0 = fem.Function(fibers_function_space, name="s0")
    n0 = fem.Function(fibers_function_space, name="n0")

    f0.assign(interpolate(fiber_expression, fibers_function_space))
    s0.assign(interpolate(sheet_expression, fibers_function_space))
    n0.assign(interpolate(sheet_normal_expression, fibers_function_space))

    fiber_directions = FiberDirections(fiber=f0, sheet=s0, sheet_normal=n0)
    print("built fiber, sheet and sheet_normal directions")

    return fiber_directions


def save_mesh_to_files(
    mesh: Mesh, bnds: MeshFunction, path_to_save: str
) -> None:
    path = pathlib.Path(path_to_save)
    path.mkdir(parents=True, exist_ok=True)

    mesh_path = path.joinpath("pvd_format/ellipsoid_domain.pvd")
    mesh_vtk = File(str(mesh_path))
    mesh_vtk << mesh
    mesh_vtk << bnds


def save_fibers_to_files(fibers: FiberDirections, path_to_save: str) -> None:
    """Save fibers to files in H5, VTK formats."""
    # TODO: save to PETSC binaries
    mesh = fibers.fiber.function_space.mesh
    path = pathlib.Path(path_to_save)
    path.mkdir(parents=True, exist_ok=True)

    directions = [fibers.fiber, fibers.sheet, fibers.sheet_normal]
    names = ["fiber", "sheet", "sheet_normal"]

    for name, direction in zip(names, directions):
        path_to_xdmf = path.joinpath(f"xdmf_format/ellipsoid_{name}.xdmf")
        path_to_vtx = path.joinpath(f"bp_format/ellipsoid_{name}.bp")

        with io.XDMFFile(mesh.comm, str(path_to_xdmf), "w") as xdmf:
            xdmf.write_mesh(mesh)
            # xdmf.write(direction, "/" + name)
            xdmf.write_function(direction)

        with io.VTXWriter(mesh.comm, str(path_to_vtx), [direction]) as vtx:
            vtx.write(0.0)

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
    mesh, mesh_function = read_mesh(path_to_mesh)
    transmural_distance = transmural_distance_problem(mesh, mesh_function, deg)
    fibers = build_orientations(mesh, transmural_distance, deg)
    save_mesh_to_files(mesh, mesh_function, path_to_save)
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
    # TODO
    # check for mesh
    # if mesh not provided, create ellipsoid domain
    args = get_parser().parse_args()
    # print(args)
    main(args.path_to_mesh, args.function_space, args.path_to_save)
