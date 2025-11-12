# file: build_truncated_icosa_sphere_rect_boxes.py
"""
Scene builder for a truncated-icosahedron based structure on a sphere, plus supports and a loaded OBJ arm.

What this script produces
- A triangulated sphere mesh centered at `center` with radius `base_radius`.
- 60 circular pads on the sphere, one along each truncated-icosahedron vertex direction.
- 60 oriented rectangular boxes (solid prisms) standing normal to the sphere at each pad.
- 4 ground supports composed of an angled strut and a shortened vertical leg that can extend underground.
- Optionally, a static robotic arm loaded from an OBJ file.

Coordinate system and units
- Right-handed world: +Z is up. Plane URDF is at z = 0.
- All distances are in meters. Angles in degrees in CLI, internally converted to radians.
- Visual meshes and collision meshes are placed using PyBullet world coordinates.

Performance and numerical stability
- The sphere tessellation defaults to 256 × 512 which is dense and heavy. Reduce if you only need previews.
- Edge detection on the icosahedron uses a tolerance scaled to the minimum edge length to avoid FP noise.
- Vector normalization assumes nonzero vectors. Caller ensures valid directions.

Authoring tips
- Cross section for pads and boxes is derived from `circle_angle_deg` and either `base_radius` or a locked reference.
- If you scale the sphere via `base_radius` without locking cross section, pads and boxes will scale accordingly.
- OBJ loading with concave collision must remain static. Set mass > 0 only with a convex collision shape.
"""

import math
import os
import time
import argparse
from typing import List, Tuple

import pybullet as p
import pybullet_data


def golden_ratio() -> float:
    """Return the golden ratio φ = (1 + sqrt(5)) / 2.

    Used to parametrize a canonical embedding of an icosahedron with coordinates
    drawn from {±1, ±φ, 0}. This embedding yields equal edge lengths after normalization.
    """
    return (1.0 + 5.0 ** 0.5) / 2.0


def icosahedron_vertices() -> List[Tuple[float, float, float]]:
    """Return the 12 vertex coordinates of a canonical icosahedron before normalization.

    Construction
    - Start from permutations of (0, ±1, ±φ) across axes.
    - This is a standard coordinate set that yields equal-length edges.

    Output
    - List of 12 3D tuples. Not unit length. Downstream code normalizes as needed.
    """
    phi = golden_ratio()
    a, b = 1.0, phi
    return [
        (0.0,  a,  b), (0.0,  a, -b), (0.0, -a,  b), (0.0, -a, -b),
        ( a,  b, 0.0), ( a, -b, 0.0), (-a,  b, 0.0), (-a, -b, 0.0),
        ( b, 0.0,  a), ( b, 0.0, -a), (-b, 0.0,  a), (-b, 0.0, -a)
    ]


def _dist(a, b) -> float:
    """Euclidean distance between two 3D points a and b.

    This uses the L2 norm √((dx)^2 + (dy)^2 + (dz)^2).
    """
    return ((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2) ** 0.5


def _normalize(v: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """Return a unit vector in the direction of v.

    Assumes v is not the zero vector. Caller must ensure a nonzero length.
    Raises ZeroDivisionError if v is zero length.
    """
    n = (v[0]**2 + v[1]**2 + v[2]**2) ** 0.5
    return (v[0]/n, v[1]/n, v[2]/n)


def icosahedron_edges(verts: List[Tuple[float, float, float]]) -> List[Tuple[int, int]]:
    """Identify edges of the icosahedron by scanning all pairwise distances.

    Strategy
    - Compute distances for all vertex pairs i < j.
    - The smallest nonzero distance dmin corresponds to the true edge length.
    - Return all pairs whose distance is within a tiny relative tolerance of dmin.

    Why this works
    - In a regular icosahedron all edges share the same length.
    - Non-edge pairs are strictly longer, so thresholding by dmin isolates edges.

    Complexity
    - O(n^2) over 12 vertices which is trivial.

    Returns
    - List of unique index pairs (i, j) with i < j.
    """
    n = len(verts)
    dmin = None
    dists = {}
    for i in range(n):
        for j in range(i + 1, n):
            d = _dist(verts[i], verts[j])
            dists[(i, j)] = d
            if d > 0 and (dmin is None or d < dmin):
                dmin = d
    eps = dmin * 1e-6
    return [(i, j) for (i, j), d in dists.items() if abs(d - dmin) <= eps]


def truncated_icosahedron_vertex_dirs(tau: float) -> List[Tuple[float, float, float]]:
    """Compute outward unit directions for the 60 vertices of a truncated icosahedron.

    Inputs
    - tau in (0, 0.5] is the fraction along each icosahedron edge from one endpoint.
      Example: tau = 1/3 places a vertex one third of the way along each edge from both ends.

    Method
    - Get base icosahedron vertex list and edge list.
    - For each edge (vi, vj), create two points:
        P1 = (1 − tau) * vi + tau * vj
        P2 = (1 − tau) * vj + tau * vi
    - Normalize P1 and P2 to lie on the unit sphere. These are pure directions.
    - Deduplicate via rounding to 6 decimals to remove numeric duplicates.

    Output
    - List of unique unit vectors. With tau = 1/3, expect 60 directions.
    """
    base = icosahedron_vertices()
    edges = icosahedron_edges(base)
    dirs: List[Tuple[float, float, float]] = []
    seen = set()

    def _add(v):
        vn = _normalize(v)
        key = (round(vn[0], 6), round(vn[1], 6), round(vn[2], 6))
        if key not in seen:
            seen.add(key)
            dirs.append(vn)

    for i, j in edges:
        vi, vj = base[i], base[j]
        _add(((1 - tau) * vi[0] + tau * vj[0],
              (1 - tau) * vi[1] + tau * vj[1],
              (1 - tau) * vi[2] + tau * vj[2]))
        _add(((1 - tau) * vj[0] + tau * vi[0],
              (1 - tau) * vj[1] + tau * vi[1],
              (1 - tau) * vj[2] + tau * vi[2]))
    return dirs


def ortho_basis_from_dir(u: Tuple[float, float, float]) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """Build an orthonormal basis {t1, t2} perpendicular to a given unit vector u.

    Why we need this
    - To construct circular rings or box cross sections in a plane orthogonal to u,
      we need two unit tangent directions spanning that plane.

    Strategy
    - Pick a helper axis a that is not nearly colinear with u to avoid numerical issues.
    - Subtract projection of a onto u to get t1, then normalize.
    - Compute t2 as u × t1 for a right-handed set.

    Inputs
    - u is expected to be normalized already.

    Returns
    - t1, t2 which are unit, mutually perpendicular, and perpendicular to u.
    """
    ux, uy, uz = u
    a = (1.0, 0.0, 0.0) if abs(ux) < 0.5 and abs(uy) < 0.5 else (0.0, 0.0, 1.0)
    dot = a[0] * ux + a[1] * uy + a[2] * uz
    t1 = (a[0] - dot * ux, a[1] - dot * uy, a[2] - dot * uz)
    t1 = _normalize(t1)
    t2 = (uy * t1[2] - uz * t1[1], uz * t1[0] - ux * t1[2], ux * t1[1] - uy * t1[0])
    return t1, t2


def build_sphere_mesh(radius: float, lats: int, lons: int):
    """Triangulate a UV sphere into vertices and triangles.

    Parameters
    - radius: sphere radius in meters.
    - lats: number of latitude subdivisions. lats + 1 vertex rows are generated.
    - lons: number of longitude subdivisions per latitude row.

    Topology
    - Each quad cell is split into two triangles in a consistent winding.
    - The texture-style grid repeats the first column to close the seam.

    Returns
    - verts: list of 3D positions.
    - faces: list of index triplets (triangles).
    """
    verts, faces = [], []
    for i in range(lats + 1):
        theta = math.pi * i / lats
        st, ct = math.sin(theta), math.cos(theta)
        for j in range(lons):
            phi = 2.0 * math.pi * j / lons
            sp, cp = math.sin(phi), math.cos(phi)
            x = radius * st * cp
            y = radius * st * sp
            z = radius * ct
            verts.append((x, y, z))

    def idx(i, j):  # wrap j to close the seam
        return i * lons + (j % lons)

    for i in range(lats):
        for j in range(lons):
            a = idx(i, j)
            b = idx(i + 1, j)
            c = idx(i + 1, j + 1)
            d = idx(i, j + 1)
            faces.append((a, b, c))
            faces.append((a, c, d))
    return verts, faces


def build_oriented_cylinder(center, axis_dir, radius, height, segments):
    """Build a triangulated closed cylinder oriented along `axis_dir`.

    Geometry
    - The cylinder axis is along unit vector u = axis_dir / ||axis_dir||.
    - Two circular rings displaced ±height/2 along u.
    - Rings parameterized in the tangent plane spanned by {t1, t2}.

    Triangulation
    - Side wall as a strip of quads split into triangles.
    - Bottom and top caps fan from their centers for a watertight mesh.

    Returns
    - verts, faces suitable to write into an OBJ.

    Notes
    - `segments` controls ring tessellation. Larger is smoother but heavier.
    """
    u = _normalize(axis_dir)
    t1, t2 = ortho_basis_from_dir(u)
    half_h = 0.5 * height
    ring0, ring1 = [], []

    for k in range(segments):
        ang = 2.0 * math.pi * k / segments
        ca, sa = math.cos(ang), math.sin(ang)
        # radial offset in the plane orthogonal to u
        px = radius * (ca * t1[0] + sa * t2[0])
        py = radius * (ca * t1[1] + sa * t2[1])
        pz = radius * (ca * t1[2] + sa * t2[2])
        ring0.append((center[0] - half_h * u[0] + px,
                      center[1] - half_h * u[1] + py,
                      center[2] - half_h * u[2] + pz))
        ring1.append((center[0] + half_h * u[0] + px,
                      center[1] + half_h * u[1] + py,
                      center[2] + half_h * u[2] + pz))

    # cap centers
    c0 = (center[0] - half_h * u[0], center[1] - half_h * u[1], center[2] - half_h * u[2])
    c1 = (center[0] + half_h * u[0], center[1] + half_h * u[1], center[2] + half_h * u[2])

    verts = ring0 + ring1 + [c0, c1]
    faces = []
    n = segments

    # side wall
    for k in range(n):
        a = k
        b = (k + 1) % n
        c = n + (k + 1) % n
        d = n + k
        faces.append((a, b, c))
        faces.append((a, c, d))

    # bottom cap fans to c0
    bottom_center_idx = len(verts) - 2
    for k in range(n):
        faces.append((bottom_center_idx, (k + 1) % n, k))

    # top cap fans to c1
    top_center_idx = len(verts) - 1
    for k in range(n):
        faces.append((top_center_idx, n + k, n + (k + 1) % n))

    return verts, faces


def write_single_obj(path: str, parts: List[Tuple[List[Tuple[float, float, float]], List[Tuple[int, int, int]]]]):
    """Write multiple mesh parts into a single OBJ with separate vertex regions.

    Input
    - parts: list of (verts, faces) pairs.
      Faces for part k are automatically offset by the accumulated vertex count.

    OBJ specifics
    - Only positional vertices `v` and faces `f` are written.
    - No texture coordinates or normals. PyBullet loads GEOM_MESH without them.
    """
    with open(path, "w", encoding="utf-8") as f:
        f.write("# combined mesh\n")
        v_offset = 0
        for verts, faces in parts:
            for (x, y, z) in verts:
                f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
            for (a, b, c) in faces:
                f.write(f"f {a + 1 + v_offset} {b + 1 + v_offset} {c + 1 + v_offset}\n")
            v_offset += len(verts)


def ray_hit_sphere(P, d, R):
    """Compute the smallest positive intersection t for the parametric ray P + t*d with a sphere of radius R at the origin.

    Derivation
    - Solve ||P + t d||^2 = R^2. This yields a quadratic At^2 + Bt + C = 0 with:
        A = d·d
        B = 2 P·d
        C = P·P − R^2
    - Return the smallest t > 0 of the two roots. None if no real positive intersection.

    Usage
    - Used to locate where a ground support ray toward the target meets the sphere.
    """
    px, py, pz = P
    dx, dy, dz = d
    A = dx * dx + dy * dy + dz * dz
    B = 2 * (px * dx + py * dy + pz * dz)
    C = px * px + py * py + pz * pz - R * R
    disc = B * B - 4 * A * C
    if disc <= 0:
        return None
    t1 = (-B - math.sqrt(disc)) / (2 * A)
    t2 = (-B + math.sqrt(disc)) / (2 * A)
    ts = [t for t in (t1, t2) if t > 0]
    return min(ts) if ts else None


def mat3_to_quat(r):
    """Convert a 3×3 rotation matrix to a quaternion [x, y, z, w].

    Numerics
    - Branch on the trace to avoid catastrophic cancellation when the diagonal is small.
    - Assumes r is orthonormal. No re-orthogonalization is performed here.
    """
    m00, m01, m02 = r[0]
    m10, m11, m12 = r[1]
    m20, m21, m22 = r[2]
    tr = m00 + m11 + m22
    if tr > 0.0:
        S = (tr + 1.0) ** 0.5 * 2.0
        qw = 0.25 * S
        qx = (m21 - m12) / S
        qy = (m02 - m20) / S
        qz = (m10 - m01) / S
    elif m00 > m11 and m00 > m22:
        S = (1.0 + m00 - m11 - m22) ** 0.5 * 2.0
        qw = (m21 - m12) / S
        qx = 0.25 * S
        qy = (m01 + m10) / S
        qz = (m02 + m20) / S
    elif m11 > m22:
        S = (1.0 + m11 - m00 - m22) ** 0.5 * 2.0
        qw = (m02 - m20) / S
        qx = (m01 + m10) / S
        qy = 0.25 * S
        qz = (m12 + m21) / S
    else:
        S = (1.0 + m22 - m00 - m11) ** 0.5 * 2.0
        qw = (m10 - m01) / S
        qx = (m02 + m20) / S
        qy = (m12 + m21) / S
        qz = 0.25 * S
    return [qx, qy, qz, qw]


def euler_to_quat(r, p_, y):
    """Convert roll pitch yaw (in radians) to a quaternion [x, y, z, w].

    Convention
    - Intrinsic rotations in the order roll X, pitch Y, yaw Z.
    - Matches common robotics conventions for PyBullet.
    """
    cr = math.cos(r * 0.5)
    sr = math.sin(r * 0.5)
    cp = math.cos(p_ * 0.5)
    sp = math.sin(p_ * 0.5)
    cy = math.cos(y * 0.5)
    sy = math.sin(y * 0.5)
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return [qx, qy, qz, qw]


def main():
    """Build meshes, spawn PyBullet bodies, optionally load an arm OBJ, and run a short sim."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-radius", type=float, default=3.2)
    ap.add_argument("--circle-angle-deg", type=float, default=2.5)
    ap.add_argument("--circle-height", type=float, default=0.25)
    ap.add_argument("--hex-length", type=float, default=0.5)
    ap.add_argument("--rect-half-width", type=float, default=None)
    ap.add_argument("--rect-half-length", type=float, default=None)
    ap.add_argument("--lock-cross-section", action="store_true")

    # Arm placement
    ap.add_argument("--arm-obj", type=str, default="Rmk3.obj", help="robotic arm OBJ filename in same folder")
    ap.add_argument("--arm-pos", type=float, nargs=3, default=[5.0, 0.0, 0.0], help="x y z world position")
    ap.add_argument("--arm-rpy", type=float, nargs=3, default=[90.0, 0.0, -143.0], help="roll pitch yaw in degrees")
    ap.add_argument("--arm-scale", type=float, default=0.003, help="uniform scale for the OBJ")
    ap.add_argument("--arm-mass", type=float, default=0.0, help="mass 0 makes it static")
    args = ap.parse_args()

    # Scene scale and layout
    base_radius = 2.3  # fixed smaller sphere here. Use CLI to drive other parameters.
    circle_angle_deg = args.circle_angle_deg
    circle_height = args.circle_height
    circle_segments = 48
    hex_length = args.hex_length
    sphere_lats, sphere_lons = 256, 512
    center = [0.0, 0.0, 3.5]  # raise sphere center above plane
    tau = 1.0 / 3  # truncation fraction for vertex directions

    # Support geometry parameters
    support_radius = 0.22
    support_segments = 36
    support_target_offset = 0.6  # how far below ground the angled strut aims in Z
    support_clearance = 0.02     # gap between strut tip and sphere surface
    support_shorten_factor = 0.65  # where the vertical leg begins along the angled strut
    underground_depth = 0.30       # how far the vertical leg extends below the ground plane

    # Cross section sizing for pads and boxes
    # If unlocked, this follows current `base_radius`. If locked, uses a reference radius for predictable scale.
    circle_radius_from_base = 3.2 * math.sin(math.radians(circle_angle_deg))
    circle_radius_locked = 3.2 * math.sin(math.radians(circle_angle_deg))
    circle_radius = circle_radius_locked if args.lock_cross_section else circle_radius_from_base

    # Box half extents in the two tangent directions at each pad
    rect_half_width = args.rect_half_width if args.rect_half_width is not None else 2.0 * circle_radius
    rect_half_length = args.rect_half_length if args.rect_half_length is not None else 2.0 * circle_radius

    # Output mesh files for sphere base and supports
    obj_base = os.path.abspath("solid_sphere_with_corner_circles.obj")
    obj_sup = os.path.abspath("sphere_supports_short_vert.obj")

    # Directions to 60 pads
    def dirs_corners():
        return truncated_icosahedron_vertex_dirs(tau=tau)

    # Build base: sphere plus 60 short cylinders forming pads
    parts_base = []
    sphere_verts, sphere_faces = build_sphere_mesh(base_radius, sphere_lats, sphere_lons)
    parts_base.append((sphere_verts, sphere_faces))

    cylinder_tops: List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = []
    for u in dirs_corners():
        # Surface point on the sphere along unit direction u
        surf_point = (base_radius * u[0], base_radius * u[1], base_radius * u[2])
        # Pad cylinder center sits halfway along its height outward from the surface
        center_circle = (surf_point[0] + 0.5 * circle_height * u[0],
                         surf_point[1] + 0.5 * circle_height * u[1],
                         surf_point[2] + 0.5 * circle_height * u[2])
        cv, cf = build_oriented_cylinder(center_circle, u, circle_radius, circle_height, circle_segments)
        parts_base.append((cv, cf))

        # Top center of the pad, used to anchor the base of the box that extrudes along u
        top_center = (center_circle[0] + 0.5 * circle_height * u[0],
                      center_circle[1] + 0.5 * circle_height * u[1],
                      center_circle[2] + 0.5 * circle_height * u[2])
        cylinder_tops.append((top_center, u))

    # Write the combined base mesh that PyBullet will load as a single GEOM_MESH
    write_single_obj(obj_base, parts_base)

    # Build supports: four feet around the ground projected toward an underground target
    parts_sup = []
    ground_z_local = -center[2]  # plane height in the sphere local frame
    target = (0.0, 0.0, -support_target_offset)  # below the ground plane
    vertical_dist = target[2] - ground_z_local
    r_ground = abs(vertical_dist)  # radius around origin where the feet sit on the ground

    feet = [
        ( r_ground, 0.0, ground_z_local),
        (-r_ground, 0.0, ground_z_local),
        (0.0,  r_ground, ground_z_local),
        (0.0, -r_ground, ground_z_local),
    ]

    for foot in feet:
        # Ray from the ground foot toward the target under ground
        dir_vec = _normalize((target[0] - foot[0], target[1] - foot[1], target[2] - foot[2]))
        # Intersect this ray with the sphere to find where the angled strut meets the sphere
        t_hit = ray_hit_sphere(foot, dir_vec, base_radius)
        if t_hit is None:
            continue

        # Back off a small clearance so the strut does not penetrate the sphere mesh
        t_top = t_hit - support_clearance
        top_point = (foot[0] + t_top * dir_vec[0],
                     foot[1] + t_top * dir_vec[1],
                     foot[2] + t_top * dir_vec[2])

        # Choose the split point along the angled strut where the vertical leg starts
        t_end = max(0.0, support_shorten_factor * t_top)
        end_point = (foot[0] + t_end * dir_vec[0],
                     foot[1] + t_end * dir_vec[1],
                     foot[2] + t_end * dir_vec[2])

        # Angled segment from end_point up to top_point
        angled_center = ((end_point[0] + top_point[0]) * 0.5,
                         (end_point[1] + top_point[1]) * 0.5,
                         (end_point[2] + top_point[2]) * 0.5)
        angled_len = _dist(end_point, top_point)
        sv, sf = build_oriented_cylinder(angled_center, dir_vec, support_radius, angled_len, support_segments)
        parts_sup.append((sv, sf))

        # Vertical leg goes down from end_point to below the ground plane by underground_depth
        bottom_z = ground_z_local - underground_depth
        vert_len = end_point[2] - bottom_z
        vert_center = (end_point[0], end_point[1], bottom_z + 0.5 * vert_len)
        vv, vf = build_oriented_cylinder(vert_center, (0.0, 0.0, 1.0), support_radius, vert_len, support_segments)
        parts_sup.append((vv, vf))

    # Write supports mesh
    write_single_obj(obj_sup, parts_sup)

    # PyBullet setup and visualization
    physics_client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")
    p.resetDebugVisualizerCamera(
        cameraDistance=base_radius * 3.0,
        cameraYaw=35.0,
        cameraPitch=-20.0,
        cameraTargetPosition=center,
    )

    # Load base mesh as a single static body that uses a concave trimesh collision
    col_base = p.createCollisionShape(p.GEOM_MESH, fileName=obj_base, flags=p.GEOM_FORCE_CONCAVE_TRIMESH)
    vis_base = p.createVisualShape(p.GEOM_MESH, fileName=obj_base, rgbaColor=[0.5, 0.5, 0.5, 1.0])
    p.createMultiBody(0, col_base, vis_base, basePosition=center, baseOrientation=[0, 0, 0, 1])

    # Build and place 60 oriented rectangular boxes on top of the pads
    # Half extents correspond to tangent directions t1 and t2, height along u
    box_half_extents = [rect_half_width, rect_half_length, 0.5 * hex_length]
    box_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=box_half_extents)
    box_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=box_half_extents, rgbaColor=[1.0, 0.55, 0.0, 1.0])

    for top_center, u in cylinder_tops:
        t1, t2 = ortho_basis_from_dir(u)
        # Rotation matrix with columns [t1 t2 u] aligns box local axes to world
        R = [[t1[0], t2[0], u[0]],
             [t1[1], t2[1], u[1]],
             [t1[2], t2[2], u[2]]]
        q = mat3_to_quat(R)

        # Box center is offset half its height along u from the pad top
        center_box = (
            top_center[0] + 0.5 * hex_length * u[0],
            top_center[1] + 0.5 * hex_length * u[1],
            top_center[2] + 0.5 * hex_length * u[2],
        )

        # Convert from sphere local frame to world by adding `center`
        world_pos = (center[0] + center_box[0],
                     center[1] + center_box[1],
                     center[2] + center_box[2])

        p.createMultiBody(
            baseMass=0.0,  # static decorative solids
            baseCollisionShapeIndex=box_col,
            baseVisualShapeIndex=box_vis,
            basePosition=world_pos,
            baseOrientation=q,
        )

    # Load supports as a single static concave mesh
    col_sup = p.createCollisionShape(p.GEOM_MESH, fileName=obj_sup, flags=p.GEOM_FORCE_CONCAVE_TRIMESH)
    vis_sup = p.createVisualShape(p.GEOM_MESH, fileName=obj_sup, rgbaColor=[0.35, 0.35, 0.35, 1.0])
    p.createMultiBody(0, col_sup, vis_sup, basePosition=center, baseOrientation=[0, 0, 0, 1])

    # Robotic arm OBJ as a static concave mesh
    arm_path = os.path.abspath(args.arm_obj)
    if not os.path.isfile(arm_path):
        raise FileNotFoundError(f"Cannot find OBJ file: {arm_path}")

    rx, ry, rz = [math.radians(v) for v in args.arm_rpy]
    arm_quat = euler_to_quat(rx, ry, rz)
    arm_pos = tuple(args.arm_pos)
    mesh_scale = [args.arm_scale, args.arm_scale, args.arm_scale]

    # Concave trimesh collision must be static in PyBullet
    if args.arm_mass != 0.0:
        raise ValueError("arm-mass must be 0 for GEOM_FORCE_CONCAVE_TRIMESH. Use 0 or switch to a convex collision.")

    arm_col = p.createCollisionShape(
        p.GEOM_MESH,
        fileName=arm_path,
        meshScale=mesh_scale,
        flags=p.GEOM_FORCE_CONCAVE_TRIMESH
    )
    arm_vis = p.createVisualShape(
        p.GEOM_MESH,
        fileName=arm_path,
        meshScale=mesh_scale,
        rgbaColor=[0.8, 0.8, 0.8, 1.0]
    )
    p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=arm_col,
        baseVisualShapeIndex=arm_vis,
        basePosition=arm_pos,
        baseOrientation=arm_quat,
    )

    # Run a short sim loop to keep the GUI open and allow camera interaction
    end_time = time.time() + 60.0
    while time.time() < end_time:
        p.stepSimulation()
        time.sleep(1.0 / 240.0)

    p.disconnect()


if __name__ == "__main__":
    main()
