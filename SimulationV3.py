import math
import os
import time
import argparse
from typing import List, Tuple

import pybullet as p
import pybullet_data

def golden_ratio() -> float:
    return (1.0 + 5.0 ** 0.5) / 2.0

def icosahedron_vertices():
    phi = golden_ratio()
    a, b = 1.0, phi
    return [
        (0.0,  a,  b), (0.0,  a, -b), (0.0, -a,  b), (0.0, -a, -b),
        ( a,  b, 0.0), ( a, -b, 0.0), (-a,  b, 0.0), (-a, -b, 0.0),
        ( b, 0.0,  a), ( b, 0.0, -a), (-b, 0.0,  a), (-b, 0.0, -a)
    ]

def _dist(a, b): return ((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2) ** 0.5
# ecluidean distance between 2 points in 3D

def _normalize(v):
# Turns a 3D vector into a unit vector pointing the same way
    n = (v[0]**2 + v[1]**2 + v[2]**2) ** 0.5
    return (v[0]/n, v[1]/n, v[2]/n)

def icosahedron_edges(verts):
# Finds which vertex pairs are edges by detecting the smallest nonzero pairwise distance and returning all pairs at that distance.
    n=len(verts); dmin=None; dists={}
    for i in range(n):
        for j in range(i+1,n):
            d=_dist(verts[i],verts[j]); dists[(i,j)]=d
            if d>0 and (dmin is None or d<dmin): dmin=d
    eps=dmin*1e-6
    return [(i,j) for (i,j),d in dists.items() if abs(d-dmin)<=eps]

def truncated_icosahedron_vertex_dirs(tau):
# Generates unit direction vectors to the 60 truncated-icosahedron corner locations by sampling points along every icosahedron edge at a fraction tau from each end normalizing.
    base=icosahedron_vertices(); edges=icosahedron_edges(base)
    dirs=[]; seen=set()
    def _add(v):
        vn=_normalize(v); key=(round(vn[0],6),round(vn[1],6),round(vn[2],6))
        if key not in seen: seen.add(key); dirs.append(vn)
    for i,j in edges:
        vi,vj=base[i],base[j]
        _add(((1-tau)*vi[0]+tau*vj[0],(1-tau)*vi[1]+tau*vj[1],(1-tau)*vi[2]+tau*vj[2]))
        _add(((1-tau)*vj[0]+tau*vi[0],(1-tau)*vj[1]+tau*vi[1],(1-tau)*vj[2]+tau*vi[2]))
    return dirs

def ortho_basis_from_dir(u):
    ux,uy,uz=u
    a=(1.0,0.0,0.0) if abs(ux)<0.5 and abs(uy)<0.5 else (0.0,0.0,1.0)
    dot=a[0]*ux+a[1]*uy+a[2]*uz
    t1=(a[0]-dot*ux,a[1]-dot*uy,a[2]-dot*uz); t1=_normalize(t1)
    t2=(uy*t1[2]-uz*t1[1],uz*t1[0]-ux*t1[2],ux*t1[1]-uy*t1[0])
    return t1,t2

def build_sphere_mesh(radius: float, lats: int, lons: int):
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
    def idx(i, j): return i * lons + (j % lons)
    for i in range(lats):
        for j in range(lons):
            a = idx(i, j); b = idx(i + 1, j); c = idx(i + 1, j + 1); d = idx(i, j + 1)
            faces.append((a, b, c)); faces.append((a, c, d))
    return verts, faces

def build_oriented_cylinder(center, axis_dir, radius, height, segments):
    u = _normalize(axis_dir); t1, t2 = ortho_basis_from_dir(u)
    half_h = 0.5 * height
    ring0, ring1 = [], []
    for k in range(segments):
        ang = 2.0 * math.pi * k / segments
        ca, sa = math.cos(ang), math.sin(ang)
        px = radius * (ca * t1[0] + sa * t2[0])
        py = radius * (ca * t1[1] + sa * t2[1])
        pz = radius * (ca * t1[2] + sa * t2[2])
        ring0.append((center[0] - half_h*u[0] + px, center[1] - half_h*u[1] + py, center[2] - half_h*u[2] + pz))
        ring1.append((center[0] + half_h*u[0] + px, center[1] + half_h*u[1] + py, center[2] + half_h*u[2] + pz))
    c0 = (center[0] - half_h*u[0], center[1] - half_h*u[1], center[2] - half_h*u[2])
    c1 = (center[0] + half_h*u[0], center[1] + half_h*u[1], center[2] + half_h*u[2])
    verts = ring0 + ring1 + [c0, c1]
    faces = []
    n = segments
    for k in range(n):
        a = k; b = (k + 1) % n; c = n + (k + 1) % n; d = n + k
        faces.append((a, b, c)); faces.append((a, c, d))
    bottom_center_idx = len(verts) - 2
    for k in range(n): faces.append((bottom_center_idx, (k + 1) % n, k))
    top_center_idx = len(verts) - 1
    for k in range(n): faces.append((top_center_idx, n + k, n + (k + 1) % n))
    return verts, faces

def write_single_obj(path: str, parts: List[Tuple[List[Tuple[float, float, float]], List[Tuple[int, int, int]]]]):
    with open(path, "w", encoding="utf-8") as f:
        f.write("# combined mesh\n")
        v_offset = 0
        for verts, faces in parts:
            for (x, y, z) in verts:
                f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
            for (a, b, c) in faces:
                f.write(f"f {a+1+v_offset} {b+1+v_offset} {c+1+v_offset}\n")
            v_offset += len(verts)

def ray_hit_sphere(P, d, R):
    px,py,pz = P; dx,dy,dz = d
    A = dx*dx + dy*dy + dz*dz
    B = 2*(px*dx + py*dy + pz*dz)
    C = px*px + py*py + pz*pz - R*R
    disc = B*B - 4*A*C
    if disc <= 0: return None
    t1 = (-B - math.sqrt(disc)) / (2*A)
    t2 = (-B + math.sqrt(disc)) / (2*A)
    ts = [t for t in (t1, t2) if t > 0]
    return min(ts) if ts else None

def mat3_to_quat(r):
    m00,m01,m02 = r[0]
    m10,m11,m12 = r[1]
    m20,m21,m22 = r[2]
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
    cr = math.cos(r*0.5); sr = math.sin(r*0.5)
    cp = math.cos(p_*0.5); sp = math.sin(p_*0.5)
    cy = math.cos(y*0.5); sy = math.sin(y*0.5)
    qw = cr*cp*cy + sr*sp*sy
    qx = sr*cp*cy - cr*sp*sy
    qy = cr*sp*cy + sr*cp*sy
    qz = cr*cp*sy - sr*sp*cy
    return [qx, qy, qz, qw]

def main():
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

    base_radius = 2.3
    circle_angle_deg = args.circle_angle_deg
    circle_height = args.circle_height
    circle_segments = 48
    hex_length = args.hex_length
    sphere_lats, sphere_lons = 256, 512
    center = [0.0, 0.0, 3.5]
    tau = 1.0 / 3

    support_radius = 0.22
    support_segments = 36
    support_target_offset = 0.6
    support_clearance = 0.02
    support_shorten_factor = 0.65
    underground_depth = 0.30

    circle_radius_from_base = 3.2 * math.sin(math.radians(circle_angle_deg))
    circle_radius_locked = 3.2 * math.sin(math.radians(circle_angle_deg))
    circle_radius = circle_radius_locked if args.lock_cross_section else circle_radius_from_base

    rect_half_width  = args.rect_half_width  if args.rect_half_width  is not None else 2.0 * circle_radius
    rect_half_length = args.rect_half_length if args.rect_half_length is not None else 2.0 * circle_radius

    obj_base = os.path.abspath("solid_sphere_with_corner_circles.obj")
    obj_sup  = os.path.abspath("sphere_supports_short_vert.obj")

    def dirs_corners():
        return truncated_icosahedron_vertex_dirs(tau=tau)

    parts_base = []
    sphere_verts, sphere_faces = build_sphere_mesh(base_radius, sphere_lats, sphere_lons)
    parts_base.append((sphere_verts, sphere_faces))

    cylinder_tops: List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = []
    for u in dirs_corners():
        surf_point = (base_radius * u[0], base_radius * u[1], base_radius * u[2])
        center_circle = (surf_point[0] + 0.5 * circle_height * u[0],
                         surf_point[1] + 0.5 * circle_height * u[1],
                         surf_point[2] + 0.5 * circle_height * u[2])
        cv, cf = build_oriented_cylinder(center_circle, u, circle_radius, circle_height, circle_segments)
        parts_base.append((cv, cf))
        top_center = (center_circle[0] + 0.5 * circle_height * u[0],
                      center_circle[1] + 0.5 * circle_height * u[1],
                      center_circle[2] + 0.5 * circle_height * u[2])
        cylinder_tops.append((top_center, u))
    write_single_obj(obj_base, parts_base)

    parts_sup = []
    ground_z_local = -center[2]
    target = (0.0, 0.0, -support_target_offset)
    vertical_dist = target[2] - ground_z_local
    r_ground = abs(vertical_dist)
    feet = [
        ( r_ground, 0.0, ground_z_local),
        (-r_ground, 0.0, ground_z_local),
        (0.0,  r_ground, ground_z_local),
        (0.0, -r_ground, ground_z_local),
    ]

    for foot in feet:
        dir_vec = _normalize((target[0]-foot[0], target[1]-foot[1], target[2]-foot[2]))
        t_hit = ray_hit_sphere(foot, dir_vec, base_radius)
        if t_hit is None:
            continue
        t_top = t_hit - support_clearance
        top_point = (foot[0] + t_top * dir_vec[0],
                     foot[1] + t_top * dir_vec[1],
                     foot[2] + t_top * dir_vec[2])
        t_end = max(0.0, support_shorten_factor * t_top)
        end_point = (foot[0] + t_end * dir_vec[0],
                     foot[1] + t_end * dir_vec[1],
                     foot[2] + t_end * dir_vec[2])

        angled_center = ((end_point[0] + top_point[0]) * 0.5,
                         (end_point[1] + top_point[1]) * 0.5,
                         (end_point[2] + top_point[2]) * 0.5)
        angled_len = _dist(end_point, top_point)
        sv, sf = build_oriented_cylinder(angled_center, dir_vec, support_radius, angled_len, support_segments)
        parts_sup.append((sv, sf))

        bottom_z = ground_z_local - underground_depth
        vert_len = end_point[2] - bottom_z
        vert_center = (end_point[0], end_point[1], bottom_z + 0.5 * vert_len)
        vv, vf = build_oriented_cylinder(vert_center, (0.0, 0.0, 1.0), support_radius, vert_len, support_segments)
        parts_sup.append((vv, vf))
    write_single_obj(obj_sup, parts_sup)

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

    col_base = p.createCollisionShape(p.GEOM_MESH, fileName=obj_base, flags=p.GEOM_FORCE_CONCAVE_TRIMESH)
    vis_base = p.createVisualShape(p.GEOM_MESH, fileName=obj_base, rgbaColor=[0.5, 0.5, 0.5, 1.0])
    p.createMultiBody(0, col_base, vis_base, basePosition=center, baseOrientation=[0,0,0,1])

    box_half_extents = [rect_half_width, rect_half_length, 0.5 * hex_length]
    box_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=box_half_extents)
    box_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=box_half_extents, rgbaColor=[1.0, 0.55, 0.0, 1.0])

    for top_center, u in cylinder_tops:
        t1, t2 = ortho_basis_from_dir(u)
        R = [[t1[0], t2[0], u[0]],
             [t1[1], t2[1], u[1]],
             [t1[2], t2[2], u[2]]]
        q = mat3_to_quat(R)
        center_box = (
            top_center[0] + 0.5 * hex_length * u[0],
            top_center[1] + 0.5 * hex_length * u[1],
            top_center[2] + 0.5 * hex_length * u[2],
        )
        world_pos = (center[0] + center_box[0],
                     center[1] + center_box[1],
                     center[2] + center_box[2])
        p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=box_col,
            baseVisualShapeIndex=box_vis,
            basePosition=world_pos,
            baseOrientation=q,
        )

    col_sup = p.createCollisionShape(p.GEOM_MESH, fileName=obj_sup, flags=p.GEOM_FORCE_CONCAVE_TRIMESH)
    vis_sup = p.createVisualShape(p.GEOM_MESH, fileName=obj_sup, rgbaColor=[0.35, 0.35, 0.35, 1.0])
    p.createMultiBody(0, col_sup, vis_sup, basePosition=center, baseOrientation=[0,0,0,1])

    # Robotic arm OBJ
    arm_path = os.path.abspath(args.arm_obj)
    if not os.path.isfile(arm_path):
        raise FileNotFoundError(f"Cannot find OBJ file: {arm_path}")

    rx, ry, rz = [math.radians(v) for v in args.arm_rpy]
    arm_quat = euler_to_quat(rx, ry, rz)
    arm_pos = tuple(args.arm_pos)
    mesh_scale = [args.arm_scale, args.arm_scale, args.arm_scale]

    # Keep concave mesh static
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

    end_time = time.time() + 60.0
    while time.time() < end_time:
        p.stepSimulation()
        time.sleep(1.0 / 240.0)
    p.disconnect()

if __name__ == "__main__":
    main()
