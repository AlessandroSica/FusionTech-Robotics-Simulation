import math
import os
import time
from typing import List, Tuple, Set

import pybullet as p
import pybullet_data

# ---------- Math / geometry helpers ----------
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
def _normalize(v):
    n = (v[0]**2 + v[1]**2 + v[2]**2) ** 0.5
    return (v[0]/n, v[1]/n, v[2]/n)

def icosahedron_edges(verts):
    n=len(verts); dmin=None; dists={}
    for i in range(n):
        for j in range(i+1,n):
            d=_dist(verts[i],verts[j]); dists[(i,j)]=d
            if d>0 and (dmin is None or d<dmin): dmin=d
    eps=dmin*1e-6
    return [(i,j) for (i,j),d in dists.items() if abs(d-dmin)<=eps]

def truncated_icosahedron_vertex_dirs(tau: float = 1.0/3.0):
    base=icosahedron_vertices(); edges=icosahedron_edges(base)
    dirs=[]; seen=set()
    def _add(v):
        vn=_normalize(v); key=(round(vn[0],6),round(vn[1],6),round(vn[2],6))
        if key not in seen: seen.add(key); dirs.append(vn)
    for i,j in edges:
        vi,vj=base[i],base[j]
        _add(((1-tau)*vi[0]+tau*vj[0],(1-tau)*vi[1]+tau*vj[1],(1-tau)*vi[2]+tau*vj[2]))
        _add(((1-tau)*vj[0]+tau*vi[0],(1-tau)*vj[1]+tau*vi[1],(1-tau)*vj[2]+tau*vi[2]))
    return dirs  # 60

def ortho_basis_from_dir(u):
    ux,uy,uz=u
    a=(1.0,0.0,0.0) if abs(ux)<0.5 and abs(uy)<0.5 else (0.0,0.0,1.0)
    dot=a[0]*ux+a[1]*uy+a[2]*uz
    t1=(a[0]-dot*ux,a[1]-dot*uy,a[2]-dot*uz); t1=_normalize(t1)
    t2=(uy*t1[2]-uz*t1[1],uz*t1[0]-ux*t1[2],ux*t1[1]-uy*t1[0])
    return t1,t2

# ---------- Mesh builders ----------
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

def build_oriented_hex_prism(bottom_center, axis_dir, circumradius, length):
    u = _normalize(axis_dir); t1, t2 = ortho_basis_from_dir(u)
    top_center = (bottom_center[0] + length * u[0],
                  bottom_center[1] + length * u[1],
                  bottom_center[2] + length * u[2])
    ring_bot, ring_top = [], []
    for k in range(6):
        ang = 2.0 * math.pi * k / 6.0
        ca, sa = math.cos(ang), math.sin(ang)
        px = circumradius * (ca * t1[0] + sa * t2[0])
        py = circumradius * (ca * t1[1] + sa * t2[1])
        pz = circumradius * (ca * t1[2] + sa * t2[2])
        ring_bot.append((bottom_center[0] + px, bottom_center[1] + py, bottom_center[2] + pz))
        ring_top.append((top_center[0] + px, top_center[1] + py, top_center[2] + pz))
    verts = ring_bot + ring_top
    faces = []
    n = 6
    for k in range(n):
        a = k; b = (k + 1) % n; c = n + (k + 1) % n; d = n + k
        faces.append((a, b, c)); faces.append((a, c, d))
    for k in range(1, n - 1): faces.append((0, k + 1, k))
    for k in range(1, n - 1): faces.append((n, n + k, n + k + 1))
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

# ---- Line→sphere hit (returns smallest t>0 for P + t*d) ----
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

# ---------- Parameters ----------
base_radius = 3.2
circle_angle_deg = 2.5
circle_height = 0.25
circle_segments = 48
hex_length = 0.7
sphere_lats, sphere_lons = 256, 512
center = [0.0, 0.0, 4.5]
tau = 1.0 / 3.0

# supports
support_radius = 0.22
support_segments = 36
support_target_offset = 0.6
support_clearance = 0.02
support_shorten_factor = 0.65   # <1 → angled pillar shorter
underground_depth = 0.30        # how far below ground the vertical goes

obj_base = os.path.abspath("solid_sphere_with_corner_circles.obj")
obj_hex  = os.path.abspath("corner_hex_prisms.obj")
obj_sup  = os.path.abspath("sphere_supports_short_vert.obj")

# ---------- Directions ----------
def dirs_corners():
    return truncated_icosahedron_vertex_dirs(tau=tau)

# ---------- Build base (sphere + cylinders) ----------
parts_base = []
sphere_verts, sphere_faces = build_sphere_mesh(base_radius, sphere_lats, sphere_lons)
parts_base.append((sphere_verts, sphere_faces))

circle_radius = base_radius * math.sin(math.radians(circle_angle_deg))
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

# ---------- Hex prisms ----------
parts_hex = []
for top_center, u in cylinder_tops:
    hv, hf = build_oriented_hex_prism(top_center, u, 3 * circle_radius, hex_length)
    parts_hex.append((hv, hf))
write_single_obj(obj_hex, parts_hex)

# ---------- NEW: 4 supports: shorter + vertical-to-underground ----------
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

    # choose lower end above ground
    t_end = max(0.0, support_shorten_factor * t_top)
    end_point = (foot[0] + t_end * dir_vec[0],
                 foot[1] + t_end * dir_vec[1],
                 foot[2] + t_end * dir_vec[2])

    # angled segment: end_point -> top_point
    angled_center = ((end_point[0] + top_point[0]) * 0.5,
                     (end_point[1] + top_point[1]) * 0.5,
                     (end_point[2] + top_point[2]) * 0.5)
    angled_len = _dist(end_point, top_point)
    sv, sf = build_oriented_cylinder(angled_center, dir_vec, support_radius, angled_len, support_segments)
    parts_sup.append((sv, sf))

    # vertical segment: from end_point straight down past ground
    bottom_z = ground_z_local - underground_depth
    vert_len = end_point[2] - bottom_z
    vert_center = (end_point[0], end_point[1], bottom_z + 0.5 * vert_len)
    vv, vf = build_oriented_cylinder(vert_center, (0.0, 0.0, 1.0), support_radius, vert_len, support_segments)
    parts_sup.append((vv, vf))

write_single_obj(obj_sup, parts_sup)

# ---------- PyBullet setup ----------
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

# Base (gray)
col_base = p.createCollisionShape(p.GEOM_MESH, fileName=obj_base, flags=p.GEOM_FORCE_CONCAVE_TRIMESH)
vis_base = p.createVisualShape(p.GEOM_MESH, fileName=obj_base, rgbaColor=[0.5, 0.5, 0.5, 1.0])
body_base = p.createMultiBody(0, col_base, vis_base, basePosition=center, baseOrientation=[0,0,0,1])
p.changeVisualShape(body_base, -1, specularColor=[0.0, 0.0, 0.0])

# Hex prisms (orange)
col_hex = p.createCollisionShape(p.GEOM_MESH, fileName=obj_hex, flags=p.GEOM_FORCE_CONCAVE_TRIMESH)
vis_hex = p.createVisualShape(p.GEOM_MESH, fileName=obj_hex, rgbaColor=[1.0, 0.55, 0.0, 1.0])
body_hex = p.createMultiBody(0, col_hex, vis_hex, basePosition=center, baseOrientation=[0,0,0,1])
p.changeVisualShape(body_hex, -1, specularColor=[0.0, 0.0, 0.0])

# Supports (dark steel)
col_sup = p.createCollisionShape(p.GEOM_MESH, fileName=obj_sup, flags=p.GEOM_FORCE_CONCAVE_TRIMESH)
vis_sup = p.createVisualShape(p.GEOM_MESH, fileName=obj_sup, rgbaColor=[0.35, 0.35, 0.35, 1.0])
body_sup = p.createMultiBody(0, col_sup, vis_sup, basePosition=center, baseOrientation=[0,0,0,1])
p.changeVisualShape(body_sup, -1, specularColor=[0.15, 0.15, 0.15])

# Simulation
end_time = time.time() + 60.0
while time.time() < end_time:
    p.stepSimulation()
    time.sleep(1.0 / 240.0)

p.disconnect()
