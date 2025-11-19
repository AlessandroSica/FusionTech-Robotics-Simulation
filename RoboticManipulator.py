# file: irb6700_pybullet_demo.py
"""
Quick start:
1) Get the model (needs ROS tools or just 'xacro' CLI):
   git clone https://github.com/ros-industrial/abb_irb6700_support.git
   pip install xacro
   # Example variant (200/2.60). List xacros under abb_irb6700_support/urdf/
   python - <<'PY'
import os, subprocess, sys
pkg = "abb_irb6700_support"
xacro = os.path.join(pkg, "urdf", "irb6700_200_260_macro.xacro")  # pick variant
out   = os.path.join(pkg, "urdf", "irb6700_200_260.urdf")
subprocess.check_call(["xacro", xacro, f"robot_name:=irb6700_200_260", f"prefix:=",
                       f"use_old_inertia:=false"])
# xacro prints to stdout → redirect:
# xacro ... > abb_irb6700_support/urdf/irb6700_200_260.urdf
PY

   # Or do it manually:
   # xacro abb_irb6700_support/urdf/irb6700_200_260_macro.xacro \
   #   robot_name:=irb6700_200_260 > abb_irb6700_support/urdf/irb6700_200_260.urdf

2) Run this demo and point --urdf to the generated file:
   python irb6700_pybullet_demo.py --urdf abb_irb6700_support/urdf/irb6700_200_260.urdf
"""
import time
import math
import argparse
from typing import List, Tuple

import pybullet as p
import pybullet_data


def get_revolute_joints(body_id: int) -> List[int]:
    jids = []
    for j in range(p.getNumJoints(body_id)):
        ji = p.getJointInfo(body_id, j)
        if ji[2] in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
            jids.append(j)
    return jids


def end_effector_index(body_id: int) -> int:
    # Heuristic: last link that moves (often final joint)
    n = p.getNumJoints(body_id)
    if n == 0:
        return -1
    return n - 1


def ik_goto(body_id: int, ee_idx: int, joint_indices: List[int],
           target_pos: Tuple[float, float, float],
           target_rpy: Tuple[float, float, float],
           steps: int = 240,
           max_force: float = 2000.0) -> None:
    orn = p.getQuaternionFromEuler(target_rpy)
    sol = p.calculateInverseKinematics(
        body_id, ee_idx, target_pos, orn,
        maxNumIterations=200, residualThreshold=1e-4
    )
    sol = sol[:len(joint_indices)]
    p.setJointMotorControlArray(
        body_id, joint_indices, p.POSITION_CONTROL,
        targetPositions=sol,
        positionGains=[0.2]*len(joint_indices),
        velocityGains=[1.0]*len(joint_indices),
        forces=[max_force]*len(joint_indices),
    )
    for _ in range(steps):
        p.stepSimulation()
        time.sleep(1.0/240.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", type=str, required=True,
                    help="Path to IRB6700 URDF generated from the xacro (ROS-Industrial).")
    ap.add_argument("--fixed-base", action="store_true", default=True)
    ap.add_argument("--gui", action="store_true", default=True)
    ap.add_argument("--ee-rpy", type=float, nargs=3, default=[math.pi, 0.0, 0.0],
                    help="End-effector orientation in radians (roll pitch yaw).")
    args = ap.parse_args()

    cid = p.connect(p.GUI if args.gui else p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetDebugVisualizerCamera(cameraDistance=2.6, cameraYaw=55, cameraPitch=-20,
                                 cameraTargetPosition=[1.1, 0, 1.1])
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.setGravity(0, 0, -9.81)
    p.setPhysicsEngineParameter(numSolverIterations=200)
    p.setTimeStep(1.0/240.0)

    p.loadURDF("plane.urdf")
    robot_uid = p.loadURDF(args.urdf, basePosition=[0, 0, 0], useFixedBase=args.fixed_base)
    joint_indices = get_revolute_joints(robot_uid)
    ee_idx = end_effector_index(robot_uid)

    # Home above base
    home = (1.4, 0.0, 1.5)   # tune based on variant reach
    ik_goto(robot_uid, ee_idx, joint_indices, home, tuple(args.ee_rpy), steps=480, max_force=3000.0)

    # Sweep a large arc in front of the robot
    radius = 0.6
    center = (1.6, 0.0, 1.4)
    for k in range(240):
        th = math.radians(k)
        px = center[0] + radius*math.cos(th)
        py = center[1] + 0.4*math.sin(th)
        pz = center[2] + 0.2*math.sin(2*th)
        ik_goto(robot_uid, ee_idx, joint_indices, (px, py, pz), tuple(args.ee_rpy), steps=8, max_force=3000.0)

    time.sleep(1.0)
    p.disconnect()


if __name__ == "__main__":
    main()
