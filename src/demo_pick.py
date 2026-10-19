"""最小デモ: Franka が薬を1個つかんでケースのマスへ入れ、mp4 に記録する。"""
import os, sys, numpy as np, genesis as gs
backend = gs.gpu if "--gpu" in sys.argv else gs.cpu
gs.init(backend=backend, logging_level="warning")
scene = gs.Scene(sim_options=gs.options.SimOptions(dt=0.01, substeps=4), show_viewer=False,
                 rigid_options=gs.options.RigidOptions(box_box_detection=True))
scene.add_entity(gs.morphs.Plane())
pill = scene.add_entity(gs.morphs.Box(size=(0.03, 0.03, 0.03), pos=(0.55, -0.15, 0.015)),
                        surface=gs.surfaces.Default(color=(0.9, 0.1, 0.1)))
# case cell: 4 thin walls
cx, cy = 0.55, 0.2
for dx, dy, sx, sy in [(0.05,0,0.01,0.11),(-0.05,0,0.01,0.11),(0,0.05,0.11,0.01),(0,-0.05,0.11,0.01)]:
    scene.add_entity(gs.morphs.Box(size=(sx, sy, 0.04), pos=(cx+dx, cy+dy, 0.02), fixed=True),
                     surface=gs.surfaces.Default(color=(0.3, 0.5, 0.9)))
franka = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
cam = scene.add_camera(res=(640, 480), pos=(1.4, 0.0, 0.9), lookat=(0.5, 0.0, 0.1), fov=40, GUI=False)
scene.build()
motors, fingers = np.arange(7), np.arange(7, 9)
franka.set_dofs_kp(np.array([4500,4500,3500,3500,2000,2000,2000,100,100]))
franka.set_dofs_kv(np.array([450,450,350,350,200,200,200,10,10]))
ee = franka.get_link("hand")
os.makedirs("out", exist_ok=True)
cam.start_recording(save_to_filename="out/demo_pick.mp4", fps=50)
def step(n):
    for _ in range(n):
        scene.step(); cam.render()
def move(pos, grip, n=80):
    q = franka.inverse_kinematics(link=ee, pos=np.array(pos), quat=np.array([0, 1, 0, 0]))
    q[-2:] = grip
    path = franka.plan_path(qpos_goal=q, num_waypoints=n) if hasattr(franka, "plan_path") else None
    if path is not None:
        for wp in path:
            franka.control_dofs_position(wp); scene.step(); cam.render()
    else:
        franka.control_dofs_position(q); step(n)
OPEN = 0.04
px, py, _ = pill.get_pos().cpu().numpy()
move([px, py, 0.25], OPEN)
move([px, py, 0.13], OPEN, 60)
franka.control_dofs_position(q:=franka.get_dofs_position()[:7], motors)
franka.control_dofs_force(np.array([-1.0, -1.0]), fingers); step(60)
q = franka.inverse_kinematics(link=ee, pos=np.array([px, py, 0.3]), quat=np.array([0,1,0,0]))
franka.control_dofs_position(q[:-2], motors); step(100)
q = franka.inverse_kinematics(link=ee, pos=np.array([cx, cy, 0.3]), quat=np.array([0,1,0,0]))
franka.control_dofs_position(q[:-2], motors); step(150)
franka.control_dofs_force(np.array([0.0, 0.0]), fingers)
franka.control_dofs_position(np.array([OPEN, OPEN]), fingers); step(120)
pp = pill.get_pos().cpu().numpy()
ok = abs(pp[0]-cx) < 0.045 and abs(pp[1]-cy) < 0.045
print("pill final pos", pp, "IN_CELL" if ok else "MISS")
cam.stop_recording()
