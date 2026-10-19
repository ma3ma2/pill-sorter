"""服薬仕分けシミュレーション（ステップ2：複数の薬を曜日×朝夜のケースへ）。

流れ: 処方テキスト → planner で作業リスト → 机の上の薬を1つずつ掴んで該当マスへ
      → 置いた位置から各マスの中身を判定して正誤を表示。
実行: python src/sorter.py  [--gpu]
"""
import sys

import genesis as gs
import numpy as np

from config import CONFIG
from planner import (
    CASE_X0,
    CASE_Y0,
    CELL,
    DAYS,
    PILL,
    SLOTS,
    WALL_H,
    cell_center,
    cell_of_xy,
    make_plan,
)


def _pad16(x):
    """動画コーデック都合で縦横を16の倍数にそろえる。"""
    h, w = x.shape[:2]
    ph, pw = (-h) % 16, (-w) % 16
    if ph or pw:
        x = np.pad(x, ((0, ph), (0, pw), (0, 0)), mode="edge")
    return x


RX = CONFIG["prescription"]["main"]
COLORS = {k: tuple(v) for k, v in CONFIG["pills"]["colors"].items()}
DOWN = np.array([0, 1, 0, 0])                 # 手先を真下に向ける姿勢
GRIP_Z, LIFT_Z = CONFIG["robot"]["grip_z"], CONFIG["robot"]["lift_z"]   # 掴む/運ぶ高さ
PLACE_Z = CONFIG["robot"]["place_z"]                     # 放す高さ（跳ねて外へ出ないように低くする）


def cell_of(pos):
    """物体の位置からどのマスに入っているかを返す（外なら None）。"""
    if pos[2] >= 0.03:
        return None
    return cell_of_xy(pos[0], pos[1])


def build_scene(tasks):
    scene = gs.Scene(sim_options=gs.options.SimOptions(dt=0.01, substeps=4), show_viewer=False,
                     rigid_options=gs.options.RigidOptions(box_box_detection=True))
    scene.add_entity(gs.morphs.Plane())
    # ケース: 外枠＋仕切り（高さは config の wall_h、既定8mm）
    W, H = len(SLOTS) * CELL, len(DAYS) * CELL
    wall = gs.surfaces.Default(color=(0.85, 0.85, 0.9))
    for k in range(len(SLOTS) + 1):
        scene.add_entity(gs.morphs.Box(size=(0.005, H, WALL_H), pos=(CASE_X0 + k * CELL, CASE_Y0 + H / 2, WALL_H / 2), fixed=True), surface=wall)
    for k in range(len(DAYS) + 1):
        scene.add_entity(gs.morphs.Box(size=(W, 0.005, WALL_H), pos=(CASE_X0 + W / 2, CASE_Y0 + k * CELL, WALL_H / 2), fixed=True), surface=wall)
    # 薬: 作業リストの数だけ手前に2列で並べる
    pills = []
    for n, t in enumerate(tasks):
        x, y = 0.38 + (n % 2) * 0.07, -0.25 + (n // 2) * 0.08
        e = scene.add_entity(gs.morphs.Box(size=(PILL,) * 3, pos=(x, y, PILL / 2)), surface=gs.surfaces.Default(color=COLORS[t["pill"]]))
        pills.append(e)
    franka = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
    cam = scene.add_camera(res=(960, 540), pos=(1.45, 0.0, 1.0), lookat=(0.5, 0.0, 0.05), fov=42, GUI=False)
    top = scene.add_camera(res=(480, 540), pos=(0.64, 0.0, 0.9), lookat=(0.64, 0.0, 0.0), up=(1, 0, 0), fov=40, GUI=False)
    scene.build()
    franka.set_dofs_kp(np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]))
    franka.set_dofs_kv(np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]))
    return scene, franka, pills, cam, top


class Robot:
    def __init__(self, scene, franka, frames):
        self.s, self.f, self.ee, self.frames = scene, franka, franka.get_link("hand"), frames
        self.arm, self.fing = np.arange(7), np.arange(7, 9)
        self.home = self.f.get_dofs_position()[:7].clone()   # 初期姿勢（ケースから離れている）

    def park(self, n=70):
        """撮影前にアームを初期姿勢へ戻し、ケースを隠さないようにする。"""
        self.f.control_dofs_position(self.home, self.arm)
        self.step(n)

    def step(self, n):
        for k in range(n):
            self.s.step()
            if k % 2 == 0:
                self.frames()                  # 2ステップに1枚録画（動画を短く）

    def goto(self, x, y, z, n=90):
        q = self.f.inverse_kinematics(link=self.ee, pos=np.array([x, y, z]), quat=DOWN)
        self.f.control_dofs_position(q[:7], self.arm)
        self.step(n)

    def grip(self, close):
        if close:
            self.f.control_dofs_force(np.array([-1.0, -1.0]), self.fing)
        else:
            self.f.control_dofs_position(np.array([0.04, 0.04]), self.fing)
        self.step(50)

    def pick_place(self, src, dst):
        self.grip(False)
        self.goto(*src, LIFT_Z)
        self.goto(*src, GRIP_Z, 60)
        self.grip(True)
        self.goto(*src, LIFT_Z, 60)
        self.goto(*dst, LIFT_Z, 110)
        self.goto(*dst, PLACE_Z, 50)
        self.grip(False)
        self.goto(*dst, LIFT_Z, 40)


def main(rx=RX, out="out/sorter.mp4"):
    gs.init(backend=gs.gpu if "--gpu" in sys.argv else gs.cpu, logging_level="warning")
    np.random.seed(CONFIG["video"]["seed"])          # 再現性
    tasks, how = make_plan(rx)
    print(f"[plan:{how}]", tasks)
    scene, franka, pills, cam, top = build_scene(tasks)
    import imageio
    writer = imageio.get_writer(out, fps=CONFIG["video"]["fps"])
    def frames():
        a = cam.render()[0]
        b = top.render()[0]
        writer.append_data(_pad16(np.concatenate([a, b], axis=1)))
    bot = Robot(scene, franka, frames)
    bot.step(30)
    for t, p in zip(tasks, pills):
        src = p.get_pos().cpu().numpy()[:2]
        bot.pick_place(src, cell_center(t["day"], t["slot"]))
    bot.step(40)
    writer.close()
    # 判定: 各薬が指定どおりのマスに入ったか
    ok = 0
    for t, p in zip(tasks, pills):
        got = cell_of(p.get_pos().cpu().numpy())
        hit = got == (t["day"], t["slot"])
        ok += hit
        print(f"  {t['pill']:6s} → {t['day']}-{t['slot']}  実際: {got}  {'OK' if hit else 'NG'}")
    print(f"正答 {ok}/{len(tasks)}")
    import json
    import os as _os
    try:
        _os.makedirs("out", exist_ok=True)
        with open("out/sort_result.json", "w", encoding="utf-8") as f:
            json.dump({"plan": len(tasks), "ok": ok, "mode": how}, f, ensure_ascii=False)
    except Exception:  # noqa: BLE001, S110  # 結果の保存は失敗してもよい
        pass


if __name__ == "__main__":
    main()
