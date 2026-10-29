"""Stage 7: toss and catch (`hand_env.TossCatchController` with `TOSS_CATCH_PARAMS`). Runs the cycle
from 9 starting ball positions (+-8mm in x and y) and at ball mass x0.8 / x1.2, up to `--cycles`
toss+catch cycles each, and reports cycles completed, toss heights, peak actuator torque, and the
failure reason if any.

Usage:
    python3 eval_stage7_toss_catch.py --cycles 100
"""
import argparse
import itertools

import mujoco as mj
import numpy as np

from hand_env import TOSS_CATCH_PARAMS, TOSS_SCENE, TOSS_SPAWN, TossCatchController, place_balls


def run(cycles, dx=0.0, dy=0.0, mass_scale=1.0, seconds=200.0):
    m = mj.MjModel.from_xml_path(TOSS_SCENE)
    bid = m.body("ball").id
    m.body_mass[bid] *= mass_scale
    m.body_inertia[bid] *= mass_scale
    d = mj.MjData(m)
    place_balls(m, d, {"ball": TOSS_SPAWN["ball"] + np.array([dx, dy, 0.0])})
    ctl = TossCatchController(**TOSS_CATCH_PARAMS)
    dt = m.opt.timestep
    d.ctrl[:] = ctl.cup
    for _ in range(int(0.5 / dt)):
        mj.mj_step(m, d)
    peak = 0.0
    lo, hi = m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]
    for _ in range(int(seconds / dt)):
        d.ctrl[:] = np.clip(ctl.step(d, dt), lo, hi)
        mj.mj_step(m, d)
        peak = max(peak, float(np.abs(d.actuator_force).max()))
        if ctl.failed or ctl.cycles >= cycles:
            break
    return ctl, peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles", type=int, default=100)
    args = ap.parse_args()
    conds = [(dx, dy, 1.0) for dx, dy in itertools.product((-0.008, 0.0, 0.008), repeat=2)]
    conds += [(0.0, 0.0, 0.8), (0.0, 0.0, 1.2)]
    total, all_apex, peak_all = 0, [], 0.0
    for dx, dy, ms in conds:
        ctl, peak = run(args.cycles, dx, dy, ms)
        total += ctl.cycles
        all_apex += ctl.apexes
        peak_all = max(peak_all, peak)
        apex = np.array(ctl.apexes) * 100
        print(f"start offset ({dx * 1000:+.0f},{dy * 1000:+.0f})mm mass x{ms:.1f}: {ctl.cycles}/{args.cycles} "
              f"cycles, toss height {apex.mean():.1f}cm (min {apex.min():.1f}, max {apex.max():.1f})"
              + (f" -- FAILED: {ctl.failed}" if ctl.failed else ""))
    print(f"TOTAL {total}/{args.cycles * len(conds)} cycles, mean toss height "
          f"{np.mean(all_apex) * 100:.1f}cm, peak actuator torque {peak_all:.2f} N*m")


if __name__ == "__main__":
    main()
