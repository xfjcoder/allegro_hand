"""Stage 6, second RL design iteration: same goal as rl_env_baoding.py (real two-ball
circulation, not a bounded wobble or a one-time shift), on the larger, more stable ball scene
(assets/scene_two_balls_large.xml) with two changes motivated directly by what baoding01/02 and
the scripted leapfrog experiments showed (see README Stage 6):

1. **Grace periods instead of instant termination on contact/rate violations.** Every scripted
   "real progress" attempt (orbit_wave_ctrl, the leapfrog sequence) passes through BRIEFLY risky
   transitional states on the way to genuine progress -- a ball momentarily losing a fingertip
   contact as it's nudged past the other, or a brief rate spike during a push. `BaodingEnv`
   (rl_env_baoding.py) terminates instantly on any single bad step, which empirically produces a
   risk-averse policy: `baoding02` took one "safe" partial shift and then went static rather than
   risk the termination penalty by continuing. Position-based retention (a ball actually ending
   up far from where it should be) still terminates immediately -- that's a real failure, not a
   transient. Contact/rate violations now need to PERSIST for `GRACE_STEPS` before terminating.

2. **An explicit milestone bonus for crossing NEW rotation thresholds.** Dense per-step reward
   alone doesn't distinguish "one shift, then stop" from "keep going" -- both earn the same total
   for the same net displacement. Added `MILESTONE_BONUS`, paid once per episode the first time
   |cumulative orbit| crosses each new multiple of `MILESTONE_RAD` (default 90 degrees) --
   directly rewards continuing past wherever the "safe, stop here" point would otherwise be.
"""
import gymnasium as gym
import mujoco as mj
import numpy as np
from gymnasium import spaces

from hand_env import (BAODING_GRASP_TARGET_LARGE, BALL_REST_POS_LARGE, MAX_BALL_SEPARATION_LARGE,
                       MIN_BALL_SEPARATION_LARGE, N_FINGER_JOINTS, TWO_BALL_SCENE_LARGE,
                       ball_pair_angle_rad, ball_separation_ok, balls_retained,
                       hand_contact_count, ramp_to_target, settle_object)

SCENE = TWO_BALL_SCENE_LARGE
CONTROL_DT = 0.02
EPISODE_SECONDS = 10.0  # longer than Stage 5/baoding01-02's 6s -- more room for multiple laps
EPISODE_STEPS = int(EPISODE_SECONDS / CONTROL_DT)

MAX_JOINT_RATE = 0.05
# `fingertip_contact_count` (capsule-tip-only) turned out too marginal for a long hold: even a
# completely static BAODING_GRASP_TARGET_LARGE genuinely loses one fingertip contact (real soft-
# actuator creep over ~6.5s, not noise) that would fail every episode regardless of policy.
# `hand_contact_count` (any hand link) is far more robust -- measured c1=3, c2=4 rock-stable over
# a 400-step/8s static hold -- so thresholds here have real margin, unlike v1's boundary-hugging 2.
MIN_HAND_CONTACTS_PER_BALL = 1
MIN_HAND_CONTACTS_TOTAL = 3
MAX_ORBIT_RATE = 8.0
MIN_ACTIVE_JOINT_VEL = 0.1
MAX_SEPARATION_DRIFT = 0.015  # meters -- slightly more headroom than v1's 0.012, calibrated
# against the leapfrog scripted experiment's own real-progress-before-failure range (~0.02 over
# several reps) so legitimate progress isn't immediately flagged as drift.
GRACE_STEPS = 15  # 0.3s -- a contact/rate violation must persist this long to terminate

MILESTONE_RAD = np.pi / 4  # 45 degrees -- v1 (90deg) hit exactly ONE milestone (baoding_v2_01:
# 117 degrees, a real visible partial circulation -- balls went from side-by-side to stacked --
# then settled into a permanent, very stable static hold rather than attempting a second
# milestone). Smaller increments give more frequent reward shaping toward continuing.
MILESTONE_BONUS = 25.0  # up from 5.0 -- 8 milestones (one full 360-degree lap) would net 200,
# which should dominate the one-time 8.0 drop penalty if the policy is willing to risk one
# failed attempt per several successful laps' worth of reward.

W_ORBIT = 1.0
W_ACTION = 0.01
DROP_PENALTY = 8.0


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class BaodingEnvV2(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, scene=SCENE, episode_steps=EPISODE_STEPS):
        super().__init__()
        self.model = mj.MjModel.from_xml_path(scene)
        self.data = mj.MjData(self.model)
        self.episode_steps = episode_steps
        self.n_phys_steps = int(round(CONTROL_DT / self.model.opt.timestep))

        self.ctrl_lo = self.model.actuator_ctrlrange[:, 0].copy()
        self.ctrl_hi = self.model.actuator_ctrlrange[:, 1].copy()

        self.action_space = spaces.Box(-1.0, 1.0, shape=(N_FINGER_JOINTS,), dtype=np.float32)
        obs_dim = N_FINGER_JOINTS * 3 + 7 + 6 + 7 + 6
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)

        self._t = 0
        self._angle_prev = None
        self._bad_streak = 0
        self._cum_orbit = 0.0
        self._next_milestone = MILESTONE_RAD

    def _obs(self):
        return np.concatenate([
            self.data.qpos[:N_FINGER_JOINTS],
            self.data.qvel[:N_FINGER_JOINTS],
            self.data.ctrl.copy(),
            self.data.qpos[N_FINGER_JOINTS:N_FINGER_JOINTS + 7],
            self.data.qvel[N_FINGER_JOINTS:N_FINGER_JOINTS + 6],
            self.data.qpos[N_FINGER_JOINTS + 7:N_FINGER_JOINTS + 14],
            self.data.qvel[N_FINGER_JOINTS + 6:N_FINGER_JOINTS + 12],
        ]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mj.mj_resetData(self.model, self.data)
        mj.mj_forward(self.model, self.data)
        settle_object(self.model, self.data)
        ramp_to_target(self.model, self.data, BAODING_GRASP_TARGET_LARGE, ramp_seconds=1.0,
                        hold_seconds=0.3)
        self._t = 0
        self._angle_prev = ball_pair_angle_rad(self.data)
        _, self._sep0 = ball_separation_ok(self.data, MIN_BALL_SEPARATION_LARGE,
                                            MAX_BALL_SEPARATION_LARGE)
        self._bad_streak = 0
        self._cum_orbit = 0.0
        self._next_milestone = MILESTONE_RAD
        return self._obs(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        target_delta = action * MAX_JOINT_RATE
        new_ctrl = np.clip(self.data.ctrl + target_delta, self.ctrl_lo, self.ctrl_hi)
        prev_ctrl = self.data.ctrl.copy()
        self.data.ctrl[:] = new_ctrl
        for _ in range(self.n_phys_steps):
            mj.mj_step(self.model, self.data)

        angle_cur = ball_pair_angle_rad(self.data)
        orbit_step = _wrap(angle_cur - self._angle_prev)
        self._angle_prev = angle_cur
        orbit_rate = orbit_step / CONTROL_DT
        self._cum_orbit += orbit_step

        pos_retained, dists = balls_retained(self.data, rest_pos=BALL_REST_POS_LARGE)
        sep_ok, sep = ball_separation_ok(self.data, MIN_BALL_SEPARATION_LARGE,
                                          MAX_BALL_SEPARATION_LARGE)
        drift_ok = abs(sep - self._sep0) <= MAX_SEPARATION_DRIFT
        c1 = hand_contact_count(self.model, self.data, "ball1")
        c2 = hand_contact_count(self.model, self.data, "ball2")
        contacts_ok = (c1 >= MIN_HAND_CONTACTS_PER_BALL and c2 >= MIN_HAND_CONTACTS_PER_BALL and
                       (c1 + c2) >= MIN_HAND_CONTACTS_TOTAL)
        not_flung = abs(orbit_rate) <= MAX_ORBIT_RATE

        # Position retention is a hard, immediate failure -- a ball actually ending up far from
        # where it belongs isn't a "transient risky state," it's dropped. Contact/rate violations
        # get a grace period (see module docstring).
        transient_ok = sep_ok and drift_ok and contacts_ok and not_flung
        if pos_retained and transient_ok:
            self._bad_streak = 0
        elif pos_retained:
            self._bad_streak += 1
        retained = pos_retained and self._bad_streak < GRACE_STEPS
        actively_moving = np.linalg.norm(self.data.qvel[:N_FINGER_JOINTS]) >= MIN_ACTIVE_JOINT_VEL

        action_penalty = W_ACTION * float(np.sum((new_ctrl - prev_ctrl) ** 2))
        orbit_reward = np.clip(orbit_step, -MAX_ORBIT_RATE * CONTROL_DT, MAX_ORBIT_RATE * CONTROL_DT)
        reward = W_ORBIT * float(orbit_reward) * actively_moving - action_penalty

        milestone_hit = False
        while abs(self._cum_orbit) >= self._next_milestone:
            reward += MILESTONE_BONUS
            self._next_milestone += MILESTONE_RAD
            milestone_hit = True

        self._t += 1
        terminated = not retained
        if terminated:
            reward -= DROP_PENALTY
        truncated = self._t >= self.episode_steps

        info = {"retained": retained, "dist1": dists["ball1"], "dist2": dists["ball2"],
                "separation": sep, "separation_drift": sep - self._sep0,
                "n_contacts_1": c1, "n_contacts_2": c2,
                "orbit_step_rad": orbit_step, "orbit_rate": orbit_rate,
                "cum_orbit_rad": self._cum_orbit, "bad_streak": self._bad_streak,
                "milestone_hit": milestone_hit, "actively_moving": bool(actively_moving)}
        if terminated or truncated:
            info["success"] = retained
        return self._obs(), reward, terminated, truncated, info
