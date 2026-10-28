"""Stage 6's RL environment: can a learned policy make the two-ball pair (assets/scene_two_balls.
xml) actually ORBIT around each other -- real Baoding-ball motion -- where both scripted
primitives tried (hand_env.twist_ctrl, hand_env.orbit_wave_ctrl) only managed a small bounded
repositioning before flatlining (see README Stage 6)?

Built on Stage 5's hard-won lessons (see README/memory Stage 5 -- three successive exploits:
zero-contact coast, rate-capped-but-still-flung coast, frozen-hand coast on inertia), applied
PROACTIVELY here instead of discovered one at a time:
  1. Per-ball position retention (`hand_env.balls_retained`).
  2. Per-ball AND combined minimum fingertip contact (`hand_env.fingertip_contact_count`).
  3. Ball-pair separation bounded (`hand_env.ball_separation_ok`) -- a Stage-6-specific risk
     Stage 5 never had: flinging one ball far from the other on a wide arc would also read as
     "the pair angle changed a lot" without being anything like real orbiting.
  4. Orbit-rate cap, both reward-clipped and termination-enforced.
  5. Reward gated on active finger motion (a frozen hand coasting on inertia earns nothing).
  6. Separation DRIFT from the episode's own starting separation, bounded (added after baoding01
     -- see README Stage 6: the absolute bound in (3) wasn't tight enough to stop ball1 being
     gradually squeezed/pushed sideways out of the grip over several seconds, which reads as "the
     pair angle changed a lot" in the raw metric without the balls ever actually circling a
     common center at constant mutual distance -- confirmed by rendering the rollout and watching
     ball1 visibly get pushed out to the side, not orbited).

Reward is dense, per-control-step signed change in `hand_env.ball_pair_angle_rad` (wrap-delta
accumulated every step, same reasoning as Stage 5's `twist_angle_rad`: a raw angle alone can't
tell one real orbit from "wiggled and came back").
"""
import gymnasium as gym
import mujoco as mj
import numpy as np
from gymnasium import spaces

from hand_env import (BAODING_GRASP_TARGET, N_FINGER_JOINTS, ball_pair_angle_rad,
                       ball_separation_ok, balls_retained, fingertip_contact_count,
                       ramp_to_target, settle_object)

SCENE = "assets/scene_two_balls.xml"
CONTROL_DT = 0.02  # 10 physics steps (dt=0.002) per control step
EPISODE_SECONDS = 6.0
EPISODE_STEPS = int(EPISODE_SECONDS / CONTROL_DT)

MAX_JOINT_RATE = 0.05  # rad/control-step per actuator -- Stage 5's gait02/03-proven-safe value
MIN_FINGERTIP_CONTACTS_PER_BALL = 1  # matches what BAODING_GRASP_TARGET itself achieves (1, 2)
MIN_FINGERTIP_CONTACTS_TOTAL = 2
# Measured the scripted orbit_wave_ctrl's own peak per-step orbit rate for calibration (same
# methodology as Stage 5's MAX_TWIST_RATE): 2.84 rad/s. ~3x margin for a genuinely faster learned
# technique, less generous than Stage 5's 4x given the fling-exploit history.
MAX_ORBIT_RATE = 8.0  # rad/s
MIN_ACTIVE_JOINT_VEL = 0.1  # rad/s, L2 norm of the 16 finger-joint qvels -- Stage 5's proven value
# baoding01's ejection drifted separation smoothly from 0.039m (start) to ~0.051m within 3.6s,
# well inside the absolute [0.025, 0.065] band above -- real orbiting should hold roughly
# CONSTANT mutual distance (that's what distinguishes "circling a shared center" from "one ball
# sliding away"), so this bounds drift from each episode's OWN starting separation instead of a
# fixed global band. Calibrated against orbit_wave_ctrl's own natural wobble (0.038-0.050m
# observed over 2 scripted cycles, so ~0.012m of legitimate variation already happens).
MAX_SEPARATION_DRIFT = 0.012  # meters, from the episode's own reset-time separation

W_ORBIT = 1.0
W_ACTION = 0.01
DROP_PENALTY = 8.0


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class BaodingEnv(gym.Env):
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
        # qpos, qvel, prev_ctrl (16 each) + ball1 qpos/qvel (7+6) + ball2 qpos/qvel (7+6)
        obs_dim = N_FINGER_JOINTS * 3 + 7 + 6 + 7 + 6
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)

        self._t = 0
        self._angle_prev = None

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
        ramp_to_target(self.model, self.data, BAODING_GRASP_TARGET, ramp_seconds=1.0,
                        hold_seconds=0.3)
        self._t = 0
        self._angle_prev = ball_pair_angle_rad(self.data)
        _, self._sep0 = ball_separation_ok(self.data)
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

        pos_retained, dists = balls_retained(self.data)
        sep_ok, sep = ball_separation_ok(self.data)
        drift_ok = abs(sep - self._sep0) <= MAX_SEPARATION_DRIFT
        c1 = fingertip_contact_count(self.model, self.data, object_body_name="ball1")
        c2 = fingertip_contact_count(self.model, self.data, object_body_name="ball2")
        contacts_ok = (c1 >= MIN_FINGERTIP_CONTACTS_PER_BALL and
                       c2 >= MIN_FINGERTIP_CONTACTS_PER_BALL and
                       (c1 + c2) >= MIN_FINGERTIP_CONTACTS_TOTAL)
        not_flung = abs(orbit_rate) <= MAX_ORBIT_RATE
        retained = pos_retained and sep_ok and drift_ok and contacts_ok and not_flung
        actively_moving = np.linalg.norm(self.data.qvel[:N_FINGER_JOINTS]) >= MIN_ACTIVE_JOINT_VEL

        action_penalty = W_ACTION * float(np.sum((new_ctrl - prev_ctrl) ** 2))
        orbit_reward = np.clip(orbit_step, -MAX_ORBIT_RATE * CONTROL_DT, MAX_ORBIT_RATE * CONTROL_DT)
        reward = W_ORBIT * float(orbit_reward) * actively_moving - action_penalty

        self._t += 1
        terminated = not retained
        if terminated:
            reward -= DROP_PENALTY
        truncated = self._t >= self.episode_steps

        info = {"retained": retained, "dist1": dists["ball1"], "dist2": dists["ball2"],
                "separation": sep, "separation_drift": sep - self._sep0,
                "n_contacts_1": c1, "n_contacts_2": c2,
                "orbit_step_rad": orbit_step, "orbit_rate": orbit_rate,
                "actively_moving": bool(actively_moving)}
        if terminated or truncated:
            info["success"] = retained
        return self._obs(), reward, terminated, truncated, info
