"""Stage 5's RL environment: can a learned policy do what Stage 4's scripted finger-gaiting
attempt couldn't -- break past Stage 3's ~23-25 degree simultaneous-motion rotation ceiling by
learning WHEN and WHICH finger to release/recontact (real contact planning), instead of a fixed
release-to-same-target regrasp?

Full privileged state as input (joint + object pose/velocity), no vision -- same staged
discipline as barrett_wam/panda_vision_grasp's own first RL attempts: show the easier,
fully-observed version of the problem is solvable before adding a harder observation model.

Reward is dense, per-control-step signed rotation about the object's vertical axis (via
`hand_env.twist_angle_rad`'s swing-twist decomposition, accumulated rather than a one-shot
start-to-now angle -- see that function's docstring for why `quat_angle_deg` can't tell a real
multi-turn spin from "came back to start"), plus a hard termination+penalty if the object is
dropped. This directly extends Stage 3/4's own success metric (cumulative rotation, same
retention criterion) rather than inventing a new one.
"""
import gymnasium as gym
import mujoco as mj
import numpy as np
from gymnasium import spaces

from hand_env import (N_FINGER_JOINTS, POWER_GRASP_TARGET, fingertip_contact_count,
                       object_retained, ramp_to_target, settle_object, twist_angle_rad)

SCENE = "assets/scene_right.xml"
CONTROL_DT = 0.02  # 10 physics steps (dt=0.002) per control step
EPISODE_SECONDS = 6.0
EPISODE_STEPS = int(EPISODE_SECONDS / CONTROL_DT)
# gait01 (first attempt, see README Stage 5) found a severe reward-hacking exploit at
# MAX_JOINT_RATE=0.15: within ~0.3s it splayed all 4 fingers' base joints wide open, flicked the
# object loose, and let it coast with ZERO fingertip contacts (verified via
# hand_env.fingertip_contact_count) for the rest of a 100s rollout -- "retained" only because
# object_retained's position-distance check can't tell a held object from one geometrically
# boxed into the open palm by sheer luck. Cut 3x slower so a multi-joint "explode open in one
# step" motion isn't reachable within a single control step; MIN_FINGERTIP_CONTACTS below closes
# the actual hole (a real grip signal), this is belt-and-suspenders against the same failure mode.
MAX_JOINT_RATE = 0.05  # rad/control-step per actuator
MIN_FINGERTIP_CONTACTS = 2
# gait02 (second attempt, see README Stage 5) found the NEXT exploit once gait01's was closed:
# a wide (but not fully open) splay that keeps exactly 2 fingertips touching while flicking the
# object to instantaneous rotation rates up to ~6.8 rad/s in a ~1.3s burst, then coasting (near-
# zero rate) until it eventually drifts past the position threshold ~30s later. Measured the
# scripted Stage 3 twist primitive's own peak rate for comparison: 0.37 rad/s. A fling-then-coast
# is not finger-gaiting under any reading of the term, so this caps the per-step rate directly
# (4x the scripted baseline's peak -- generous headroom for a genuinely faster learned technique,
# nowhere near what either exploit reached) rather than patching another geometric symptom.
MAX_TWIST_RATE = 1.5  # rad/s -- instantaneous |rot_step / CONTROL_DT| limit
# gait03 (third attempt, see README Stage 5) passed every check above (contacts stayed at 2-3,
# rate stayed under the 1.5 rad/s cap) and STILL wasn't real gaiting: measured finger joint
# velocity during its long "steady rotation" stretch (t=20-30s) and found it averaging 0.009
# rad/s -- the hand was essentially FROZEN. The one wide-splay reconfiguration in the first
# ~1s imparted enough momentum/compliant-contact creep for the object to keep slowly rolling and
# rotating on its own for the next ~45 seconds while the policy did nothing, right up until
# position drift finally failed the retention check. None of gait01/02/03's checks catch this
# because none of them require the rotation to be CAUSED by ongoing finger motion, only that it
# exist within some rate/contact bound. This gates the reward on exactly that: no finger motion,
# no rotation credit, however much the object happens to still be spinning from inertia.
MIN_ACTIVE_JOINT_VEL = 0.1  # L2 norm of the 16 finger-joint qvels, rad/s -- well separated from
# both the measured passive-coast floor (~0.01-0.012) and measured active-motion levels (~1.4-3.8
# during gait03's own opening move), so this doesn't accidentally penalize genuine slow gaiting.

W_ROT = 1.0
W_ACTION = 0.01
DROP_PENALTY = 8.0


class GaitingEnv(gym.Env):
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
        obs_dim = N_FINGER_JOINTS * 3 + 7 + 6  # qpos, qvel, prev_ctrl, obj qpos, obj qvel
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)

        self._t = 0
        self._q_prev = None

    def _obs(self):
        return np.concatenate([
            self.data.qpos[:N_FINGER_JOINTS],
            self.data.qvel[:N_FINGER_JOINTS],
            self.data.ctrl.copy(),
            self.data.qpos[N_FINGER_JOINTS:N_FINGER_JOINTS + 7],
            self.data.qvel[N_FINGER_JOINTS:N_FINGER_JOINTS + 6],
        ]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mj.mj_resetData(self.model, self.data)
        mj.mj_forward(self.model, self.data)
        settle_object(self.model, self.data)
        ramp_to_target(self.model, self.data, POWER_GRASP_TARGET, ramp_seconds=1.0, hold_seconds=0.3)
        self._t = 0
        self._q_prev = self.data.body("object").xquat.copy()
        return self._obs(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        target_delta = action * MAX_JOINT_RATE
        new_ctrl = np.clip(self.data.ctrl + target_delta, self.ctrl_lo, self.ctrl_hi)
        prev_ctrl = self.data.ctrl.copy()
        self.data.ctrl[:] = new_ctrl
        for _ in range(self.n_phys_steps):
            mj.mj_step(self.model, self.data)

        q_cur = self.data.body("object").xquat.copy()
        rot_step = twist_angle_rad(self._q_prev, q_cur)
        self._q_prev = q_cur

        pos_retained, dist = object_retained(self.data)
        n_contacts = fingertip_contact_count(self.model, self.data)
        twist_rate = rot_step / CONTROL_DT
        not_flung = abs(twist_rate) <= MAX_TWIST_RATE
        retained = pos_retained and n_contacts >= MIN_FINGERTIP_CONTACTS and not_flung
        actively_moving = np.linalg.norm(self.data.qvel[:N_FINGER_JOINTS]) >= MIN_ACTIVE_JOINT_VEL
        action_penalty = W_ACTION * float(np.sum((new_ctrl - prev_ctrl) ** 2))
        # Clip the rotation reward itself to the rate cap -- a step that violates it is about to
        # terminate+get penalized below anyway, but without this an action landing exactly on the
        # boundary step could still bank one oversized reward tick first. Gate it on active
        # finger motion too (gait03's finding) -- rotation coasting on inertia with a frozen hand
        # earns nothing, however fast or slow it happens to still be going.
        rot_reward = np.clip(rot_step, -MAX_TWIST_RATE * CONTROL_DT, MAX_TWIST_RATE * CONTROL_DT)
        reward = W_ROT * float(rot_reward) * actively_moving - action_penalty

        self._t += 1
        terminated = not retained
        if terminated:
            reward -= DROP_PENALTY
        truncated = self._t >= self.episode_steps

        info = {"retained": retained, "dist": dist, "rot_step_rad": rot_step,
                "n_contacts": n_contacts, "twist_rate": twist_rate,
                "actively_moving": bool(actively_moving)}
        if terminated or truncated:
            info["success"] = retained
        return self._obs(), reward, terminated, truncated, info
