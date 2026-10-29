"""Stage 6, fifth RL design iteration: the 4-finger claw on bigger (60mm) balls, with reward keyed to
WHICH FACE and WHICH SEGMENT of each digit touches the balls.

User, watching the BaodingEnvV4 (`baoding_v4_05`) GIF: "latest .gif show odd finger position,
thumb's outside of tip against ball, and middle finger use bottom part to message balls, balls
should be inside of all fingers, use mainly inside middle part of finger with help with inside of
top finger to manuevar balls to cross each other, bigger balls should be used so that all finger
are used in the baoding trick".

Classifying every `baoding_v4_05` contact by segment and face (`hand_env.classify_ball_contacts`)
confirmed all of it: `th_tip` touched with its BACK 96% of steps, `mf` almost only via
`mf_proximal`/`mf_base`, palmar-medial contact rare on every digit. V4's `touching_digits` counted
a digit as "engaged" no matter which face or segment touched, so its whole engagement mechanism was
satisfied by exactly the posture the user objected to.

Changes from `rl_env_baoding_v4.py`:

1. **Scene `scene_two_balls_xl.xml`** (0.03m radius balls resting in the palm) and starting grasp
   `BAODING_GRASP_TARGET_XL`, found by a static search scored on palmar medial/distal contact for
   all four digits (see its docstring in hand_env.py).
2. **"Engaged" now means a GOOD contact** (`hand_env.contact_quality`): the palmar face of a
   digit's medial/distal segment, or its fingertip pad/end. The per-digit idle timeout and the
   idle-approach cost from V4 are kept but tick on good contact only, so a thumb touching with its
   back, or a middle finger touching only with its root, now counts as idle.
3. **Per-step face/segment shaping**: a bonus per good digit and a further bonus per digit whose
   palmar MEDIAL is on a ball (the "mainly inside middle part"; both gated on `actively_moving`,
   the V4 static-contact-loophole lesson), a penalty per segment touching with its BACK, a smaller
   penalty per proximal/base segment touching (allowed -- a ball can rest against the finger
   roots in a bowl grip -- but not what should do the massaging).
4. **Retention is radial from the palm center, not distance from each ball's own rest spot.** V2-V4
   all required each ball to stay within 0.05m of where IT started; with 60mm balls a genuine swap
   moves each ball ~0.07m (to the other's starting spot), so that check would have terminated
   exactly the motion the task asks for. Each ball must instead stay within `MAX_RADIAL` of the
   episode's own starting center (midpoint of the two balls after the grasp closes) and between
   `MIN_Z`/`MAX_Z` (not fallen through, not popped out over the fingers).

Orbit reward (per-ball angle around that center, same-direction-matched, smaller magnitude
credited), milestones on the matched accumulator, idle penalty, grace-gated termination: unchanged
from V3/V4, none of it implicated in this failure.

**Never trained to a result.** The starting grasp went through two more rounds of user review
while this env was being built (`BAODING_GRASP_TARGET_XL_CAGE` -> the fanned `BAODING_GRASP_TARGET_XL`
-> `BAODING_GRASP_TARGET_XL_CUP`, see hand_env.py and README Stage 6); two training runs were
started on earlier grasps and stopped, and the project then moved to Stage 7 (toss and catch) at
the user's call. Kept as the documented stopping point, smoke-tested only.
"""
import gymnasium as gym
import mujoco as mj
import numpy as np
from gymnasium import spaces

from hand_env import (BALL_RADIUS_XL, BALL_SPAWN_XL, BAODING_GRASP_TARGET_XL, DIGIT_PREFIXES,
                       N_FINGER_JOINTS, TWO_BALL_SCENE_XL, contact_quality, hand_contact_count,
                       place_balls, ramp_to_target)

SCENE = TWO_BALL_SCENE_XL
CONTROL_DT = 0.02
EPISODE_SECONDS = 20.0
EPISODE_STEPS = int(EPISODE_SECONDS / CONTROL_DT)
GRASP_RAMP_SECONDS = 0.8
GRASP_HOLD_SECONDS = 2.0

MAX_JOINT_RATE = 0.05
MAX_ORBIT_RATE = 8.0
MIN_ACTIVE_JOINT_VEL = 0.1
GRACE_STEPS = 15

# Each ball's center vs the episode's starting center: at rest they sit ~0.031-0.035m out (one
# radius plus a little), so 0.055m means a ball has genuinely left the bowl. Margin cost starts
# at 0.045m, same "bite before the cliff" role as V4's MARGIN_PENALTY_THRESHOLD.
MAX_RADIAL = 0.055
MARGIN_RADIAL = 0.045
W_MARGIN = 20.0
MIN_Z = 0.0
MAX_Z = 0.09
MIN_SEPARATION = 2 * BALL_RADIUS_XL - 0.004
MAX_SEPARATION = 0.10

MILESTONE_RAD = np.pi / 4
MILESTONE_BONUS = 25.0
W_ORBIT = 1.0
W_ACTION = 0.01
DROP_PENALTY = 8.0
IDLE_PENALTY = 0.05

MAX_DIGIT_IDLE_SECONDS = 3.0
MAX_DIGIT_IDLE_STEPS = int(MAX_DIGIT_IDLE_SECONDS / CONTROL_DT)
DIGIT_IDLE_GRACE_FRAC = 0.3
W_DIGIT_IDLE = 0.5

W_GOOD = 0.02
# Extra on top of W_GOOD for palmar MEDIAL contact specifically -- user: "use mainly inside middle
# part of finger with help with inside of top finger".
W_MEDIAL = 0.02
W_DORSAL = 0.05
W_ROOT = 0.01


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class BaodingEnvV5(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, scene=SCENE, episode_steps=EPISODE_STEPS, grasp_target=None):
        super().__init__()
        self.model = mj.MjModel.from_xml_path(scene)
        self.data = mj.MjData(self.model)
        self.episode_steps = episode_steps
        self.grasp_target = BAODING_GRASP_TARGET_XL if grasp_target is None else grasp_target
        self.n_phys_steps = int(round(CONTROL_DT / self.model.opt.timestep))
        self.ctrl_lo = self.model.actuator_ctrlrange[:, 0].copy()
        self.ctrl_hi = self.model.actuator_ctrlrange[:, 1].copy()
        self.action_space = spaces.Box(-1.0, 1.0, shape=(N_FINGER_JOINTS,), dtype=np.float32)
        obs_dim = N_FINGER_JOINTS * 3 + 7 + 6 + 7 + 6
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
        self._digit_idle = {p: 0 for p in DIGIT_PREFIXES}

    def _obs(self):
        return np.concatenate([
            self.data.qpos[:N_FINGER_JOINTS],
            self.data.qvel[:N_FINGER_JOINTS],
            self.data.ctrl.copy(),
            self.data.qpos[N_FINGER_JOINTS:N_FINGER_JOINTS + 14],
            self.data.qvel[N_FINGER_JOINTS:N_FINGER_JOINTS + 12],
        ]).astype(np.float32)

    def _angle(self, name):
        p = self.data.body(name).xpos
        return np.arctan2(p[1] - self.center[1], p[0] - self.center[0])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mj.mj_resetData(self.model, self.data)
        place_balls(self.model, self.data, BALL_SPAWN_XL)
        ramp_to_target(self.model, self.data, self.grasp_target,
                       ramp_seconds=GRASP_RAMP_SECONDS, hold_seconds=GRASP_HOLD_SECONDS)
        self.center = 0.5 * (self.data.body("ball1").xpos + self.data.body("ball2").xpos)
        self._t = 0
        self._angle1_prev, self._angle2_prev = self._angle("ball1"), self._angle("ball2")
        self._bad_streak = 0
        self._cum1 = self._cum2 = self._matched_cum = 0.0
        self._next_milestone = MILESTONE_RAD
        self._digit_idle = {p: 0 for p in DIGIT_PREFIXES}
        return self._obs(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        prev_ctrl = self.data.ctrl.copy()
        new_ctrl = np.clip(prev_ctrl + action * MAX_JOINT_RATE, self.ctrl_lo, self.ctrl_hi)
        self.data.ctrl[:] = new_ctrl
        for _ in range(self.n_phys_steps):
            mj.mj_step(self.model, self.data)

        a1, a2 = self._angle("ball1"), self._angle("ball2")
        step1, step2 = _wrap(a1 - self._angle1_prev), _wrap(a2 - self._angle2_prev)
        self._angle1_prev, self._angle2_prev = a1, a2
        self._cum1 += step1
        self._cum2 += step2
        same_dir = (step1 * step2) > 0
        matched_step = (step1 if abs(step1) < abs(step2) else step2) if same_dir else 0.0
        self._matched_cum += matched_step
        orbit_rate = max(abs(step1), abs(step2)) / CONTROL_DT

        radial, z = {}, {}
        for b in ("ball1", "ball2"):
            p = self.data.body(b).xpos
            radial[b] = float(np.linalg.norm(p[:2] - self.center[:2]))
            z[b] = float(p[2])
        pos_ok = all(radial[b] < MAX_RADIAL and MIN_Z < z[b] < MAX_Z for b in radial)
        sep = float(np.linalg.norm(self.data.body("ball1").xpos - self.data.body("ball2").xpos))
        sep_ok = MIN_SEPARATION <= sep <= MAX_SEPARATION
        c1 = hand_contact_count(self.model, self.data, "ball1")
        c2 = hand_contact_count(self.model, self.data, "ball2")
        contacts_ok = c1 >= 1 and c2 >= 1
        not_flung = orbit_rate <= MAX_ORBIT_RATE

        good, medial, n_dorsal, n_root = contact_quality(self.model, self.data)
        for p in DIGIT_PREFIXES:
            self._digit_idle[p] = 0 if p in good else self._digit_idle[p] + 1
        digits_ok = max(self._digit_idle.values()) <= MAX_DIGIT_IDLE_STEPS

        if pos_ok and sep_ok and contacts_ok and not_flung and digits_ok:
            self._bad_streak = 0
        elif pos_ok:
            self._bad_streak += 1
        retained = pos_ok and self._bad_streak < GRACE_STEPS
        actively_moving = bool(np.linalg.norm(self.data.qvel[:N_FINGER_JOINTS]) >= MIN_ACTIVE_JOINT_VEL)

        orbit_reward = float(np.clip(matched_step, -MAX_ORBIT_RATE * CONTROL_DT,
                                     MAX_ORBIT_RATE * CONTROL_DT))
        margin_penalty = W_MARGIN * sum(max(0.0, radial[b] - MARGIN_RADIAL) for b in radial)
        digit_idle_penalty = W_DIGIT_IDLE * sum(
            max(0.0, idle / MAX_DIGIT_IDLE_STEPS - DIGIT_IDLE_GRACE_FRAC)
            for idle in self._digit_idle.values())
        reward = (W_ORBIT * orbit_reward * actively_moving
                  + (W_GOOD * len(good) + W_MEDIAL * len(medial)) * actively_moving
                  - W_DORSAL * n_dorsal - W_ROOT * n_root
                  - W_ACTION * float(np.sum((new_ctrl - prev_ctrl) ** 2))
                  - margin_penalty - digit_idle_penalty
                  - (0.0 if actively_moving else IDLE_PENALTY))

        milestone_hit = False
        while abs(self._matched_cum) >= self._next_milestone:
            reward += MILESTONE_BONUS
            self._next_milestone += MILESTONE_RAD
            milestone_hit = True

        self._t += 1
        terminated = not retained
        if terminated:
            reward -= DROP_PENALTY
        truncated = self._t >= self.episode_steps

        info = {"retained": retained, "radial1": radial["ball1"], "radial2": radial["ball2"],
                "z1": z["ball1"], "z2": z["ball2"], "separation": sep,
                "n_contacts_1": c1, "n_contacts_2": c2, "orbit_rate": orbit_rate,
                "cum1_rad": self._cum1, "cum2_rad": self._cum2,
                "matched_cum_rad": self._matched_cum, "bad_streak": self._bad_streak,
                "milestone_hit": milestone_hit, "actively_moving": actively_moving,
                "good_digits": sorted(good), "medial_digits": sorted(medial),
                "n_dorsal": n_dorsal, "n_root": n_root,
                "digit_idle_max": max(self._digit_idle.values())}
        if terminated or truncated:
            info["success"] = retained
        return self._obs(), reward, terminated, truncated, info
