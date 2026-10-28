"""Stage 6, third RL design iteration. User: "can we put rl on four balls orbiting scheme" ->
clarified to mean RL on the existing 2-ball / 4-finger "crossover" scheme just found by scripted
search (see README Stage 6 and `hand_env.BAODING_GRASP_TARGET_LARGE_CROSSOVER`'s docstring), not
literally four balls. Two changes from `rl_env_baoding_v2.py` (BaodingEnvV2), both motivated
directly by what that scripted search found:

1. **Starting grasp is `BAODING_GRASP_TARGET_LARGE_CROSSOVER`**, not `BAODING_GRASP_TARGET_LARGE`.
   The scripted search found ball2 was permanently caged by mf's own resting curl (0.85) regardless
   of dynamic modulation; loosening it to 0.4 was the actual unlock. Starting the policy from the
   already-freed pose means it doesn't have to rediscover that from scratch.

2. **Reward and milestones are keyed to the per-ball angle (`hand_env.ball_angle_rad`), not the
   pair-vector angle (`ball_pair_angle_rad`) BaodingEnvV2 used.** That same scripted search found
   the pair-vector metric can drift substantially from an ASYMMETRIC wobble -- one ball doing all
   the moving while the other sits still -- without the reward signal ever reflecting it. This is
   the RL analog of the same failure mode, and the same bug could let a PPO policy "cheat" the
   exact same way (spin ball1 in place, leave ball2 pinned, collect reward from the pair-vector
   drift). Fixed structurally, not just by observation: each step's orbit reward is the SMALLER-
   magnitude of the two balls' own signed angular steps, and ONLY if they share the same sign (true
   circulation requires both balls' own angle advancing in the same rotational sense around the
   palm center, confirmed by the scripted result: ball1_cum=-64deg, ball2_cum=-172deg, same sign).
   If the two balls move in opposite directions, or wildly different amounts, the reward credits
   whichever ball is lagging, not whichever is easiest to spin -- a policy can only accumulate
   reward by making genuine progress on BOTH balls together.

Training history within this design (kept here, not just in README, because each fix targets a
specific failure the previous run actually exhibited):

- **baoding_v3_01**: milestones were keyed to min(|cum1|, |cum2|) -- raw per-ball cumulative angle,
  NOT the same-direction-matched amount the per-step reward credits. Real same-direction
  circulation (ball1=-127deg, ball2=-97deg) but stopped moving at t~16s and drifted to failure at
  18-27s (reward is neutral once idle, regardless of which pose it freezes at -- the base grasp's
  OWN frozen pose was separately confirmed rock-stable for 28s+, so this was policy-specific, not
  an environment flaw).
- **baoding_v3_02** (added `MARGIN_PENALTY_THRESHOLD`/`W_MARGIN`, a small penalty for either ball's
  distance exceeding 0.04m): survived longer (49.7s) but the milestone bug let it bank bonus via
  OPPOSITE-direction motion (ball1=-36deg, ball2=+65deg) that earned zero per-step reward -- not
  real circulation, a reward hack enabled by the min(|cum1|,|cum2|) milestone gating.
- **baoding_v3_03** (added `IDLE_PENALTY`, a flat per-step cost while not actively moving): same
  opposite-direction hack, failed even sooner (15.2s) -- confirms the milestone gating was the
  actual bug, not an insufficient idle disincentive.
- **Fix**: milestones are now keyed to `self._matched_cum`, the running sum of `matched_step`
  (the SAME quantity the per-step reward uses) -- a policy can only earn milestone bonus through
  genuine same-direction progress, closing the hack.

Everything else (observation space, action space, grace-period contact/rate termination, drop
penalty, action-change penalty) is unchanged from BaodingEnvV2 -- that part of the design wasn't
implicated in any of these findings.
"""
import gymnasium as gym
import mujoco as mj
import numpy as np
from gymnasium import spaces

from hand_env import (BAODING_GRASP_TARGET_LARGE_CROSSOVER, BALL_REST_POS_LARGE,
                       MAX_BALL_SEPARATION_LARGE, MIN_BALL_SEPARATION_LARGE, N_FINGER_JOINTS,
                       TWO_BALL_SCENE_LARGE, ball_angle_rad, ball_separation_ok, balls_retained,
                       hand_contact_count, ramp_to_target, settle_object)

SCENE = TWO_BALL_SCENE_LARGE
CONTROL_DT = 0.02
# baoding_v3_04 (with the milestone fix above) achieved real same-direction circulation
# (ball1=82deg, ball2=129deg) but stopped moving at t~16s -- PAST this env's own 10s/500-step
# training horizon, i.e. genuinely out-of-distribution behavior the policy never had a reason to
# handle. Doubled to 20s so a trained policy actually experiences "what happens if I stop here"
# within its own training episodes, instead of only ever being evaluated on it.
EPISODE_SECONDS = 20.0
EPISODE_STEPS = int(EPISODE_SECONDS / CONTROL_DT)

MAX_JOINT_RATE = 0.05
MIN_HAND_CONTACTS_PER_BALL = 1
MIN_HAND_CONTACTS_TOTAL = 3
MAX_ORBIT_RATE = 8.0
MIN_ACTIVE_JOINT_VEL = 0.1
MAX_SEPARATION_DRIFT = 0.015
GRACE_STEPS = 15

MILESTONE_RAD = np.pi / 4  # 45 degrees, same as V2
MILESTONE_BONUS = 25.0

W_ORBIT = 1.0
W_ACTION = 0.01
DROP_PENALTY = 8.0

# Added after a first training run (baoding_v3_01) reproduced the "one safe partial attempt, then
# go static" failure noted when V2's milestone bonus was designed -- but in a new form: once
# per-step joint velocity drops below MIN_ACTIVE_JOINT_VEL, `actively_moving` zeroes the orbit
# reward, so the policy has no pressure over WHICH idle pose it stops at. Measured directly that
# the scripted crossover grasp's own frozen pose is rock-stable (dist1 <0.006m) for 28+ seconds,
# so the environment isn't inherently drifty -- the policy just happened to stop somewhere that
# wasn't. `MARGIN_PENALTY_THRESHOLD` (0.04m, just below `balls_retained`'s hard 0.05m cutoff) with
# a small per-step penalty on the excess discourages drifting toward the cliff without fighting
# genuine circulation, which was measured (same scripted grasp) to peak at dist2=0.041m mid-orbit.
MARGIN_PENALTY_THRESHOLD = 0.04
W_MARGIN = 20.0

# Added after baoding_v3_02 (the margin-penalty run) still failed, this time from `separation`
# itself (ball1<->ball2 distance, not either ball's own position) slowly creeping until
# `separation_drift` crossed MAX_SEPARATION_DRIFT at t=49.7s -- a SECOND, different slow-drift
# failure from a SECOND different frozen idle pose. Root cause common to both: once milestones are
# banked within the 10s training episode, `actively_moving` gates reward to ~0 either way, so
# stopping is reward-NEUTRAL regardless of which exact pose it freezes at -- nothing in training
# ever selected for "the pose I stop at should be a genuine equilibrium," only for "stopping here
# doesn't cost anything within a 10s horizon." A flat per-step tax while idle removes that
# neutrality: continuing to make real matched progress must outearn the tax, not just equal it.
IDLE_PENALTY = 0.05


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class BaodingEnvV3(gym.Env):
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
        self._angle1_prev = None
        self._angle2_prev = None
        self._bad_streak = 0
        self._cum1 = 0.0
        self._cum2 = 0.0
        self._matched_cum = 0.0
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
        ramp_to_target(self.model, self.data, BAODING_GRASP_TARGET_LARGE_CROSSOVER,
                        ramp_seconds=1.0, hold_seconds=0.3)
        self._t = 0
        self._angle1_prev = ball_angle_rad(self.data, "ball1")
        self._angle2_prev = ball_angle_rad(self.data, "ball2")
        _, self._sep0 = ball_separation_ok(self.data, MIN_BALL_SEPARATION_LARGE,
                                            MAX_BALL_SEPARATION_LARGE)
        self._bad_streak = 0
        self._cum1 = 0.0
        self._cum2 = 0.0
        self._matched_cum = 0.0
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

        angle1_cur = ball_angle_rad(self.data, "ball1")
        angle2_cur = ball_angle_rad(self.data, "ball2")
        step1 = _wrap(angle1_cur - self._angle1_prev)
        step2 = _wrap(angle2_cur - self._angle2_prev)
        self._angle1_prev, self._angle2_prev = angle1_cur, angle2_cur
        self._cum1 += step1
        self._cum2 += step2

        # Structural fix for the pair-vector cheat found in the scripted search: only credit
        # reward when BOTH balls move the same rotational direction, and credit the SMALLER
        # magnitude of the two -- a policy can't accumulate reward by spinning one ball while the
        # other sits pinned.
        same_dir = (step1 * step2) > 0
        matched_step = (step1 if abs(step1) < abs(step2) else step2) if same_dir else 0.0
        self._matched_cum += matched_step
        orbit_rate = max(abs(step1), abs(step2)) / CONTROL_DT

        pos_retained, dists = balls_retained(self.data, rest_pos=BALL_REST_POS_LARGE)
        sep_ok, sep = ball_separation_ok(self.data, MIN_BALL_SEPARATION_LARGE,
                                          MAX_BALL_SEPARATION_LARGE)
        drift_ok = abs(sep - self._sep0) <= MAX_SEPARATION_DRIFT
        c1 = hand_contact_count(self.model, self.data, "ball1")
        c2 = hand_contact_count(self.model, self.data, "ball2")
        contacts_ok = (c1 >= MIN_HAND_CONTACTS_PER_BALL and c2 >= MIN_HAND_CONTACTS_PER_BALL and
                       (c1 + c2) >= MIN_HAND_CONTACTS_TOTAL)
        not_flung = orbit_rate <= MAX_ORBIT_RATE

        transient_ok = sep_ok and drift_ok and contacts_ok and not_flung
        if pos_retained and transient_ok:
            self._bad_streak = 0
        elif pos_retained:
            self._bad_streak += 1
        retained = pos_retained and self._bad_streak < GRACE_STEPS
        actively_moving = np.linalg.norm(self.data.qvel[:N_FINGER_JOINTS]) >= MIN_ACTIVE_JOINT_VEL

        action_penalty = W_ACTION * float(np.sum((new_ctrl - prev_ctrl) ** 2))
        orbit_reward = np.clip(matched_step, -MAX_ORBIT_RATE * CONTROL_DT,
                                MAX_ORBIT_RATE * CONTROL_DT)
        margin_penalty = W_MARGIN * (max(0.0, dists["ball1"] - MARGIN_PENALTY_THRESHOLD) +
                                      max(0.0, dists["ball2"] - MARGIN_PENALTY_THRESHOLD))
        idle_penalty = 0.0 if actively_moving else IDLE_PENALTY
        reward = (W_ORBIT * float(orbit_reward) * actively_moving - action_penalty -
                  margin_penalty - idle_penalty)

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

        info = {"retained": retained, "dist1": dists["ball1"], "dist2": dists["ball2"],
                "separation": sep, "separation_drift": sep - self._sep0,
                "n_contacts_1": c1, "n_contacts_2": c2,
                "orbit_step1_rad": step1, "orbit_step2_rad": step2, "orbit_rate": orbit_rate,
                "cum1_rad": self._cum1, "cum2_rad": self._cum2,
                "matched_cum_rad": self._matched_cum, "bad_streak": self._bad_streak,
                "milestone_hit": milestone_hit, "actively_moving": bool(actively_moving)}
        if terminated or truncated:
            info["success"] = retained
        return self._obs(), reward, terminated, truncated, info
