"""Stage 6, fourth RL design iteration. User, watching the BaodingEnvV3 (`baoding_v3_05`) demo
GIF: "baoding_rl_v3.gif file show only two finger are working on two balls, idea is that 4 fingers
make bowl to claw shape with two balls in it, then four fingers massage two balls cross each other
and go around and around, can we design a policy network to do that?"

Diagnosed with a per-fingertip contact trace of `baoding_v3_05` (not just `hand_contact_count`'s
raw body COUNT, which can't see WHICH bodies): `th_tip` touched NEITHER ball for the entire 60s
rollout, and `mf` only ever showed up as a passive `mf_proximal` backstop (96.5% of steps) rather
than an active digit. The trained policy had quietly regressed to the ORIGINAL ff/rf-only 2-finger
diagonal stroke this whole stage's "four-finger orbit" work was meant to move past -- nothing in
BaodingEnvV3's reward or termination required otherwise: palm + 2 active digits already clears
`MIN_HAND_CONTACTS_TOTAL=3` with zero pressure to engage the other two.

Two root causes found, both fixed here:

1. **The static starting grasp never actually reached thumb contact.** `BaodingEnvV3.reset()`
   called `ramp_to_target(..., hold_seconds=0.3)`; a static sweep (holding the exact same
   `BAODING_GRASP_TARGET_LARGE_CROSSOVER` ctrl target longer, nothing else changed) showed the
   thumb is still mid-swing at 0.3s and only arrives at genuine `th_tip`-on-ball2 contact by
   ~0.5s, settling there through at least hold=3.0s (not fighting physics, just needed more time
   to get there). **Fixed: `hold_seconds` raised to 1.0** -- the episode now actually STARTS from
   a real 3-digit cage (ff+palm on ball1, rf+th+palm on ball2; mf stays loosened/uncaged per
   `BAODING_GRASP_TARGET_LARGE_CROSSOVER`'s own fix, free to be the dynamic "pass-through" digit),
   matching the user's "4 fingers make a bowl/claw shape with two balls in it" starting picture
   instead of handing the policy an undercooked pose it then had no reason to improve on.

2. **Nothing penalized a digit going completely idle.** Added `hand_env.touching_digits` (digit-
   IDENTITY contact, not just a raw count) and a per-digit idle-duration tracker: each digit's
   `steps-since-last-touched-either-ball` resets to 0 whenever it's in contact, else increments.
   If any digit's idle streak exceeds `MAX_DIGIT_IDLE_STEPS` (3 seconds), the episode terminates
   the same grace-gated way as the other retention checks -- directly ruling out the exact
   `baoding_v3_05` failure (one digit sitting out 100% of a 60s rollout). A small continuous
   `DIGIT_ENGAGEMENT_BONUS` per currently-active digit beyond 2 gives PPO a smooth gradient toward
   using 3-4 digits, not just a hard cliff only encountered at the 3-second timeout.

Reward/milestone structure (per-ball angle, same-direction-matched orbit credit, milestones keyed
to the matched accumulator) is otherwise unchanged from `rl_env_baoding_v3.py` -- that part of the
design was never implicated in this failure; see that file's own docstring for its history.

**baoding_v4_01**: both fixes above confirmed working -- a per-fingertip trace showed all 4
digits genuinely active (ff=100%, mf=99%, rf=69%, th=97% of steps, vs v3_05's th=0%/mf=passive).
Circulation is real AND much faster than anything in V3 (ball1=76deg, ball2=194deg within 11s,
~12deg/s combined vs v3_05's eventual ~1.3deg/s) -- genuinely closer to "massage", not a slow
stroke. But it fails at t=11s: a verbose per-step trace showed `dist1`/`dist2` (each ball's own
distance from rest) drifting outward essentially from step 0, even while all 4 digits stayed
engaged (0.006m at t=0 -> 0.025m by t=7.4s, a steady climb DURING active, correctly-matched
circulation, not a freeze at a bad idle pose like every V3 failure). `MARGIN_PENALTY_THRESHOLD`
(0.04, tuned for V3's much slower circulation, which peaked at a measured 0.041m) never applies
until the ball is already most of the way to the 0.05m hard cutoff, so it gave no gradient against
this early, continuous drift -- the faster churn this round discovered appears to have a real
secular outward cost the V3-era threshold was never calibrated to resist. Lowered to 0.025 for
baoding_v4_02, changing nothing else, to test whether earlier pressure on this specific drift (not
a new mechanism, just an earlier-biting version of the one that already worked for V3) is enough
without suppressing genuine circulation.

**baoding_v4_02**: the tighter threshold worked cleanly -- dist1/dist2 stayed under 0.013m for the
entire 26s it survived (vs v4_01's climb to 0.05m), position drift fully resolved. But it then hit
the OTHER new V4 mechanism: it settled into a quiet, perfectly stable 3-digit hold and let the
thumb idle climb for the whole second half of the episode, terminating at the idle-timeout exactly
as designed -- confirms that mechanism works, but also that `DIGIT_ENGAGEMENT_BONUS` (0.02/step
difference between 3 and 4 active digits) was too weak against the now-larger margin penalty to
stop PPO drifting toward "mostly comply." Added `DIGIT_IDLE_GRACE_FRAC`/`W_DIGIT_IDLE`: a
continuous per-step cost once any digit's own idle streak passes 30% of its budget, ramping to
0.35/step (more than double a max-rate orbit step) right before the timeout -- same fix pattern as
`IDLE_PENALTY` already was for the whole-hand freeze, now applied per-digit.

**baoding_v4_03**: positions AND all 4 digits' contact held for the full 90s eval (first fully
stable run in this design) -- but circulation collapsed: ball1=+32deg, ball2=-19.5deg, OPPOSITE
signs, and a per-step trace showed `actively_moving` goes False at t~10s and essentially freezes
there (cum1/cum2 barely creep for the remaining 80s). Root cause: `touching_digits` is a pure
CONTACT check, not a motion check -- sitting perfectly still with all 4 digits resting on the
balls satisfies `engagement_bonus` exactly as well as actively massaging with all 4, while
avoiding every risk (margin penalty, digit-idle cost, action penalty) that attempting real
circulation carries. The V4 digit-engagement fix had accidentally created a new, MORE attractive
passive equilibrium than the one V3's IDLE_PENALTY was built to discourage. Fixed by gating
`engagement_bonus` on `actively_moving`, the same way `orbit_reward` already is -- static,
motionless contact no longer earns it, closing the exact loophole.

**baoding_v4_04**: closing the loophole worked -- first run to survive the full 90s eval WITHOUT
freezing (no `actively_moving=False` plateau), all 4 digits still 100% engaged throughout. But
`matched_cum` only reached ~12-14deg early on and then plateaued for the rest of the episode:
ball2 kept moving a lot (+72deg net) while ball1 just oscillated in a small arc, occasionally
matching ball2's direction briefly then reversing, netting near-zero further matched progress.
Stability and digit-engagement were both solved this round; what's left is whether more training
from this already-stable base finds a genuinely circulating gait, rather than changing the reward
yet again on an unproven hypothesis. Resumed training (not a fresh run) for another 4M steps.

**baoding_v4_05** (resumed from v4_04's checkpoint): the best result in this stage to date. Ran a
150s eval (not just 90s) and it never terminates: all 4 digits stay engaged the whole time,
`dist2`/`separation` rise to hover right at ~0.045-0.050m (the 0.05m hard cutoff) without ever
crossing it, and `matched_cum` -- genuine same-direction progress on BOTH balls, the strictest
metric this whole design exists to protect -- climbs to 77deg by t=75s and then holds in a stable
~70deg band through t=150s (ball1 oscillates gently 2-15deg, ball2 sustains 50-65deg, matching
direction consistently rather than the earlier opposite-sign wobbles). This is the first V4 run
where additional training, not a new mechanism, was the right next step -- the previous 4 rounds
each fixed a distinct structural gap (thumb contact, position drift, idle-digit abandonment, the
static-contact loophole); once all four were closed, PPO had a clean optimization landscape left
to actually improve circulation in, and did.
"""
import gymnasium as gym
import mujoco as mj
import numpy as np
from gymnasium import spaces

from hand_env import (BAODING_GRASP_TARGET_LARGE_CROSSOVER, BALL_REST_POS_LARGE, DIGIT_PREFIXES,
                       MAX_BALL_SEPARATION_LARGE, MIN_BALL_SEPARATION_LARGE, N_FINGER_JOINTS,
                       TWO_BALL_SCENE_LARGE, ball_angle_rad, ball_separation_ok, balls_retained,
                       hand_contact_count, ramp_to_target, settle_object, touching_digits)

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
# baoding_v4_01 (see module docstring): the faster, genuinely-4-digit circulation this round
# unlocked drifts dist1/dist2 outward from step 0, well before 0.04m -- lowered so the same
# existing mechanism bites earlier, instead of only in the last ~1s before the 0.05m hard cutoff.
MARGIN_PENALTY_THRESHOLD = 0.025
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

# New in V4 -- see module docstring point 2. 3 seconds was picked to clearly separate "a digit
# briefly released mid-cycle while the orbit passes it by" (the scripted four_finger_orbit_ctrl's
# own quadrature cycle is ~1.5-2s per digit) from "a digit has been structurally abandoned," which
# is what `baoding_v3_05` actually did to the thumb for an entire 60s rollout.
MAX_DIGIT_IDLE_SECONDS = 3.0
MAX_DIGIT_IDLE_STEPS = int(MAX_DIGIT_IDLE_SECONDS / CONTROL_DT)
# Small on purpose relative to W_ORBIT's per-step scale (orbit_reward maxes out at
# MAX_ORBIT_RATE*CONTROL_DT = 0.16) -- meant as a smooth gradient toward using more digits, not a
# replacement for the hard idle-timeout termination above, which is what actually rules the
# 2-finger regression out.
DIGIT_ENGAGEMENT_BONUS = 0.02

# Added after baoding_v4_02 (MARGIN_PENALTY_THRESHOLD tightened to 0.025) fixed the position-drift
# failure (dist1/dist2 stayed <0.013m for the full 26s it survived) but hit the OTHER new V4
# mechanism instead: it settled into a quiet, perfectly stable 3-digit hold and let the thumb idle
# climb steadily for the whole second half of the episode, terminating at the idle-timeout exactly
# as designed. Root cause: DIGIT_ENGAGEMENT_BONUS only distinguishes 3-vs-4 active digits by 0.02
# reward/step, dwarfed by W_MARGIN's penalty scale -- nothing made a digit APPROACHING its 3s
# budget cost more than one that just released. Continuous: once a digit's own idle streak passes
# DIGIT_IDLE_GRACE_FRAC of its budget (0.9s -- past the ~1.5-2s mid-cycle release the scripted
# quadrature motion needs, so this doesn't fight genuine transitional releases), a per-step cost
# ramps linearly, reaching 0.35/step (double a max-rate orbit_reward) right before the digit would
# actually time out -- the same "stopping/idling must cost more the longer it continues" fix
# `IDLE_PENALTY` already was for the whole-hand freeze, now applied per-digit.
DIGIT_IDLE_GRACE_FRAC = 0.3
W_DIGIT_IDLE = 0.5


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class BaodingEnvV4(gym.Env):
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
        self._digit_idle = {p: 0 for p in DIGIT_PREFIXES}

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
                        ramp_seconds=1.0, hold_seconds=1.0)
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
        self._digit_idle = {p: 0 for p in DIGIT_PREFIXES}
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

        active_digits = (touching_digits(self.model, self.data, object_body_name="ball1") |
                          touching_digits(self.model, self.data, object_body_name="ball2"))
        for p in DIGIT_PREFIXES:
            self._digit_idle[p] = 0 if p in active_digits else self._digit_idle[p] + 1
        digits_ok = max(self._digit_idle.values()) <= MAX_DIGIT_IDLE_STEPS

        transient_ok = sep_ok and drift_ok and contacts_ok and not_flung and digits_ok
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
        engagement_bonus = (DIGIT_ENGAGEMENT_BONUS * max(0, len(active_digits) - 2) *
                             actively_moving)
        digit_idle_penalty = W_DIGIT_IDLE * sum(
            max(0.0, idle / MAX_DIGIT_IDLE_STEPS - DIGIT_IDLE_GRACE_FRAC)
            for idle in self._digit_idle.values())
        reward = (W_ORBIT * float(orbit_reward) * actively_moving - action_penalty -
                  margin_penalty - idle_penalty + engagement_bonus - digit_idle_penalty)

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
                "milestone_hit": milestone_hit, "actively_moving": bool(actively_moving),
                "active_digits": sorted(active_digits), "digit_idle_max": max(self._digit_idle.values())}
        if terminated or truncated:
            info["success"] = retained
        return self._obs(), reward, terminated, truncated, info
