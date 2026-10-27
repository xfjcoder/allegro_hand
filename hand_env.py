"""Shared constants and helpers for the Allegro Hand grasp/manipulation stages, reused by every
eval_stageN.py / demo_stageN.py script -- same role as panda_vision_grasp's grasp_env.py.

**Why "grasp stability under perturbation", not "lift", is this project's success metric**:
Stage 1 found the vendored scene's free-floating object already falls into a stable resting
equilibrium with the hand fully OPEN (ctrl=0) -- the default-open finger pose's own slight
gravity sag forms a shallow passive basin. That makes "the object doesn't fall" a trivial,
already-solved null-policy outcome, and this fixed-base hand has no wrist/arm to literally
raise the object's height by translating -- so neither "stays put" nor "gets higher" is a
meaningful task goal here. What IS meaningful and non-trivial: whether a grasp can resist an
external perturbation that the passive pose can't. Confirmed directly (see README Stage 2):
the passive open-hand pose survives small pushes but is overwhelmed by push magnitude ~0.8-1.0
m/s (success rate 8/8 -> 4/8 -> 0/8 as magnitude rises from 0.6 to 1.0), while an actively
curled power-grasp pose retains the object at every tested magnitude up to 2.0 m/s.
"""
import mujoco as mj
import numpy as np

JOINT_NAMES = [f"{f}j{j}" for f in ("ff", "mf", "rf", "th") for j in range(4)]
N_FINGER_JOINTS = 16
# jnt_range for the "curl" joints (index 1/2/3 within each finger's own 4-joint group) --
# verified identical across ff/mf/rf's shared `proximal`/`medial`/`distal` default classes, and
# separately for the thumb's own `thumb_proximal`/`thumb_medial`/`thumb_distal` classes (see
# assets/right_hand.xml's <default> block).
FINGER_CURL_RANGES = {1: (-0.196, 1.61), 2: (-0.174, 1.709), 3: (-0.227, 1.618)}
THUMB_CURL_RANGES = {1: (-0.105, 1.163), 2: (-0.189, 1.644), 3: (-0.162, 1.719)}
THUMB_BASE_RANGE = (0.263, 1.396)
OBJECT_REST_POS = np.array([0.0, 0.0, 0.03])  # Stage 1's measured passive-settle position


def make_grasp_target(curl_frac, thumb_base, base_spread=0.0):
    """Builds a 16-dim ctrl target: ff/mf/rf's 3 curl joints (proximal/medial/distal) set to
    curl_frac of their own range (0=open, 1=fully curled), base joints (ab/adduction) at
    +-base_spread (confirmed empirically -- see README Stage 2 -- that base_spread=0, i.e. no
    spread, gives the TIGHTEST cage around the object; spreading in either direction only moves
    fingertips farther from center for this object/pose), and the thumb's own base joint set to
    an absolute target (its range doesn't have a natural "0 = open" point like the fingers'
    base joints do) plus the same curl_frac on its 3 curl joints."""
    target = np.zeros(N_FINGER_JOINTS)
    base_vals = [-base_spread, 0.0, base_spread]  # ff, mf, rf
    for f in range(3):
        target[f * 4] = base_vals[f]
        for j in (1, 2, 3):
            lo, hi = FINGER_CURL_RANGES[j]
            target[f * 4 + j] = lo + curl_frac * (hi - lo)
    target[12] = thumb_base
    for j in (1, 2, 3):
        lo, hi = THUMB_CURL_RANGES[j]
        target[12 + j] = lo + curl_frac * (hi - lo)
    return target


# Found by a direct static (massless, forward-kinematics-only) search over curl_frac/thumb_base/
# base_spread for the configuration that brings all 4 fingertips closest to the object's center
# (see README Stage 2) -- this is the "power grasp" target used by every later stage's scripted
# baseline and by the RL action space's own anchor/reference pose.
POWER_GRASP_TARGET = make_grasp_target(curl_frac=0.95, thumb_base=1.3, base_spread=0.0)


def settle_object(model, data, settle_seconds=1.0):
    """Holds the hand open (ctrl=0) for settle_seconds so the included free-floating object
    falls from its spawn position ([0,0,0.1], per scene_right.xml) into its passive resting
    equilibrium (measured ~[0,0,0.031], see Stage 1) before any grasp attempt begins."""
    dt = model.opt.timestep
    for _ in range(int(settle_seconds / dt)):
        data.ctrl[:] = 0.0
        mj.mj_step(model, data)


def ramp_to_target(model, data, target, ramp_seconds=1.0, hold_seconds=0.5):
    """Linearly ramps ctrl from its current value to `target` over ramp_seconds (a sudden jump
    to a fully-curled target can make fingers slap through the object rather than close around
    it), then holds at `target` for hold_seconds to let the grasp stabilize."""
    dt = model.opt.timestep
    start = data.ctrl.copy()
    n_ramp = int(ramp_seconds / dt)
    for i in range(n_ramp):
        alpha = (i + 1) / n_ramp
        data.ctrl[:] = start + alpha * (target - start)
        mj.mj_step(model, data)
    for _ in range(int(hold_seconds / dt)):
        data.ctrl[:] = target
        mj.mj_step(model, data)


def apply_push(data, push_vel):
    """Adds push_vel (a 3-vector, m/s) directly to the object's free-joint linear velocity --
    the object's qvel occupies indices [16:19] (3 linear + 3 angular, right after the 16 finger
    joints' own 1 DOF each; confirmed via jnt_dofadr, NOT the qpos layout's own indices, which
    this project's Stage 2 exploration initially mixed up -- see git history)."""
    data.qvel[16:19] += push_vel


def object_retained(data, threshold=0.05):
    """True if the object is still within `threshold` meters of its passive-rest position --
    this project's grasp-stability success criterion (see this module's docstring)."""
    dist = np.linalg.norm(data.body("object").xpos - OBJECT_REST_POS)
    return dist < threshold, dist


def quat_angle_deg(q1, q2):
    """Shortest rotation angle (degrees) between two world-frame quaternions (mjtNum wxyz)."""
    dot = np.clip(abs(np.dot(q1, q2)), 0.0, 1.0)
    return np.degrees(2 * np.arccos(dot))


def twist_ctrl(base_ctrl, amp, phase):
    """Stage 3's scripted reorientation primitive: the power grasp's own ctrl target, with the
    first finger's base (ab/adduction) joint (`ffj0`, ctrl index 0) and the ring finger's base
    joint (`rfj0`, index 8) driven sinusoidally in OPPOSITE directions -- found empirically (see
    README Stage 3) to induce a real but BOUNDED object rotation. A first few oscillation cycles
    can look like it's still climbing (don't stop early and call it "saturated" -- confirmed
    directly that a too-short sweep at amp=0.4 looked like unbounded linear growth through 20
    cycles before actually flattening by 50), but it reliably converges to a hard ceiling around
    ~23-25 degrees for amp>=0.3, independent of amplitude (even the joint's own max range, 0.47,
    converges to the same ~23.5 degree ceiling) -- a static-friction-cone limit of simultaneous-
    contact motion, not unbounded reorientation. Arbitrary/large-angle reorientation needs
    finger-gaiting (releasing and re-contacting at new positions), Stage 4's topic."""
    c = base_ctrl.copy()
    c[0] = amp * np.sin(phase)
    c[8] = -amp * np.sin(phase)
    return c
