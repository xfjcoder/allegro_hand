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

# Stage 6: assets/scene_two_balls.xml -- two 0.015m-radius spheres (half the single-object
# scene's ellipsoid scale) spawned offset +-0.018m in y so they land side by side without
# interpenetrating, instead of the single centered ellipsoid the scenes above use.
TWO_BALL_SCENE = "assets/scene_two_balls.xml"
# Measured directly (same discipline as OBJECT_REST_POS): open-hand passive settle position for
# each ball, used as this pair's own retention reference.
BALL_REST_POS = {
    "ball1": np.array([0.0, -0.0185, 0.0257]),
    "ball2": np.array([0.0, 0.0185, 0.0257]),
}


def balls_retained(data, threshold=0.05, rest_pos=None):
    """Same role as object_retained, but for a two-ball scene -- both balls must stay within
    `threshold` of their OWN measured rest position. `rest_pos` defaults to the original
    scene_two_balls.xml's `BALL_REST_POS`; pass `BALL_REST_POS_LARGE` for
    scene_two_balls_large.xml. Returns (both_retained, per-ball distances)."""
    rest_pos = BALL_REST_POS if rest_pos is None else rest_pos
    dists = {name: np.linalg.norm(data.body(name).xpos - rest)
             for name, rest in rest_pos.items()}
    return all(d < threshold for d in dists.values()), dists


def ball_pair_angle_rad(data):
    """Angle (world XY plane) of the vector from ball1 to ball2 -- this project's orbit metric:
    sustained CHANGE in this angle over time means the pair is actually circulating around each
    other (real Baoding-ball motion), not just jittering in place. Unwrap/accumulate step-to-step
    deltas the same way Stage 5 accumulates `twist_angle_rad`, for the same reason (a raw angle
    alone can't distinguish one full orbit from zero)."""
    v = data.body("ball2").xpos - data.body("ball1").xpos
    return np.arctan2(v[1], v[0])


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

# Stage 6: POWER_GRASP_TARGET was tuned for ONE centered object and holds the two-ball pair
# badly -- checked directly: ball1 gets ZERO fingertip contact under it (only touches the palm),
# ball2 gets 1. Found this instead via the same static curl_frac/spread search, scored on total
# hand-ball contact count for BOTH balls (not just one): same curl_frac=0.95, but base_spread=0.5
# (vs. 0.0 -- a WIDER spread is needed to reach two side-by-side balls instead of one centered
# one) gives ball1=1, ball2=2 fingertip contacts with both well retained (<0.01m from rest).
BAODING_GRASP_TARGET = make_grasp_target(curl_frac=0.95, thumb_base=1.3, base_spread=0.5)
MIN_BALL_SEPARATION = 0.025  # meters -- below the balls' own 0.03m diameter would mean overlap
MAX_BALL_SEPARATION = 0.065  # meters -- natural range under grasp+scripted motion was 0.037-0.050


def ball_separation_ok(data, min_sep=None, max_sep=None):
    """`min_sep`/`max_sep` default to the original scene_two_balls.xml's bounds; pass
    `MIN/MAX_BALL_SEPARATION_LARGE` for scene_two_balls_large.xml."""
    min_sep = MIN_BALL_SEPARATION if min_sep is None else min_sep
    max_sep = MAX_BALL_SEPARATION if max_sep is None else max_sep
    dist = np.linalg.norm(data.body("ball1").xpos - data.body("ball2").xpos)
    return min_sep <= dist <= max_sep, dist


# Stage 6 continued: user -- "maybe relative larger balls, so finger movement can make ball push
# each other and pass by". scene_two_balls_large.xml, 0.02m radius balls (up from 0.015m).
# BAODING_GRASP_TARGET's own curl/spread (0.95/0.5) was tuned for the SMALLER balls' geometry;
# re-searched for this size and found something simpler works: a gentler curl (0.85) with NO
# spread needed (0.0) already brings the bigger balls into genuine contact (confirmed via
# data.contact -- ball1<->ball2 registers) and holds rock-stable for 10+ seconds (position drift
# <0.001m), unlike the smaller balls which needed the special BAODING_CONTACT_TARGET just to
# touch at all.
TWO_BALL_SCENE_LARGE = "assets/scene_two_balls_large.xml"
BAODING_GRASP_TARGET_LARGE = make_grasp_target(curl_frac=0.85, thumb_base=1.3, base_spread=0.0)
# Measured directly, same discipline as BALL_REST_POS: open-hand passive settle position (at
# settle_object's standard 1.0s duration -- the open-hand pose doesn't reach a true equilibrium
# at this ball size, it keeps creeping slowly off the palm indefinitely, but every real use
# immediately follows settle_object with a grasp, which DOES hold rock-stable, so this snapshot
# is the right "rest" reference for retention purposes, same role BALL_REST_POS already plays).
BALL_REST_POS_LARGE = {
    "ball1": np.array([0.0014, -0.0210, 0.0307]),
    "ball2": np.array([0.0014, 0.0210, 0.0307]),
}
MIN_BALL_SEPARATION_LARGE = 0.030  # meters -- below the balls' own 0.04m diameter means overlap
MAX_BALL_SEPARATION_LARGE = 0.075  # meters -- touching is ~0.04m; generous headroom above that


# User: "the current baoding .gif show two are separated with distance, no contact, no push each
# other" -- correct; BAODING_GRASP_TARGET's separation (~0.041m) is wider than the balls' own
# touching distance (~0.030m, sum of radii). Found this by sweeping curl/spread with the middle
# finger's curl FORCED to 0 (open) -- it was the one wedging itself into the gap between the
# balls and holding them apart; with it retracted and a narrower spread, ff/rf/th alone bring the
# balls into genuine geom-geom contact (confirmed via data.contact, see README Stage 6). Tested
# whether this closes the orbiting gap: it doesn't -- `twist_ctrl` on this contact grasp settles
# into a small (~4-5 degree) stable wobble while actually touching, then the contact becomes
# unstable after several cycles and one ball gets violently ejected (separation jumps ~4x in one
# step). Contact between the balls wasn't the missing ingredient for circulation.
BAODING_CONTACT_TARGET = make_grasp_target(curl_frac=0.95, thumb_base=1.3, base_spread=0.15)
BAODING_CONTACT_TARGET[4:8] = make_grasp_target(curl_frac=0.0, thumb_base=1.3,
                                                 base_spread=0.15)[4:8]


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
    this project's grasp-stability success criterion (see this module's docstring).

    **Known gap, found during Stage 5 (see README)**: this alone does NOT guarantee the object is
    actually held by the fingers -- an object wedged/rattling in a gap between splayed-open
    fingers can stay within `threshold` of its rest position without any real multi-finger grip.
    For any task where the POLICY can freely reshape the hand (Stage 5's RL, unlike Stages 2-4's
    fixed scripted poses), pair this with `fingertip_contact_count` below and require >=2."""
    dist = np.linalg.norm(data.body("object").xpos - OBJECT_REST_POS)
    return dist < threshold, dist


FINGERTIP_BODY_NAMES = ("ff_tip", "mf_tip", "rf_tip", "th_tip")


def _collision_geom_id(model, body_name):
    """Each fingertip body has 2 geoms (see assets/right_hand.xml): a visual mesh (contype=0,
    doesn't register contacts) followed by the actual collision capsule -- confirmed directly via
    model.geom_bodyid/geom_type on the vendored, unmodified scene_right.xml. Looked up by body
    name (not hardcoded index) so this survives any future asset re-vendor."""
    body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body_name)
    start, num = model.body_geomadr[body_id], model.body_geomnum[body_id]
    return start + (1 if num > 1 else 0)


def fingertip_contact_count(model, data, object_geom_id=None, object_body_name="object"):
    """Number of DISTINCT fingertips (0-4) currently in contact with `object_body_name` -- a real
    multi-finger-grip signal, unlike `object_retained`'s position-only check (see its docstring
    for why that alone isn't enough once the policy can reshape the hand itself). Takes an
    explicit body name (not hardcoded to Stage 2-5's single "object") so Stage 6's two-ball scene
    can call this once per ball."""
    if object_geom_id is None:
        object_geom_id = _collision_geom_id(model, object_body_name)
    tip_geom_ids = {_collision_geom_id(model, name) for name in FINGERTIP_BODY_NAMES}
    touching = set()
    for i in range(data.ncon):
        c = data.contact[i]
        if c.geom1 == object_geom_id and c.geom2 in tip_geom_ids:
            touching.add(c.geom2)
        elif c.geom2 == object_geom_id and c.geom1 in tip_geom_ids:
            touching.add(c.geom1)
    return len(touching)


def hand_contact_count(model, data, object_body_name):
    """Number of DISTINCT hand bodies (any link, not just fingertip capsules -- see
    `fingertip_contact_count` for that stricter version) currently in contact with
    `object_body_name`. Added for Stage 6's larger-ball RL env after finding that
    `fingertip_contact_count` alone is too marginal for a long static hold: measured directly
    that even a completely motionless `BAODING_GRASP_TARGET_LARGE` grasp genuinely loses one
    fingertip contact (not flickering noise -- a real, permanent transition) after ~6.5 seconds
    of soft-actuator creep, which would fail every episode regardless of policy behavior. A ball
    resting against a proximal/medial segment plus the palm is still meaningfully "held" even
    without fingertip-specific contact, and this is far more robust against that creep since
    there's much more redundant contact area to lose before the count actually drops."""
    object_geom_id = _collision_geom_id(model, object_body_name)
    other_ball = "ball2" if object_body_name == "ball1" else "ball1"
    other_geom_id = _collision_geom_id(model, other_ball) if other_ball in (
        "ball1", "ball2") else None
    touching = set()
    for i in range(data.ncon):
        c = data.contact[i]
        for a, b in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
            if a == object_geom_id and b != other_geom_id:
                body_id = model.geom_bodyid[b]
                body_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, body_id)
                if body_name != "world":
                    touching.add(body_id)
    return len(touching)


def quat_angle_deg(q1, q2):
    """Shortest rotation angle (degrees) between two world-frame quaternions (mjtNum wxyz).
    Only meaningful up to 180 degrees (quaternions double-cover rotations) -- can't tell a
    genuine multi-turn spin from a return-to-start. Use `twist_angle_rad` + per-step
    accumulation (see below) for Stage 5's "did we actually spin past the Stage 3/4 ceiling"
    question, where the whole point is to exceed a single half-turn."""
    dot = np.clip(abs(np.dot(q1, q2)), 0.0, 1.0)
    return np.degrees(2 * np.arccos(dot))


def _quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def _quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def twist_angle_rad(q_prev, q_cur, axis=np.array([0.0, 0.0, 1.0])):
    """Signed incremental rotation (radians) of q_cur relative to q_prev, about world `axis`,
    via swing-twist decomposition (the correct way to isolate "how much did it spin about THIS
    axis" from a quaternion that may also include off-axis wobble -- a naive dot-product
    projection of the vector part conflates the two). Meant to be called every control step and
    SUMMED, unlike `quat_angle_deg` -- that's how Stage 5's RL env/eval can detect genuine
    multi-turn rotation (e.g. 400 degrees of net spin), which a single start-to-now quat_angle_deg
    call could never report (it saturates at 180 and would read that spin as "coming back")."""
    q_rel = _quat_mul(q_cur, _quat_conj(q_prev))
    w, v = q_rel[0], q_rel[1:]
    s = np.dot(v, axis)
    angle = 2 * np.arctan2(abs(s), w)
    return angle if s >= 0 else -angle


def regrasp_finger(model, data, finger_ctrl_slice, hold_target, release_seconds=0.4,
                    hold_seconds=0.3):
    """Stage 4's scripted finger-gaiting attempt: releases one finger (its 4 ctrl indices, e.g.
    `slice(8, 12)` for the ring finger) back to 0 (open) while the rest of `hold_target` keeps
    the other fingers' grip, then re-curls that finger back to `hold_target`'s own value for it
    -- a minimal "let go, recontact" regrasp. Confirmed (see README Stage 4) this does NOT
    extend Stage 3's twist primitive past its own ~23-25 degree rotation ceiling, however many
    times it's repeated -- a real negative result, not a missing feature of this helper."""
    release = hold_target.copy()
    release[finger_ctrl_slice] = 0.0
    n = int(release_seconds / model.opt.timestep)
    start = data.ctrl.copy()
    for i in range(n):
        alpha = (i + 1) / n
        data.ctrl[:] = start + alpha * (release - start)
        mj.mj_step(model, data)
    for _ in range(int(hold_seconds / model.opt.timestep)):
        data.ctrl[:] = release
        mj.mj_step(model, data)
    start2 = data.ctrl.copy()
    for i in range(n):
        alpha = (i + 1) / n
        data.ctrl[:] = start2 + alpha * (hold_target - start2)
        mj.mj_step(model, data)
    for _ in range(int(hold_seconds / model.opt.timestep)):
        data.ctrl[:] = hold_target
        mj.mj_step(model, data)


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


def orbit_wave_ctrl(base_ctrl, amp, phase):
    """Stage 6's first scripted attempt at orbiting the two-ball pair: a traveling wave of curl
    amplitude across the three non-thumb fingers' proximal joints (120 degrees of phase offset
    between ff/mf/rf), intended to act like a rotating paddle that sequentially nudges the balls
    around the palm. **Confirmed NOT to produce sustained orbiting** (see README Stage 6) -- the
    ball-pair angle (`ball_pair_angle_rad`) shifts a few degrees on the first cycle and then
    flatlines, same qualitative failure as reusing `twist_ctrl` directly on this scene. Kept as a
    documented negative result, not deleted, same as `regrasp_finger`'s Stage 4 role."""
    c = base_ctrl.copy()
    for f, offset in enumerate((0.0, 2 * np.pi / 3, 4 * np.pi / 3)):
        idx = f * 4 + 1  # proximal curl joint within this finger's 4-joint group
        lo, hi = FINGER_CURL_RANGES[1]
        mod = amp * np.sin(phase + offset)
        c[idx] = np.clip(base_ctrl[idx] + mod * (hi - lo), lo, hi)
    return c


# Stage 6, trying "sequenced contact planning" (as opposed to RL) for the two-ball-circulation
# problem: ff is on ball2's side, rf is on ball1's side (mf sits centered, between the two balls
# -- see assets/scene_two_balls_large.xml). FF_SLICE/RF_SLICE are each finger's 3 curl joints.
FF_CURL_SLICE = slice(1, 4)
RF_CURL_SLICE = slice(9, 12)
MF_PROXIMAL_IDX = 5


def diagonal_stroke_ctrl(base_ctrl, amp_lat, amp_curl, phase, thumb_amp_base=0.0,
                          thumb_amp_curl=0.0):
    """Stage 6 continued -- user observation from the orbit_wave_ctrl/leapfrog GIFs: "only three
    fingers make efforts and the thumb not involved much, and only make balls spin, not push
    balls to opposite direction and cross each other, can finger move in cross direction...?"

    Correct diagnosis: `orbit_wave_ctrl` and `leapfrog_half_cycle` ONLY modulate curl (flexion,
    idx 1/2/3 per finger) -- pure RADIAL motion toward/away from the palm center. That's exactly
    the motion that rolls a ball in place (Stage 6's "ball spin" finding) but has no component
    in the tangential/lateral direction needed to actually drag a ball sideways across the palm.
    This primitive instead drives each finger's base/spread joint (idx 0/8, ab-adduction --
    LATERAL motion) and its own proximal curl joint in quadrature (90 degree phase offset), so
    each fingertip traces a small ELLIPSE/diagonal stroke (lateral-and-radial at once) instead of
    pulsing straight in and out. ff and rf are driven with opposite lateral phase so they sweep
    complementary diagonals (one finger's stroke carries a ball toward where the other is
    retreating from), and the thumb can optionally be added into the same quadrature pattern
    (thumb_amp_base/thumb_amp_curl -- both default 0, i.e. thumb stays at `base_ctrl`'s target,
    matching the GIFs the user was looking at) instead of sitting out the whole motion."""
    c = base_ctrl.copy()
    lo1, hi1 = FINGER_CURL_RANGES[1]
    c[0] = base_ctrl[0] + amp_lat * np.sin(phase)
    c[1] = np.clip(base_ctrl[1] + amp_curl * (hi1 - lo1) * np.cos(phase), lo1, hi1)
    c[8] = base_ctrl[8] - amp_lat * np.sin(phase)
    c[9] = np.clip(base_ctrl[9] + amp_curl * (hi1 - lo1) * np.cos(phase + np.pi), lo1, hi1)
    if thumb_amp_base or thumb_amp_curl:
        lo_t, hi_t = THUMB_CURL_RANGES[1]
        c[12] = np.clip(base_ctrl[12] + thumb_amp_base * np.sin(phase + np.pi / 2),
                         *THUMB_BASE_RANGE)
        c[13] = np.clip(base_ctrl[13] + thumb_amp_curl * (hi_t - lo_t) * np.cos(phase + np.pi / 2),
                         lo_t, hi_t)
    return c


FOUR_DIGIT_BASE_IDX = {"ff": 0, "mf": 4, "rf": 8}
FOUR_DIGIT_CURL_IDX = {"ff": 1, "mf": 5, "rf": 9}
FOUR_DIGIT_PHASE_OFFSET = {"ff": 0.0, "mf": np.pi / 2, "rf": np.pi, "th": 3 * np.pi / 2}


def four_finger_orbit_ctrl(base_ctrl, amp_lat, amp_curl, phase, thumb_amp_base=0.0,
                            thumb_amp_curl=0.0):
    """Stage 6 continued -- user, watching `diagonal_stroke_ctrl`'s GIF: "middle finger and thumb
    block balls cross each other, can 4 fingers work in a coordinate way". Correct: that
    primitive only ever moves ff/rf -- mf and th sit rigidly at `base_ctrl`'s fixed target the
    whole time, so they act as a static wall the balls have to be squeezed past rather than a gap
    that opens for them.

    This instead gives all FOUR digits (ff, mf, rf, th) the same lateral+curl quadrature stroke
    `diagonal_stroke_ctrl` used for just ff/rf, but phased a quarter-cycle apart from each other
    (`FOUR_DIGIT_PHASE_OFFSET`, 0/90/180/270 degrees) -- a rotating gate: at any instant exactly
    one digit is near its own curl minimum (maximally retracted, i.e. "open") while the others are
    nearer their own curl maximum (engaged, holding), and which digit is open sweeps around all
    four in sequence as `phase` advances, instead of the fixed ff/rf-only opening pattern before."""
    c = base_ctrl.copy()
    lo1, hi1 = FINGER_CURL_RANGES[1]
    for name, base_idx in FOUR_DIGIT_BASE_IDX.items():
        curl_idx = FOUR_DIGIT_CURL_IDX[name]
        local = phase + FOUR_DIGIT_PHASE_OFFSET[name]
        c[base_idx] = base_ctrl[base_idx] + amp_lat * np.sin(local)
        c[curl_idx] = np.clip(base_ctrl[curl_idx] + amp_curl * (hi1 - lo1) * np.cos(local), lo1,
                               hi1)
    if thumb_amp_base or thumb_amp_curl:
        lo_t, hi_t = THUMB_CURL_RANGES[1]
        local = phase + FOUR_DIGIT_PHASE_OFFSET["th"]
        c[12] = np.clip(base_ctrl[12] + thumb_amp_base * np.sin(local), *THUMB_BASE_RANGE)
        c[13] = np.clip(base_ctrl[13] + thumb_amp_curl * (hi_t - lo_t) * np.cos(local), lo_t, hi_t)
    return c


def leapfrog_half_cycle(model, data, base_ctrl, release_slice, push_idx, release_amt=0.2,
                         push_amt=0.25, duration_seconds=0.6, hold_seconds=0.2):
    """One half of a "leapfrog" attempt at genuine circulation (not a smooth periodic oscillation
    like twist_ctrl/orbit_wave_ctrl, which settle to a fixed point -- see their own docstrings):
    partially release one side's grip (`release_slice`, e.g. `RF_CURL_SLICE` to let ball1 roll
    toward where ball2 was) while the centered mf finger pushes harder (`push_idx`) to nudge the
    near ball past the other, then returns to `base_ctrl` to re-secure. Calling this twice with
    roles swapped (release RF+push, then release FF+push) is one full leapfrog cycle.

    **Confirmed (see README Stage 6) this gets real progress (~20-28 degrees per attempt,
    tuning-dependent) but is NOT yet stable over many repeated cycles** -- every parameter
    combination tried eventually ejects a ball (separation jumps ~4x in one step) after a handful
    of cycles. Kept as a documented partial-progress result, the scripted counterpart to the RL
    approach's own "real but bounded" findings, not deleted despite the instability."""
    dt = model.opt.timestep
    n_half = int(duration_seconds / dt)
    for t in range(n_half):
        c = base_ctrl.copy()
        c[release_slice] = base_ctrl[release_slice] * (1 - release_amt)
        c[push_idx] = min(base_ctrl[push_idx] + push_amt, 1.6)
        data.ctrl[:] = c
        mj.mj_step(model, data)
    data.ctrl[:] = base_ctrl
    for _ in range(int(hold_seconds / dt)):
        mj.mj_step(model, data)
