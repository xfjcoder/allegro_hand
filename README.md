# Allegro Hand Dexterous Manipulation

A fixed-base Allegro Hand V3 (16-DOF, 4-finger) dexterous manipulation project in MuJoCo --
started after closing out `barrett_wam` (Barrett WAM arm + BarrettHand) and `panda_vision_grasp`
(Franka Panda vision-based grasping).

## Scoping decisions

- **Platform: Allegro Hand V3, fixed base, no arm.** `mujoco_menagerie` (vendored locally at
  `~/robo/mujoco_menagerie`) already has a ready-to-use `wonik_allegro` package -- Franka/UR5/
  Kinova/Shadow Hand/Allegro Hand models were all already available there (noted in
  `barrett_wam`'s own scoping decisions, which is why that project built a from-scratch Barrett
  WAM model instead). This project picks up one of those already-available dexterous options
  instead: Allegro Hand (16 DOF, 4-finger) over Shadow Hand (24 DOF, 5-finger) -- simpler and the
  more commonly used option in manipulation RL research, a better fit for a first dexterous-
  manipulation project than the harder, higher-DOF Shadow Hand.
- **Hand-only, fixed base, not mounted on an arm.** Isolates the dexterous manipulation problem
  itself (in-hand reorientation, finger-gaiting, regrasping) from reach/approach control, which
  `panda_vision_grasp` and `barrett_wam` already covered at length with simpler end-effectors.
  Revisit mounting on an arm (UR5e or Kinova, also available in `mujoco_menagerie`) only if a
  later stage's results suggest reach matters.
- **Simulator: MuJoCo**, reusing the same physics engine and venv every sibling project already
  uses (`~/robo/venv`, `mujoco>=3.1.0`, satisfies this package's own `>=2.2.2` requirement).

## Staged plan

1. **Vendor the Allegro Hand model, verify it loads and actually moves under its own actuators
   -- DONE.** `mujoco_menagerie`'s `wonik_allegro` package is already a curated, ready-to-use
   MJCF model (16 position-controlled actuators across 4 fingers, fixed-base palm, no arm) --
   copied wholesale into `assets/` (1.5MB, far smaller than `panda_vision_grasp`'s 33MB Franka
   vendoring) rather than re-derived, same reasoning as that project: this package is already
   solved, no genuine build step needed here unlike `barrett_wam`'s from-scratch Barrett WAM.

   **Verified with real closed-loop physics stepping (`mj_step` under gravity), not just forward
   kinematics**: each of the 16 finger joints commanded to its own `[low, high]` range and back
   to 0 one at a time (holding the others at their last commanded target) -- confirms real
   position-actuator tracking. `scene_right.xml` (also vendored unmodified) already includes a
   floor and a free-floating graspable ellipsoid object spawned 10cm above the palm; held the
   hand open (`ctrl=0`) for 1.2s under gravity afterward and confirmed the object **falls into
   the open hand's workspace and settles at `z=0.031m`** (not through it to the floor at
   `z=-0.1m`) -- a real, repeatable resting configuration to build a grasp/manipulation task on,
   not a scene that needs a table or other support added.

   **Found a real, repeatable steady-state gravity sag, not hidden** (same discipline as
   `panda_vision_grasp`'s Stage 1): holding all 16 finger joints at a commanded target of 0 settles
   most joints within ~0.04 rad of that target, EXCEPT the thumb's base joint (`thj0`), which
   settles at **0.261 rad (~15 degrees) off its zero target** -- its `thumb_base` actuator class
   has a much lower effective stiffness relative to the torque the thumb's own mass/geometry
   puts on it at this pose than the other three fingers' `base` class joints do. Not fixed here
   (no gain retuning or gravity-compensation feedforward added): the actual manipulation policy
   planned for later stages observes/acts through this same actuator, so it has every opportunity
   to learn around this bias as part of the task itself -- revisit only if a later stage's
   results suggest this sag is actually a limiting factor, not before.

   <p align="center">
     <img src="media/stage1_kinematic_sweep.gif" width="360">
   </p>
   <p align="center"><sub>Stage 1: each of the 16 finger joints swept through its own range, then
   the included object settling into the open hand under gravity.</sub></p>

2. **First task: grasp stability under perturbation, not "lift" -- DONE.** User: "let's do all
   them one by one" (in-hand reorientation / from-rest grasp-lift / finger-gaiting, offered as
   the three natural next-task options). Starting with grasp/lift, the simplest of the three --
   but a literal vertical "lift" doesn't translate to a fixed-base hand with no wrist/arm to
   raise anything, AND Stage 1 already found the included object falls into a STABLE passive
   resting equilibrium with the hand fully open (`ctrl=0`) -- the default-open pose's own
   slight gravity sag forms a shallow basin that already holds the object without any grasp at
   all. That makes "does the object stay put" a trivial, already-solved null-policy outcome,
   not a real task to build on.

   **Reframed the task as: does an active grasp retain the object against a perturbation that
   the passive pose can't?** A real, non-trivial, and directly measurable question. Found a
   "power grasp" ctrl target (`hand_env.py`'s `POWER_GRASP_TARGET`) via a direct static search
   over curl fraction / thumb-base angle / finger base-joint spread for the configuration that
   brings all 4 fingertips closest to the object's center (curl=0.95 of each curl joint's own
   range, thumb base=1.3 rad, **zero** base-joint spread -- spreading the three fingers'
   ab/adduction joints in EITHER direction, counter to intuition, only moved their fingertips
   farther from the object for this pose, not closer).

   **`eval_stage2_grasp_stability.py`: paired comparison (identical random push direction, 8
   trials per magnitude) of the passive open-hand pose against the power grasp, sweeping push
   speed injected directly into the object's free-joint velocity:**

   | push magnitude (m/s) | open-hand retained | power-grasp retained |
   |---|---|---|
   | 0.6 | 8/8 | 8/8 |
   | 0.8 | 4/8 | 8/8 |
   | 1.0 | 0/8 | 8/8 |
   | 1.2 | 0/8 | 8/8 |
   | 1.5 | 0/8 | 8/8 |
   | 2.0 | 0/8 | 8/8 |

   **Clean, reproducible result**: the passive pose's retention collapses from 8/8 to 0/8
   exactly in the 0.6-1.0 m/s range, while the power grasp retains the object at every tested
   magnitude up to 2.0 m/s (3x the magnitude that fully breaks the passive pose) -- confirming
   this is a real, active grasp effect, not just a relabeling of the same passive equilibrium.
   This is Stage 2's scripted/privileged-state feasibility check (direct `qvel` injection, full
   state access) -- the reference number a later learned policy (closing fingers from pixels/
   touch sensing, reacting to a variety of objects/perturbations, not just this one tuned
   static target) would need to match or beat.

   <p align="center">
     <img src="media/stage2_grasp_stability.gif" width="500">
   </p>
   <p align="center"><sub>Same push (1.0 m/s), identical start state -- left: open-hand pose
   loses the object; right: power grasp retains it.</sub></p>

## Repository layout

```
allegro_hand/
├── assets/
│   ├── right_hand.xml, scene_right.xml   # vendored from mujoco_menagerie, see NOTICE.md
│   ├── assets/                           # meshes the above MJCF references
│   ├── LICENSE.wonik_allegro
│   ├── CHANGELOG.wonik_allegro.md
│   └── NOTICE.md
├── demo_stage1_kinematic_sweep.py         # Stage 1: joint-range sweep + object settle, item 1
├── hand_env.py                            # shared constants/helpers, item 2
├── eval_stage2_grasp_stability.py         # Stage 2: open-hand vs power-grasp push sweep, item 2
├── demo_stage2_grasp.py                   # Stage 2: side-by-side push-response GIF, item 2
├── media/
│   ├── stage1_kinematic_sweep.gif
│   └── stage2_grasp_stability.gif
└── README.md
```
