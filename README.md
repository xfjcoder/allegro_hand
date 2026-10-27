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

3. **Second task: in-hand reorientation via simultaneous finger motion only -- DONE, a real
   bounded limit found.** From the Stage 2 power grasp, oscillate two opposing fingers' base
   (ab/adduction) joints (`ffj0`/`rfj0`) in antiphase (`hand_env.py`'s `twist_ctrl`) and measure
   the object's cumulative rotation (quaternion angle from its grasped starting orientation).

   **Caught a real mistake before reporting it, not after**: a first sweep at only 6-20
   oscillation cycles made amplitude 0.4 look like unbounded, continuously-accumulating rotation
   (still climbing ~0.8 degrees/cycle with no sign of flattening at cycle 20) -- which would have
   meant simultaneous finger motion alone is enough for arbitrary reorientation, no finger-gaiting
   needed. Ran it out to 50 cycles before writing that down, and it DOES flatten -- just much
   later, and to a much higher ceiling (~23 degrees, not ~8) than the low-amplitude runs showed.
   `eval_stage3_reorientation.py` defaults to 50 cycles for exactly this reason.

   | oscillation amplitude | rotation @ cycle 25 | rotation @ cycle 50 | retained every cycle |
   |---|---|---|---|
   | 0.1 | 11.0deg | 14.5deg (still rising) | yes |
   | 0.2 | 11.0deg | 10.7deg (plateaued) | yes |
   | 0.3 | 14.7deg | 22.2deg | yes |
   | 0.4 | 22.8deg | 23.4deg (flat) | yes |
   | 0.47 (joint's own max range) | 23.3deg | 23.5deg (flat) | yes |

   **Finding: this primitive has a hard rotation ceiling around ~23-25 degrees, independent of
   oscillation amplitude** -- even driving the joint to its own maximum range (0.47) converges to
   the same ~23.5 degree bound as amplitude 0.4, not further. This is a static-friction-cone limit
   of simultaneous-contact motion (the grasped contact points can only "stick-slip" so far before
   the geometry runs out of room to advance further under this specific motion), not a tunable
   parameter -- confirms that genuinely large-angle or arbitrary in-hand reorientation needs
   finger-GAITING (releasing and re-contacting at new positions), not just richer simultaneous
   finger motion. Object retention held at every amplitude and cycle count tested -- this bounded
   twist never risks dropping the object, unlike what a full re-gaiting controller will have to
   manage.

   <p align="center">
     <img src="media/stage3_reorientation.gif" width="360">
   </p>
   <p align="center"><sub>Stage 3: the twist primitive (amplitude 0.4) rotating the held object
   toward its ~23 degree ceiling over 25 cycles.</sub></p>

4. **Third task: finger-gaiting (release-and-recontact a finger to extend reorientation range)
   -- tried, a real negative result with a traced cause.** Stage 3 ended with a clear
   hypothesis: its ~23-25 degree ceiling is a static-friction-cone limit of SIMULTANEOUS-contact
   motion, and should be breakable by releasing a finger and recontacting it at a fresh position
   once the object has rotated -- the textbook definition of finger-gaiting.

   **`eval_stage4_finger_gaiting.py`: interleaves twist blocks with a `regrasp_finger` call
   (release one finger back to open, then re-curl it to the same power-grasp target) between
   blocks, same total twist-cycle budget as a no-gait control.** Tried gaiting the ring finger
   (one of the two ACTIVELY twisting fingers), the middle finger (one of the two STATIC anchor
   fingers), and -- in exploration not kept as a committed script -- recontacting at a
   deliberately SHIFTED base-joint target rather than the original one. Repeated the regrasp up
   to 4 times (50 total twist cycles) to rule out "just hadn't gaited enough yet."

   | condition (50 total twist cycles) | final rotation |
   |---|---|
   | no gait (control) | 23.4 degrees |
   | gait ring finger (same target) | 23.9 degrees |
   | gait middle finger (same target) | 23.8 degrees |
   | gait ring finger (shifted target, exploration only) | 18.0 degrees (worse) |

   **All three committed conditions converge to within ~0.5 degrees of the SAME ceiling as no
   gaiting at all -- repeated regrasping does not extend Stage 3's rotation limit.** Root cause:
   releasing a finger and recurling it back to the exact same joint target doesn't actually
   reposition its CONTACT POINT on the object by very much (the object barely moves during the
   brief 3-finger-hold release window), so the "fresh contact" a real gait needs never actually
   materializes -- confirmed directly that even shifting the recontact target to a different
   joint angle made things WORSE, not better, since that just moves the finger to a less
   favorable grip angle rather than a deliberately-planned fresh contact point. A real fix would
   need to plan WHERE on the object's current surface to recontact (a genuine contact-planning
   problem), not just blindly reopen-and-reclose a finger -- exactly the kind of problem a
   learned (RL) controller is suited for and open-loop scripting isn't. Object retention held
   through every attempt (no drops), so this is a clean "didn't help" result, not a crash.

   <p align="center">
     <img src="media/stage4_finger_gaiting.gif" width="360">
   </p>
   <p align="center"><sub>Stage 4: twist blocks interleaved with ring-finger regrasps (visible
   release/reclose) -- still converges to the same ~23 degree ceiling as no gaiting at all.</sub></p>

   **This closes out the three tasks from "let's do all them one by one":** grasp stability
   (Stage 2, a clean positive result), reorientation (Stage 3, a real bounded limit found), and
   finger-gaiting (Stage 4, a real negative result with the cause traced to naive regrasping not
   being genuine contact planning). The natural next step, flagged but not started, is a learned
   policy for the contact-planning problem Stage 4 exposed -- not more hand-scripting.

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
├── eval_stage3_reorientation.py           # Stage 3: twist-primitive amplitude/cycle sweep, item 3
├── demo_stage3_reorientation.py           # Stage 3: rotation-to-ceiling GIF, item 3
├── eval_stage4_finger_gaiting.py          # Stage 4: regrasp vs no-gait comparison, item 4
├── demo_stage4_finger_gaiting.py          # Stage 4: visible regrasp attempt GIF, item 4
├── media/
│   ├── stage1_kinematic_sweep.gif
│   ├── stage2_grasp_stability.gif
│   ├── stage3_reorientation.gif
│   └── stage4_finger_gaiting.gif
└── README.md
```
