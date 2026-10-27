`right_hand.xml`, `scene_right.xml`, and everything under `assets/` are vendored unmodified from
[`google-deepmind/mujoco_menagerie`](https://github.com/google-deepmind/mujoco_menagerie)'s
`wonik_allegro` package (local checkout: `~/robo/mujoco_menagerie`), licensed BSD-2-Clause
(`LICENSE.wonik_allegro`, see also `CHANGELOG.wonik_allegro.md` for that package's own derivation
history from the upstream Allegro Hand V3 URDF). Copied in wholesale, same reasoning as
`panda_vision_grasp`'s vendoring of `franka_emika_panda`: this package is already a curated,
ready-to-use MJCF model (16 position-controlled actuators across 4 fingers, fixed-base palm, a
`scene_right.xml` that already includes a floor and a graspable free-floating ellipsoid object)
-- unlike `barrett_wam`'s Barrett WAM, no genuine build step was needed here.

Everything else in this repo (scenes/tasks built on top of this model, training code, docs) is
original to this project.
