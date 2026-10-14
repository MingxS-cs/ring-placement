# Panda model attribution

The `panda` model and mesh assets are from
Google DeepMind's MuJoCo Menagerie, revision
`d7b481bac63d3d600ab404ee74f7474c5efe8a67`:
https://github.com/google-deepmind/mujoco_menagerie/tree/d7b481bac63d3d600ab404ee74f7474c5efe8a67/franka_emika_panda

The Panda model's Apache-2.0 license is retained at
`panda/LICENSE`. Original XML and assets are
unmodified. `ring_learning/sim.py` derives an in-memory scene, changes contact
settings and finger geometry, replaces the gripper servo, adds gravity
compensation and the task environment, and removes the original keyframe.
The tabletop, annuli, camera and fingertip extensions are task-specific.
