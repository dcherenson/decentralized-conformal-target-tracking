Place the vehicle meshes for the PyBullet renderer in this directory.

Expected files:
- `ugv.obj`
- `uav.obj`

Mesh conventions:
- Units should be meters.
- Forward should point along `+X`.
- Left should point along `+Y`.
- Up should point along `+Z`.
- The mesh origin should be near the vehicle body center.
- Keep the geometry reasonably compact and triangulated.

Recommended approximate dimensions:
- `ugv.obj`: 0.4 m to 0.7 m long, 0.25 m to 0.45 m wide, 0.15 m to 0.30 m tall.
- `uav.obj`: 0.3 m to 0.8 m rotor span, 0.08 m to 0.20 m body height.

Notes:
- The renderer falls back to primitive shapes if these files are absent.
- If the meshes are not centered or not scaled in meters, use the renderer CLI
  scale and height flags to compensate.
