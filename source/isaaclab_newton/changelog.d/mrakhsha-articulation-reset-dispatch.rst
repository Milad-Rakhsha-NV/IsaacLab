Fixed
^^^^^

* Fixed Newton articulation root-state writes with Torch inputs by selecting the
  root writer kernel specialization for the environment index dtype.
* Restored shared cache invalidation and selective forward kinematics after root
  and joint state writes, including velocity-only writes. Fixed-base root
  velocity writes now remain a no-op, and moving a fixed root notifies the solver
  of the changed joint anchor.
* Kept DVI asset state bindings current with odd substep counts when CUDA graphs
  are disabled.
* Restored selective MJWarp solver-history reset at the forward and physics-step
  boundaries without overwriting authored joint state or other worlds' history.
