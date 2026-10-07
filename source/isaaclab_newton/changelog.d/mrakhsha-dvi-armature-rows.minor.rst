Added
^^^^^

* Expose experimental ``DVISolverCfg.use_armature_rows`` for revolute rotor
  inertia with Newton DVI's sparse direct joint solver. The option defaults to
  false and may change without the normal deprecation period. It requires finite,
  nonnegative armature, with zero armature on enabled non-revolute joints.
  Contact and joint-limit coupling remains iterative; enabling the flag does not
  change actuator settings or guarantee contact accuracy at a finite coupling count.
