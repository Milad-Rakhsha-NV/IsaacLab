Fixed
^^^^^

* Restored root pose and velocity kernel dispatch for rigid-object resets,
  including both 32-bit and 64-bit environment indices. This fixes cube resets
  in manipulation tasks after the DVI integration merge.
* Clarified contact convergence diagnostics and documented using ``res4`` with
  APGD cone projection to avoid selecting an incorrect cone solution using a
  normal-only complementarity diagnostic.
