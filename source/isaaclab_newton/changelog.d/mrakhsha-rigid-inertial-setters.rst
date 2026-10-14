Fixed
^^^^^

* Restored rigid-object and collection mass/inertia setters used by manipulation
  randomization. Updated selected inverse properties, preserved unselected bodies,
  and handled static mass transitions for indexed and masked writes.

* Initialized the required Newton Jacobian row map on first dynamics query,
  preventing a startup crash in operational-space-control tasks.
