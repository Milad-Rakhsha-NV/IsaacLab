Removed
^^^^^^^

* Removed the experimental ``DVISolverCfg.use_armature_rows`` option and its
  Newton solver forwarding. Remove this field from runtime configurations;
  DVI uses its previous body-inertia armature approximation. Archived run
  configurations retain their original values for provenance.
