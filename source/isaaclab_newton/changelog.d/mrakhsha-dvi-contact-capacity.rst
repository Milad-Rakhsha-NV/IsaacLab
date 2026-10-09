Fixed
~~~~~

* Resolve automatic collision capacity before constructing the DVI solver.
  Large replicated batches without an explicit contact budget previously kept
  the solver's 1000-contact fallback and silently omitted remaining contacts.
