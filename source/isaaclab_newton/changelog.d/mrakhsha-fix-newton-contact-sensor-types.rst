Fixed
^^^^^

* Fixed :class:`~isaaclab_newton.sensors.contact_sensor.ContactSensor` name
  resolution in ``_create_buffers`` to handle Newton's per-sensor scalar
  ``sensing_obj_type`` / ``counterpart_type`` strings. Sensor and filter
  object names were previously resolved to ``"MATCH_ANY"`` for every entry,
  which caused regex-based ``SceneEntityCfg.body_names`` lookups to fail.
