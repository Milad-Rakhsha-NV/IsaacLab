# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Policy mapping utilities for schema joint order transformations."""

import copy
import numpy as np
import torch
import os
import yaml
import tempfile
from typing import cast

import omni.usd
import isaaclab.sim as sim_utils

try:
    from tensordict import TensorDictBase as _TensorDictBase  # type: ignore
    TensorDictBase = _TensorDictBase
except Exception:
    TensorDictBase = tuple()  # fallback: isinstance(obs, TensorDictBase) will be False


class SchemaJointOrderHelperBase:
    """Base class for schema joint order helpers with common functionality."""

    def __init__(self, base_env, schema_override_names: list[str] | None = None):
        self.base_env = base_env
        self._schema_override_names = list(schema_override_names) if schema_override_names else None

    def _get_scene_articulation_and_joint_names(self):
        """Get articulation and joint names from the current environment."""
        scene = self.base_env.scene
        # Prefer common key 'robot', else fallback to the first articulation
        if hasattr(scene, "articulations") and isinstance(scene.articulations, dict) and len(scene.articulations) > 0:
            if "robot" in scene.articulations:
                art = scene.articulations["robot"]
            else:
                art = next(iter(scene.articulations.values()))
            
            return art, list(art.joint_names)
        return None, None

    def _get_schema_joint_names(self, art) -> list[str] | None:
        """Get joint names from USD Isaac Robot Schema."""
        try:
            # Resolve the robot prim in the first environment
            first_robot_prim = sim_utils.find_first_matching_prim(art.cfg.prim_path)
            if first_robot_prim is None:
                return None
            stage = omni.usd.get_context().get_stage()
            prim = first_robot_prim
            # Import here to avoid hard dependency if schema package is unavailable
            from usd.schema.isaac import robot_schema  # type: ignore

            joints = robot_schema.utils.GetAllRobotJoints(stage, prim, False)
            schema_joint_names = []
            for j in joints:
                # joints may be prims or have GetPrim(); robustly extract name
                try:
                    p = j.GetPrim() if hasattr(j, "GetPrim") else j
                    name = p.GetPath().pathString.rsplit("/", 1)[-1]
                except Exception:
                    name = str(j)
                schema_joint_names.append(name)
            return schema_joint_names
        except Exception:
            return None

    def _debug_enabled(self) -> bool:
        try:
            return str(os.environ.get("SIM2SIM_DEBUG", "0")).lower() in ("1", "true", "yes")
        except Exception:
            return False

    def _debug_print_mappings(
        self,
        where: str,
        engine_joint_names: list[str],
        schema_joint_names: list[str],
        mappings: tuple[list[int] | None, list[int] | None],
        term_names: list[str] | None,
        term_dims: list[int] | None,
        joint_related_indices: list[int] | None,
        obs_perm: torch.Tensor | None,
        action_indices: torch.Tensor | None,
    ) -> None:
        if not self._debug_enabled():
            return
        try:
            print(f"[SIM2SIM][{where}] engine_joint_names ({len(engine_joint_names)}): {engine_joint_names}")
            print(f"[SIM2SIM][{where}] schema_joint_names ({len(schema_joint_names)}): {schema_joint_names}")
            if mappings and mappings[0] is not None and mappings[1] is not None:
                _ets = cast(list[int], mappings[0])
                _ste = cast(list[int], mappings[1])
                pair_str = ", ".join(
                    [
                        f"{name}: eng={i} -> sch={_ets[i]} | sch->eng={_ste[_ets[i]]}"
                        for i, name in enumerate(engine_joint_names)
                    ]
                )
                print(f"[SIM2SIM][{where}] mapping pairs: {pair_str}")
            if hasattr(self, "_direct_env_term_offsets"):
                print(f"[SIM2SIM][{where}] direct env offsets: {getattr(self, '_direct_env_term_offsets')}")
            if term_names is not None and term_dims is not None:
                print(f"[SIM2SIM][{where}] term_names: {term_names}")
                print(f"[SIM2SIM][{where}] term_dims: {term_dims}")
            if joint_related_indices is not None:
                print(f"[SIM2SIM][{where}] joint_related_indices: {joint_related_indices}")
            if obs_perm is not None:
                print(f"[SIM2SIM][{where}] obs_perm.shape: {tuple(obs_perm.shape)}")
                # Print first 64 entries for brevity
                print(f"[SIM2SIM][{where}] obs_perm[:64]: {obs_perm[:64].tolist()}")
            if action_indices is not None:
                print(f"[SIM2SIM][{where}] action_indices: {action_indices.tolist()}")
            # Actuated DOFs info if present
            try:
                if hasattr(self.base_env, "actuated_dof_indices"):
                    adi = list(getattr(self.base_env, "actuated_dof_indices"))
                    adi_names = [engine_joint_names[i] for i in adi if i < len(engine_joint_names)]
                    print(f"[SIM2SIM][{where}] actuated_dof_indices ({len(adi)}): {adi}")
                    print(f"[SIM2SIM][{where}] actuated_dof_names: {adi_names}")
            except Exception as e:
                print(f"[SIM2SIM][{where}] actuated_dof_indices debug failed: {e}")
        except Exception as e:
            print(f"[SIM2SIM][{where}] debug print failed: {e}")

    def _get_observation_terms_info(self):
        """Extract observation manager information."""
        if hasattr(self.base_env, "observation_manager"):
            obs_mgr = self.base_env.observation_manager
            if ("policy" not in obs_mgr.active_terms) or ("policy" not in obs_mgr.group_obs_term_dim):
                return None, None
            term_names = list(obs_mgr.active_terms["policy"])  # list[str]
            term_dims = [int(np.prod(d)) for d in obs_mgr.group_obs_term_dim["policy"]]
            return term_names, term_dims
        elif self._is_direct_env():
            # Handle direct environments by analyzing the observation structure
            return self._get_direct_env_observation_info()
        return None, None

    def _get_term_type(self, term_func):
        """Get the observation type of a term function."""
        # Check if function has the generic_io_descriptor with observation_type
        if hasattr(term_func, '__wrapped__'):
            # For decorated functions, check for observation_type attribute
            if hasattr(term_func, 'observation_type'):
                return getattr(term_func, 'observation_type')
        
        # Check function name patterns as ultimate fallback
        func_name = getattr(term_func, '__name__', str(term_func))
        if 'joint_pos' in func_name or 'joint_vel' in func_name:
            return 'JointState'
        elif 'action' in func_name:
            return 'Action'
            
        return None

    def _is_direct_env(self):
        """Check if the environment is a direct environment."""
        # Check if it's a direct environment (doesn't have observation_manager)
        return not hasattr(self.base_env, "observation_manager") and hasattr(self.base_env, "_get_observations")

    def _get_actuated_indices(self) -> list[int] | None:
        """Return actuated DOF indices if available (for direct envs)."""
        try:
            if hasattr(self.base_env, "actuated_dof_indices"):
                adi = list(getattr(self.base_env, "actuated_dof_indices"))
                return [int(i) for i in adi]
        except Exception:
            pass
        return None

    def _get_direct_env_observation_info(self):
        """Extract observation information from direct environments by analyzing observation structure."""
        try:
            # Get a sample observation to analyze structure
            obs = self.base_env._get_observations()
            if not isinstance(obs, dict) or "policy" not in obs:
                return None, None
            
            policy_obs = obs["policy"]
            if not hasattr(policy_obs, "shape") or len(policy_obs.shape) < 2:
                return None, None
            
            # For direct environments, we need to manually identify joint-related segments
            # This is environment-specific, so we'll use heuristics based on known patterns
            return self._parse_direct_env_observation_structure(policy_obs.shape[-1])
        except Exception:
            return None, None

    def _parse_direct_env_observation_structure(self, total_obs_dim):
        """Parse direct environment observation structure to identify joint-related terms."""
        # Get joint information from the environment
        art, joint_names = self._get_scene_articulation_and_joint_names()
        if art is None or not joint_names:
            return None, None
        
        num_joints = len(joint_names)
        
        # Try to get the number of actions more reliably
        num_actions = num_joints  # Default fallback
        if hasattr(self.base_env, "num_actions"):
            num_actions = self.base_env.num_actions
        elif hasattr(self.base_env.cfg, "action_space"):
            if isinstance(self.base_env.cfg.action_space, int):
                num_actions = self.base_env.cfg.action_space
            elif hasattr(self.base_env.cfg.action_space, "shape"):
                shape = self.base_env.cfg.action_space.shape
                num_actions = shape[0] if isinstance(shape, (list, tuple)) else shape
        elif hasattr(self.base_env, "num_hand_dofs"):
            num_actions = self.base_env.num_hand_dofs
        
        # For InHandManipulation environment, the structure is:
        # [joint_pos(num_joints), joint_vel(num_joints), ..., actions(num_actions)]
        # We know joint_pos and joint_vel are at the beginning, actions at the end
        
        joint_related_terms = []
        
        # Joint positions at offset 0
        joint_related_terms.append(("joint_pos", num_joints, 0))
        
        # Joint velocities at offset num_joints
        joint_related_terms.append(("joint_vel", num_joints, num_joints))
        
        # Actions at the end
        action_offset = total_obs_dim - num_actions
        if action_offset >= 2 * num_joints:  # Make sure there's space for joint_pos and joint_vel
            joint_related_terms.append(("actions", num_actions, action_offset))
        
        if len(joint_related_terms) < 2:  # At least joint_pos and joint_vel
            return None, None
        
        # Store the offsets for later use
        self._direct_env_term_offsets = {term[0]: term[2] for term in joint_related_terms}
        
        # Convert to the format expected by the rest of the code
        term_names = [term[0] for term in joint_related_terms]
        term_dims = [term[1] for term in joint_related_terms]
        
        return term_names, term_dims

    def _validate_joint_terms(self, joint_names: list[str], term_names: list[str], term_dims: list[int]):
        """Find all joint-related observation terms (JointState and Action types)."""
        if hasattr(self.base_env, "observation_manager"):
            # Manager-based environment
            obs_mgr = self.base_env.observation_manager
            if "policy" not in obs_mgr._group_obs_term_cfgs:
                return None
                
            term_cfgs = obs_mgr._group_obs_term_cfgs["policy"]
            
            # Find all joint-related terms (JointState and Action types)
            joint_related_indices = []
            
            for i, (term_name, term_cfg) in enumerate(zip(term_names, term_cfgs)):
                if not hasattr(term_cfg, 'func'):
                    continue
                    
                term_type = self._get_term_type(term_cfg.func)
                
                # All joint-related terms use the same permutation
                if term_type in ['JointState', 'Action']:
                    # Validate size matches number of joints
                    num_joints = len(joint_names)
                    if term_dims[i] == num_joints:
                        joint_related_indices.append(i)
            
            # Need at least one joint-related term
            if not joint_related_indices:
                return None
            
            return joint_related_indices
        elif self._is_direct_env():
            # Direct environment - identify joint-related terms by name and size
            joint_related_indices = []
            num_joints = len(joint_names)
            
            for i, (term_name, term_dim) in enumerate(zip(term_names, term_dims)):
                # Check if this is a joint-related term
                if term_name in ['joint_pos', 'joint_vel'] and term_dim == num_joints:
                    joint_related_indices.append(i)
                elif term_name == 'actions':
                    # Actions are joint-related for manipulation tasks
                    joint_related_indices.append(i)
            
            # Need at least one joint-related term
            if not joint_related_indices:
                return None
            
            return joint_related_indices
        
        return None

    def _build_joint_index_mappings(self, engine_joint_names: list[str], schema_joint_names: list[str]):
        """Build bidirectional mappings between engine and schema joint orders."""
        # Filter schema list to only those joints that exist in engine
        schema_filtered = [n for n in schema_joint_names if n in engine_joint_names]
        if len(schema_filtered) != len(engine_joint_names):
            return None, None
        
        engine_index = {n: i for i, n in enumerate(engine_joint_names)}
        schema_index = {n: i for i, n in enumerate(schema_filtered)}
        
        # engine_to_schema: for each engine joint index, what schema index should it map to
        engine_to_schema = [schema_index[n] for n in engine_joint_names]
        # schema_to_engine: for each schema joint index, what engine index should it map to  
        schema_to_engine = [engine_index[n] for n in schema_filtered]
        
        return engine_to_schema, schema_to_engine

    def _compute_observation_permutation(self, mappings, term_names: list[str], term_dims: list[int], joint_related_indices: list[int], for_import: bool):
        """Compute observation permutation. Direction controlled by for_import parameter."""
        engine_to_schema, schema_to_engine = mappings

        # Build flat observation permutation
        if self._is_direct_env() and hasattr(self, '_direct_env_term_offsets'):
            # For direct environments, use the actual offsets from parsed structure
            offsets = [self._direct_env_term_offsets.get(term_name, 0) for term_name in term_names]
            # For direct envs, total_obs should be the actual observation dimension, not sum of joint terms
            total_obs = max(offsets[i] + term_dims[i] for i in range(len(term_names)))
        else:
            # For manager-based environments, use cumulative sum
            offsets = np.cumsum([0] + term_dims[:-1]).tolist()
            total_obs = int(np.sum(term_dims))
        
        obs_perm = np.arange(total_obs)

        # Apply permutation to joint-related terms, handling 'actions' specially for direct envs
        actuated_indices = self._get_actuated_indices() if self._is_direct_env() else None
        pos_in_act = None
        if actuated_indices is not None:
            pos_in_act = {int(e): int(k) for k, e in enumerate(actuated_indices)}

        for term_index in joint_related_indices:
            start = offsets[term_index]
            length = term_dims[term_index]
            
            if length == 0:
                print(f"[WARN] Skipping term {term_names[term_index]} with zero length")
                continue
                
            if start + length > total_obs:
                print(f"[WARN] Term {term_names[term_index]} extends beyond observation bounds: start={start}, length={length}, total={total_obs}")
                continue
            
            term_name = term_names[term_index] if term_names is not None else ""
            if term_name == 'actions' and actuated_indices is not None and pos_in_act is not None:
                # Special handling: actions are in actuated order in direct envs
                if for_import:
                    # We want schema-ordered previous actions from actuated-ordered input
                    # perm_slice[s] = position k in actuated where engine index == schema_to_engine[s]
                    perm_slice = np.array([pos_in_act[int(schema_to_engine[s])] for s in range(length)], dtype=np.int64)
                else:
                    # We want actuated-ordered actions from schema-ordered input
                    # perm_slice[k] = schema index for engine index of actuated[k]
                    perm_slice = np.array([engine_to_schema[int(actuated_indices[k])] for k in range(length)], dtype=np.int64)
            else:
                if for_import:
                    # For importing: build schema-ordered obs by selecting engine indices per schema index
                    perm_slice = np.array(schema_to_engine, dtype=np.int64)
                else:
                    # For exporting: build sim-ordered obs using inverse mapping (engine index -> schema index)  
                    perm_slice = np.array(engine_to_schema, dtype=np.int64)
            
            if len(perm_slice) != length:
                print(f"[WARN] Permutation slice length {len(perm_slice)} doesn't match term length {length} for {term_names[term_index]}")
                continue
            
            obs_perm[start : start + length] = start + perm_slice

        return torch.as_tensor(obs_perm, dtype=torch.long), engine_to_schema, schema_to_engine

class SchemaImportHelper(SchemaJointOrderHelperBase):
    """Helper to import policies from schema joint order representation to engine representation."""
    
    def __init__(self, base_env, schema_override_names: list[str] | None = None):
        super().__init__(base_env, schema_override_names)
        # filled by compute()
        self.obs_perm = None  # 1D LongTensor to map engine obs -> schema obs (for input to policy)
        self.action_perm = None  # 1D LongTensor to map schema actions -> engine actions (for output from policy)
        
    def compute(self) -> bool:
        """Compute the permutation mappings for importing from schema representation."""
        # Get engine joint names
        art, engine_joint_names = self._get_scene_articulation_and_joint_names()
        if art is None or not engine_joint_names:
            return False
            
        # Get schema joint names
        schema_joint_names = self._schema_override_names if self._schema_override_names is not None else self._get_schema_joint_names(art)
        if not schema_joint_names:
            return False
            
        # Build index mappings
        mappings = self._build_joint_index_mappings(engine_joint_names, schema_joint_names)
        if mappings[0] is None:
            return False
        
        # print("engine_to_schema", mappings[0])
        # print("schema_to_engine", mappings[1])
        
        # Get observation terms info
        term_names, term_dims = self._get_observation_terms_info()
        if term_names is None or term_dims is None:
            return False

        # Validate joint terms
        joint_related_indices = self._validate_joint_terms(engine_joint_names, term_names, term_dims)
        if joint_related_indices is None:
            return False

        # Compute observation permutation (for import, for_import=True)
        self.obs_perm, engine_to_schema, schema_to_engine = self._compute_observation_permutation(
            mappings, term_names, term_dims, joint_related_indices, for_import=True
        )
        # For import: map schema actions -> env actions
        # In direct envs, actions are ordered by actuated_dof_indices
        actuated_indices = self._get_actuated_indices() if self._is_direct_env() else None
        if actuated_indices is not None:
            action_indices = [engine_to_schema[int(actuated_indices[k])] for k in range(len(actuated_indices))]
        else:
            action_indices = list(engine_to_schema)
        self.action_perm = torch.as_tensor(action_indices, dtype=torch.long)

        # Debug info
        self._debug_print_mappings(
            where="IMPORT",
            engine_joint_names=engine_joint_names,
            schema_joint_names=schema_joint_names,
            mappings=(engine_to_schema, schema_to_engine),
            term_names=term_names,
            term_dims=term_dims,
            joint_related_indices=joint_related_indices,
            obs_perm=self.obs_perm,
            action_indices=self.action_perm,
        )
        
        # print("obs_perm", self.obs_perm)
        # print("action_perm", self.action_perm)
        return True

class SchemaExportHelper(SchemaJointOrderHelperBase):
    """Export helper: compute observation/action reordering from Engine to Schema joint orders."""

    def __init__(self, base_env, policy_module, normalizer, schema_override_names: list[str] | None = None):
        super().__init__(base_env, schema_override_names)
        self.policy_module = policy_module
        self.normalizer = normalizer
        self.is_recurrent = getattr(policy_module, "is_recurrent", False)
        # filled by compute()
        self.obs_perm = None  # 1D LongTensor of size num_obs to map schema-ordered obs -> sim-ordered obs
        self.action_out_indices = None  # 1D LongTensor of size num_actions to map sim actions -> schema order

    def compute(self) -> bool:
        """Compute permutations for export."""
        # Get sim joint names
        art, sim_joint_names = self._get_scene_articulation_and_joint_names()
        if art is None or not sim_joint_names:
            return False
        
        # Get schema joint names
        schema_joint_names = self._schema_override_names if self._schema_override_names is not None else self._get_schema_joint_names(art)
        if not schema_joint_names:
            return False
        
        # Build index mappings
        mappings = self._build_joint_index_mappings(sim_joint_names, schema_joint_names)
        if mappings[0] is None:
            return False
        
        # Get observation terms info
        term_names, term_dims = self._get_observation_terms_info()
        if term_names is None or term_dims is None:
            return False

        # Validate joint terms
        joint_related_indices = self._validate_joint_terms(sim_joint_names, term_names, term_dims)
        if joint_related_indices is None:
            return False

        # Compute observation permutation (for export, for_import=False)
        self.obs_perm, engine_to_schema, schema_to_engine = self._compute_observation_permutation(
            mappings, term_names, term_dims, joint_related_indices, for_import=False
        )
        # For export: action_out_indices maps sim actions -> schema order
        # In direct envs, sim actions are in actuated_dof_indices order
        actuated_indices = self._get_actuated_indices() if self._is_direct_env() else None
        if actuated_indices is not None:
            pos_in_act = {int(e): int(k) for k, e in enumerate(actuated_indices)}
            action_indices = [pos_in_act[int(schema_to_engine[s])] for s in range(len(schema_to_engine))]
        else:
            # Original used: schema_to_sim, which is equivalent to our schema_to_engine
            action_indices = list(schema_to_engine)
        self.action_out_indices = torch.as_tensor(action_indices, dtype=torch.long)

        # Debug info
        self._debug_print_mappings(
            where="EXPORT",
            engine_joint_names=sim_joint_names,
            schema_joint_names=schema_joint_names,
            mappings=(engine_to_schema, schema_to_engine),
            term_names=term_names,
            term_dims=term_dims,
            joint_related_indices=joint_related_indices,
            obs_perm=self.obs_perm,
            action_indices=self.action_out_indices,
        )
        
        # print("obs_perm", self.obs_perm)
        # print("action_out_indices", self.action_out_indices)
        return True


class SchemaOrderedTorchPolicyExporter(torch.nn.Module):
    """Exporter that wraps policy to accept schema-ordered obs and emit schema-ordered actions."""

    def __init__(self, policy, normalizer, perm_helper: SchemaExportHelper):
        super().__init__()
        if getattr(policy, "is_recurrent", False):
            raise NotImplementedError("Schema-ordered export supports only non-recurrent policies.")
        # Ensure permutations are available for type-checkers and runtime
        assert perm_helper.obs_perm is not None
        assert perm_helper.action_out_indices is not None
        # deep copy actor/student
        if hasattr(policy, "actor"):
            self.actor = (
                torch.nn.Sequential(*[m for m in policy.actor.children()])
                if isinstance(policy.actor, torch.nn.Sequential)
                else copy.deepcopy(policy.actor)
            )
        elif hasattr(policy, "student"):
            self.actor = (
                torch.nn.Sequential(*[m for m in policy.student.children()])
                if isinstance(policy.student, torch.nn.Sequential)
                else copy.deepcopy(policy.student)
            )
        else:
            raise ValueError("Policy does not have an actor/student module.")
        # copy normalizer
        self.normalizer = copy.deepcopy(normalizer) if normalizer else torch.nn.Identity()
        # store permutations
        self.register_buffer("obs_perm", perm_helper.obs_perm.clone())
        self.register_buffer("action_out_indices", perm_helper.action_out_indices.clone())

    def _apply_obs_perm(self, x: torch.Tensor) -> torch.Tensor:
        # print("applying mapping from schema to sim with obs_perm", self.obs_perm)
        return x.index_select(dim=1, index=self.obs_perm)

    def _apply_action_perm(self, actions_sim: torch.Tensor) -> torch.Tensor:
        # print("applying mapping from sim to schema with action_out_indices", self.action_out_indices)
        return actions_sim.index_select(dim=1, index=self.action_out_indices)

    def forward(self, x):
        x = self._apply_obs_perm(x)
        actions_sim = self.actor(self.normalizer(x))
        return self._apply_action_perm(actions_sim)

    @torch.jit.export
    def reset(self):
        pass


def export_robot_schema_policy(
    base_env,
    runner,
    policy_nn,
    normalizer,
    export_model_dir: str,
    robot_schema_file: str | None,
):
    """Export schema-ordered policy artifacts.

    Exports:
    - JIT wrapper that accepts schema-ordered observations and emits schema-ordered actions (policy_schema_order.pt)
    - Runner checkpoint with weights remapped to schema order (policy_runner_schema_order.pt)
    """
    try:
        schema_override = None
        if robot_schema_file:
            with open(robot_schema_file) as f:
                cfg_yaml = yaml.safe_load(f)
            key = "robot_schema_joint_names"
            if key not in cfg_yaml:
                raise KeyError(f"Key '{key}' not found in YAML {robot_schema_file}")
            schema_override = list(cfg_yaml[key])

        perm_helper = SchemaExportHelper(base_env, policy_nn, normalizer, schema_override_names=schema_override)
        if perm_helper.compute():
            # Export schema-ordered JIT policy
            schema_jit = SchemaOrderedTorchPolicyExporter(policy_nn, normalizer, perm_helper)
            schema_jit.to("cpu")
            traced = torch.jit.script(schema_jit)
            schema_jit_path = os.path.join(export_model_dir, "policy_schema_order.pt")
            traced.save(schema_jit_path)
            print("[INFO] Exported schema-ordered JIT policy to:", schema_jit_path)

            # Additionally export a runner-compatible checkpoint for convenience
            try:
                runner_ckpt_path = os.path.join(export_model_dir, "policy_runner_schema_order.pt")

                # First save the original runner to get the proper checkpoint format
                # Use temporary directory to avoid side-effects in export directory
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_runner_path = os.path.join(temp_dir, "temp_runner.pt")

                    # Temporarily set up logging attributes for the original save
                    orig_log_dir = getattr(runner, 'log_dir', None)
                    orig_logger_type = getattr(runner, 'logger_type', None)
                    try:
                        if not hasattr(runner, 'logger_type'):
                            runner.logger_type = "tensorboard"
                        if getattr(runner, 'log_dir', None) is None:
                            runner.log_dir = temp_dir

                        # Save and load checkpoint to obtain proper serialization format
                        runner.save(temp_runner_path)
                        checkpoint = torch.load(temp_runner_path, map_location='cpu')
                    finally:
                        if orig_log_dir is not None:
                            runner.log_dir = orig_log_dir
                        elif hasattr(runner, 'log_dir'):
                            runner.log_dir = None
                        if orig_logger_type is not None:
                            runner.logger_type = orig_logger_type
                        elif hasattr(runner, 'logger_type') and orig_logger_type is None:
                            try:
                                delattr(runner, 'logger_type')
                            except AttributeError:
                                pass

                # Apply schema mapping to the checkpoint weights
                schema_checkpoint = copy.deepcopy(checkpoint)
                temp_policy = copy.deepcopy(policy_nn)
                
                # Debug: show what keys are in the original checkpoint
                print(f"[DEBUG] Original checkpoint keys: {list(checkpoint.keys())}")
                if "model_state_dict" in checkpoint:
                    model_keys = [k for k in checkpoint["model_state_dict"].keys() if 'norm' in k]
                    print(f"[DEBUG] Normalizer-related keys in model_state_dict: {model_keys}")

                obs_perm = perm_helper.obs_perm
                action_out_indices = perm_helper.action_out_indices
                # Guard: obs_perm can be None in type inference; ensure it exists
                assert obs_perm is not None
                inv_obs_perm = torch.empty_like(obs_perm)
                inv_obs_perm[obs_perm] = torch.arange(
                    obs_perm.numel(), device=obs_perm.device, dtype=obs_perm.dtype
                )

                # Skip normalizer reordering since we're excluding all normalizer keys
                # The target environment will initialize its own normalizers

                # Reorder first and last linear layers in actor/student
                actor_module = getattr(temp_policy, "actor", None) or getattr(temp_policy, "student", None)
                if actor_module is not None:
                    with torch.no_grad():
                        first_linear = None
                        last_linear = None
                        for m in actor_module.modules():
                            if isinstance(m, torch.nn.Linear):
                                if first_linear is None:
                                    first_linear = m
                                last_linear = m

                        if first_linear is not None:
                            idx = inv_obs_perm.to(first_linear.weight.device)
                            first_linear.weight.data = first_linear.weight.data.index_select(1, idx)

                        if last_linear is not None:
                            aidx = cast(torch.Tensor, action_out_indices).to(last_linear.weight.device)
                            last_linear.weight.data = last_linear.weight.data.index_select(0, aidx)
                            if last_linear.bias is not None:
                                last_linear.bias.data = last_linear.bias.data.index_select(0, aidx)

                # Completely remove all normalizer keys to avoid compatibility issues
                # The target environment will use its own normalizer structure
                temp_state_dict = temp_policy.state_dict()
                filtered_state_dict = {}
                
                normalizer_keys_excluded = []
                
                for k, v in temp_state_dict.items():
                    # Completely skip all normalizer keys to avoid compatibility issues
                    if 'normalizer' in k:
                        normalizer_keys_excluded.append(k)
                        continue
                    else:
                        filtered_state_dict[k] = v
                
                print(f"[INFO] Excluded {len(normalizer_keys_excluded)} normalizer keys from schema checkpoint")
                print(f"[INFO] Target environment will use its own normalizer structure")
                
                schema_checkpoint["model_state_dict"] = filtered_state_dict
                
                # Handle obs_norm_state_dict - either reorder existing or create from runner
                if "obs_norm_state_dict" in checkpoint:
                    print(f"[INFO] Found obs_norm_state_dict in original checkpoint")
                    original_obs_norm = checkpoint["obs_norm_state_dict"]
                    reordered_obs_norm = {}
                    
                    # Apply the same observation permutation to the normalizer statistics
                    for k, v in original_obs_norm.items():
                        if isinstance(v, torch.Tensor) and v.dim() == 1 and v.numel() == obs_perm.numel():
                            # This is likely a per-observation statistic that needs reordering
                            reordered_obs_norm[k] = v.index_select(0, obs_perm.to(v.device))
                            print(f"[INFO] Reordered obs_norm key: {k}")
                        else:
                            # Keep scalar values as-is
                            reordered_obs_norm[k] = v
                    
                    schema_checkpoint["obs_norm_state_dict"] = reordered_obs_norm
                    print(f"[INFO] Preserved and reordered obs_norm_state_dict")
                else:
                    # Create obs_norm_state_dict from the policy's normalizer or runner's normalizer
                    print(f"[INFO] No obs_norm_state_dict in original checkpoint, creating from available normalizer")
                    
                    # Try to get normalizer from different sources
                    normalizer_source = None
                    original_obs_norm = None
                    
                    # First try: runner's obs_normalizer
                    if hasattr(runner, 'obs_normalizer') and runner.obs_normalizer is not None:
                        normalizer_source = "runner.obs_normalizer"
                        original_obs_norm = runner.obs_normalizer.state_dict()
                    # Second try: policy's actor_obs_normalizer 
                    elif hasattr(policy_nn, 'actor_obs_normalizer') and policy_nn.actor_obs_normalizer is not None:
                        normalizer_source = "policy.actor_obs_normalizer"
                        original_obs_norm = policy_nn.actor_obs_normalizer.state_dict()
                    # Third try: extract from original checkpoint's model_state_dict
                    elif any(k.startswith('actor_obs_normalizer.') for k in checkpoint["model_state_dict"].keys()):
                        normalizer_source = "original_checkpoint_actor_normalizer"
                        original_obs_norm = {}
                        for k, v in checkpoint["model_state_dict"].items():
                            if k.startswith('actor_obs_normalizer.'):
                                # Remove the prefix to get the normalizer key
                                norm_key = k.replace('actor_obs_normalizer.', '')
                                original_obs_norm[norm_key] = v
                    
                    if original_obs_norm is not None:
                        try:
                            print(f"[INFO] Using normalizer from: {normalizer_source}")
                            reordered_obs_norm = {}
                            
                            # Apply the same observation permutation to the normalizer statistics
                            for k, v in original_obs_norm.items():
                                if isinstance(v, torch.Tensor) and v.dim() == 1 and v.numel() == obs_perm.numel():
                                    # This is likely a per-observation statistic that needs reordering
                                    reordered_obs_norm[k] = v.index_select(0, obs_perm.to(v.device))
                                    print(f"[INFO] Reordered obs_norm key: {k}")
                                else:
                                    # Keep scalar values as-is
                                    reordered_obs_norm[k] = v
                            
                            schema_checkpoint["obs_norm_state_dict"] = reordered_obs_norm
                            print(f"[INFO] Created and reordered obs_norm_state_dict from {normalizer_source}")
                        except Exception as e:
                            print(f"[WARN] Failed to create obs_norm_state_dict from {normalizer_source}: {e}")
                    else:
                        print(f"[WARN] No normalizer found in runner, policy, or checkpoint")
                
                # Handle privileged_obs_norm_state_dict if needed
                if "privileged_obs_norm_state_dict" in checkpoint:
                    print(f"[INFO] Found privileged_obs_norm_state_dict in original checkpoint")
                    # Preserve the privileged normalizer as-is (may not need reordering)
                    schema_checkpoint["privileged_obs_norm_state_dict"] = checkpoint["privileged_obs_norm_state_dict"]
                    print(f"[INFO] Preserved privileged_obs_norm_state_dict")
                else:
                    # Try to create privileged_obs_norm_state_dict from critic normalizer
                    print(f"[INFO] No privileged_obs_norm_state_dict in original checkpoint, checking for critic normalizer")
                    
                    # Try to get critic normalizer from different sources
                    critic_normalizer_source = None
                    critic_obs_norm = None
                    
                    # First try: extract from original checkpoint's model_state_dict
                    if any(k.startswith('critic_obs_normalizer.') for k in checkpoint["model_state_dict"].keys()):
                        critic_normalizer_source = "original_checkpoint_critic_normalizer"
                        critic_obs_norm = {}
                        for k, v in checkpoint["model_state_dict"].items():
                            if k.startswith('critic_obs_normalizer.'):
                                # Remove the prefix to get the normalizer key
                                norm_key = k.replace('critic_obs_normalizer.', '')
                                critic_obs_norm[norm_key] = v
                    # Second try: policy's critic_obs_normalizer
                    elif hasattr(policy_nn, 'critic_obs_normalizer') and policy_nn.critic_obs_normalizer is not None:
                        critic_normalizer_source = "policy.critic_obs_normalizer"
                        critic_obs_norm = policy_nn.critic_obs_normalizer.state_dict()
                    
                    if critic_obs_norm is not None:
                        try:
                            print(f"[INFO] Using critic normalizer from: {critic_normalizer_source}")
                            # For privileged obs, we might not need reordering if it's state-based
                            # But apply the same reordering to be safe
                            reordered_critic_norm = {}
                            for k, v in critic_obs_norm.items():
                                if isinstance(v, torch.Tensor) and v.dim() == 1 and v.numel() == obs_perm.numel():
                                    reordered_critic_norm[k] = v.index_select(0, obs_perm.to(v.device))
                                    print(f"[INFO] Reordered privileged_obs_norm key: {k}")
                                else:
                                    reordered_critic_norm[k] = v
                            
                            schema_checkpoint["privileged_obs_norm_state_dict"] = reordered_critic_norm
                            print(f"[INFO] Created and reordered privileged_obs_norm_state_dict from {critic_normalizer_source}")
                        except Exception as e:
                            print(f"[WARN] Failed to create privileged_obs_norm_state_dict from {critic_normalizer_source}: {e}")
                    else:
                        # Create empty privileged normalizer as fallback
                        if "obs_norm_state_dict" in schema_checkpoint:
                            print(f"[INFO] Creating privileged_obs_norm_state_dict as copy of obs_norm_state_dict")
                            schema_checkpoint["privileged_obs_norm_state_dict"] = schema_checkpoint["obs_norm_state_dict"].copy()
                        else:
                            print(f"[WARN] No critic normalizer found and no obs_norm_state_dict to copy")
                
                torch.save(schema_checkpoint, runner_ckpt_path)
                print("[INFO] Exported schema-ordered runner checkpoint to:", runner_ckpt_path)
            except Exception as e:
                print(f"[WARN] Failed to export schema-ordered runner checkpoint: {e}")
        else:
            print("[WARN] Could not compute schema joint order mapping; skipping schema-ordered exports.")
    except Exception as e:
        print(f"[WARN] Schema-ordered export failed: {e}")


def import_robot_schema_policy(
    base_env,
    robot_schema_file: str | None,
):
    """Return observation and action remap callables for schema import.

    Returns a tuple: (obs_remap_fn, action_remap_fn). On failure, returns (None, None).
    """
    obs_remap_fn = None
    action_remap_fn = None
    try:
        schema_override = None
        if robot_schema_file:
            with open(robot_schema_file) as f:
                cfg_yaml = yaml.safe_load(f)
            key = "robot_schema_joint_names"
            if key not in cfg_yaml:
                raise KeyError(f"Key '{key}' not found in YAML {robot_schema_file}")
            schema_override = list(cfg_yaml[key])

        import_helper = SchemaImportHelper(base_env, schema_override_names=schema_override)
        if import_helper.compute():
            print("[INFO] Successfully computed schema import mappings")
            # Stabilize types for static analysis
            obs_perm_t = cast(torch.Tensor, import_helper.obs_perm)
            action_perm_t = cast(torch.Tensor, import_helper.action_perm)

            def _obs_remap_fn(obs):
                # TensorDict support
                if isinstance(obs, TensorDictBase):
                    if "policy" in obs.keys():
                        obs_copy = obs.clone()
                        obs_copy["policy"] = obs_copy["policy"].index_select(
                            dim=1, index=obs_perm_t.to(obs_copy["policy"].device)
                        )
                        return obs_copy
                    else:
                        print("[WARN] TensorDict missing 'policy' key; skipping remap")
                        return obs
                # dict-like
                if isinstance(obs, dict):
                    if "policy" in obs:
                        obs_copy = obs.copy()
                        obs_copy["policy"] = obs_copy["policy"].index_select(
                            dim=1, index=obs_perm_t.to(obs_copy["policy"].device)
                        )
                        return obs_copy
                    else:
                        print("[WARN] Dict missing 'policy' key; skipping remap")
                        return obs
                # tensor
                if hasattr(obs, "index_select"):
                    return obs.index_select(dim=1, index=obs_perm_t.to(obs.device))

                print(f"[WARN] Unsupported observation type for remapping: {type(obs)}")
                return obs

            def _action_remap_fn(actions):
                return actions.index_select(dim=1, index=action_perm_t.to(actions.device))

            obs_remap_fn = _obs_remap_fn
            action_remap_fn = _action_remap_fn
            print("[INFO] Schema import remapping functions enabled")
        else:
            print("[WARN] Could not compute schema joint order mapping for import; using original policy without remapping.")
    except Exception as e:
        print(f"[WARN] Schema import failed: {e}")
    return obs_remap_fn, action_remap_fn
