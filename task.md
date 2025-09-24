The task is as follows. 
Currently we export a jit and onnx neural networks from our environments. These NN map observations to actions; The observation and actions are defined in velocity_env_cfg.py.

Important note is, that these actions and observation depend on some ordering of the joints. For instance, joint_pos is the joint position vector for all the joints. see 
    @property
    def joint_names(self) -> list[str]:
        """Ordered names of joints in articulation."""
        return self._root_newton_view.joint_dof_names

in articulation.py for the method that returns the joint order in the simulator.

The same ordering applies to joint_vel, actions in the observation terms. The actions terms are simply actions similar to the last term in the observation but still it depends on this joint ordering. 

Our policies are exported from observation>actions based on this joint order. 
What I was is to use a different order "just for exporting the policy". This should not modify the code outside of play.py in anyway. It should not affect training. Only when we export the policy I want to use a different order. 
This seems to require rearranging the NN itself. 

So here assume we have some joint odering. This ordering is read from the IsaacRobotSchema. For instance, for UR10e robot one can fine joint order from 


from pxr import Usd, UsdGeom
import omni.usd
#For Legacy reasons, we need to import the schema from the usd.schema.isaac package
from usd.schema.isaac import robot_schema
stage = omni.usd.get_context().get_stage()
prim = stage.GetPrimAtPath("/World/ur10e")
robot_tree = robot_schema.utils.GenerateRobotLinkTree(stage, prim)
robot_schema.utils.PrintRobotTree(robot_tree)
links = robot_schema.utils.GetAllRobotLinks(stage, prim, False)
joints = robot_schema.utils.GetAllRobotJoints(stage, prim, False)
print(joints)


So the task here is two thing; Find if the robot schema provides any joint ordering for the robot that is trained. If so, read that joint order, use that joint order to reorganize the policy and export a policy that is consistent with the robot schema joint order. 


@play.py @play.py 
I have two play scripts here
These are playing RL policies for two different engines. Physx and Newton.
This is still not fully developed. 
The background idea is in @task.md. high-level idea is that we want to use robot schema as an intermediate joint order to export and import policies;
meaning EngineA -> Engine B is replaced with Engine A > robot-schema > EngineB.
This means for each play script we needed to "export" the engine policy to schema format (which is done right  now for both engines). But also, we need to import from an arbitrary representation like robot-schema and convert to the engine playing the policy.
If you notice, to test the implementation I enabled exporting based on a given YAML files; essentially set robot-representation as engine-B representation to test the exporting functionality. Now I want to implement the other side of this, meaing the importing functionality. 

Assume we have a policy, and the policy provides actions and observation int he schema represeantation. We need to remap back to the playing engine now. You can use the exisiting schema related function for reading the joint order from robot schema. However, also implement a functionality to import from a yaml file joint order for testing purposes. 

The way I am going to test this is then as follows: 
In engine A I am going to rely on identity mapping (meaning robot schema is assumed to be the same as the engine A joint order). Then I have a policy that needs to be remapped to the current engine representation

This could be as simple as @rsl_rl_transfer.py, i.e. remap obs and remap actions based on the mapping that implement.





======================================= Direct Envs
@policy_mapping_helpers.py @play.py @policy_mapping_helpers.py @play.py 
if you go over these files you will see that they provide Imort/export of policies from other engine/project. physx from lab and lab from physx.
_get_observation_terms_info however, work with manager-based environments; i.e.  envs in source/isaaclab_tasks/isaaclab_tasks/manager_based (for both projects)
I want to enable this sim2sim transfer of policies for direct envs as well i.e. the ones in source/isaaclab_tasks/isaaclab_tasks/direct (for both projects)
specifically, I am interested in this env 
source/isaaclab_tasks/isaaclab_tasks/direct/inhand_manipulation/inhand_manipulation_env.py
for example the observation function is 
compute_full_observations which includes some joint related  terms that need mapping 
 unscale(self.hand_dof_pos, self.hand_dof_lower_limits, self.hand_dof_upper_limits),
                self.cfg.vel_obs_scale * self.hand_dof_vel,
for manager based env this was easy to deduce (for instance here based on KinematicObsGroupCfg you could check the type of the term and see if this is joint-related term or not)

Now the task is to ensure the sim2sim pipeline works for direct env as well

@policy_mapping_helpers.py @policy_mapping_helpers.py 
you put together the changes in these files to address the question in 
The policy can be exported on the IL-physX side and be imported on the IsaacLab side but the policy doesn't look correct. This indicates something isn't setup correctly.
what I do on the IL-PhysX side is to "conda activate IL" and then run 
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py --task=Isaac-Repose-Cube-Allegro-Direct-v0 --num_envs=32 --export_robot_schema_policy --robot_schema_file ../IsaacLab/scripts/newton_sim2sim/mappings/sim2sim_alegro.yaml
this exports the policy
later on the IsaacLAb side what I do is "conda deactivate" and then run 
clear; ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py  --task=Isaac-Repose-Cube-Allegro-Direct-v0 --num_envs=32 --checkpoint ../IL-PhysX/logs/rsl_rl/allegro_hand/2025-09-19_17-54-35/exported/policy_runner_schema_order.pt --import_robot_schema_policy --robot_schema_file ../IsaacLab/scripts/newton_sim2sim/mappings/sim2sim_alegro.yaml
use this information to debug the problem. 
One thing that does look suspicious to me is the use of 
 and  .
Maybe print the indices to make sure it is what you expect? The observations seem to be using hand_dof_pos/vel etc but the actions seems like use indexing of actuated_dof_indices which I don't understand if it has any effect. 

