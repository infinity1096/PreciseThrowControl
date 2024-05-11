import sapien.core as sapien
from sapien.utils.viewer import Viewer
import numpy as np
from transforms3d.euler import euler2quat
import numpy as np
import csv 
import tqdm 
from matplotlib import pyplot as plt
from typing import List
from pathlib import Path
from PIL import Image  
import os 


def mass_to_density_sphere(mass, radius=0.02):
    volume = 4/3 * np.pi * radius**3
    return mass/volume

def mass_to_density_box(mass, half_size):
    volume = np.prod(half_size) * 2
    return mass/volume

def create_robot(scene: sapien.Scene, l1, l2, m1,m2, ball_radius=0.04, rod_thickness=0.01, base_size=0.1, debug_visual=False):
    builder: sapien.ArticulationBuilder = scene.create_articulation_builder()
    
    base: sapien.LinkBuilder = builder.create_link_builder()
    base.set_name('base')
    base.add_box_visual(half_size=[base_size]*3, color=[0.5, 0.5, 0.5])
    base.add_box_collision(half_size=[base_size]*3)
    base.set_collision_groups(0,0,0,101)
    # currently have no collision 
    
    link1 = builder.create_link_builder(base)
    link1.set_name('link1')
    link1.add_sphere_collision(radius=ball_radius, pose=sapien.Pose([0, 0, l1]), density=mass_to_density_sphere(m1, ball_radius))
    if debug_visual:
        link1.add_box_visual(half_size=[rod_thickness, rod_thickness, l1], color=[1.0,0.0,0.0])
        link1.add_sphere_visual(radius=ball_radius, color=[1.0,0.0,0.0], pose=sapien.Pose([0, 0, l1]))
    else: 
        link1.add_capsule_visual(pose=sapien.Pose([0,0,0],[0.7071068, 0, 0.7071068, 0]),radius=ball_radius, half_length=l1, color=[0.5,0.5,0.5])
    link1.set_joint_name('joint1')
    link1.set_joint_properties(
        "revolute",
        limits = [[-3*np.pi/2, np.pi/2]],
        pose_in_parent = sapien.Pose([0, 0, base_size]),
        pose_in_child = sapien.Pose([0, 0, -l1])
    )
    link1.set_collision_groups(0,0,0,100)
    
    link2 = builder.create_link_builder(link1)
    link2.set_name('link2')
    link2.add_sphere_collision(radius=ball_radius, pose=sapien.Pose([0, 0, l2]), density=mass_to_density_sphere(m2, ball_radius))
    if debug_visual:
        link2.add_box_visual(half_size=[rod_thickness, rod_thickness, l2], color=[0.0,1.0,0.0])
        link2.add_sphere_visual(radius=ball_radius, color=[0.0,1.0,0.0], pose=sapien.Pose([0, 0, l2]))
    else:
        link2.add_capsule_visual(pose=sapien.Pose([0,0,0],[0.7071068, 0,0.7071068, 0]),radius=ball_radius, half_length=l2, color=[0.8,0.8,0.8])
        link2.add_box_visual(half_size=[ball_radius,ball_radius/2, ball_radius/2], color=[0.5,0.5,0.5], pose=sapien.Pose([0,0,l2+ball_radius]))
        link2.add_box_visual(half_size=[ball_radius/5,ball_radius/2, ball_radius], color=[0.5,0.5,0.5], pose=sapien.Pose([ball_radius,0,l2+ball_radius+ball_radius]))
        link2.add_box_visual(half_size=[ball_radius/5,ball_radius/2, ball_radius], color=[0.5,0.5,0.5], pose=sapien.Pose([-ball_radius,0,l2+ball_radius+ball_radius]))
    
    link2.set_joint_name('joint2')
    link2.set_joint_properties(
        "revolute",
        limits = [[-np.pi, np.pi]],
        pose_in_parent = sapien.Pose([0, 0, l1]),
        pose_in_child = sapien.Pose([0, 0, -l2])
    )
    link2.set_collision_groups(0,0,0,103)
    
    robot = builder.build(fix_root_link=True)
    robot.set_name('robot')
    return robot 

def get_joints_dict(articulation: sapien.Articulation):
    joints = articulation.get_joints()
    joint_names =  [joint.name for joint in joints]
    assert len(joint_names) == len(set(joint_names)), 'Joint names are assumed to be unique.'
    return {joint.name: joint for joint in joints}

def create_throw_object(scene: sapien.Scene, obj_type: str, geom_params:np.ndarray, mesh_dae: str = None, mesh_scale: np.ndarray=np.ones(3), mesh_pose: sapien.Pose = sapien.Pose(), mass: float = 1.0):
    if obj_type == "sword":
        density = mass_to_density_sphere(mass,0.02)
        builder: sapien.ActorBuilder = scene.create_actor_builder()
        builder.add_sphere_collision(pose=sapien.Pose([0,0,-0.25]),radius=0.05, density=density)
        builder.add_box_collision(pose=sapien.Pose([0,0,0]),half_size=[0.01,0.01,0.25], density=1e-10)
        
        if mesh_dae is not None:
            builder.add_visual_from_file(mesh_dae, scale=mesh_scale, pose=mesh_pose)
        else:
            builder.add_box_visual(pose=sapien.Pose([0,0,0]), half_size=[0.01,0.01,0.25], color=[0.0,0.0,1.0])
            builder.add_sphere_visual(pose=sapien.Pose([0,0,-0.25]), radius=0.02, color=[0.0,0.0,1.0])

    elif obj_type == "sphere":
        density = mass_to_density_sphere(mass, geom_params[-1]/2)
        builder: sapien.ActorBuilder = scene.create_actor_builder()
        builder.add_sphere_visual(radius=geom_params[-1], color=[0.0,0.0,1.0])
        builder.add_sphere_collision(radius=geom_params[-1]/2, density=density)
    actor = builder.build()
    actor.set_name("throw_object")
    return actor
    
def create_target(scene: sapien.Scene, pose: sapien.Pose):
    # make a white and red target with square shape
    layer0 = create_box(scene, pose, [0.6, 0.6, 0.1], color=[1, 0, 0], name="target0", static=True)
    layer1 = create_box(scene, pose, [0.5, 0.5, 0.101], color=[1, 1, 1], name="target1", static=True)
    layer2 = create_box(scene, pose, [0.4, 0.4, 0.102], color=[1, 0, 0], name="target2", static=True)
    layer3 = create_box(scene, pose, [0.3, 0.3, 0.103], color=[1, 1, 1], name="target3", static=True)
    layer4 = create_box(scene, pose, [0.2, 0.2, 0.104], color=[1, 0, 0], name="target4", static=True)
    layer5 = create_box(scene, pose, [0.1, 0.1, 0.105], color=[0,0,0], name="target5", static=True)
    return [layer0, layer1, layer2, layer3, layer4, layer5]

def create_box(
    scene: sapien.Scene, pose: sapien.Pose, half_size, color=None, name="", static=False, mesh_file_args=None
) -> sapien.Actor:
    half_size = np.array(half_size)
    builder: sapien.ActorBuilder = scene.create_actor_builder()
    physical_material = scene.create_physical_material(
        static_friction=0.0,
        dynamic_friction=0.0,
        restitution=0.0,
    )
    builder.add_box_collision(half_size=half_size, material=physical_material)  # Add collision shape
    if mesh_file_args is not None:
        mesh_file = mesh_file_args["file"]
        mesh_scale = mesh_file_args["scale"]
        mesh_pose = mesh_file_args["pose"]
        builder.add_visual_from_file(filename=mesh_file, pose=mesh_pose, scale=mesh_scale)
    else:
        builder.add_box_visual(half_size=half_size, color=color)  # Add visual shape
    mass = 0.02
    # make a diagonal inertia matrix
    inertia = np.zeros(3)
    inertia[0] = 2  * mass * (half_size[1] ** 2 + half_size[2] ** 2)
    inertia[1] = 2  * mass * (half_size[0] ** 2 + half_size[2] ** 2)
    inertia[2] = 2  * mass * (half_size[0] ** 2 + half_size[1] ** 2)
    inertia_pose = sapien.Pose([0, 0, 0], [1, 0, 0, 0])
    builder.set_mass_and_inertia(mass, inertia_pose, inertia)
    if not static:
        box: sapien.Actor = builder.build(name=name)
    else:
        box: sapien.Actor = builder.build_static(name=name)
    # Or you can set_name after building the actor
    # box.set_name(name)
    pos = pose[:3]
    rot = pose[3:]
    pose = sapien.Pose(pos, rot)
    box.set_pose(pose)
    return box
    
    
def throw_obj(args, stiffness_1=1000, stiffness_2=1000, damp1=10, damp2=10, view=False):
    take_picture = args.take_picture
    engine = sapien.Engine()
    renderer = sapien.SapienRenderer()
    engine.set_renderer(renderer)

    scene_config = sapien.SceneConfig()
    scene_config.default_static_friction = 10
    scene_config.default_dynamic_friction = 10
    scene_config.default_restitution = 0.1
    scene_config.gravity = [0, 0, -9.81]
    scene = engine.create_scene(scene_config)
    scene.set_timestep(1 / args.frame_per_second)
    scene.add_ground(altitude=0)
    
    manipulator = create_robot(scene, l1=args.l1, l2=args.l2,m1=args.m1,m2=args.m2, ball_radius=args.radius, debug_visual=args.debug_visual)
    manipulator.set_pose(sapien.Pose(p=[0, 0, 1.0]))
    if view:
        viewer = Viewer(renderer)
        viewer.set_scene(scene)

        viewer.set_camera_xyz(x=-4, y=0, z=1)
        viewer.set_camera_rpy(r=0, p=-np.arctan2(2, 2), y=0)
        viewer.window.set_camera_parameters(near=0.05, far=100, fovy=1)
        viewer.focus_entity(manipulator)
        viewer.toggle_pause(True)


    # make camera
    near, far = 0.1, 100
    width, height = 1920, 1080
    camera = scene.add_camera(
        name="camera",
        width=width,
        height=height,
        fovy=np.deg2rad(45),
        near=near,
        far=far,
    )
    camera.set_pose(sapien.Pose(p=[3.5, 2, 2], q=[0,-0.0087265, 0, 0.9999619]))
    
    
    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5], shadow=True)
    
    joints = get_joints_dict(manipulator)
    if args.use_pd:
        joints["joint1"].set_drive_property(stiffness=stiffness_1, damping=damp1)
        joints["joint2"].set_drive_property(stiffness=stiffness_2, damping=damp2)
    else:
        joints["joint1"].set_drive_property(stiffness=0, damping=0)
        joints["joint2"].set_drive_property(stiffness=0, damping=0)
    limits = np.array([[-np.pi, np.pi], [-np.pi, np.pi]])
    steps = 0
    
    target = create_target(scene, np.array([0,4,1.0,1,0,0,0]))
    
    throw_obj_height = args.l2
    geom_params = np.array([0.02,0.02,throw_obj_height])
    xa,xb,k,u = None, None, None, None
    release_time = 0.0
    save_dir = None 
    pd_scale = 1.0
    if args.object_type == "sword":
        throw_obj = create_throw_object(scene, "sword", geom_params, mesh_dae=str(Path(__file__).resolve().parent / "sword_small.dae"), mesh_scale=np.array([0.03,0.03,0.03]), mesh_pose=sapien.Pose([0,0,-throw_obj_height],[ 0.7071068, 0, 0, 0.7071068]))
        xa = str(Path(__file__).resolve().parent / "sword" / "Xa.csv")
        xb = str(Path(__file__).resolve().parent / "sword" / "Xb.csv")
        ks = str(Path(__file__).resolve().parent / "sword" / "K.csv")
        u = str(Path(__file__).resolve().parent / "sword" / "U.csv")
        release_time = 0.725
        save_dir = Path(__file__).resolve().parent / "sword" / "images"
        if not args.use_pd:
            save_dir = Path(__file__).resolve().parent / "sword" / "images_lqr"
        pd_scale = 1.4
        
    elif args.object_type == "sphere":
        throw_obj_height= args.radius   
        geom_params[-1] = throw_obj_height
        throw_obj = create_throw_object(scene, "sphere", geom_params)
        xa = str(Path(__file__).resolve().parent / "ball" / "Xa.csv")
        xb = str(Path(__file__).resolve().parent / "ball" / "Xb.csv")
        ks = str(Path(__file__).resolve().parent / "ball" / "K.csv")
        u = str(Path(__file__).resolve().parent / "ball" / "U.csv")
        release_time = 0.725
        save_dir = Path(__file__).resolve().parent / "ball"/ "images"
        if not args.use_pd:
            save_dir = Path(__file__).resolve().parent / "ball" / "images_lqr"
        pd_scale = 1.3
        
    if take_picture:
        os.makedirs(str(save_dir), exist_ok=True)


    thrown = False    
    link1 = manipulator.get_links()[1]
    link2 = manipulator.get_links()[2]

    throw_obj.set_pose(sapien.Pose([0, (args.l1*2+args.l2*2+geom_params[-1]+0.1), 0.2]))
    
    sticky_gripper = scene.create_drive(throw_obj, sapien.Pose([0,0,-throw_obj_height]), manipulator.get_links()[-1], sapien.Pose([0,0,args.l2+args.radius]))
    sticky_gripper.set_x_properties(1000, 10)
    sticky_gripper.set_y_properties(1000, 10)
    sticky_gripper.set_z_properties(1000, 10)
    
    sticky_gripper.lock_motion(True, True, True,True, True,True)
    
    manipulator.set_qpos([-np.pi/2,0])
    manipulator.set_qvel([0,0])
    manipulator.set_qacc([0,0])
    
    throw_obj.set_angular_velocity([0,0,0])
    throw_obj.set_velocity([0,0,0])

    # while not viewer.closed:
    #     manipulator.set_drive_target([np.pi/2,0])
    #     manipulator.set_drive_velocity_target([0,0])
    #     qf = manipulator.compute_passive_force(external=False)
    #     manipulator.set_qf(qf)
    #     scene.step()
    #     scene.update_render()
    #     viewer.render()  
    
    # all_refs = None 
    # with open(xa,"r") as f:
    #     reader = csv.reader(f)
    #     all_refs = np.array([x for x in reader])
    #     all_refs = all_refs[1:]
        
    # all_ks = None 
    # with open(ks,"r") as f:
    #     reader = csv.reader(f)
    #     all_ks = np.array([x for x in reader])
    #     all_ks = all_ks[1:]
    #     all_ks = all_ks.astype(np.float32)
    #     all_ks = all_ks.reshape(len(all_ks), 2, 4)

    # with open("/home/ycyao/tossing/tossing_final/ball/U_interp.csv","r") as f:
    #     reader = csv.reader(f)
    #     all_controls = np.array([x for x in reader])
    #     all_controls = all_controls[1:]
        
    all_controls = np.loadtxt(u)
    all_ks = np.loadtxt(ks)
    all_refs = np.loadtxt(xa)
    # all_ks2 = all_ks.reshape(-1,4,2)
    all_ks = all_ks.reshape(-1,4,2).transpose(0,2,1)
    
    with open(xb,"r") as f:
        reader = csv.reader(f)
        ball_ref = np.array([x for x in reader])
        ball_ref = ball_ref[1:, :4]
        ball_ref = ball_ref.astype(np.float32)
            
    recorded_joint_positions = []
    recorded_joint_velocities = []
    recorded_obj_positions = []
    recorded_obj_velocities = []
    
    # warm start
    for w_step in range(10):
        qf = manipulator.compute_passive_force(gravity=True,coriolis_and_centrifugal=True,external=False)
        manipulator.set_qf(qf)
        scene.step()
        scene.update_render()
        if take_picture:
            camera.take_picture()
            rgba = camera.get_float_texture("Color")
            rgba_img = (rgba * 255).clip(0, 255).astype("uint8")
            rgba_pil = Image.fromarray(rgba_img)
            rgba_pil.save(str(save_dir / f"{w_step}.png"))

        if view:
            viewer.render()
            
    for steps in range(150):
        joint_speeds = manipulator.get_qvel()
        joint_positions = manipulator.get_qpos()
        # hack: set the first joint to minus pi/2 to compensate the initial pose difference
        joint_positions[0] += np.pi/2
        qf = np.zeros(2)
        if steps < len(all_controls):
            if not args.use_pd:
                row = all_controls[steps]
                joint1_torque = float(row[0])
                joint2_torque = float(row[1])
                qf = manipulator.compute_passive_force(gravity=True,coriolis_and_centrifugal=False,external=False)
                qf[0] += joint1_torque
                qf[1] += joint2_torque
                
                if args.use_tv_lqr:
                    ref = all_refs[steps-1]
                    ref = ref.astype(np.float32)
                    combined_states = np.concatenate([joint_positions, joint_speeds])
                    diff = combined_states - ref
                    k = all_ks[steps-1]
                    qf = np.dot(-k, diff) + qf
                # qf *= 0
                # qf = manipulator.compute_passive_force(gravity=True,external=False)
                manipulator.set_qf(qf)

                # if steps * (1/args.frame_per_second ) >= 0.725 and not thrown:
                if steps * (1/args.frame_per_second ) >= release_time and not thrown:
                    scene.remove_drive(sticky_gripper)
                    thrown = True 
            else:
                qf = manipulator.compute_passive_force(gravity=True,external=False)
                
                target_pos = all_refs[steps][:2].astype(np.float32)
                target_vel = all_refs[steps][2:].astype(np.float32)
                joints["joint1"].set_drive_target(target_pos[0]-np.pi/2)
                joints["joint2"].set_drive_target(target_pos[1])
                joints["joint1"].set_drive_velocity_target(target_vel[0]*pd_scale)
                joints["joint2"].set_drive_velocity_target(target_vel[1]*pd_scale)
                
                manipulator.set_qf(qf)
                # if steps * (1/args.frame_per_second ) >= 0.975 and not thrown:
                if steps * (1/args.frame_per_second ) >= release_time and not thrown:
                    scene.remove_drive(sticky_gripper)
                    thrown = True 
                
            recorded_joint_positions.append(joint_positions)
            recorded_joint_velocities.append(joint_speeds)
            recorded_obj_positions.append(throw_obj.get_pose().p)
            recorded_obj_velocities.append(throw_obj.get_velocity())        
        
        scene.step()
        scene.update_render()
        if take_picture:
            camera.take_picture()
            rgba = camera.get_float_texture("Color")
            rgba_img = (rgba * 255).clip(0, 255).astype("uint8")
            rgba_pil = Image.fromarray(rgba_img)
            rgba_pil.save(str(save_dir / f"{steps+10}.png"))
        
        if view:
            viewer.render()
        if thrown:
            contacts = scene.get_contacts()
            for contact in contacts:
                if (contact.actor0.name == "throw_object" and contact.actor1.name == "ground") or (contact.actor0.name=="throw_object" and contact.actor1 in target):
                    obj_location = throw_obj.get_pose().p
                    target_location = target[0].get_pose().p
                    distance = np.linalg.norm(obj_location-target_location)
                    throw_obj.set_velocity([0,0,0])
                    throw_obj.set_angular_velocity([0,0,0])
                    
 
    recorded_joint_positions = np.array(recorded_joint_positions)
    recorded_joint_velocities = np.array(recorded_joint_velocities)
    
    # plot the reference joints vs the actual joints in the same graph. One line for each joint. Do a graph for positions and another for velocities
    fig, axs = plt.subplots(2, 1, figsize=(20, 10))
    for i in range(2):
        axs[0].plot(recorded_joint_positions[:, i], label=f"Actual Joint {i} Pos")
        axs[0].plot(all_refs[:, i], label=f"Reference Joint {i} Pos")
        axs[1].plot(recorded_joint_velocities[:, i], label=f"Actual Joint {i} Vel")
        axs[1].plot(all_refs[:, i+2], label=f"Reference Joint {i} Vel")
    axs[0].legend()
    axs[1].legend()
    fig.suptitle("Joint positions and velocities")
    # save as pdf
    plt.savefig(str(save_dir / "joint_positions_velocities.pdf"))
    
    
def parse_args():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--l1", type=float, default=0.25)
    parser.add_argument("--l2", type=float, default=0.25)
    parser.add_argument("--m1", type=float, default=5)
    parser.add_argument("--m2", type=float, default=5)
    parser.add_argument("-ot", "--object_type", type=str, default="sphere")
    parser.add_argument("--m_obj", type=float, default=1.0)
    parser.add_argument("-r", "--radius", type=float, default=0.04)
    parser.add_argument("-dv", "--debug_visual", action="store_true")
    parser.add_argument("-fps", "--frame_per_second", type=int, default=40)
    parser.add_argument("-pd", "--use_pd", action="store_true")
    parser.add_argument("-tvlqr", "--use_tv_lqr", action="store_true")
    parser.add_argument("-tp", "--take_picture", action="store_true")
    parser.add_argument("-v", "--view", action="store_true")
    args = parser.parse_args()
    return args

                
if __name__ == '__main__':
    stiffness_candidates = list(range(200,400,20))
    damping_candidates = list(range(1,5,1))
    args = parse_args()
    best_st1, best_st2, best_dp1, best_dp2 = 2000,2000,600,600

    throw_obj(args, best_st1, best_st2, best_dp1, best_dp2, view=args.view)