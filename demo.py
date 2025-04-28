import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import cv2
import pybullet as p
import ast
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from utils.panda_pushing_env import PandaImageSpacePushingEnv
from utils.visualizers import GIFVisualizer, NotebookVisualizer
from utils.utils import *
from sindy.SINDy import RoboSINDy, SindyDataset, NormalizationTransform, PushingImgSpaceController, img_space_pushing_cost_function
from e2c.E2C import RoboE2C, GloboE2C, PushingImgSpaceController_E2C
from utils.panda_pushing_env import TARGET_POSE_FREE, TARGET_POSE_OBSTACLES, OBSTACLE_HALFDIMS, OBSTACLE_CENTRE, BOX_SIZE
from utils.process_data_ import process_data

# def visualize_trajectories(txt_file, start_state, target_state):
#     # Load trajectories from file
#     trajectories = []
#     with open(txt_file, 'r') as f:
#         for line in f:
#             s = line.strip()
#             if not s:
#                 continue
#             # Convert array([...]) to list [...]
#             s = s.replace('array([', '[').replace('])', ']')
#             pts = ast.literal_eval(s)  # list of [x, y, θ]
#             traj = np.array(pts)       # shape (T, 3)

#             # Prepend the start pose
#             start_pose = start_state
#             traj = np.vstack([start_pose, traj])

#             trajectories.append(traj)

#     # Plot
#     fig, ax = plt.subplots()

#     size = 0.1
#     x0, y0 = start_state[0], start_state[1]
#     goal_x, goal_y = target_state[0], target_state[1]
#     rect = Rectangle((x0 - size/2, y0 - size/2), size, size,
#                      linewidth=1, edgecolor='black', facecolor='none')
#     goal_rect = Rectangle((goal_x - size/2, goal_y - size/2), size, size,
#                           linewidth=1, edgecolor='green', facecolor='none')
#     ax.add_patch(rect)
#     ax.add_patch(goal_rect)

#     for traj in trajectories:
#         if (traj[-1, 1] >= 0.07 or traj[-1,1] <= -0.03) or (traj[-1, 0] >= 0.46 or traj[-1,0] <= 0.34):
#             ax.plot(traj[:, 0], traj[:, 1], color='red', linewidth=1.0)
#             ax.plot(traj[:, 0], traj[:, 1], color='lightblue', linewidth=1.0, alpha=0.8)
#         else:
#             ax.plot(traj[:, 0], traj[:, 1], color='lightblue', linewidth=1.0)

#     deviations = []
#     for traj in trajectories:
#         nominal_traj_x = np.linspace(0.4, 0.4, num=traj.shape[0])
#         nominal_traj_y = np.linspace(0.3, 0.05, num=traj.shape[0])
#         nominal_traj = np.vstack([nominal_traj_x, nominal_traj_y]).T

#         dists = np.linalg.norm(traj[:, :2] - nominal_traj, axis=1)

#         deviations.append(np.mean(dists))

#     average_deviation = np.mean(deviations)
#     print(f"Average deviation from nominal path {txt_file}: {average_deviation:.4f}\n")

#     ax.set_aspect('equal', 'box')
#     ax.set_xlabel('X (m)')
#     ax.set_xlim(0.1, 0.7)
#     ax.set_ylim(-0.1, 0.5)
#     ax.set_ylabel('Y (m)')
    
#     plt.show(block=False)
#     plt.pause(3)
#     plt.close()

def load_sindy_model(model_path, latent_dim, batch_size):
    """
    Load the SINDy model from the specified path.
    """
    sindy_model = RoboSINDy(input_dim=32*32*1, batch_size=batch_size, latent_dim=latent_dim)
    sindy_model.load_state_dict(torch.load(model_path))
    sindy_model.eval()
    print(f"SINDY Model Loaded from {model_path}")
    return sindy_model

def main():

    ################## CONSTANTS ############################
    batch_size = 64
    val_fraction = 0.2
    current_working_dir = os.getcwd()
    sindy_model_latent2_path = current_working_dir + "/trained_models/sindy_model_v2_latent2.pt"
    sindy_model_latent3_path = current_working_dir + "/trained_models/sindy_model_v4_latent3.pt"
    globoe2c_model_path = current_working_dir + "/trained_models/global_e2c_3dim_v2.pt"
    roboe2c_path = current_working_dir + "/trained_models/robo_e2c_3dim_v2.pt"
    num_steps_max = 15

    start_states = [
        np.array([0.4, 0.0, 0.0]),
        # np.array([0.4, 0.3, -np.pi/2]),
        np.array([0.3, 0.3, np.pi]),
        # np.array([0.6, 0.1, -np.pi/4]),
        # np.array([0.4, 0.3, -np.pi/2]),
    ]
    target_states = [
        np.array([0.7, -0.0, 0.0]),
        # np.array([0.4, -0.0, 0.0]),
        np.array([0.0, 0.3, 0.0]),
        # np.array([0.8, -0.0, 0.0]),
        # np.array([0.3, -0.1, -np.pi/2]),
    ]
    
    #########################################################

    ################### CONSTRUCT DATASET ###################

    dataset_path = "datasets/collected_data_large_push.npy"
    dt = 1/240.0 # time step in pybullet


    data_npy = np.load(dataset_path, allow_pickle=True)

    samples = []
    for item in data_npy:
        
        states = item['states']
        actions = item['actions']
        state_derivatives = []
        for i in range(1, len(states)-1):
            state_derivative = (states[i+1] - states[i-1]) / dt
            state_derivatives.append(state_derivative)
        state_derivatives = np.array(state_derivatives)

        for i, state in enumerate(states[1:-1]):
            sample = {
                'state': state,
                'action': actions[i],
                'state_derivative': state_derivatives[i]
            }
            samples.append(sample)

    dataset = SindyDataset(data=samples)

    val_size = int(val_fraction * len(dataset))
    train_size = len(dataset) - val_size
    train_data, val_data = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_data, batch_size=batch_size)
    val_loader = DataLoader(val_data, batch_size=batch_size)

    tot_train_states = []
    tot_train_state_derivatives = []
    for i in range(len(train_loader.dataset)):
        s = train_loader.dataset[i]['states']   
        sd = train_loader.dataset[i]['state_derivatives']  
        tot_train_states.append(s.unsqueeze(0))
        tot_train_state_derivatives.append(sd.unsqueeze(0))
    tot_train_states = torch.cat(tot_train_states,dim=0)  
    tot_train_state_derivatives = torch.cat(tot_train_state_derivatives, dim=0)  # (N,C,H,W)

    mean_s = tot_train_states .mean(dim=(0,2,3))
    std_s = tot_train_states .std( dim=(0,2,3))
    mean_sd = tot_train_state_derivatives.mean(dim=(0,2,3))
    std_sd = tot_train_state_derivatives.std( dim=(0,2,3))

    normalization_constants = {
    'mean_state': mean_s,              
    'std_state':  std_s,               
    'mean_state_derivative': mean_sd,  
    'std_state_derivative': std_sd,    
    }

    norm_tr = NormalizationTransform(normalization_constants)
    train_data.dataset.transform = norm_tr
    val_data.dataset.transform = norm_tr

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)

    ################################################

    ################# LOAD SINDY LATENT 2 MODEL AND RUN DEMO ####################

    sindy_latent2_model = load_sindy_model(sindy_model_latent2_path, latent_dim=2, batch_size=batch_size)
    sindy_latent2_model.xi_coefficients.data = mask_xi_matrix(sindy_latent2_model.xi_coefficients.data)

    print("Xi coefficients:")
    print(sindy_latent2_model.xi_coefficients.data.numpy())

    for i in range(len(start_states)):
        start_state = start_states[i]
        target_state = target_states[i]

        
        env = PandaImageSpacePushingEnv(visualizer=None, render_non_push_motions=False,  camera_heigh=800, camera_width=800, render_every_n_steps=5, grayscale=True, target_pose_vis=target_state, start_state=start_state, model_label="RoboSINDy 2D")
        env.object_target_pose = env._planar_pose_to_world_pose(target_state)
        state_0 = env.reset()
        controller = PushingImgSpaceController(env, sindy_latent2_model, img_space_pushing_cost_function, normalization_constants, num_samples=200, horizon=20)
        
        state = state_0

        trajectory = []
        goal_reached = False
        for i in range(num_steps_max):

            action = controller.control(state)
            state, reward, done, _ = env.step(action)

            # check if we have reached the goal
            end_pose = env.get_object_pos_planar()
            trajectory.append(end_pose)
            goal_distance = np.linalg.norm(end_pose[:2]-target_state[:2]) # evaluate only position, not orientation
            goal_reached = goal_distance < BOX_SIZE/1.5
            if done or goal_reached:
                break

    ##################################################################################################################################

    #breathe!! stop and smell the flowers!
    time.sleep(2)

    ################# LOAD SINDY LATENT 3 MODEL AND RUN DEMO ####################

    sindy_latent2_model = load_sindy_model(sindy_model_latent3_path, latent_dim=3, batch_size=batch_size)
    sindy_latent2_model.xi_coefficients.data = mask_xi_matrix(sindy_latent2_model.xi_coefficients.data)

    print("Xi coefficients:")
    print(sindy_latent2_model.xi_coefficients.data.numpy())

    for i in range(len(start_states)):
        start_state = start_states[i]
        target_state = target_states[i]

        
        env = PandaImageSpacePushingEnv(visualizer=None, render_non_push_motions=False,  camera_heigh=800, camera_width=800, render_every_n_steps=5, grayscale=True, target_pose_vis=target_state, start_state=start_state, model_label="RoboSINDy 3D")
        env.object_target_pose = env._planar_pose_to_world_pose(target_state)
        state_0 = env.reset()
        controller = PushingImgSpaceController(env, sindy_latent2_model, img_space_pushing_cost_function, normalization_constants, num_samples=200, horizon=20)
        
        state = state_0

        trajectory = []
        goal_reached = False
        for i in range(num_steps_max):

            action = controller.control(state)
            state, reward, done, _ = env.step(action)

            # check if we have reached the goal
            end_pose = env.get_object_pos_planar()
            trajectory.append(end_pose)
            goal_distance = np.linalg.norm(end_pose[:2]-target_state[:2]) # evaluate only position, not orientation
            goal_reached = goal_distance < BOX_SIZE/1.5
            if done or goal_reached:
                break

    #####################################################################################################################################################

    #breathe!! stop and smell the flowers!
    time.sleep(2)

    ########################################### LOAD GLOBOE2C MODEL AND RUN DEMO ###################################################################

    globo_model = GloboE2C(latent_dim=3, action_dim=3, num_channels=1)

    globo_model.load_state_dict(torch.load(globoe2c_model_path,weights_only=True))

    train_loader,val_loader,norm_constants = process_data(data_npy, batch_size)
    
    for i in range(len(start_states)):
        start_state = start_states[i]
        target_state = target_states[i]

        
        env = PandaImageSpacePushingEnv(visualizer=None, render_non_push_motions=False,  camera_heigh=800, camera_width=800, render_every_n_steps=5, grayscale=True, target_pose_vis=target_state, start_state=start_state, model_label="GloboE2C")
        env.object_target_pose = env._planar_pose_to_world_pose(target_state)
        state_0 = env.reset()
        controller = PushingImgSpaceController_E2C(env, globo_model, img_space_pushing_cost_function, norm_constants, num_samples=200, horizon=20)
        
        state = state_0

        trajectory = []
        goal_reached = False
        for i in range(num_steps_max):

            action = controller.control(state)
            state, reward, done, _ = env.step(action)

            # check if we have reached the goal
            end_pose = env.get_object_pos_planar()
            trajectory.append(end_pose)
            goal_distance = np.linalg.norm(end_pose[:2]-target_state[:2]) # evaluate only position, not orientation
            goal_reached = goal_distance < BOX_SIZE/1.5
            if done or goal_reached:
                break



    #####################################################################################################################################################

    #breathe!! stop and smell the flowers!
    time.sleep(2)    

    ########################################### LOAD ROBOE2C MODEL AND RUN DEMO ###################################################################

    roboe2c_model = RoboE2C(latent_dim=3, action_dim=3, num_channels=1)

    roboe2c_model.load_state_dict(torch.load(roboe2c_path,weights_only=True))

    # train_loader,val_loader,norm_constants = process_data(data_npy, batch_size)
    
    for i in range(len(start_states)):
        start_state = start_states[i]
        target_state = target_states[i]

        
        env = PandaImageSpacePushingEnv(visualizer=None, render_non_push_motions=False,  camera_heigh=800, camera_width=800, render_every_n_steps=5, grayscale=True, target_pose_vis=target_state, start_state=start_state, model_label="RoboE2C")
        env.object_target_pose = env._planar_pose_to_world_pose(target_state)
        state_0 = env.reset()
        controller = PushingImgSpaceController_E2C(env, roboe2c_model, img_space_pushing_cost_function, norm_constants, num_samples=200, horizon=20)
        
        state = state_0

        trajectory = []
        goal_reached = False
        for i in range(num_steps_max):

            action = controller.control(state)
            state, reward, done, _ = env.step(action)

            # check if we have reached the goal
            end_pose = env.get_object_pos_planar()
            trajectory.append(end_pose)
            goal_distance = np.linalg.norm(end_pose[:2]-target_state[:2]) # evaluate only position, not orientation
            goal_reached = goal_distance < BOX_SIZE/1.5
            if done or goal_reached:
                break

    #####################################################################################################################################################

    print("\nThe demo for all models have been run successfully.\n")

    print("*********** Thank you :) ***********\n")

        
        

if __name__ == "__main__":
    main()








