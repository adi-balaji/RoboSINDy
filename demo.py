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

from utils.panda_pushing_env import PandaImageSpacePushingEnv
from utils.visualizers import GIFVisualizer, NotebookVisualizer
from utils.utils import *
from sindy.SINDy import RoboSINDy, SindyDataset, NormalizationTransform, PushingImgSpaceController, img_space_pushing_cost_function
from utils.panda_pushing_env import TARGET_POSE_FREE, TARGET_POSE_OBSTACLES, OBSTACLE_HALFDIMS, OBSTACLE_CENTRE, BOX_SIZE

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
    sindy_model_latent2_path = "trained_models/sindy_model_v2_latent2.pt"
    sindy_model_latent3_path = "trained_models/sindy_model_v4_latent3.pt"
    globoe2c_model_path = "trained_models/global_e2c_3dim_v2.pt"
    num_trials = 3
    num_steps_max = 15
    
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

    ################# LOAD SINDY LATENT 2 MODEL ####################

    sindy_latent2_model = load_sindy_model(sindy_model_latent2_path, latent_dim=2, batch_size=batch_size)
    sindy_latent2_model.xi_coefficients.data = mask_xi_matrix(sindy_latent2_model.xi_coefficients.data)

    print("Xi coefficients:")
    print(sindy_latent2_model.xi_coefficients.data.numpy())

    ################################################################

    ################# VISUALIZE PUSHING TRAJECTORY ####################

    start_state = np.array([0.4, 0.3, -np.pi/2])
    target_state = np.array([0.4, -0.0, 0.0])

    for i in range(num_trials):
        env = PandaImageSpacePushingEnv(visualizer=None, render_non_push_motions=False,  camera_heigh=800, camera_width=800, render_every_n_steps=5, grayscale=True, target_pose_vis=target_state, start_state=start_state)
        env.object_target_pose = env._planar_pose_to_world_pose(target_state)
        state_0 = env.reset()
        controller = PushingImgSpaceController(env, sindy_latent2_model, img_space_pushing_cost_function, normalization_constants, num_samples=50, horizon=10)
        
        state = state_0

        trajectory = []
        goal_reached = False
        for i in range(num_steps_max):

            frame = env._render_image(camera_pos=[0.55, -0.35, 0.2],
                                            camera_orn=[0, -40, 0],
                                            camera_width=env.camera_width,
                                            camera_height=env.camera_height,
                                            distance=1.5)

            frame = frame.transpose(1, 2, 0)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow('Pushing', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            action = controller.control(state)
            state, reward, done, _ = env.step(action)

            # check if we have reached the goal
            end_pose = env.get_object_pos_planar()
            trajectory.append(end_pose)
            goal_distance = np.linalg.norm(end_pose[:2]-target_state[:2]) # evaluate only position, not orientation
            goal_reached = goal_distance < BOX_SIZE/2
            if done or goal_reached:
                break

        trajectory_file = "sindy_latent2_trajs.txt"
        write_traj_to_file(trajectory=trajectory, filename=trajectory_file)
        #########################################################

if __name__ == "__main__":
    main()








