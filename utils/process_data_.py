# From HW 5 learning_latent_dynamics.py

import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
# from mppi import MPPI
# from panda_pushing_env import TARGET_POSE_FREE, TARGET_POSE_OBSTACLES, OBSTACLE_HALFDIMS, OBSTACLE_CENTRE, BOX_SIZE

# TARGET_POSE_FREE_TENSOR = torch.as_tensor(TARGET_POSE_FREE, dtype=torch.float32)
# TARGET_POSE_OBSTACLES_TENSOR = torch.as_tensor(TARGET_POSE_OBSTACLES, dtype=torch.float32)
# OBSTACLE_CENTRE_TENSOR = torch.as_tensor(OBSTACLE_CENTRE, dtype=torch.float32)[:2]
# OBSTACLE_HALFDIMS_TENSOR = torch.as_tensor(OBSTACLE_HALFDIMS, dtype=torch.float32)[:2]


def collect_data_random_trajectory(env, num_trajectories=1000, trajectory_length=10):
    """
    Collect data from the provided environment using uniformly random exploration.
    :param env: Gym Environment instance.
    :param num_trajectories: <int> number of data to be collected.
    :param trajectory_length: <int> number of state transitions to be collected
    :return: collected data: List of dictionaries containing the state-action trajectories.
    Each trajectory dictionary should have the following structure:
        {'states': states,
        'actions': actions}
    where
        * states is a numpy array of shape (trajectory_length+1, 32, 32, num_channels) containing the states [x_0, ...., x_T]
        * actions is a numpy array of shape (trajectory_length, actions_size) containing the actions [u_0, ...., u_{T-1}]
    Each trajectory is:
        x_0 -> u_0 -> x_1 -> u_1 -> .... -> x_{T-1} -> u_{T_1} -> x_{T}
        where x_0 is the state after resetting the environment with env.reset()
    Our states should be of type np.uint8, our actions of type np.float32.
    """
    collected_data = None
    # --- Your code here
    collected_data = []
    for i in range(num_trajectories):
      x0 = env.reset()
      states = np.zeros([trajectory_length+1, x0.shape[0], x0.shape[1], x0.shape[2]], dtype=np.uint8)
      actions = np.zeros([trajectory_length, 3], dtype=np.float32)
      states[0] = x0
      for j in range(trajectory_length):
        u_j = env.action_space.sample()
        x_j, reward, done, info = env.step(u_j)
        states[j+1] = x_j
        actions[j] = u_j
        if (done):
          break
      traj = {'states': states, 'actions': actions}
      collected_data.append(traj)
   

      


    # ---
    return collected_data


class NormalizationTransform(object):

    def __init__(self, norm_constants):
        self.norm_constants = norm_constants
        self.mean = norm_constants['mean']
        self.std = norm_constants['std']

    def __call__(self, sample):
        """
        Transform the sample by normalizing the 'states' using the provided normalization constants.
        :param sample: dictionary containing {'states', 'actions'}
        :return:
        """
        # --- Your code here
        # print("_call_")
        # sample['states'] = (sample['states'] - self.mean) / self.std
        sample['states'] = self.normalize_state(sample['states'])

        # print("_call_ return")
        # ---
        return sample

    def inverse(self, sample):
        """
        Transform the sample by de-normalizing the 'states' using the provided normalization constants.
        :param sample: dictionary containing {'states', 'actions'}
        :return:
        """
        # --- Your code here
        # sample['states'] = sample['states'] * self.std + self.mean
        sample['states'] = self.denormalize_state(sample['states'])

        # ---
        return sample

    def normalize_state(self, state):
        """
        Normalize the state using the provided normalization constants.
        :param state: <torch.tensor> of shape (..., num_channels, 32, 32)
        :return: <torch.tensor> of shape (..., num_channels, 32, 32)
        """
        # --- Your code here
        # print(state.shape)
        state = (state - self.mean) / self.std

        # ---
        return state

    def denormalize_state(self, state_norm):
        """
        Denormalize the state using the provided normalization constants.
        :param state_norm: <torch.tensor> of shape (..., num_channels, 32, 32)
        :return: <torch.tensor> of shape (..., num_channels, 32, 32)
        """
        # --- Your code here
        state = state_norm * self.std + self.mean


        # ---
        return state
    

def process_data(collected_data, batch_size):

    train_data = None
    val_data = None
    normalization_constants = {
        'mean': None,
        'std': None,
    }

    dataset = MultiStepDynamicsDataset(collected_data, 1)
    
    train_data, val_data = torch.utils.data.random_split(dataset, [0.8, 0.2])

    states = []
    for dictionary in train_data.dataset:
      for state in dictionary['states']:
        states.append(state)


    states_np = np.array(states, dtype=np.float32)
    mean = np.mean(states_np)
    std = np.std(states_np)
    normalization_constants['mean'] = torch.tensor(mean)
    normalization_constants['std'] = torch.tensor(std)

    norm_tr = NormalizationTransform(normalization_constants)
    train_data.dataset.transform = norm_tr
    val_data.dataset.transform = norm_tr

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)

    return train_loader, val_loader, normalization_constants

def process_data_multiple_step(collected_data, batch_size=500, num_steps=4):
    """
    Process the collected data and returns a DataLoader for train and one for validation.
    The data provided is a list of trajectories (like collect_data_random output).
    Each DataLoader must load dictionary as
    {'states': x_t,x_{t+1}, ... , x_{t+num_steps}
     'actions': u_t, ..., u_{t+num_steps-1},
    }
    where:
     states: torch.float32 tensor of shape (batch_size, num_steps+1, state_size)
     actions: torch.float32 tensor of shape (batch_size, num_steps, state_size)

    Each DataLoader must load dictionary dat
    The data should be split in a 80-20 training-validation split.

    :param collected_data:
    :param batch_size: <int> size of the loaded batch.
    :param num_steps: <int> number of steps to load the multistep data.

    :return train_loader: <torch.utils.data.DataLoader> for training
    :return val_loader: <torch.utils.data.DataLoader> for validation
    :return normalization_constants: <dict> containing the mean and std of the states.

    Hints:
     - Pytorch provides data tools for you such as Dataset and DataLoader and random_split
     - You should implement MultiStepDynamicsDataset below.
        This class extends pytorch Dataset class to have a custom data format.
    """
    train_data = None
    val_data = None
    normalization_constants = {
        'mean': None,
        'std': None,
    }
    # Your implemetation needs to do the following:
    #  1. Initialize dataset
    #  2. Split dataset,
    #  3. Estimate normalization constants for the train dataset states.
    # --- Your code here
    dataset = MultiStepDynamicsDataset(collected_data, num_steps)
    
    train_data, val_data = torch.utils.data.random_split(dataset, [0.8, 0.2])

    states = []
    for dictionary in train_data.dataset:
      for state in dictionary['states']:
        states.append(state)


    states_np = np.array(states, dtype=np.float32)
    mean = np.mean(states_np)
    std = np.std(states_np)
    normalization_constants['mean'] = torch.tensor(mean)
    normalization_constants['std'] = torch.tensor(std)

    # ---
    norm_tr = NormalizationTransform(normalization_constants)
    train_data.dataset.transform = norm_tr
    val_data.dataset.transform = norm_tr

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)

    return train_loader, val_loader, normalization_constants



class MultiStepDynamicsDataset(Dataset):
    """
    Dataset containing multi-step dynamics data.
    Each data sample is a dictionary containing (state, action, next_state) in the form:
    {'states':[x_{t}, x_{t+1},..., x_{t+num_steps} ] -- initial state of the multistep torch.float32 tensor of shape (state_size,)
     'actions': [u_t,..., u_{t+num_steps-1}] -- actions applied in the multi-step.
                torch.float32 tensor of shape (num_steps, action_size)
    }

    Observation: If num_steps=1, this dataset is equivalent to SingleStepDynamicsDataset.
    """

    def __init__(self, collected_data, num_steps=4, transform=None):
        self.data = collected_data
        self.trajectory_length = self.data[0]['actions'].shape[0] - num_steps + 1
        self.num_steps = num_steps
        self.transform = transform

    def __len__(self):
        return len(self.data) * self.trajectory_length

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

    def __getitem__(self, item):
        """
        Return the data sample corresponding to the index <item>.
        :param item: <int> index of the data sample to produce.
            It can take any value in range 0 to self.__len__().
        :return: data sample corresponding to encoded as a dictionary with keys (states, actions).
        The class description has more details about the format of this data sample.
        """
        sample = {
            'states': None,
            'actions': None,
        }
        # --- Your code here
        data_ind = item // self.trajectory_length
        traj_ind = item % self.trajectory_length
        
        sample['states'] = torch.tensor(self.data[data_ind]['states'][traj_ind:traj_ind+self.num_steps+1], dtype=torch.float32).permute([0,3,1,2])
        sample['actions'] = torch.tensor(self.data[data_ind]['actions'][traj_ind:traj_ind+self.num_steps], dtype=torch.float32)
        
        if self.transform is not None:
          sample = self.transform(sample)
        # ---
        return sample