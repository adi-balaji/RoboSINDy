"""
Originally adapted from HW5

Embed to Control (E2C) Implementation

"""
import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from utils.mppi import MPPI
from utils.panda_pushing_env import TARGET_POSE_FREE, TARGET_POSE_OBSTACLES, OBSTACLE_HALFDIMS, OBSTACLE_CENTRE, BOX_SIZE

TARGET_POSE_FREE_TENSOR = torch.as_tensor(TARGET_POSE_FREE, dtype=torch.float32)
TARGET_POSE_OBSTACLES_TENSOR = torch.as_tensor(TARGET_POSE_OBSTACLES, dtype=torch.float32)
OBSTACLE_CENTRE_TENSOR = torch.as_tensor(OBSTACLE_CENTRE, dtype=torch.float32)[:2]
OBSTACLE_HALFDIMS_TENSOR = torch.as_tensor(OBSTACLE_HALFDIMS, dtype=torch.float32)[:2]


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


class VAELoss(nn.Module):
    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta  # Weight of the KL divergence term

    def forward(self, x_hat, x, mu, logvar):
        """
        Compute the VAE loss.
        vae_loss = MSE(x, x_hat) + beta * KL(N(\mu, \sigma), N(0, 1))
        where KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param x: <torch.tensor> ground truth tensor of shape (batch_size, state_size)
        :param x_hat: <torch.tensor> reconstructed tensor of shape (batch_size, state_size)
        :param mu: <torch.tensor> of shape (batch_size, state_size)
        :param logvar: <torch.tensor> of shape (batch_size, state_size)
        :return: <torch.tensor> scalar loss
        """
        loss = None
        # --- Your code here
        std = torch.exp(0.5 * logvar)
        kl = torch.log(1.0/std) + (std*std + mu*mu)/2.0 - 0.5
        kl_loss = kl.sum(dim=1).mean()
        mse = F.mse_loss(x_hat,x)
        loss = mse + self.beta * kl_loss

        # ---
        return loss


# TODO: change to ELBO
class MultiStepLoss(nn.Module): 
    def __init__(self, state_loss_fn, latent_loss_fn, alpha=0.1):
        super().__init__()
        self.state_loss = state_loss_fn
        self.latent_loss = latent_loss_fn
        self.alpha = alpha

    def forward(self, model, states, actions):
        """
        Compute the multi-step loss resultant of multi-querying the model from (state, action) and comparing the predictions with targets.
        Here the loss is computed based on 3 terms:
         - reconstruction loss: enforces good encoding and reconstruction of the states (no dynamics).
         - latent loss: enforces dynamics on latent space by matching the state encodings with the dynamics on latent space.
         - prediction loss: enforces reconstruction of the predicted states by matching the predictions with the targets.

         :param model: <nn.Module> model to be trained.
         :param states: <torch.tensor> tensor of shape (batch_size, traj_size + 1, state_size)
         :param actions: <torch.tensor> tensor of shape (batch_size, traj_size, action_size)
        """
        # compute reconstruction loss -- compares the encoded-decoded states with the original states
        rec_loss = 0.
        # --- Your code here
        latent_values = model.encode(states)
        decoded_states = model.decode(latent_values)

        # mse = 0.0
        # for i in range(states.shape[1]):
        #   mse += self.state_loss(decoded_states[:,i], states[:,i])
        # rec_loss = mse / float(states.shape[1])
        rec_loss = self.state_loss(decoded_states, states)


        # ---
        # propagate dynamics on latent space as well as reconstructed states
        pred_latent_values = []
        pred_states = []
        prev_z = latent_values[:, 0, :]  # get initial latent value
        prev_state = states[:, 0, :]  # get initial state value
        for t in range(actions.shape[1]):
            next_z = None
            next_state = None
            # --- Your code here
            next_z = model.latent_dynamics(prev_z, actions[:,t])
            pred_latent_values.append(next_z)
            next_state = model(prev_state, actions[:,t])
            pred_states.append(next_state)

            # ---
            prev_z = next_z
            prev_state = next_state
        pred_states = torch.stack(pred_states, dim=1)
        pred_latent_values = torch.stack(pred_latent_values, dim=1)
        # compute prediction loss -- compares predicted state values with the given states
        pred_loss = 0.
        # --- Your code here
        # mse = 0.0
        # for i in range(1, states.shape[1]):
        #   mse += self.state_loss(pred_states[:,i-1], states[:,i])
        # pred_loss = mse / float(states.shape[1]-1)
        pred_loss = self.state_loss(pred_states, states[:,1:])
        # ---

        # compute latent loss -- compares predicted latent values with the encoded latent values for states
        lat_loss = 0.
        # --- Your code here
        # mse = 0.0
        # for i in range(1, states.shape[1]):
        #   mse += self.latent_loss(pred_latent_values[:,i-1], latent_values[:,i])
        #   # mse += self.state_loss(decoded_states[i], states[i])
        # lat_loss = mse / float(states.shape[1]-1)

        lat_loss = self.latent_loss(pred_latent_values, latent_values[:,1:])
        # ---

        multi_step_loss = rec_loss + pred_loss + self.alpha * lat_loss

        return multi_step_loss


class StateEncoder(nn.Module):
    """
    Embeds the state into a latent space.
    State shape: (..., num_channels, 32, 32)
    latent shape: (..., latent_dim)
    Check the notebook for more details about the architecture.
    """

    def __init__(self, latent_dim, num_channels=3):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_channels = num_channels
        # --- Your code here
        self.conv1 = nn.Conv2d(num_channels, 4, [5,5], 1, 0, 1)
        self.relu = nn.ReLU()
        self.mp1 = nn.MaxPool2d([2,2], 2)
        self.conv2 = nn.Conv2d(4, 4, [5,5], 1, 0, 1)
        self.mp2 = nn.MaxPool2d([2,2], 2)
        self.flatten = nn.Flatten()
        self.l1 = nn.Linear(100,100)
        self.l2 = nn.Linear(100,latent_dim)

        # ---

    def forward(self, state):
        """
        :param state: <torch.Tensor> of shape (..., num_channels, 32, 32)
        :return latent_state: <torch.Tensor> of shape (..., latent_dim)
        """
        latent_state = None
        input_shape = state.shape
        state = state.reshape(-1, self.num_channels, 32, 32)
        # --- Your code here
        x = self.conv1(state)
        x = self.relu(x)
        x = self.mp1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.mp2(x)
        x = self.flatten(x)
        x = self.l1(x)
        x = self.relu(x)
        latent_state = self.l2(x)

        # ---
        # convert to original multi-batch shape
        latent_state = latent_state.reshape(*input_shape[:-3], self.latent_dim)
        return latent_state


class StateVariationalEncoder(nn.Module):
    """
    Embeds the state into a latent space.
    State shape: (..., num_channels, 32, 32)
    latent shape: (..., latent_dim)
    Check the notebook for more details about the architecture.
    """

    def __init__(self, latent_dim, num_channels=3):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_channels = num_channels
        # --- Your code here
        # self.kernel1 = torch.randn((out_channels, in_channels,  kernel_size, kernel_size))
        self.conv1 = nn.Conv2d(num_channels, 4, [5,5], 1, 0, 1)
        self.relu = nn.ReLU()
        self.mp1 = nn.MaxPool2d([2,2], 2)
        self.conv2 = nn.Conv2d(4, 4, [5,5], 1, 0, 1)
        self.mp2 = nn.MaxPool2d([2,2], 2)
        self.flatten = nn.Flatten()
        self.l1 = nn.Linear(100,100)
        self.lmu = nn.Linear(100,latent_dim)
        self.lstd = nn.Linear(100,latent_dim)

        # ---

    def forward(self, state):
        """
        :param state: <torch.Tensor> of shape (..., num_channels, 32, 32)
        :return: 2 <torch.Tensor>
          :mu: <torch.Tensor> of shape (..., latent_dim)
          :log_var: <torch.Tensor> of shape (..., latent_dim)
        """
        mu = None
        log_var = None
        input_shape = state.shape
        state = state.reshape(-1, self.num_channels, 32, 32)
        # --- Your code here
        x = self.conv1(state)
        x = self.relu(x)
        x = self.mp1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.mp2(x)
        x = self.flatten(x)
        # print(x.shape)
        x = self.l1(x)
        x = self.relu(x)
        mu = self.lmu(x)
        log_var = self.lstd(x)


        # ---
        # convert to original multi-batch shape
        mu = mu.reshape(*input_shape[:-3], self.latent_dim)
        log_var = log_var.reshape(*input_shape[:-3], self.latent_dim)
        return mu, log_var

    def reparameterize(self, mu, logvar):
        """
        Reparametrization trick to sample from N(mu, std) from N(0,1)
        :param mu: <torch.Tensor> of shape (..., latent_dim)
        :param logvar: <torch.Tensor> of shape (..., latent_dim)
        :return: <torch.Tensor> of shape (..., latent_dim)
        """
        sampled_latent_state = None
        # --- Your code here
        eps = np.random.normal()
        sampled_latent_state = mu + eps * logvar

        # ---
        return sampled_latent_state


class StateDecoder(nn.Module):
    """
    Reconstructs the state from a latent space.
    Check the notebook for more details about the architecture.
    """

    def __init__(self, latent_dim, num_channels=3):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_channels = num_channels
        # --- Your code here
        self.l1 = nn.Linear(latent_dim, 500)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(500,500)
        self.l3 = nn.Linear(500, num_channels * 32 * 32)


        # ---

    def forward(self, latent_state):
        """
        :param latent_state: <torch.Tensor> of shape (..., latent_dim)
        :return decoded_state: <torch.Tensor> of shape (..., num_channels, 32, 32)
        """
        decoded_state = None
        input_shape = latent_state.shape
        latent_state = latent_state.reshape(-1, self.latent_dim)

        # --- Your code here
        x = self.l1(latent_state)
        x = self.relu(x)
        x = self.l2(x)
        x = self.relu(x)
        decoded_state = self.l3(x)



        # ---

        decoded_state = decoded_state.reshape(*input_shape[:-1], self.num_channels, 32, 32)

        return decoded_state


class StateVAE(nn.Module):
    """
    State AutoEncoder
    """

    def __init__(self, latent_dim, num_channels=3):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_channels = num_channels
        self.encoder = StateVariationalEncoder(latent_dim, num_channels)
        self.decoder = StateDecoder(latent_dim, num_channels)

    def forward(self, state):
        """
        :param state: <torch.Tensor> of shape (..., num_channels, 32, 32)
        :return:
            reconstructed_state: <torch.Tensor> of shape (..., num_channels, 32, 32)
            mu: <torch.Tensor> of shape (..., latent_dim)
            log_var: <torch.Tensor> of shape (..., latent_dim)
            latent_state: <torch.Tensor> of shape (..., latent_dim)
        """
        reconstructed_state = None  # decoded states from the latent_state
        mu, log_var = None, None  # mean and log variance obtained from encoding state
        latent_state = None  # sample from the latent space feeded to the decoder
        # --- Your code here
        mu, log_var = self.encoder(state)
        latent_state = self.reparameterize(mu,log_var)
        reconstructed_state = self.decode(latent_state)

        # ---
        return reconstructed_state, mu, log_var, latent_state

    def encode(self, state):
        """
        :param state: <torch.Tensor> of shape (..., num_channels, 32, 32)
        :return: <torch.Tensor> of shape (..., latent_dim)
        """
        latent_state = None
        # --- Your code here
        mu,log_var= self.encoder(state)
        latent_state = self.reparameterize(mu,log_var)

        # ---
        return latent_state

    def decode(self, latent_state):
        """
        :param latent_state: <torch.Tensor> of shape (..., latent_dim)
        :return: <torch.Tensor> of shape (..., num_channels, 32, 32)
        """
        reconstructed_state = None
        # --- Your code here
        reconstructed_state = self.decoder(latent_state)

        # ---
        return reconstructed_state

    def reparameterize(self, mu, logvar):
        return self.encoder.reparameterize(mu, logvar)


class LatentDynamicsModel(nn.Module):

    def __init__(self, latent_dim, action_dim, num_channels=3):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.num_channels = num_channels
        self.encoder = StateEncoder(latent_dim, num_channels)
        self.decoder = StateDecoder(latent_dim, num_channels)
        self.latent_dynamics_model = nn.Sequential(
          nn.Linear(latent_dim+action_dim, 200),
          nn.ReLU(),
          nn.Linear(200, 200),
          nn.ReLU(),
          nn.Linear(200, 2*latent_dim + latent_dim*action_dim + latent_dim)
        )
    

        # ---

    def forward(self, state, action):
        """
        Compute next_state resultant of applying the provided action to provided state
        :param state: torch tensor of shape (..., num_channels, 32, 32)
        :param action: torch tensor of shape (..., action_dim)
        :return: next_state: torch tensor of shape (..., num_channels, 32, 32)
        """
        next_state = None
        # --- Your code here
        latent_state = self.encode(state)
        next_latent_state = self.latent_dynamics(latent_state, action)
        next_state = self.decode(next_latent_state)

        # ---
        return next_state

    def unwrap_linear_coeffs(self, flattened_coeffs):
        """
        :param next_latent_state: torch tensor of shap (..., 2*latent_dim + latent_dim*action_dim + latent_dim)
        """
        a1_end = self.latent_dim
        a2_end = 2*self.latent_dim
        b_end = a2_end + self.latent_dim*self.action_dim
        A = torch.eye(self.latent_dim) + torch.bmm(flattened_coeffs[:,:a1_end].unsqueeze(), flattened_coeffs[:,a1_end:a2_end].unsqueeze().transpose(-2,-1))
        B = torch.reshape(flattened_coeffs[:,a2_end:b_end], [flattened_coeffs.shape[0], self.latent_dim, self.action_dim])
        b = flattened_coeffs[:,b_end:]

        return A, B, b

    def encode(self, state):
        """
        Encode a state into the latent space
        :param state: torch tensor of shape (..., num_channels, 32, 32)
        :return: latent_state: torch tensor of shape (..., latent_dim)
        """
        latent_state = None
        # --- Your code here
        latent_state = self.encoder(state)

        # ---
        return latent_state

    def decode(self, latent_state):
        """
        Decode a latent state into the original space.
        :param latent_state: torch tensor of shape (..., latent_dim)
        :return: state: torch tensor of shape (..., num_channels, 32, 32)
        """
        state = None
        # --- Your code here
        state = self.decoder(latent_state)

        # ---
        return state

    def latent_dynamics(self, latent_state, action):
        """
        z(t+1) = A@z(t) + B@a(t) + b
        """
        next_latent_state = None
        # --- Your code here
        linear_terms_flattened = self.latent_dynamics_model(torch.cat([latent_state, action], -1))
        A, B, b = self.unwrap_linear_coeffs(linear_terms_flattened)
        next_latent_state = torch.bmm(A, latent_state) + torch.bmm(B, action) + b
        # ---
        return next_latent_state


def latent_space_pushing_cost_function(latent_state, action, target_latent_state):
    """
    Compute the state cost for MPPI on a setup without obstacles in latent space.
    :param state: torch tensor of shape (B, latent_dim)
    :param action: torch tensor of shape (B, action_size)
    :param target_latent_state: torch tensor of shape (latent_dim,)
    :return: cost: torch tensor of shape (B,) containing the costs for each of the provided states
    """
    cost = None
    # --- Your code here
    diff = latent_state - target_latent_state
    cost = (diff * diff).sum(dim=-1)

    # ---
    return cost


def img_space_pushing_cost_function(state, action, target_state):
    """
    Compute the state cost for MPPI on a setup without obstacles in state space (images).
    :param state: torch tensor of shape (B, num_channels, w, h)
    :param action: torch tensor of shape (B, action_size)
    :param target_state: torch tensor of shape (num_channels, w, h)
    :return: cost: torch tensor of shape (B,) containing the costs for each of the provided states
    """
    cost = None
    # --- Your code here
    # batch_target = target_state.repeat(0, state.shape[0], 1, 1)
    sum_cost = []
    cost = torch.zeros(state.shape[0])
    for i,s in enumerate(state):
      cost[i] = F.mse_loss(s,target_state)

    # ---
    return cost


class PushingImgSpaceController(object):
    """
    MPPI-based controller
    Since you implemented MPPI on HW2, here we will give you the MPPI implementation.
    You will just need to implement the dynamics and tune the hyperparameters and cost functions.
    """

    def __init__(self, env, model, cost_function, norm_constants, num_samples=100, horizon=10):
        self.env = env
        self.model = model
        self.norm_constants = norm_constants
        self.target_state = torch.as_tensor(self.env.get_target_state(), dtype=torch.float32).permute(2, 0, 1)
        self.target_state_norm = (self.target_state - self.norm_constants['mean']) / self.norm_constants['std']
        self.cost_function = cost_function
        # MPPI Hyperparameters:
        # --- You may need to tune them
        state_dim = env.observation_space.shape[0]
        u_min = torch.from_numpy(env.action_space.low)
        u_max = torch.from_numpy(env.action_space.high)
        noise_sigma = 0.1 * torch.eye(env.action_space.shape[0])
        lambda_value = 0.01
        # ---
        self.mppi = MPPI(self._compute_dynamics,
                         self._compute_costs,
                         nx=state_dim,
                         num_samples=num_samples,
                         horizon=horizon,
                         noise_sigma=noise_sigma,
                         lambda_=lambda_value,
                         u_min=u_min,
                         u_max=u_max)

    def _compute_dynamics(self, state, action):
        """
        Compute next_state using the dynamics model self.model and the provided state and action tensors
        :param state: torch tensor of shape (B, wrapped_state_size)
        :param action: torch tensor of shape (B, action_size)
        :return: next_state: torch tensor of shape (B, wrapped_state_size) containing the predicted states from the learned model.
        """
        next_state = None
        # --- Your code 
        # print("compute dynamics call")
        # print(action.shape)
        uw_state = self._unwrap_state(state)
        # print(uw_state.shape)
        w_next_state = self.model(uw_state, action)

        next_state = self._wrap_state(w_next_state)
        # print("compute dynamics return")

        # ---
        return next_state

    def _compute_costs(self, state, action):
        """
        Compute the cost for each state-action pair.
        You need to call self.cost_function to compute the cost.
        :param state: torch tensor of shape (B, wrapped_state_size)
        :param action: torch tensor of shape (B, action_size)
        :return: cost: torch tensor of shape (B,) containing the costs for each of the provided states
        """
        cost = None
        # --- Your code here
        # print("compute costs call")
        uw_state = self._unwrap_state(state)

        cost = self.cost_function(uw_state, action, self.target_state_norm)

        # ---
        return cost

    def control(self, state):
        """
        Query MPPI and return the optimal action given the current state <state>
        :param state: numpy array of shape (height, width, num_channels) representing current state
        :return: action: numpy array of shape (action_size,) representing optimal action to be sent to the robot.
        TO DO:
         - Prepare the state so it can be sent to the mppi controller. Note that MPPI works with torch tensors.
         - You may need to normalize the state to the same space used for training the model.
         - Unpack the mppi returned action to the desired format.
        """
        action = None
        state_tensor = None
        # --- Your code here
        # print(state.shape)
        temp_tensor = torch.tensor(state, dtype=torch.float32).permute([-1,0,1])
        norm_state = (temp_tensor - self.norm_constants['mean']) / self.norm_constants['std']
        state_tensor = self._wrap_state(norm_state)

        # ---
        action_tensor = self.mppi.command(state_tensor)
        # --- Your code here
        # print(action_tensor.shape)
        action = action_tensor.detach().cpu().numpy()

        # ---
        return action

    def _wrap_state(self, state):
        # convert state from shape (..., num_channels, height, width) to shape (..., num_channels*height*width)
        wrapped_state = None
        # --- Your code here
        wrapped_state = state.flatten(-3)

        # ---
        return wrapped_state

    def _unwrap_state(self, wrapped_state):
        # convert state from shape (..., num_channels*height*width) to shape (..., num_channels, height, width)
        state = None
        # --- Your code here
        desired_shape = []
        shape = self.target_state.shape
        w_shape = wrapped_state.shape
        for i in range(len(w_shape)-1):
          desired_shape.append(w_shape[i])
        for i in range(len(shape)):
          desired_shape.append(shape[i])
        
        state = wrapped_state.reshape(desired_shape)


        # ---
        return state



class PushingLatentController(object):
    """
    MPPI-based controller
    Since you implemented MPPI on HW2, here we will give you the MPPI implementation.
    You will just need to implement the dynamics and tune the hyperparameters and cost functions.
    """
    def __init__(self, env, model, cost_function, norm_constants, num_samples=100, horizon=10):
        self.env = env
        self.model = model
        self.norm_constants = norm_constants
        self.target_state = torch.as_tensor(self.env.get_target_state(), dtype=torch.float32).permute(2, 0, 1)
        self.target_state_norm = (self.target_state - self.norm_constants['mean']) / self.norm_constants['std']
        self.latent_target_state = self.model.encode(self.target_state_norm)
        self.cost_function = cost_function
        # MPPI Hyperparameters:
        # --- You may need to tune them
        state_dim = model.latent_dim  # Note that the state size is the latent dimension of the model
        u_min = torch.from_numpy(env.action_space.low)
        u_max = torch.from_numpy(env.action_space.high)
        noise_sigma = 0.1 * torch.eye(env.action_space.shape[0])
        lambda_value = 0.01
        # ---
        self.mppi = MPPI(self._compute_dynamics,
                         self._compute_costs,
                         nx=state_dim,
                         num_samples=num_samples,
                         horizon=horizon,
                         noise_sigma=noise_sigma,
                         lambda_=lambda_value,
                         u_min=u_min,
                         u_max=u_max)
    def _compute_dynamics(self, state, action):
        """
        Compute next_state using the dynamics model self.model and the provided state and action tensors
        :param state: torch tensor of shape (B, latent_dim)
        :param action: torch tensor of shape (B, action_size)
        :return: next_state: torch tensor of shape (B, latent_dim) containing the predicted states from the learned model.
        """
        next_state = None
        # --- Your code here
        next_state = self.model.latent_dynamics(state, action)
        
        # ---
        return next_state
    def _compute_costs(self, state, action):
        """
        Compute the cost for each state-action pair.
        You need to call self.cost_function to compute the cost.
        :param state: torch tensor of shape (B, latent_dim)
        :param action: torch tensor of shape (B, action_size)
        :return: cost: torch tensor of shape (B,) containing the costs for each of the provided states
        """
        cost = None
        # --- Your code here
        cost = self.cost_function(state, action, self.latent_target_state)
        # ---
        return cost
    def control(self, state):
        """
        Query MPPI and return the optimal action given the current state <state>
        :param state: numpy array of shape (height, width, num_channels) representing current state
        :return: action: numpy array of shape (action_size,) representing optimal action to be sent to the robot.
        TO DO:
         - Prepare the state so it can be sent to the mppi controller. Note that MPPI works with torch tensors.
         - You may need to normalize the state to the same space used for training the model.
         - Unpack the mppi returned action to the desired format.
        """
        action = None
        state_tensor = None
        # --- Your code here
        temp_tensor = torch.tensor(state, dtype=torch.float32).permute([-1,0,1])
        norm_state = (temp_tensor - self.norm_constants['mean']) / self.norm_constants['std']
        # state_tensor = torch.tensor(state, dtype=torch.float32).permute([-1,0,1])
        latent_tensor = self.model.encode(norm_state)


        # ---
        action_tensor = self.mppi.command(latent_tensor)
        # --- Your code here
        action = action_tensor.detach().cpu().numpy()
        # ---
        return action

# =========== AUXILIARY FUNCTIONS AND CLASSES HERE ===========
# --- Your code here


# ---
# ============================================================