import sys
import torch
import numpy as np
from torch import nn
from torch import optim
import tqdm
from torch.autograd.functional import jvp
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from utils.mppi import MPPI
from utils.panda_pushing_env import TARGET_POSE_FREE, TARGET_POSE_OBSTACLES, OBSTACLE_HALFDIMS, OBSTACLE_CENTRE, BOX_SIZE

class NormalizationTransform:
    def __init__(self, norm_constants, eps=1e-8):
        self.mean_state  = norm_constants['mean_state']             
        self.std_state   = norm_constants['std_state']              
        self.mean_state_derivative = norm_constants['mean_state_derivative']  
        self.std_state_derivative = norm_constants['std_state_derivative'] 
        self.eps = eps

    def __call__(self, sample):
        s, sd = sample['states'], sample['state_derivatives']  

        m_s  = self.mean_state .view(-1,1,1)
        st_s = self.std_state  .view(-1,1,1)
        m_sd = self.mean_state_derivative.view(-1,1,1)
        st_sd = self.std_state_derivative.view(-1,1,1)

        sample['states'] = (s  - m_s) / (st_s + self.eps)
        sample['state_derivatives'] = (sd - m_sd)/(st_sd+ self.eps)
        return sample

    def inverse(self, sample):
        s, sd = sample['states'], sample['state_derivatives']
        m_s  = self.mean_state.view(-1,1,1)
        st_s = self.std_state.view(-1,1,1)
        m_sd = self.mean_state_derivative.view(-1,1,1)
        st_sd= self.std_state_derivative .view(-1,1,1)

        sample['states'] = s * (st_s + self.eps) + m_s
        sample['state_derivatives'] = sd * (st_sd + self.eps) + m_sd
        return sample

class SindyDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)
    
    def __getitem__(self, item):
        data_sample = self.data[item]

        action_sample = data_sample['action']
        state_sample = data_sample['state']
        state_derivative_sample = data_sample['state_derivative']

        sample = {
            'actions': None,
            'states': None,
            'state_derivatives': None
        }


        sample['actions'] = torch.tensor(action_sample, dtype=torch.float32)
        sample['states'] = torch.tensor(state_sample, dtype=torch.float32).permute(2, 0, 1)
        sample['state_derivatives'] = torch.tensor(state_derivative_sample, dtype=torch.float32).permute(2,0,1)

        if self.transform is not None:
          sample = self.transform(sample)

        return sample

class RoboSINDy(nn.Module):
    def __init__(self, input_dim, batch_size=32, latent_dim=2):
        
        super().__init__()

        # CONSTANTS
        self.latent_dim = latent_dim # can only be 1,2,3

        #set num_functions_in_library according to the latent_dim
        self.num_functions_in_library = 0
        if self.latent_dim == 2:
            self.num_functions_in_library = 6
        elif self.latent_dim == 3:
            self.num_functions_in_library = 14
        elif self.latent_dim == 1:
            self.num_functions_in_library = 2
        else:
            raise ValueError("latent_dim must be 1, 2 or 3")
        
        self.rec_loss_reg = 1.0
        self.sindy_x_reg = 5e-4
        self.sindy_z_reg = 5e-5
        self.sparsity_reg = 5e-5


        self.batch_size = batch_size
        self.theta_z = torch.zeros((batch_size, self.num_functions_in_library))
        self.xi_coefficients = nn.Parameter(torch.ones(self.num_functions_in_library, self.latent_dim))

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Sigmoid(),
            nn.Linear(128, 64),
            nn.Sigmoid(),
            nn.Linear(64, 32),
            nn.Sigmoid(),
            nn.Linear(32, self.latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 32),
            nn.Sigmoid(),
            nn.Linear(32, 64),
            nn.Sigmoid(),
            nn.Linear(64, 128),
            nn.Sigmoid(),
            nn.Linear(128, input_dim)
        )


        #xavier initialization
        # for m in self.modules():
        #     if isinstance(m, nn.Linear):
        #         nn.init.xavier_uniform_(m.weight)
        #         if m.bias is not None:
        #             nn.init.zeros_(m.bias)


    # def compute_theta(self, z):
        # """
        # Compute the function library theta(z) matrix for the given latent variable z.
        # For latent dim 2, theta_z will be of shape (batch_size, 6) with the following columns: 
        # [1, z1, z2, z1*z2, z1^2, z2^2]
        # """
    #     self.theta_z = torch.cat(
    #         (torch.ones((self.batch_size, 1)), z, z[:, 0].unsqueeze(1) * z[:, 1].unsqueeze(1), z**2), dim=1
    #     )
    #     return self.theta_z

    def compute_theta(self, z):
        """
        Compute the function library theta(z) matrix for the given latent variable z.
        For latent dim 2, theta_z will be of shape (batch_size, 6) with the following columns: 
        [1, z1, z2, z1*z2, z1^2, z2^2]
        """

        # if block to handle different latent dimensions

        if self.latent_dim == 2:
            N = z.size(0)
            z1, z2 = z[:, [0]], z[:, [1]]
            return torch.cat([
                torch.ones(N,1,device=z.device),
                z1, z2,
                z1*z2,
                z1**2, z2**2
            ], dim=1)
        
        elif self.latent_dim == 3:
            N = z.size(0)
            z1, z2, z3 = z[:, [0]], z[:, [1]], z[:, [2]]
            return torch.cat([
                torch.ones(N,1,device=z.device),
                z1, z2, z3,
                z1*z2, z1*z3, z2*z3,
                z1**2, z2**2, z3**2,
                z1*z2*z3, z1**3, z2**3, z3**3
            ], dim=1)
        
        elif self.latent_dim == 1:
            N = z.size(0)
            z1 = z[:, [0]]
            return torch.cat([
                torch.ones(N,1,device=z.device),
                z,
            ], dim=1)

    def forward(self, x):
        """
        Returns latent variable z, reconstructed data x_hat, and z_next from latent dynamics
        """
        x = x.view(self.batch_size, -1)
        
        z = self.encoder(x)
        x_hat = self.decoder(z)
        theta_z = self.compute_theta(z)
        z_dot_pred = theta_z @ self.xi_coefficients

        # check if any of these are nan
        # if torch.isnan(x).any():
        #     print("x nan!!!")
        return x, x_hat, z, z_dot_pred, theta_z
    
    def loss_function(self, x, x_dot, x_hat, z, z_dot_pred, theta_z):
        """
        Compute SINDy dynamics loss
        """
        x = x.requires_grad_(True)
        x_dot = x_dot.reshape(self.batch_size, -1)
        
        rec_loss = torch.mean((x - x_hat)**2)

        _, z_dot_true = jvp(self.encoder, (x,), (x_dot,)) # z_dot_true from encoder and input
        sindy_loss_z = torch.mean((z_dot_true - z_dot_pred)**2)

        _, x_dot_rec = jvp(self.decoder, (z,), (z_dot_pred,)) # x_dot_rec from decoder and z_dot_pred
        sindy_loss_x = torch.mean((x_dot - x_dot_rec)**2)

        sparsity = torch.mean(torch.abs(self.xi_coefficients))
        

        return (rec_loss * self.rec_loss_reg) + (sindy_loss_z * self.sindy_z_reg) + (sindy_loss_x * self.sindy_x_reg) + (sparsity * self.sparsity_reg)
        
    def train_model(self, dataloader, num_epochs=1000, learning_rate=0.001):
        
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        criterion = self.loss_function

        for epoch in range(num_epochs):
            optimizer.zero_grad()

            sample = next(iter(dataloader))
            x = sample['states']
            
            x, x_hat, z, z_dot_pred, theta_z = self.forward(x)
            loss = criterion(x, sample['state_derivatives'], x_hat, z, z_dot_pred, theta_z)
            loss.backward()
            optimizer.step()

            if epoch % 500 == 0:
                #set xi coefficients that are less that 0.1 to 0 by multiplying a mask
                mask = torch.abs(self.xi_coefficients) > 0.2
                self.xi_coefficients.data *= mask.float()
                print(f"\rEpoch {epoch}/{num_epochs}  Loss: {loss}")
            

        # set xi coefficients that are less that 0.1 to 0 by multiplying a mask
        mask = torch.abs(self.xi_coefficients) > 0.2
        self.xi_coefficients.data *= mask.float()
        print(f"Final Loss: {loss.item()}")

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

    cost = torch.mean((state - target_state.unsqueeze(0))**2, dim=(1,2,3))

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
        self.target_state_norm = (self.target_state - self.norm_constants['mean_state']) / self.norm_constants['std_state']
        self.cost_function = cost_function
        # MPPI Hyperparameters:
        # --- You may need to tune them
        state_dim = env.observation_space.shape[0]
        u_min = torch.from_numpy(env.action_space.low)
        u_max = torch.from_numpy(env.action_space.high)
        noise_sigma = 50 * torch.eye(env.action_space.shape[0])
        lambda_value = 1e-5
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
        # --- Your code here

        x = self._wrap_state(state)
        
        z = self.model.encoder(x)

        # SINDy dynamics
        theta_z = self.model.compute_theta(z)
        z_dot_pred = theta_z @ self.model.xi_coefficients
        z_next = z + (z_dot_pred / 240.0)
        next_state = self.model.decoder(z_next)

        # Parsimonious dynamics inferred from the SINDy model in the coefficient matrix
        # z_dot = torch.cat([z[:, [0]] * 0.9017911, z[:, [1]] * 1.7897408], dim=1)
        # z_next = z + (z_dot / 240.0)
        # next_state = self.model.decoder(z_next)

        # next_state = torch.zeros_like(x)



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

        state = self._unwrap_state(state)
        cost = self.cost_function(state, action, self.target_state_norm)

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

        state_tensor = torch.tensor(state, dtype=torch.float32).permute(2, 0, 1)
        state_tensor = (state_tensor - self.norm_constants['mean_state']) / self.norm_constants['std_state']

        # ---
        action_tensor = self.mppi.command(state_tensor)
        # --- Your code here


        action = action_tensor.detach().numpy()


        # ---
        return action

    def _wrap_state(self, state):
        # convert state from shape (..., num_channels, height, width) to shape (..., num_channels*height*width)
        wrapped_state = None
        # --- Your code here

        wrapped_state = state.reshape(-1, self.target_state.shape[0] * self.target_state.shape[1] * self.target_state.shape[2])

        # ---
        return wrapped_state

    def _unwrap_state(self, wrapped_state):
        # convert state from shape (..., num_channels*height*width) to shape (..., num_channels, height, width)
        state = None
        # --- Your code here

        state = wrapped_state.reshape(-1, self.target_state.shape[0], self.target_state.shape[1], self.target_state.shape[2])


        # ---
        return state



            