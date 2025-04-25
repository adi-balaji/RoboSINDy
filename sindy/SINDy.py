import torch
import numpy as np
from torch import nn
from torch import optim
import tqdm
from torch.autograd.functional import jvp
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

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
    def __init__(self, input_dim, batch_size=32):
        
        super().__init__()

        # CONSTANTS
        self.latent_dim = 2
        self.num_functions_in_library = 6 # Must set according to the latent_dim!
        self.rec_loss_reg = 1.0
        self.sindy_x_reg = 5e-4
        self.sindy_z_reg = 5e-5
        self.sparsity_reg = 1e-5


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
        N = z.size(0)
        z1, z2 = z[:, [0]], z[:, [1]]
        return torch.cat([
            torch.ones(N,1,device=z.device),
            z1, z2,
            z1*z2,
            z1**2, z2**2
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
                mask = torch.abs(self.xi_coefficients) > 0.1
                self.xi_coefficients.data *= mask.float()
                print(f"Epoch {epoch}/{num_epochs}, Loss: {loss.item()}")



            