import torch
import numpy as np
from torch import nn
from torch import optim

class RoboSINDy(nn.Module):
    def __init__(self, input_dim, batch_size=32):
        
        super(RoboSINDy, self).__init__()

        # CONSTANTS
        self.latent_dim = 2
        self.num_functions_in_library = 6 # Must set according to the latent_dim!
        self.rec_loss_reg = 1.0
        self.sindy_x_reg = 5e-4
        self.sindy_z_reg = 5e-5
        self.sparsity_reg = 1e-5


        self.batch_size = batch_size
        self.theta_z = torch.zeros((batch_size, self.num_functions_in_library))
        self.xi_coefficients = torch.ones((self.num_functions_in_library, self.latent_dim))

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


    def compute_theta(self, z):
        """
        Compute the function library theta(z) matrix for the given latent variable z.
        For latent dim 2, theta_z will be of shape (batch_size, 6) with the following columns: 
        [1, z1, z2, z1*z2, z1^2, z2^2]
        """
        self.theta_z = torch.cat(
            (torch.ones((self.batch_size, 1)), z, z[:, 0].unsqueeze(1) * z[:, 1].unsqueeze(1), z**2), dim=1
        )
        return self.theta_z

    def forward(self, x):
        """
        Returns latent variable z, reconstructed data x_hat, and z_next from latent dynamics
        """
        x = x.view(self.batch_size, -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)

        theta_z = self.compute_theta(z)
        z_next = theta_z @ self.xi_coefficients

        return x, z, x_hat, z_next
    
    def loss_function(self, x, z, x_hat, z_next):
        """
        Compute SINDy dynamics loss
        """
        return 1.0
        

    def train_model(self, dataloader, epochs=1000, learning_rate=0.001):
        
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        criterion = self.loss_function

        for epoch in range(epochs):
            optimizer.zero_grad()

            sample = next(iter(dataloader))
            x = sample['state']
            x, z, x_hat, z_next = self.forward(x)
            loss = criterion(x, z, x_hat, z_next)
            loss.backward()
            optimizer.step()

            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{epochs}, Loss: {loss.item()}")



            