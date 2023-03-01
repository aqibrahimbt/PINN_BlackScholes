import numpy as np
import torch
import torch.nn as nn
from Numerics.BlackScholes.EuropeanOptions.CNEu import CNEu as FDM
import time as tm
from data import initialize_data


class BlackScholesMertonModel(nn.Module):

    def __init__(self, num_layers=100, num_units=50):
        super(BlackScholesMertonModel, self).__init__()

        self.fc = nn.ModuleList([nn.Linear(2 if i == 0 else num_units, num_units) for i in range(num_layers)])
        self.bn = nn.ModuleList([nn.BatchNorm1d(num_units) for _ in range(num_layers)])
        self.act = nn.ModuleList([nn.ReLU() for _ in range(num_layers)])

        self.fc[-1] = nn.Linear(num_units, 1)

    def forward(self, x):
        for i in range(len(self.fc) - 1):
            x = self.fc[i](x)
            x = self.bn[i](x)
            x = self.act[i](x)
        x = self.fc[-1](x)
        return x


if __name__ == '__main__':
    S0 = 100  # initial stock price
    exercise_price = 100
    sigma = 0.4  # volatility
    r = 0.03  # interest rate
    dividend = 0.00
    tau = 1  # time to expiration
    M = 500  # Stock steps
    N = 600  # time steps
    Smax = 500
    is_call = True
    N_b = 100
    N_exp = 1000
    N_f = 10000
    lb = [0, 0]
    ub = [500, tau]
    u0 = 0

    X_f, X_b, X_exp, u_exp = initialize_data(N_b, N_exp, N_f, lb, ub, exercise_price, tau)

    t, S = np.meshgrid(np.linspace(0, 1, N + 1), np.linspace(0, Smax, M + 1))
    option = FDM(S0, exercise_price, r, 0, tau, sigma, Smax, M, N, u0, is_call)
    option.price()
    option_fde_prices = option.grid

    # get values given time and stocks as inputs from the FDE scheme
    u_collocation = []
    for instance in X_f:
        time = instance[0]
        stock_price = instance[1]
        stock_price = int(stock_price.item())
        time = int(time.item() * 200)
        u_collocation_val = torch.Tensor(np.array([(np.round(option_fde_prices[stock_price, time], 3))])).type(torch.FloatTensor)
        u_collocation.append(u_collocation_val)

    u_collocation = torch.Tensor(u_collocation).type(torch.FloatTensor)
    u_collocation = torch.reshape(u_collocation, (-1, 1))
    f_collocation = torch.zeros(N_f, 1)
    u_boundary = torch.zeros(N_b, 1)

    print('... Initializing the Neural Network ...')
    model = BlackScholesMertonModel()
    device = torch.device("cuda:0")

    # # initialize the weights using Kaiming normal and bias with zeros
    for m in model.modules():
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            torch.nn.init.constant_(m.bias, 0)

    MAX_EPOCHS = int(5000)
    LRATE = 1e-2

    # use Adam for training
    optimizer = torch.optim.Adam(model.parameters(), lr=LRATE)
    X_f.requires_grad = True

    model.to(device)

    # send everything to GPU.
    X_f = X_f.to(device)
    X_b = X_b.to(device)
    X_exp = X_exp.to(device)
    u_boundary = u_boundary.to(device)
    u_exp = u_exp.to(device)
    u_collocation = u_collocation.to(device)
    f_collocation = f_collocation.to(device)

    loss_history_function = []
    loss_history_f = []
    loss_history_boundary = []
    loss_history_exp = []

    begin_time = tm.time()

    print("Learning Rate for this Round of Training : ", LRATE)
    print("....................................................")
    for epoch in range(MAX_EPOCHS):
        # boundary loss
        with torch.no_grad():
            rand_index = torch.randperm(n=len(X_b), device=device)

        X_b_shuffle = X_b[rand_index]
        u_boundary_shuffle = u_boundary[rand_index]
        u_b_pred = model(X_b_shuffle)
        mse_u_b = torch.nn.MSELoss()(u_b_pred, u_boundary_shuffle)

        # expiration time loss
        with torch.no_grad():
            rand_index = torch.randperm(n=len(X_exp), device=device)

        X_exp_shuffle = X_exp[rand_index]
        u_exp_shuffle = u_exp[rand_index]
        u_exp_pred = model(X_exp_shuffle)
        u_exp_pred = u_exp_pred.to(device)
        mse_u_exp = torch.nn.MSELoss()(u_exp_pred, u_exp_shuffle)

        # collocation loss
        with torch.no_grad():
            rand_index = torch.randperm(n=len(X_f), device=device)

        X_f_shuffle = X_f[rand_index]
        f_collocation_shuffle = f_collocation[rand_index]
        u_collocation_shuffle = u_collocation[rand_index]
        u_pred = model(X_f_shuffle)
        stock_price = X_f_shuffle[:, 1:2]

        # first derivative
        u_pred_first_partials = torch.autograd.grad(u_pred.sum(), X_f_shuffle, create_graph=True, allow_unused=True)[0]
        u_pred_dt = u_pred_first_partials[:, 0:1]
        u_pred_ds = u_pred_first_partials[:, 1:2]

        # second derivative
        u_pred_second_partials = torch.autograd.grad(u_pred_ds.sum(), X_f_shuffle, create_graph=True, allow_unused=True)[0]

        u_pred_dss = u_pred_second_partials[:, 1:2]
        f_pred = u_pred_dt + (0.5 * (sigma ** 2) * (stock_price ** 2) * u_pred_dss) + ((r - dividend) * stock_price * u_pred_ds) - (
                r * u_pred)
        f_true = f_collocation_shuffle
        mse_f = 100 * torch.nn.MSELoss()(f_pred, f_true)
        # writer.add_scalar("Loss/First/asset_loss", mse_f, epoch)

        loss = mse_f + mse_u_exp + mse_u_b

        # minimization of loss function
        mse_function = torch.nn.MSELoss()(u_pred, u_collocation_shuffle).detach()

        # optimizer step
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        loss_history_f.append(mse_f / 100)
        loss_history_boundary.append(mse_u_b)
        loss_history_exp.append(mse_u_exp)
        loss_history_function.append(mse_function)

        if (epoch % 10) == 0:
            print("- - - - - - - - - - - - - - -")
            print("Epoch : ", epoch)
            print(f"Loss Residual:\t{loss_history_f[-1]:.4f}")
            print(f"Loss Boundary:\t{loss_history_boundary[-1]:.4f}")
            print(f"Loss Expiration:\t{loss_history_exp[-1]:.4f}")
            print(f"Loss Function:\t{loss_history_function[-1]:.4f}")
    print("----------------------------------------------------------")

    print('... Starting second round of training ...')
    MAX_EPOCHS_1 = int(800)
    LRATE_1 = 1e-3
    print("Learning Rate for this Round of Training : ", LRATE_1)
    print("....................................................")

    optimizer = torch.optim.Adam(model.parameters(), lr=LRATE_1)

    loss_history_function_1 = []
    loss_history_f_1 = []
    loss_history_boundary_1 = []
    loss_history_exp_1 = []

    for epoch in range(MAX_EPOCHS_1):
        # boundary loss
        with torch.no_grad():
            rand_index = torch.randperm(n=len(X_b), device=device)

        X_b_shuffle = X_b[rand_index]
        u_boundary_shuffle = u_boundary[rand_index]
        u_b_pred = model(X_b_shuffle)
        mse_u_b = torch.nn.MSELoss()(u_b_pred, u_boundary_shuffle)

        # expiration time loss
        with torch.no_grad():
            rand_index = torch.randperm(n=len(X_exp), device=device)
        X_exp_shuffle = X_exp[rand_index]
        u_exp_shuffle = u_exp[rand_index]
        u_exp_pred = model(X_exp_shuffle)
        u_exp_pred = u_exp_pred.to(device)
        mse_u_exp = torch.nn.MSELoss()(u_exp_pred, u_exp_shuffle)

        # collocation loss
        with torch.no_grad():
            rand_index = torch.randperm(n=len(X_f), device=device)

        X_f_shuffle = X_f[rand_index]
        f_collocation_shuffle = f_collocation[rand_index]
        u_collocation_shuffle = u_collocation[rand_index]
        u_pred = model(X_f_shuffle)
        stock_price = X_f_shuffle[:, 1:2]

        # first derivative
        u_pred_first_partials = torch.autograd.grad(u_pred.sum(), X_f_shuffle, create_graph=True, allow_unused=True)[0]
        u_pred_dt = u_pred_first_partials[:, 0:1]
        u_pred_ds = u_pred_first_partials[:, 1:2]

        # second derivative
        u_pred_second_partials = torch.autograd.grad(u_pred_ds.sum(), X_f_shuffle, create_graph=True, allow_unused=True)[0]
        u_pred_dss = u_pred_second_partials[:, 1:2]

        f_pred = u_pred_dt + (0.5 * (sigma ** 2) * (stock_price ** 2) * u_pred_dss) + ((r - dividend) * stock_price * u_pred_ds) - (
                r * u_pred)
        f_true = f_collocation_shuffle
        mse_f = 100 * torch.nn.MSELoss()(f_pred, f_true)

        loss = mse_f + mse_u_exp + mse_u_b

        # minimization of loss function
        mse_function = torch.nn.MSELoss()(u_pred, u_collocation_shuffle).detach()

        # optimizer step
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        loss_history_f_1.append(mse_f / 100)
        loss_history_boundary_1.append(mse_u_b)
        loss_history_exp_1.append(mse_u_exp)
        loss_history_function_1.append(mse_function)

        if (epoch % 10) == 0:
            print("- - - - - - - - - - - - - - -")
            print("Epoch : ", epoch)
            print(f"Loss Residual:\t{loss_history_f_1[-1]:.4f}")
            print(f"Loss Boundary:\t{loss_history_boundary_1[-1]:.4f}")
            print(f"Loss Expiration:\t{loss_history_exp_1[-1]:.4f}")
            print(f"Loss Function:\t{loss_history_function_1[-1]:.4f}")
    print("----------------------------------------------------------")
    end_time = tm.time()
    total_time = end_time - begin_time

    # print('saving the models')
    # TODO: Change to the relative path
    torch.save(model, 'black_scholes_gpu.pt')
    print(f"Total Runnning Time:\t{total_time:.4f}")
