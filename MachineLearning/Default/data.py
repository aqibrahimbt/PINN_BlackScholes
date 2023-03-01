import numpy as np
import torch


# data for collocation
def initialize_data(N_b, N_exp, N_f, lb, ub, exercise_price, tau):
    # time
    time_collocation = (torch.randint(low=0, high=100 * ub[1] + 1, size=(N_f, 1)) / 100).type(torch.FloatTensor)
    time_mean = torch.mean(time_collocation)
    time_std = torch.std(time_collocation)

    # stocks
    stock_price_collocation = torch.randint(low=0, high=ub[0] + 1, size=(N_f, 1)).type(torch.FloatTensor)
    stock_price_mean = torch.mean(stock_price_collocation)  # mean
    stock_price_std = torch.std(stock_price_collocation)  # standard deviation

    # time and stock data frame for collocation
    X_f = torch.cat((time_collocation, stock_price_collocation), 1)

    # normalized data frame
    time_collocation = (time_collocation - time_mean) / time_std
    stock_price_collocation = (stock_price_collocation - stock_price_mean) / stock_price_std
    # X_f_norm = torch.cat((time_collocation, stock_price_collocation), 1)

    time_boundary = (torch.randint(low=1, high=100 * ub[1] + 1, size=(N_b, 1)) / 100).type(torch.FloatTensor)

    X_b = torch.cat((time_boundary, 0 * time_boundary), 1)

    stock_price_exp = torch.randint(low=0, high=ub[0] + 1, size=(N_exp, 1)).type(torch.FloatTensor)

    option_price_exp = stock_price_exp - exercise_price
    u_exp = torch.Tensor([[max(instance, 0)] for instance in option_price_exp]).type(torch.FloatTensor)
    X_exp = torch.cat((0 * stock_price_exp + tau, stock_price_exp), 1)

    return X_f, X_b, X_exp, u_exp


# data for collocation optimized for use in the PINN coarse propagator
def initialize_data_mini(N_b, N_exp, N_f, stock_slice, time_slice, N, exercise_price):
    # time
    time_tt = np.round(np.linspace(time_slice[0], time_slice[1], N + 1), 2)
    time_collocation = torch.Tensor(time_tt[torch.randint(len(time_tt), (N_f, 1))]).type(torch.FloatTensor)

    # stocks
    stock_price_collocation = torch.randint(low=0, high=stock_slice[1] + 1, size=(N_f, 1)).type(torch.FloatTensor)

    # time and stock data frame for collocation
    X_f = torch.cat((time_collocation, stock_price_collocation), 1)

    time_boundary = torch.Tensor(time_tt[torch.randint(len(time_tt), (N_b, 1))]).type(torch.FloatTensor)
    X_b = torch.cat((time_boundary, 0 * time_boundary), 1)

    stock_price_exp = torch.randint(low=0, high=stock_slice[1] + 1, size=(N_exp, 1)).type(torch.FloatTensor)

    option_price_exp = stock_price_exp - exercise_price
    u_exp = torch.Tensor([[max(instance, 0)] for instance in option_price_exp]).type(torch.FloatTensor)
    X_exp = torch.cat((0 * stock_price_exp + time_slice[1], stock_price_exp), 1)

    return X_f, X_b, X_exp, u_exp
