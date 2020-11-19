import pandas as pd
import numpy as np
import os
import torch.utils.data
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import time
from examples.ParWH.models import ParallelWHNet
import argparse


def normal_standard_cdf(val):
    """Returns the value of the cumulative distribution function for a standard normal variable"""
    return 1/2 * (1 + torch.erf(val/np.sqrt(2)))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='training parameters')

    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    
    # Dataset constants

    amplitudes = 5  # number of different amplitudes
    realizations = 20  # number of random phase multisine realizations
    samp_per_period = 16384  # number of samples per period
    periods = 2  # number of periods
    seq_len = samp_per_period * periods  # data points per realization
    quant_delta = 0.2
    quant_noise = 0.1
    meas_intervals = np.arange(-1.0, 1.0 + quant_delta, quant_delta, dtype=np.float32)
    meas_intervals_full = np.r_[-1000, meas_intervals, 1000]


    # Training constants
    lr_ADAM = 5e-4
    lr_BFGS = 1e-2
    epochs_ADAM = 4000
    epochs_BFGS = 0#100
    epochs = epochs_ADAM + epochs_BFGS
    test_freq = 1  # print a msg every epoch
    batch_size = 95  # all in one batch
    n_skip = 100  # skip first n_skip points in loss evaluation

    # Column names in the dataset
    TAG_U = 'u'
    TAG_Y = 'y'
    DF_COL = ['amplitude', 'fs', 'lines'] + [TAG_U + str(i) for i in range(realizations)] + [TAG_Y + str(i) for i in range(realizations)] + ['?']

    # Load dataset
    dataset_list_level = ['ParWHData_Estimation_Level' + str(i) for i in range(1, amplitudes + 1)]

    df_X_lst = []
    for dataset_name in dataset_list_level:
        dataset_filename = dataset_name + '.csv'
        df_Xi = pd.read_csv(os.path.join("data", dataset_filename))
        df_Xi.columns = DF_COL
        df_X_lst.append(df_Xi)

    # Setup model
    net = ParallelWHNet()
    #model_name_load = "PWH"
    #model_folder = os.path.join("models", model_name_load)
    #net.load_state_dict(torch.load(os.path.join(model_folder, f"{model_name_load}.pt")))
    log_sigma_hat = torch.tensor(np.log(quant_delta), requires_grad=True)  # torch.randn(1, requires_grad = True)

    # Setup optimizer
    optimizer_ADAM = torch.optim.Adam([
        {'params': net.parameters(), 'lr': lr_ADAM},
        {'params': log_sigma_hat, 'lr': lr_ADAM},
    ], lr=lr_ADAM)


    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_ADAM, patience=10, factor=0.1, min_lr=1e-4, verbose=True)
    optimizer_LBFGS = torch.optim.LBFGS([par for par in net.parameters()] + [log_sigma_hat], lr=lr_BFGS)

    # Setup data loaders

    data = np.empty((amplitudes, realizations, seq_len, 2))  # Level, Realization, Time, Feat
    for amp_idx in range(amplitudes):
        for real_idx in range(realizations):
            tag_u = 'u' + str(real_idx)
            tag_y = 'y' + str(real_idx)
            df_data = df_X_lst[amp_idx][[tag_u, tag_y]]  #np.array()
            data[amp_idx, real_idx, :, :] = np.array(df_data)

    data = data.astype(np.float32)  # N_amp, N_real, seq_len, 2

    train_data = data[:, :-1, :, :]
    train_X = train_data[..., 0].reshape(-1, seq_len, 1)
    train_Y = train_data[..., 1].reshape(-1, seq_len, 1)
    train_Y_noise = train_Y + np.random.randn(*train_Y.shape)*quant_noise
    train_V = np.digitize(train_Y_noise, bins=meas_intervals)
    train_bins = meas_intervals_full[np.c_[train_V, train_V+1]] # bins of the measurement

    val_data = data[:, [-1], :, :]  # use last realization as a validation dataset
    val_X = val_data[..., 0].reshape(-1, seq_len, 1)
    val_Y = val_data[..., 1].reshape(-1, seq_len, 1)
    val_Y_noise = val_Y + np.random.randn(*val_Y.shape)*quant_noise
    val_V = np.digitize(val_Y_noise, bins=meas_intervals)
    val_bins = meas_intervals_full[np.c_[val_V, val_V+1]] # bins of the measurement

    train_ds = TensorDataset(torch.Tensor(train_X), torch.Tensor(train_bins))  # 19*5=95 samples
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    valid_ds = TensorDataset(torch.Tensor(val_X), torch.Tensor(val_bins))  # 19*5=95 samples
    valid_dl = torch.utils.data.DataLoader(valid_ds, batch_size=batch_size, shuffle=True)


    def get_loss(u, bins):
        y_hat = net(u)
        sigma_hat = torch.exp(log_sigma_hat) + 1e-3

        Phi_hat = normal_standard_cdf((bins - y_hat) / (sigma_hat)) # value of the two cumulative
        Phi_diff_hat = Phi_hat[..., [1]] - Phi_hat[..., [0]]  # integral
        Phi_diff_hat_log = torch.log(Phi_diff_hat + 1e-12)
        loss = - Phi_diff_hat_log.mean()
        return loss


    # In[Training loop]
    LOSS_ITR = []
    LOSS_TRAIN = []
    LOSS_VAL = []
    LOG_SIGMA = []
    start_time = time.time()
    for epoch in range(epochs):
        #loop = tqdm(train_dl)
        net.train()

        train_loss = torch.tensor(0.0)
        for u_batch, bins_batch in train_dl:

            def closure():
                optimizer_LBFGS.zero_grad()
                # Simulate
                loss = get_loss(u_batch, bins_batch)
                # Backward pass
                loss.backward()
                return loss

            bs = u_batch.shape[0]  # length of this batch (normally batch_size, except the last of the epoch)

            if epoch < epochs_ADAM:
                loss = optimizer_ADAM.step(closure)
            else:
                loss = optimizer_LBFGS.step(closure)

            with torch.no_grad():
                train_loss += loss * bs

            # Statistics
            LOSS_ITR.append(loss.item())
            LOG_SIGMA.append(log_sigma_hat.item())

        # Model in evaluation mode
        net.eval()

        train_loss = train_loss / len(train_ds)
        RMSE_train = torch.sqrt(train_loss)
        LOSS_TRAIN.append(train_loss.item())

        val_loss = torch.tensor(0.0)

        for u_batch, bins_batch in valid_dl:
            bs = u_batch.shape[0]
            val_loss += get_loss(u_batch, bins_batch) * bs

        val_loss = val_loss / len(valid_ds)

        LOSS_VAL.append(val_loss.item())

        sigma_hat = torch.exp(log_sigma_hat) + 1e-3
        print(f'Epoch {epoch} | Train Loss {train_loss:.6f} | Val Loss {val_loss:.6f} | Sigma_hat:{sigma_hat:.5f}')


    train_time = time.time() - start_time
    print(f"\nTrain time: {train_time:.2f}")  # 182 seconds


    # In[Save model]
    model_name = "PWH_quant"
    model_folder = os.path.join("models", model_name)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    torch.save(net.state_dict(), os.path.join(model_folder, f"{model_name}.pt"))
    torch.save(log_sigma_hat, os.path.join(model_folder, "log_sigma_hat.pt"))


    # In[detach]
#    with torch.no_grad():

#    y_hat_np = y_hat.detach().numpy()[0, :, 0]

    # In[Plot]
#    fig, ax = plt.subplots(2, 1, sharex=True)
#    ax[0].plot(t, y, 'k', label="$y$")
#    ax[0].plot(t, y_hat_np, 'r', label="$y$")

#    ax[0].legend()
#    ax[0].grid()

#    ax[1].plot(t, u, 'k', label="$u$")
#    ax[1].legend()
#    ax[1].grid()

#    plt.figure()
#    plt.plot(LOSS_ITR)
#    plt.grid(True)

    plt.figure()
    plt.plot(LOSS_TRAIN, 'r', label="train loss")
    plt.plot(LOSS_VAL, 'g', label="val loss")
    plt.grid(True)
    plt.legend()

    plt.figure()
    plt.plot(LOG_SIGMA)
    # In[Metrics]

#    idx_metric = range(0, N_per_period)
#    e_rms = util.metrics.error_rmse(y[idx_metric], y_hat_np[idx_metric])
#    fit_idx = util.metrics.fit_index(y[idx_metric], y_hat_np[idx_metric])
#    r_sq = util.metrics.r_squared(y[idx_metric], y_hat_np[idx_metric])

#    print(f"RMSE: {e_rms:.4f}V\nFIT:  {fit_idx:.1f}%\nR_sq: {r_sq:.1f}")