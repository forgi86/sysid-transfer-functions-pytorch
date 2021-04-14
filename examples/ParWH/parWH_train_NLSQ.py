import pandas as pd
import numpy as np
import os
import torch.utils.data
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import time
from examples.ParWH.models import ParallelWHNet


if __name__ == '__main__':

    # Dataset constants
    amplitudes = 5  # number of different amplitudes
    realizations = 20  # number of random phase multisine realizations
    samp_per_period = 16384  # number of samples per period
    periods = 2  # number of periods
    seq_len = samp_per_period * periods  # data points per realization

    # Training constants
    lr_ADAM = 1e-3
    lr_BFGS = 1e-1
    epochs_ADAM = 500
    epochs_BFGS = 100
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

    # Setup optimizer
    optimizer_ADAM = torch.optim.Adam(net.parameters(), lr=lr_ADAM)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_ADAM, patience=20, factor=0.5, min_lr=1e-4, verbose=True)
    optimizer_LBFGS = torch.optim.LBFGS(net.parameters(), lr=lr_BFGS)

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
    val_data = data[:, [-1], :, :]  # use last realization as a validation dataset
    val_X = val_data[..., 0].reshape(-1, seq_len, 1)
    val_Y = val_data[..., 1].reshape(-1, seq_len, 1)

    train_ds = TensorDataset(torch.Tensor(train_X), torch.Tensor(train_Y))  # 19*5=95 samples
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    valid_ds = TensorDataset(torch.Tensor(val_X), torch.Tensor(val_Y))  # 19*5=95 samples
    valid_dl = torch.utils.data.DataLoader(valid_ds, batch_size=batch_size, shuffle=True)

    def closure():

        optimizer_LBFGS.zero_grad()

        # Simulate
        y_hat = net(ub)

        # Compute fit loss
        err_fit = yb[..., n_skip:, :] - y_hat[..., n_skip:, :]
        loss = torch.mean(err_fit ** 2) * 100

        # Backward pass
        loss.backward()
        return loss

    # In[Training loop]
    LOSS_ITR = []
    LOSS_TRAIN = []
    LOSS_VAL = []
    start_time = time.time()
    for epoch in range(epochs):
        #loop = tqdm(train_dl)
        net.train()

        train_loss = torch.tensor(0.0)
        for ub, yb in train_dl:

            bs = ub.shape[0]  # length of this batch (normally batch_size, except the last of the epoch)

            if epoch < epochs_ADAM:
                loss = optimizer_ADAM.step(closure)
            else:
                loss = optimizer_LBFGS.step(closure)

            with torch.no_grad():
                train_loss += loss * bs

            # Statistics
            LOSS_ITR.append(loss.item())

        # Model in evaluation mode
        net.eval()

        # Metrics
        with torch.no_grad():

            train_loss = train_loss / len(train_ds)
            RMSE_train = torch.sqrt(train_loss)
            LOSS_TRAIN.append(train_loss.item())

            val_loss = torch.tensor(0.0)
            for ub, yb in valid_dl:

                bs = ub.shape[0]
                # Simulate
                y_hat = net(ub)

                # Compute fit loss
                err_val = yb[..., n_skip:, :] - y_hat[..., n_skip:, :]
                val_loss += torch.mean(err_val ** 2) * bs * 100

            val_loss = val_loss / len(valid_ds)
            LOSS_VAL.append(val_loss.item())

        #scheduler.step(val_loss)

        print(f'Epoch {epoch} | Train Loss {train_loss:.6f} | Validation Loss {val_loss:.6f} | Train RMSE: {RMSE_train:.2f}')

    train_time = time.time() - start_time
    print(f"\nTrain time: {train_time:.2f}")  # 182 seconds


    # In[Save model]
    model_name = "PWH_plain"
    model_folder = os.path.join("models", model_name)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    torch.save(net.state_dict(), os.path.join(model_folder, f"{model_name}.pt"))


    # In[detach]
    y_hat_np = y_hat.detach().numpy()[0, :, 0]

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

    # In[Metrics]

#    idx_metric = range(0, N_per_period)
#    e_rms = util.metrics.error_rmse(y[idx_metric], y_hat_np[idx_metric])
#    fit_idx = util.metrics.fit_index(y[idx_metric], y_hat_np[idx_metric])
#    r_sq = util.metrics.r_squared(y[idx_metric], y_hat_np[idx_metric])

#    print(f"RMSE: {e_rms:.4f}V\nFIT:  {fit_idx:.1f}%\nR_sq: {r_sq:.1f}")