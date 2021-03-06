import pandas as pd
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import torch
import util.metrics
from examples.ParWH.models import ParallelWHNet


if __name__ == '__main__':

    matplotlib.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica'], 'size': 11})

    model_name = "PWH_quant"

    # Dataset constants
    amplitudes = 5  # number of different amplitudes
    realizations = 20  # number of random phase multisine realizations
    samp_per_period = 16384  # number of samples per period
    n_skip = 1000
    periods = 1  # number of periods
    seq_len = samp_per_period * periods  # data points per realization

    # Column names in the dataset
    TAG_U = 'u'
    TAG_Y = 'y'

#    test_signal = "100mV"
#    test_signal = "325mV"
#    test_signal = "550mV"
#    test_signal = "775mV"
#    test_signal = "1000mV"
    test_signal = "ramp"

    #test_signal = "1000mV" #"ramp" #ramp"#"320mV" #"1000mV"#"ramp"
    plot_input = False

    # In[Load dataset]

    dict_test = {"100mV": 0, "325mV": 1, "550mV": 2, "775mV": 3, "1000mV": 4, "ramp": 5}
    dataset_list_level = ['ParWHData_Validation_Level' + str(i) for i in range(1, amplitudes + 1)]
    dataset_list = dataset_list_level + ['ParWHData_ValidationArrow']

    df_X_lst = []
    for dataset_name in dataset_list:
        dataset_filename = dataset_name + '.csv'
        df_Xi = pd.read_csv(os.path.join("data", dataset_filename))
        df_X_lst.append(df_Xi)


    df_X = df_X_lst[dict_test[test_signal]]  # first

    # Extract data
    y_meas = np.array(df_X['y'], dtype=np.float32)
    u = np.array(df_X['u'], dtype=np.float32)
    fs = np.array(df_X['fs'].iloc[0], dtype=np.float32)
    N = y_meas.size
    ts = 1/fs
    t = np.arange(N)*ts

    # In[Set-up model]

    net = ParallelWHNet()
    model_folder = os.path.join("models", model_name)
    net.load_state_dict(torch.load(os.path.join(model_folder, f"{model_name}.pt")))
    #log_sigma_hat = torch.load(os.path.join(model_folder, "log_sigma_hat.pt"))
    #sigma_hat = torch.exp(log_sigma_hat) + 1e-3
    # In[Predict]
    u_torch = torch.tensor(u[None, :, None],  dtype=torch.float, requires_grad=False)

    with torch.no_grad():
        y_hat = net(u_torch)

    # In[Detach]

    y_hat = y_hat.detach().numpy()[0, :, 0]

    # In[Plot]
    if plot_input:
        fig, ax = plt.subplots(2, 1, sharex=True)
        ax[0].plot(t, y_meas, 'k', label="$\mathbf{y}$")
        ax[0].plot(t, y_hat, 'b', label=r"$\mathbf{y}^{\rm sim}$")
        ax[0].plot(t, y_meas - y_hat, 'r', label="$\mathbf{e}$")
        ax[0].legend(loc="upper right")
        ax[0].set_ylabel("Voltage (V)")
        ax[0].grid()

        ax[1].plot(t, u, 'k', label="$u$")
        ax[1].legend(loc="upper right")
        ax[1].set_ylabel("Voltage (V)")
        ax[1].set_xlabel("Time (s)")
        ax[1].grid()
    else:
        fig, ax = plt.subplots(1, 1, figsize=(6, 3))
        ax.plot(t, y_meas, 'k', label="$\mathbf{y}$")
        ax.plot(t, y_hat, 'b', label=r"$\mathbf{y}^{\rm sim}$")
        ax.plot(t, y_meas - y_hat, 'r', label="$\mathbf{e}$")
        if test_signal == "ramp":
            ax.legend(loc="upper left")
        else:
            ax.legend(loc="upper right")
        ax.set_ylabel("Voltage (V)")
        ax.set_xlabel("Time (s)")
        ax.grid()

        if test_signal == "ramp":
            ax.set_xlim([0.0, 0.21])

    fig.tight_layout()
    fig_folder = "fig"
    if not os.path.exists(fig_folder):
        os.makedirs(fig_folder)
    fig.savefig(os.path.join(fig_folder, f"{model_name}_timetrace.pdf"))


    # In[Metrics]

    idx_test = range(n_skip, N)

    e_rms = 1000*util.metrics.error_rmse(y_meas[idx_test], y_hat[idx_test])
    mae = 1000 * util.metrics.error_mae(y_meas[idx_test], y_hat[idx_test])
    fit_idx = util.metrics.fit_index(y_meas[idx_test], y_hat[idx_test])
    r_sq = util.metrics.r_squared(y_meas[idx_test], y_hat[idx_test])
    u_rms = 1000*util.metrics.error_rmse(u, 0)

    print(f"RMSE: {e_rms:.2f}mV\nMAE: {mae:.2f}mV\nFIT:  {fit_idx:.1f}%\nR_sq: {r_sq:.1f}\nRMSU: {u_rms:.2f}mV")