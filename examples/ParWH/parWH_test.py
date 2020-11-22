import pandas as pd
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import control
from torchid_nb.module.lti import MimoLinearDynamicalOperator
from torchid_nb.module.static import MimoStaticNonLinearity
import util.metrics
from examples.ParWH.models import ParallelWHNet


if __name__ == '__main__':

    matplotlib.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})

    model_name = "PWH_quant"

    # Dataset constants
    amplitudes = 5  #  number of different amplitudes
    realizations = 20  # number of random phase multisine realizations
    samp_per_period = 16384  # number of samples per period
    n_skip = 1000
    periods = 1  # number of periods
    seq_len = samp_per_period * periods  # data points per realization

    # Column names in the dataset
    TAG_U = 'u'
    TAG_Y = 'y'

    test_signal = "1000mV" #"ramp" #ramp"#"320mV" #"1000mV"#"ramp"
    plot_input = False

    # In[Load dataset]

    dict_test = {"100mV": 0, "320mV": 1, "550mV": 2, "775mV": 3, "1000mV": 4, "ramp": 5}
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

    # In[Predict]
    u_torch = torch.tensor(u[None, :, None],  dtype=torch.float, requires_grad=False)

    with torch.no_grad():
        y_hat = net(u_torch)

    # In[Detach]

    y_hat = y_hat.detach().numpy()[0, :, 0]

    # In[Plot]
    if plot_input:
        fig, ax = plt.subplots(2, 1, sharex=True)
        ax[0].plot(t, y_meas, 'k', label="$y$")
        ax[0].plot(t, y_hat, 'b', label="$\hat y$")
        ax[0].plot(t, y_meas - y_hat, 'r', label="$e$")
        ax[0].legend(loc="upper right")
        ax[0].set_ylabel("Voltage (V)")
        ax[0].grid()

        ax[1].plot(t, u, 'k', label="$u$")
        ax[1].legend(loc="upper right")
        ax[1].set_ylabel("Voltage (V)")
        ax[1].set_xlabel("Time (s)")
        ax[1].grid()
    else:
        fig, ax = plt.subplots(1, 1, sharex=True)
        ax.plot(t, y_meas, 'k', label="$y$")
        ax.plot(t, y_hat, 'b', label="$\hat y$")
        ax.plot(t, y_meas - y_hat, 'r', label="$e$")
        ax.legend(loc="upper right")
        ax.set_ylabel("Voltage (V)")
        ax.grid()

    fig_folder = "fig"
    if not os.path.exists(fig_folder):
        os.makedirs(fig_folder)
    fig.savefig(os.path.join(fig_folder, f"{model_name}_timetrace.pdf"))

    # In[Inspect linear model]

    # First linear block
    # a_coeff_1 = net.G1.a_coeff.detach().numpy()
    # b_coeff_1 = net.G1.b_coeff.detach().numpy()
    # a_poly_1 = np.empty_like(a_coeff_1, shape=(2, 2, net.na_1 + 1))
    # a_poly_1[:, :, 0] = 1
    # a_poly_1[:, :, 1:] = a_coeff_1[:, :, :]
    # b_poly_1 = np.array(b_coeff_1)
    # G1_sys = control.TransferFunction(b_poly_1, a_poly_1, ts)
    #
    # plt.figure()
    # mag_G1_1, phase_G1_1, omega_G1_1 = control.bode(G1_sys[0, 0])
    # plt.figure()
    # mag_G1_2, phase_G1_2, omega_G1_2 = control.bode(G1_sys[1, 0])
    #
    # # Second linear block
    # a_coeff_2 = net.G2.a_coeff.detach().numpy()
    # b_coeff_2 = net.G2.b_coeff.detach().numpy()
    # a_poly_2 = np.empty_like(a_coeff_2, shape=(2, 1, net.na_2 + 1))
    # a_poly_2[:, :, 0] = 1
    # a_poly_2[:, :, 1:] = a_coeff_2[:, :, :]
    # b_poly_2 = np.array(b_coeff_2)
    # G2_sys = control.TransferFunction(b_poly_2, a_poly_2, ts)

    # plt.figure()
    # mag_G2_1, phase_G2_1, omega_G2_1 = control.bode(G2_sys[0, 0])
    # plt.figure()
    # mag_G2_2, phase_G2_2, omega_G2_2 = control.bode(G2_sys[0, 1])


    # In[Metrics]

    idx_test = range(n_skip, N)

    e_rms = 1000*util.metrics.error_rmse(y_meas[idx_test], y_hat[idx_test])
    mae = 1000 * util.metrics.error_mae(y_meas[idx_test], y_hat[idx_test])
    fit_idx = util.metrics.fit_index(y_meas[idx_test], y_hat[idx_test])
    r_sq = util.metrics.r_squared(y_meas[idx_test], y_hat[idx_test])
    u_rms = 1000*util.metrics.error_rmse(u, 0)

    print(f"RMSE: {e_rms:.2f}mV\nMAE: {mae:.2f}mV\nFIT:  {fit_idx:.1f}%\nR_sq: {r_sq:.1f}\nRMSU: {u_rms:.2f}mV")