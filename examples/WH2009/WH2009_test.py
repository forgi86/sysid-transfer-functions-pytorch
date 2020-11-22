import torch
import pandas as pd
import numpy as np
import os
from torchid_nb.module.lti import SisoLinearDynamicalOperator
from torchid_nb.module.static import SisoStaticNonLinearity

import matplotlib
import matplotlib.pyplot as plt
import control
import util.metrics


# In[Main]
if __name__ == '__main__':

    matplotlib.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
    # In[Settings]
    #model_name = 'model_WH_digit'
    model_name = "model_WH_proc_noise_PEM"

    # Settings
    n_b = 8
    n_a = 8

    # Column names in the dataset
    COL_F = ['fs']
    COL_U = ['uBenchMark']
    COL_Y = ['yBenchMark']

    # Load dataset
    df_X = pd.read_csv(os.path.join("data", "WienerHammerBenchmark.csv"))

    # Extract data
    y_meas = np.array(df_X[COL_Y], dtype=np.float32)
    u = np.array(df_X[COL_U], dtype=np.float32)
    fs = np.array(df_X[COL_F].iloc[0], dtype=np.float32).item()
    N = y_meas.size
    ts = 1/fs
    t = np.arange(N)*ts

    t_fit_start = 0
    t_fit_end = 100000
    t_test_start = 100000
    t_test_end = 188000
    t_skip = 1000  # skip for statistics

    # In[Instantiate models]

    # Create models
    G1 = SisoLinearDynamicalOperator(n_b=n_b, n_a=n_a, n_k=1)
    G2 = SisoLinearDynamicalOperator(n_b=n_b, n_a=n_a, n_k=0)
    F_nl = SisoStaticNonLinearity(n_hidden=10, activation='tanh')

    model_folder = os.path.join("models", model_name)
    # Create model parameters
    G1.load_state_dict(torch.load(os.path.join(model_folder, "G1.pt")))
    F_nl.load_state_dict(torch.load(os.path.join(model_folder, "F_nl.pt")))
    G2.load_state_dict(torch.load(os.path.join(model_folder, "G2.pt")))

    # In[Predict]

    u_torch = torch.tensor(u[None, :, :])
    y1_lin = G1(u_torch)
    y1_nl = F_nl(y1_lin)
    y_hat = G2(y1_nl)

    # In[Detach]
    y_hat = y_hat.detach().numpy()[0, :, :]
    y1_lin = y1_lin.detach().numpy()[0, :, :]
    y1_nl = y1_nl.detach().numpy()[0, :, :]

    # In[]
    fig_folder = "fig"
    if not os.path.exists(fig_folder):
        os.makedirs(fig_folder)

    # In[Plot]
    plt.figure()
    plt.plot(t, y_meas, 'k', label="$y$")
    plt.plot(t, y_hat, 'b', label="$\hat y$")
    plt.plot(t, y_meas - y_hat, 'r', label="$e$")
    plt.grid(True)
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (V)')
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(fig_folder, f"{model_name}_fit.pdf"))

    # In[Inspect linear model]

    n_imp = 128
    G1_num, G1_den = G1.get_tfdata()
    G1_sys = control.TransferFunction(G1_num, G1_den, ts)
    plt.figure()
    plt.title("$G_1$ impulse response")
    _, y_imp = control.impulse_response(G1_sys, np.arange(n_imp) * ts)
    #    plt.plot(G1_num)
    plt.plot(y_imp)
    plt.savefig(os.path.join("models", model_name, "G1_imp.pdf"))
    plt.figure()
    mag_G1, phase_G1, omega_G1 = control.bode(G1_sys, omega_limits=[1e2, 1e5])
    plt.suptitle("$G_1$ bode plot")
    plt.savefig(os.path.join("models", model_name, "G1_bode.pdf"))

    # G2_b = G2.G.weight.detach().numpy()[0, 0, ::-1]
    G2_num, G2_den = G2.get_tfdata()
    G2_sys = control.TransferFunction(G2_num, G2_den, ts)
    plt.figure()
    plt.title("$G_2$ impulse response")
    _, y_imp = control.impulse_response(G2_sys, np.arange(n_imp) * ts)
    plt.plot(y_imp)
    plt.savefig(os.path.join("models", model_name, "G1_imp.pdf"))
    plt.figure()
    mag_G2, phase_G2, omega_G2 = control.bode(G2_sys, omega_limits=[1e2, 1e5])
    plt.suptitle("$G_2$ bode plot")
    plt.savefig(os.path.join("models", model_name, "G2_bode.pdf"))

# In[Inspect static non-linearity]

    y1_lin_min = np.min(y1_lin)
    y1_lin_max = np.max(y1_lin)

    in_nl = np.arange(y1_lin_min, y1_lin_max, (y1_lin_max- y1_lin_min)/1000).astype(np.float32).reshape(-1, 1)

    with torch.no_grad():
        out_nl = F_nl(torch.as_tensor(in_nl))

    plt.figure()
    plt.plot(in_nl, out_nl, 'b')
    plt.plot(in_nl, out_nl, 'b')
    plt.xlabel('Static non-linearity input (-)')
    plt.ylabel('Static non-linearity input (-)')
    plt.grid(True)

    # In[Metrics]
    idx_test = range(t_test_start + t_skip, t_test_end)
    e_rms = 1000*util.metrics.error_rmse(y_meas[idx_test], y_hat[idx_test])[0]
    fit_idx = util.metrics.fit_index(y_meas[idx_test], y_hat[idx_test])[0]
    r_sq = util.metrics.r_squared(y_meas[idx_test], y_hat[idx_test])[0]

    print(f"RMSE: {e_rms:.1f}V\nFIT:  {fit_idx:.1f}%\nR_sq: {r_sq:.4f}")


    # In[Plot for paper]

    t_test_start = 140000
    len_plot = 1000

    plt.figure(figsize=(4, 3))
    plt.plot(t[t_test_start:t_test_start+len_plot], y_meas[t_test_start:t_test_start+len_plot], 'k', label="$\mathbf{y}^{\mathrm{meas}}$")
    plt.plot(t[t_test_start:t_test_start+len_plot], y_hat[t_test_start:t_test_start+len_plot], 'b--', label="$\mathbf{y}$")
    plt.plot(t[t_test_start:t_test_start+len_plot], y_meas[t_test_start:t_test_start+len_plot] - y_hat[t_test_start:t_test_start+len_plot], 'r', label="$\mathbf{e}$")
    plt.grid(True)
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (V)')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('WH_timetrace.pdf')


    # In[Bode]

    std_v = 0.1 # noise standard deviation

    # w_v = 10000
    # damp_v = 0.2
    #
    # Hu = control.TransferFunction(np.array([0, 0, w_v**2]), np.array([1, 2*damp_v*w_v, w_v**2])) + 0.1
    # Hud = control.matlab.c2d(Hu, ts)
    #
    # Hud.num[0][0] = Hud.num[0][0] / Hud.num[0][0][0]
    # Hud.den[0][0] = Hud.den[0][0] / Hud.den[0][0][0]

    r_den = 0.97  # magnitude of poles
    wo_den = 0.2  # phase of poles (approx 2.26 kHz)

    r_num = 0.95  # magnitude of zeros
    wo_num = 0.6  # phase of zeros (approx 9.78 kHz)

    H_true = control.TransferFunction([1, -2 * r_num * np.cos(wo_num), r_num ** 2], [1, -2 * r_den * np.cos(wo_den), r_den ** 2], ts)

    H_inv_learn = SisoLinearDynamicalOperator(2, 2, n_k=1)
    H_inv_learn.load_state_dict(torch.load(os.path.join(model_folder, "H_inv.pt")))

    n_imp = 128
    H_inv_num, H_inv_den = H_inv_learn.get_tfdata()
    H_inv_sys = 1 + control.TransferFunction(H_inv_num, H_inv_den, ts)
    H_sys = 1/H_inv_sys

    # In[]

    mag_H_true, phase_H_true, omega_H_true = control.bode(H_true, omega_limits=[1e2, 1e4], Hz=True, Plot=False)
    mag_H_hat, phase_H_hat, omega_H_hat = control.bode(H_sys, omega_limits=[1e2, 1e4], Hz=True, Plot=False)

    plt.figure()
    plt.semilogx(omega_H_true/2/np.pi, 20*np.log10(mag_H_true), 'k', label="$H(q)$")
    plt.semilogx(omega_H_hat/2/np.pi, 20*np.log10(mag_H_hat), 'b', label="$\hat H(q)$")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(fig_folder, f"{model_name}_H_noise.pdf"))

    # In[]
    #plt.legend()
    #plt.suptitle("$H_inv$ bode plot")
#    plt.savefig(os.path.join("models", model_name, "G1_bode.pdf"))

    plt.figure()
    control.bode([H_sys, H_true], Hz=True)


