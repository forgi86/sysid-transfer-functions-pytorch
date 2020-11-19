import numba as nb
import numpy as np
import scipy as sp
import scipy.signal


def lfiltic_vec(b, a, y, x=None):
    """
    Construct initial conditions for lfilter given input and output vectors.

    Given a linear filter (b, a) and initial conditions on the output `y`
    and the input `x`, return the initial conditions on the state vector zi
    which is used by `lfilter` to generate the output given the input.

    Parameters
    ----------
    b : array_like
        Linear filter term.
    a : array_like
        Linear filter term.
    y : array_like
        Initial conditions.

        If ``N = len(a) - 1``, then ``y = {y[-1], y[-2], ..., y[-N]}``.

        If `y` is too short, it is padded with zeros.
    x : array_like, optional
        Initial conditions.

        If ``M = len(b) - 1``, then ``x = {x[-1], x[-2], ..., x[-M]}``.

        If `x` is not given, its initial conditions are assumed zero.

        If `x` is too short, it is padded with zeros.

    Returns
    -------
    zi : ndarray
        The state vector ``zi = {z_0[-1], z_1[-1], ..., z_K-1[-1]}``,
        where ``K = max(M, N)``.

    See Also
    --------
    lfilter, lfilter_zi

    """
    N = np.size(a) - 1
    M = np.size(b) - 1
    K = max(M, N)
    y = np.asarray(y)
    batch_size = y.shape[0]

    if y.dtype.kind in 'bui':
        # ensure calculations are floating point
        y = y.astype(np.float64)
    zi = np.zeros((batch_size, K), y.dtype)
    if x is None:
        x = np.zeros((batch_size, M), y.dtype)
    else:
        x = np.asarray(x)
        L = np.shape(x)[1]
        if L < M:
            x = np.r_[x, np.zeros((batch_size, M - L))]
    L = np.shape(y)[1]
    if L < N:
        y = np.r_[y, np.zeros((batch_size, N - L))]

    for m in range(M):
        zi[:, m] = np.sum(b[m + 1:] * x[:, :M - m], axis=1)

    for m in range(N):
        zi[:, m] -= np.sum(a[m + 1:] * y[:, :N - m], axis=1)

    return zi


def lfilter_mimo(b, a, u_in):
    batch_size, seq_len, in_ch = u_in.shape
    out_ch, _, _ = a.shape
    y_out = np.zeros_like(u_in, shape=(batch_size, seq_len, out_ch))
    for out_idx in range(out_ch):
        for in_idx in range(in_ch):
            y_out[:, :, out_idx] += scipy.signal.lfilter(b[out_idx, in_idx, :], a[out_idx, in_idx, :],
                                                         u_in[:, :, in_idx], axis=-1)
    return y_out  # [B, T, O]


def lfilter_mimo_components(b, a, u_in):
    batch_size, seq_len, in_ch = u_in.shape
    out_ch, _, _ = a.shape
    y_comp_out = np.zeros_like(u_in, shape=(batch_size, seq_len, out_ch, in_ch))
    for out_idx in range(out_ch):
        for in_idx in range(in_ch):
            y_comp_out[:, :, out_idx, in_idx] = scipy.signal.lfilter(b[out_idx, in_idx, :], a[out_idx, in_idx, :], u_in[:, :, in_idx], axis=-1)
    return y_comp_out  # [B, T, O, I]


@nb.jit(["float32[:,:,:,:](float32[:,:,:],float32[:,:,:],float32[:,:,:])",
         "float64[:,:,:,:](float64[:,:,:],float64[:,:,:],float64[:,:,:])"],
        nopython=True, parallel=True, cache=True)
def lfilter_mimo_components_jit(b_poly, a_poly, u):
    batch_size = u.shape[0]
    seq_len = u.shape[1]
    in_ch = u.shape[2]
    out_ch = a_poly.shape[0]
    n_a = a_poly.shape[-1]
    n_x = n_a - 1
    d = np.zeros((batch_size, n_x, out_ch, in_ch), dtype=u.dtype)
    y_comp_out = np.empty((batch_size, seq_len, out_ch, in_ch), dtype=u.dtype)

    for batch_idx in nb.prange(batch_size):
        for time_idx in nb.prange(seq_len):
            for out_idx in nb.prange(out_ch):
                for in_idx in nb.prange(in_ch):
                    y_comp_out[batch_idx, time_idx, out_idx, in_idx] = b_poly[out_idx, in_idx, 0] * u[batch_idx, time_idx, in_idx] + d[batch_idx, 0, out_idx, in_idx]
                    for d_idx in nb.prange(n_x-1):
                        d[batch_idx, d_idx, out_idx, in_idx] = b_poly[out_idx, in_idx, d_idx + 1] * u[batch_idx, time_idx, in_idx] \
                                                               - a_poly[out_idx, in_idx, d_idx + 1] * y_comp_out[batch_idx, time_idx, out_idx, in_idx] + d[batch_idx, d_idx + 1, out_idx, in_idx]
                    d[batch_idx, n_x-1, out_idx, in_idx] = b_poly[out_idx, in_idx, n_x] * u[batch_idx, time_idx, in_idx] \
                                                           - a_poly[out_idx, in_idx, n_x] * y_comp_out[batch_idx, time_idx, out_idx, in_idx]

    return y_comp_out


@nb.jit(["float32[:,:,:,:](float32[:,:,:],float32[:,:,:],float32[:,:,:,:])",
         "float64[:,:,:,:](float64[:,:,:],float64[:,:,:],float64[:,:,:,:])"],
        nopython=True, parallel=True, cache=True)
def lfilter_mimo_components_sep_jit(bb, aa, u):
    batch_size = u.shape[0]
    seq_len = u.shape[1]
    in_ch = u.shape[3] # B, T, O, Is
    out_ch = aa.shape[0]
    n_a = aa.shape[-1]
    n_x = n_a - 1
    d = np.zeros((batch_size, n_x, out_ch, in_ch), dtype=u.dtype)
    y_comp_out = np.empty((batch_size, seq_len, out_ch, in_ch), dtype=u.dtype)

    for batch_idx in nb.prange(batch_size):
        for time_idx in nb.prange(seq_len):
            for out_idx in nb.prange(out_ch):
                for in_idx in nb.prange(in_ch):
                    y_comp_out[batch_idx, time_idx, out_idx, in_idx] = bb[out_idx, in_idx, 0] * u[batch_idx, time_idx, out_idx, in_idx] + d[batch_idx, 0, out_idx, in_idx]
                    for d_idx in nb.prange(n_x-1):
                        d[batch_idx, d_idx, out_idx, in_idx] = bb[out_idx, in_idx, d_idx + 1] * u[batch_idx, time_idx, out_idx, in_idx] \
                                                               - aa[out_idx, in_idx, d_idx + 1] * y_comp_out[batch_idx, time_idx, out_idx, in_idx] + d[batch_idx, d_idx + 1, out_idx, in_idx]
                    d[batch_idx, n_x-1, out_idx, in_idx] = bb[out_idx, in_idx, n_x] * u[batch_idx, time_idx, out_idx, in_idx] \
                                                           - aa[out_idx, in_idx, n_x] * y_comp_out[batch_idx, time_idx, out_idx, in_idx]

    return y_comp_out


@nb.jit(["float32[:,:,:,:](float32[:,:,:], float32[:,:,:])",
         "float64[:,:,:,:](float64[:,:,:],float64[:,:,:])"],
        nopython=True, parallel=True, cache=True)
def lfilter_mimo_components_bsens_jit(a_poly, u):
    batch_size = u.shape[0]
    seq_len = u.shape[1]
    in_ch = u.shape[2]
    out_ch = a_poly.shape[0]
    n_a = a_poly.shape[-1]
    n_x = n_a - 1
    d = np.zeros((batch_size, n_x, out_ch, in_ch), dtype=u.dtype)
    y_sens_b = np.empty((batch_size, seq_len, out_ch, in_ch), dtype=u.dtype)

    for batch_idx in nb.prange(batch_size):
        for time_idx in nb.prange(seq_len):
            for out_idx in nb.prange(out_ch):
                for in_idx in nb.prange(in_ch):
                    y_sens_b[batch_idx, time_idx, out_idx, in_idx] = 1.0 * u[batch_idx, time_idx, in_idx] + d[batch_idx, 0, out_idx, in_idx]
                    for d_idx in nb.prange(n_x-1):
                        d[batch_idx, d_idx, out_idx, in_idx] = - a_poly[out_idx, in_idx, d_idx + 1] * y_sens_b[batch_idx, time_idx, out_idx, in_idx] + d[batch_idx, d_idx + 1, out_idx, in_idx]
                    d[batch_idx, n_x-1, out_idx, in_idx] = -a_poly[out_idx, in_idx, n_x] * y_sens_b[batch_idx, time_idx, out_idx, in_idx]

    return y_sens_b


@nb.jit(["float32[:,:,:,:](float32[:,:,:], float32[:,:,:,:])",
         "float64[:,:,:,:](float64[:,:,:],float64[:,:,:,:])"],
        nopython=True, parallel=True, cache=True)
def lfilter_mimo_components_asens_jit(a_poly, y_comp_out):
    batch_size = y_comp_out.shape[0]
    seq_len = y_comp_out.shape[1]
    in_ch = y_comp_out.shape[3]  # B, T, O, Is
    out_ch = a_poly.shape[0]
    n_a = a_poly.shape[-1]
    n_x = n_a - 1
    d = np.zeros((batch_size, n_x, out_ch, in_ch), dtype=y_comp_out.dtype)
    y_sens_a = np.empty((batch_size, seq_len, out_ch, in_ch), dtype=y_comp_out.dtype)

    for batch_idx in nb.prange(batch_size):
        for time_idx in nb.prange(seq_len):
            for out_idx in nb.prange(out_ch):
                for in_idx in nb.prange(in_ch):
                    y_sens_a[batch_idx, time_idx, out_idx, in_idx] = d[batch_idx, 0, out_idx, in_idx]
                    d[batch_idx, 0, out_idx, in_idx] = \
                        1.0 * y_comp_out[batch_idx, time_idx, out_idx, in_idx] \
                        - a_poly[out_idx, in_idx, 1] * y_sens_a[batch_idx, time_idx, out_idx, in_idx]
                    if n_x > 1:
                        d[batch_idx, 0, out_idx, in_idx] += d[batch_idx, 1, out_idx, in_idx]
                        for d_idx in nb.prange(1, n_x-1):
                            d[batch_idx, d_idx, out_idx, in_idx] = - a_poly[out_idx, in_idx, d_idx + 1] * y_sens_a[batch_idx, time_idx, out_idx, in_idx] \
                                                                   + d[batch_idx, d_idx + 1, out_idx, in_idx]
                        d[batch_idx, n_x-1, out_idx, in_idx] = - a_poly[out_idx, in_idx, n_x] * y_sens_a[batch_idx, time_idx, out_idx, in_idx]

    return y_sens_a


@nb.jit(["float32[:,:,:](float32[:,:,:,:], float32[:,:,:], int64)",
         "float64[:,:,:](float64[:,:,:,:], float64[:,:,:], int64)"],
        nopython=True, parallel=True, cache=True)
def compute_grad_coeff_jit(sens_coeff0, grad_y, n_coeff):
    batch_size, seq_len, out_ch, in_ch = sens_coeff0.shape
    # batch_size, seq_len, out_ch = grad_y.shape
    grad_coeff = np.zeros((out_ch, in_ch, n_coeff), dtype=grad_y.dtype)
    for batch_idx in range(batch_size):
        for out_idx in range(out_ch):
            for in_idx in range(in_ch):
                for b_idx in range(n_coeff):
                    for time_idx in range(b_idx, seq_len):
                        grad_coeff[out_idx, in_idx, b_idx] += \
                            grad_y[batch_idx, time_idx, out_idx] * sens_coeff0[batch_idx, time_idx - b_idx, out_idx, in_idx]
    return grad_coeff


if __name__ == '__main__':

    batch_size = 5
    n_b = 2
    n_f = 2
    N = 100
    u_in = np.random.rand(batch_size, N, 1).astype(np.float)

    y_init = np.random.rand(*(batch_size, n_f)).astype(np.float)
    u_init = np.random.rand(*(batch_size, n_b)).astype(np.float)

    # coefficients of a 2nd order oscillator
    b_coeff = np.array([0.0706464146944544, 0], dtype=np.float)
    a_coeff = np.array([-1.87212998940304, 0.942776404097492], dtype=np.float)

    a_poly = np.r_[1.0, a_coeff]
    b_poly = np.r_[0.0, b_coeff]

    zi = lfiltic_vec(b_poly, a_poly, y_init, u_init)  # initial condition

    zi_loop = np.empty_like(zi)
    for batch_idx in range(batch_size):
        zi_loop[batch_idx, :] = sp.signal.lfiltic(b_poly, a_poly, y_init[batch_idx, :], u_init[batch_idx, :])  # initial condition


#    y_out_comp = lfilter_mimo_components_jit(b_poly, a_poly, u_in)
#    y_out = lfilter_mimo_components_channels_last
#    yout_vec, _ = sp.signal.lfilter(b_poly, f_poly, u_in, axis=0, zi=zi.T)

#    yout_loop = np.empty_like(yout_vec)
#    for batch_idx in range(batch_size):
#        yout_loop[:, batch_idx] = sp.signal.lfilter(b_poly, f_poly, u_in[:, batch_idx], axis=0, zi=zi[batch_idx, :])[0]

