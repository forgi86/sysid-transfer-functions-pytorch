import torch
import numpy as np
import scipy as sp
import scipy.signal
from util.filtering import lfilter_mimo_components_jit
from util.filtering import lfilter_mimo_components_sep_jit
from util.filtering import lfilter_mimo_components_bsens_jit
from util.filtering import lfilter_mimo_components_asens_jit
from util.filtering import compute_grad_coeff_jit


class MimoLinearDynamicalOperatorFun(torch.autograd.Function):
    r"""Applies a multi-input-multi-output linear dynamical filtering operation: :math:`y = G(u)`.

    Examples::

        >>> G = MimoLinearDynamicalOperatorFun.apply
        >>> N = 500
        >>> y_0 = torch.zeros(n_a, dtype=torch.double)
        >>> u_0 = torch.zeros(n_b, dtype=torch.double)
        >>> b_coeff = torch.tensor([0.0706464146944544, 0], dtype=torch.double, requires_grad=True)  # b_1, b_2
        >>> a_coeff = torch.tensor([-1.872112998940304, 0.942776404097492], dtype=torch.double, requires_grad=True)  # a_1, a_2
        >>> inputs = (b_coeff, a_coeff, u_in, y_0, u_0)
        >>> Y = G(*inputs)
        >>> print(Y.size())
        torch.Size([500, 1])
    """

    @staticmethod
    def forward(ctx, b_coeff, a_coeff, u_in, y_0=None, u_0=None):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """

        # detach tensors so we can cast to numpy

        b_coeff, a_coeff, u_in = b_coeff.detach(), a_coeff.detach(), u_in.detach()

        if y_0 is not None:
            y_0.detach()

        if u_0 is not None:
            u_0.detach()

        # useful parameters
        out_channels = b_coeff.shape[0]
        in_channels = b_coeff.shape[1]
        n_a = a_coeff.shape[2]
        n_b = b_coeff.shape[2]

        # construct the A(q) polynomial with coefficient a_0=1
        M = n_b  # numerator coefficients
        N = n_a + 1  # denominator coefficients
        if M > N:
            b_poly = b_coeff.numpy()
            a_poly = np.c_[np.ones_like(a_coeff, shape=(out_channels, in_channels, 1)), a_coeff, np.zeros_like(a_coeff, shape=(out_channels, in_channels, M - N))]
        elif N > M:
            b_poly = np.c_[b_coeff, np.zeros_like(b_coeff, shape=(out_channels, in_channels, N - M))]
            a_poly = np.c_[np.ones_like(a_coeff, shape=(out_channels, in_channels, 1)), a_coeff]
        else:
            b_poly = b_coeff.numpy()
            a_poly = np.c_[np.ones_like(a_coeff, shape=(out_channels, in_channels, 1)), a_coeff]

        y_out_comp = lfilter_mimo_components_jit(b_poly, a_poly, u_in.numpy())  # [B, T, O, I]
        y_out = np.sum(y_out_comp, axis=-1)  # [B, T, O]
        y_out = torch.as_tensor(y_out, dtype=u_in.dtype)
        y_out_comp = torch.as_tensor(y_out_comp)

        ctx.save_for_backward(b_coeff, a_coeff, u_in, y_0, u_0, y_out_comp)
        return y_out

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """

        debug = False
        if debug:
            import pydevd  # required to debug the backward pass. Why?!?
            pydevd.settrace(suspend=False, trace_only_current_thread=True)

        b_coeff, a_coeff, u_in, y_0, u_0, y_out_comp = ctx.saved_tensors
        grad_b = grad_a = grad_u = grad_y0 = grad_u0 = None
        dtype_np = u_in.numpy().dtype

        out_channels, in_channels, n_b = b_coeff.shape
        n_a = a_coeff.shape[2]
        n_b = b_coeff.shape[2]
        batch_size, seq_len, _ = u_in.shape

        # construct the A(q) polynomial with coefficient a_0=1
        M = n_b  # numerator coefficients
        N = n_a + 1  # denominator coefficients
        if M > N:
            b_poly = b_coeff.numpy()
            a_poly = np.c_[np.ones_like(a_coeff, shape=(out_channels, in_channels, 1)), a_coeff, np.zeros_like(a_coeff, shape=(out_channels, in_channels, M - N))]
        elif N > M:
            b_poly = np.c_[b_coeff, np.zeros_like(b_coeff, shape=(out_channels, in_channels, N - M))]
            a_poly = np.c_[np.ones_like(a_coeff, shape=(out_channels, in_channels, 1)), a_coeff]
        else:
            b_poly = b_coeff.numpy()
            a_poly = np.c_[np.ones_like(a_coeff, shape=(out_channels, in_channels, 1)), a_coeff]

        #d0_np = np.zeros_like(a_poly, shape=(out_channels, in_channels, n_a+1))
        #d0_np[:, :, 0] = 1.0

        #d1_np = np.zeros_like(a_poly, shape=(out_channels, in_channels, n_a+1))
        #d1_np[:, :, 1] = 1.0

        if ctx.needs_input_grad[0]:  # b_coeff
            # compute forward sensitivities w.r.t. the b_i parameters

            #sens_b0 = lfilter_mimo_components_jit(d0_np, a_poly, u_in.numpy())
            sens_b0 = lfilter_mimo_components_bsens_jit(a_poly, u_in.numpy())
            grad_b = compute_grad_coeff_jit(sens_b0, grad_output.numpy(), n_b)
            grad_b = torch.as_tensor(grad_b)

        if ctx.needs_input_grad[1]:  # a_coeff

            sens_a1 = lfilter_mimo_components_asens_jit(a_poly, -y_out_comp.numpy())
            #sens_a1 = lfilter_mimo_components_sep_jit(d1_np, a_poly, -y_out_comp.numpy())
            grad_a = compute_grad_coeff_jit(sens_a1, grad_output.numpy(), n_a)
            grad_a = torch.as_tensor(grad_a)


        if ctx.needs_input_grad[2]: # u_in
            # compute jacobian w.r.t. u
            grad_output_flip = grad_output.numpy()[:, ::-1, :]  # [B, T, O]
            grad_u_comp = lfilter_mimo_components_jit(b_poly.swapaxes(0, 1), a_poly.swapaxes(0, 1), grad_output_flip)
            grad_u = np.sum(grad_u_comp, axis=-1)

            grad_u = np.array(grad_u[:, ::-1, :]).astype(dtype_np)

            grad_u = torch.as_tensor(grad_u)

        return grad_b, grad_a, grad_u, grad_y0, grad_u0


if __name__ == '__main__':

    import numpy as np
    import matplotlib.pyplot as plt
    from torch.autograd import gradcheck
    from torch.autograd.gradcheck import get_numerical_jacobian, get_analytical_jacobian


    # copied from torch.autograd.gradcheck
    def istuple(obj):
        # Usually instances of PyStructSequence is also an instance of tuple
        # but in some py2 environment it is not, so we have to manually check
        # the name of the type to determine if it is a namedtupled returned
        # by a pytorch operator.
        t = type(obj)
        return isinstance(obj, tuple) or t.__module__ == 'torch.return_types'


    # copied from torch.autograd.gradcheck
    def _as_tuple(x):
        if istuple(x):
            return x
        elif isinstance(x, list):
            return tuple(x)
        else:
            return x,

    # In[Setup problem]
    in_ch = 2
    out_ch = 5
    n_b = 8
    n_a = 5
    batch_size = 8
    seq_len = 16

    # In[Create system]

    b_coeff = torch.tensor(np.random.randn(*(out_ch, in_ch, n_b)), requires_grad=True)
    a_coeff = torch.tensor(np.random.rand(*(out_ch, in_ch, n_a)), requires_grad=True)
    G = MimoLinearDynamicalOperatorFun.apply
    y_0 = torch.tensor(0*np.random.randn(*(out_ch, in_ch, n_a)))
    u_0 = torch.tensor(0*np.random.randn(*(out_ch, in_ch, n_b)))
    u_in = torch.tensor(1*np.random.randn(*(batch_size, seq_len, in_ch)), requires_grad=True)
    inputs = (b_coeff, a_coeff, u_in, y_0, u_0)

    # In[Forward pass]
    y_out = G(*inputs)

    # In[Finite difference derivatives computation]
    def G_fun(input):
        return _as_tuple(G(*input))[0]
    numerical = get_numerical_jacobian(G_fun, inputs)

    # In[Autodiff derivatives computation]
    analytical, reentrant, correct_grad_sizes = get_analytical_jacobian(inputs, y_out)
    #torch.max(numerical[0]- analytical[0])
    test = gradcheck(G, inputs, eps=1e-6, atol=1e-4, raise_exception=True)


