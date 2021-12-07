import logging
from typing import Union
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d, BatchNorm2d, BatchNorm3d
from torch.distributions.multivariate_normal import _precision_to_scale_tril


def get_nll(out_dist, targets):
    return F.nll_loss(torch.log(out_dist), targets)


@torch.no_grad()
def validate(laplace, val_loader, pred_type="glm", link_approx="probit", n_samples=100):
    laplace.model.eval()
    output_means, output_vars = list(), list()
    targets = list()
    for X, y in val_loader:
        X, y = X.to(laplace._device), y.to(laplace._device)
        out = laplace(
            X, pred_type=pred_type, link_approx=link_approx, n_samples=n_samples
        )

        if type(out) == tuple:
            output_means.append(out[0])
            output_vars.append(out[1])
        else:
            output_means.append(out)

        targets.append(y)

    if len(output_vars) == 0:
        return torch.cat(output_means, dim=0), torch.cat(targets, dim=0)
    return (
        (torch.cat(output_means, dim=0), torch.cat(output_vars, dim=0)),
        torch.cat(targets, dim=0),
    )


def parameters_per_layer(model):
    """Get number of parameters per layer.

    Parameters
    ----------
    model : torch.nn.Module

    Returns
    -------
    params_per_layer : list[int]
    """
    return [np.prod(p.shape) for p in model.parameters()]


def invsqrt_precision(M):
    """Compute ``M^{-0.5}`` as a tridiagonal matrix.

    Parameters
    ----------
    M : torch.Tensor

    Returns
    -------
    M_invsqrt : torch.Tensor
    """
    return _precision_to_scale_tril(M)


def _is_batchnorm(module):
    if (
        isinstance(module, BatchNorm1d)
        or isinstance(module, BatchNorm2d)
        or isinstance(module, BatchNorm3d)
    ):
        return True
    return False


def _is_valid_scalar(scalar: Union[float, int, torch.Tensor]) -> bool:
    if np.isscalar(scalar) and np.isreal(scalar):
        return True
    elif torch.is_tensor(scalar) and scalar.ndim <= 1:
        if scalar.ndim == 1 and len(scalar) != 1:
            return False
        return True
    return False


def kron(t1, t2):
    """Computes the Kronecker product between two tensors.

    Parameters
    ----------
    t1 : torch.Tensor
    t2 : torch.Tensor

    Returns
    -------
    kron_product : torch.Tensor
    """
    t1_height, t1_width = t1.size()
    t2_height, t2_width = t2.size()
    out_height = t1_height * t2_height
    out_width = t1_width * t2_width

    tiled_t2 = t2.repeat(t1_height, t1_width)
    expanded_t1 = (
        t1.unsqueeze(2)
        .unsqueeze(3)
        .repeat(1, t2_height, t2_width, 1)
        .view(out_height, out_width)
    )

    return expanded_t1 * tiled_t2


def diagonal_add_scalar(X, value):
    """Add scalar value `value` to diagonal of `X`.

    Parameters
    ----------
    X : torch.Tensor
    value : torch.Tensor or float

    Returns
    -------
    X_add_scalar : torch.Tensor
    """
    if not X.device == torch.device("cpu"):
        indices = torch.cuda.LongTensor([[i, i] for i in range(X.shape[0])])
    else:
        indices = torch.LongTensor([[i, i] for i in range(X.shape[0])])
    values = X.new_ones(X.shape[0]).mul(value)
    return X.index_put(tuple(indices.t()), values, accumulate=True)


def symeig(M):
    """Symetric eigendecomposition avoiding failure cases by
    adding and removing jitter to the diagonal.

    Parameters
    ----------
    M : torch.Tensor

    Returns
    -------
    L : torch.Tensor
        eigenvalues
    W : torch.Tensor
        eigenvectors
    """
    try:
        L, W = torch.linalg.eigh(M, UPLO="U")
    except RuntimeError:  # did not converge
        logging.info("SYMEIG: adding jitter, did not converge.")
        # use W L W^T + I = W (L + I) W^T
        M = M + torch.eye(M.shape[0])
        try:
            L, W = torch.linalg.eigh(M, UPLO="U")
            L -= 1.0
        except RuntimeError:
            stats = f"diag: {M.diagonal()}, max: {M.abs().max()}, "
            stats = stats + f"min: {M.abs().min()}, mean: {M.abs().mean()}"
            logging.info(f"SYMEIG: adding jitter failed. Stats: {stats}")
            exit()
    # eigenvalues of symeig at least 0
    L = L.clamp(min=0.0)
    L = torch.nan_to_num(L)
    W = torch.nan_to_num(W)
    return L, W


def block_diag(blocks):
    """Compose block-diagonal matrix of individual blocks.

    Parameters
    ----------
    blocks : list[torch.Tensor]

    Returns
    -------
    M : torch.Tensor
    """
    P = sum([b.shape[0] for b in blocks])
    M = torch.zeros(P, P)
    p_cur = 0
    for block in blocks:
        p_block = block.shape[0]
        M[p_cur : p_cur + p_block, p_cur : p_cur + p_block] = block
        p_cur += p_block
    return M
