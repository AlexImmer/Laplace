import torch

from backpack import backpack, extend, memory_cleanup
from backpack.extensions import (
    DiagGGNExact,
    DiagGGNMC,
    KFAC,
    KFLR,
    SumGradSquared,
    BatchGrad,
)
from backpack.context import CTX

from laplace.curvature import CurvatureInterface, GGNInterface, EFInterface
from laplace.matrix import Kron


class BackPackInterface(CurvatureInterface):
    """Interface for Backpack backend."""

    def __init__(self, model, likelihood, last_layer=False):
        super().__init__(model, likelihood, last_layer)
        extend(self._model)
        extend(self.lossfunc)

    @staticmethod
    def jacobians(model, x):
        """Compute Jacobians \\(\\nabla_{\\theta} f(x;\\theta)\\) at current parameter \\(\\theta\\)
        using backpack's BatchGrad per output dimension.

        Parameters
        ----------
        model : torch.nn.Module
        x : torch.Tensor
            input data `(batch, input_shape)` on compatible device with model.

        Returns
        -------
        Js : torch.Tensor
            Jacobians `(batch, parameters, outputs)`
        f : torch.Tensor
            output function `(batch, outputs)`
        """
        model = extend(model)
        to_stack = []
        for i in range(model.output_size):
            model.zero_grad()
            out = model(x)
            with backpack(BatchGrad()):
                if model.output_size > 1:
                    out[:, i].sum().backward()
                else:
                    out.sum().backward()
                to_cat = []
                for param in model.parameters():
                    to_cat.append(param.grad_batch.detach().reshape(x.shape[0], -1))
                    delattr(param, "grad_batch")
                Jk = torch.cat(to_cat, dim=1)
            to_stack.append(Jk)
            if i == 0:
                f = out.detach()

        model.zero_grad()
        CTX.remove_hooks()
        _cleanup(model)
        if model.output_size > 1:
            return torch.stack(to_stack, dim=2).transpose(1, 2), f
        else:
            return Jk.unsqueeze(-1).transpose(1, 2), f

    def gradients(self, x, y):
        """Compute gradients \\(\\nabla_\\theta \\ell(f(x;\\theta, y)\\) at current parameter
        \\(\\theta\\) using Backpack's BatchGrad.

        Parameters
        ----------
        x : torch.Tensor
            input data `(batch, input_shape)` on compatible device with model.
        y : torch.Tensor

        Returns
        -------
        loss : torch.Tensor
        Gs : torch.Tensor
            gradients `(batch, parameters)`
        """
        f = self.model(x)
        loss = self.lossfunc(f, y)
        with backpack(BatchGrad()):
            loss.backward()
        Gs = torch.cat(
            [p.grad_batch.data.flatten(start_dim=1) for p in self._model.parameters()],
            dim=1,
        )
        return Gs, loss


class BackPackGGN(BackPackInterface, GGNInterface):
    """Implementation of the `GGNInterface` using Backpack."""

    def __init__(self, model, likelihood, last_layer=False, stochastic=False):
        super().__init__(model, likelihood, last_layer)
        self.stochastic = stochastic

    def _get_diag_ggn(self):
        if self.stochastic:
            return torch.cat(
                [p.diag_ggn_mc.data.flatten() for p in self._model.parameters()]
            )
        else:
            return torch.cat(
                [p.diag_ggn_exact.data.flatten() for p in self._model.parameters()]
            )

    def _get_kron_factors(self):
        if self.stochastic:
            return Kron([p.kfac for p in self._model.parameters()])
        else:
            return Kron([p.kflr for p in self._model.parameters()])

    @staticmethod
    def _rescale_kron_factors(kron, M, N):
        # Renormalize Kronecker factor to sum up correctly over N data points with batches of M
        # for M=N (full-batch) just M/N=1
        for F in kron.kfacs:
            if len(F) == 2:
                F[1] *= M / N
        return kron

    def diag(self, X, y, **kwargs):
        context = DiagGGNMC if self.stochastic else DiagGGNExact
        f = self.model(X)
        loss = self.lossfunc(f, y)
        with backpack(context()):
            loss.backward()
        dggn = self._get_diag_ggn()

        return self.factor * loss.detach(), self.factor * dggn

    def kron(self, X, y, N, **kwargs) -> [torch.Tensor, Kron]:
        context = KFAC if self.stochastic else KFLR
        f = self.model(X)
        loss = self.lossfunc(f, y)
        with backpack(context()):
            loss.backward()
        kron = self._get_kron_factors()
        kron = self._rescale_kron_factors(kron, len(y), N)

        return self.factor * loss.detach(), self.factor * kron


class BackPackEF(BackPackInterface, EFInterface):
    """Implementation of `EFInterface` using Backpack."""

    def diag(self, X, y, **kwargs):
        f = self.model(X)
        loss = self.lossfunc(f, y)
        with backpack(SumGradSquared()):
            loss.backward()
        diag_EF = torch.cat(
            [p.sum_grad_squared.data.flatten() for p in self._model.parameters()]
        )

        return self.factor * loss.detach(), self.factor * diag_EF

    def kron(self, X, y, **kwargs):
        raise NotImplementedError("Unavailable through Backpack.")


def _cleanup(module):
    for child in module.children():
        _cleanup(child)

    setattr(module, "_backpack_extend", False)
    memory_cleanup(module)
