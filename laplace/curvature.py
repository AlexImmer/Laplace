from abc import ABC, abstractmethod
import torch
from torch.nn import MSELoss, CrossEntropyLoss

from backpack import backpack, extend
from backpack.extensions import DiagGGNExact, DiagGGNMC, KFAC, KFLR, SumGradSquared

from laplace.jacobians import Jacobians
from laplace.matrix import Kron


class CurvatureInterface(ABC):

    def __init__(self, model, likelihood):
        assert likelihood in ['regression', 'classification']
        self.likelihood = likelihood
        self.model = model
        if likelihood == 'regression':
            self.lossfunc = MSELoss(reduction='sum')
            self.factor = 0.5  # convert to standard Gauss. log N(y|f,1)
        else:
            self.lossfunc = CrossEntropyLoss(reduction='sum')
            self.factor = 1.

    @abstractmethod
    def full(self, X, y, **kwargs):
        pass

    @abstractmethod
    def kron(self, X, y, **kwargs):
        pass

    @abstractmethod
    def diag(self, X, y, **kwargs):
        pass

    def _get_full_ggn(self, Js, f, y):
        loss = self.factor * self.lossfunc(f, y)
        if self.likelihood == 'regression':
            H_ggn = torch.einsum('mkp,mkq->pq', Js, Js)
        else:
            # second derivative of log lik is diag(p) - pp^T
            ps = torch.softmax(f, dim=-1)
            H_lik = torch.diag_embed(ps) - torch.einsum('mk,mc->mck', ps, ps)
            H_ggn = torch.einsum('mcp,mck,mkq->pq', Js, H_lik, Js)
        return loss, H_ggn


class LastLayer(CurvatureInterface):

    def __init__(self, model, likelihood, backend, **kwargs):
        super().__init__(model, likelihood)
        self.backend = backend(self.model, self.likelihood, last_layer=True, **kwargs)

    def diag(self, X, y, **kwargs):
        return self.backend.diag(X, y, **kwargs)

    def kron(self, X, y, **kwargs):
        return self.backend.kron(X, y, **kwargs)

    def full(self, X, y, **kwargs):
        f, phi = self.model.forward_with_features(X)
        bsize = len(X)
        output_size = f.shape[-1]

        # calculate Jacobians using the feature vector 'phi'
        identity = torch.eye(output_size, device=X.device)
        Js = torch.einsum('kp,ij->kijp', phi, identity).reshape(bsize, output_size, -1)
        if self.model.last_layer.bias is not None:
            Js = torch.cat([Js, identity.reshape(1, output_size, -1)], dim=2)

        loss, H = self._get_full_ggn(Js, f, y)

        return loss, H


class BackPackInterface(CurvatureInterface):

    def __init__(self, model, likelihood, last_layer=False):
        super().__init__(model, likelihood)
        self.last_layer = last_layer
        extend(self.model.last_layer) if last_layer else extend(self.model)
        extend(self.lossfunc)


class BackPackGGN(BackPackInterface):
    """[summary]

    MSELoss = |y-f|_2^2 -> d/df = -2(y-f)
    log N(y|f,1) \propto 1/2|y-f|_2^2 -> d/df = -(y-f)
    --> factor for regression is 0.5 for loss and ggn
    """

    def __init__(self, model, likelihood, last_layer=False, stochastic=False):
        super().__init__(model, likelihood, last_layer)
        self.stochastic = stochastic

    def _get_diag_ggn(self):
        if self.last_layer:
            model = self.model.last_layer
        else:
            model = self.model
        if self.stochastic:
            return torch.cat([p.diag_ggn_mc.data.flatten() for p in model.parameters()])
        else:
            return torch.cat([p.diag_ggn_exact.data.flatten() for p in model.parameters()])

    def _get_kron_factors(self):
        if self.last_layer:
            model = self.model.last_layer
        else:
            model = self.model
        if self.stochastic:
            return Kron([p.kfac for p in model.parameters()])
        else:
            return Kron([p.kflr for p in model.parameters()])

    def diag(self, X, y, **kwargs):
        context = DiagGGNMC if self.stochastic else DiagGGNExact
        f = self.model(X)
        loss = self.lossfunc(f, y)
        with backpack(context()):
            loss.backward()
        dggn = self._get_diag_ggn()

        return self.factor * loss, self.factor * dggn

    def kron(self, X, y, **wkwargs):
        context = KFAC if self.stochastic else KFLR
        f = self.model(X)
        loss = self.lossfunc(f, y)
        with backpack(context()):
            loss.backward()
        kron = self._get_kron_factors()

        return self.factor * loss, self.factor * kron

    def full(self, X, y, **kwargs):
        if self.stochastic:
            raise ValueError('Stochastic approximation not implemented for full GGN.')

        Js, f = Jacobians(self.model, X)
        loss, H_ggn = self._get_full_ggn(Js, f, y)

        return loss, H_ggn


class BackPackEF(BackPackInterface):

    def diag(self, X, y, **kwargs):
        f = self.model(X)
        loss = self.lossfunc(f, y)
        with backpack(SumGradSquared()):
            loss.backward()
        diag_EF = torch.cat([p.sum_grad_squared.data.flatten()
                             for p in self.model.parameters()])

        # TODO: self.factor * 2 here? To get true grad * grad for regression
        return self.factor * loss, self.factor * diag_EF

    def kron(self, X, y, **kwargs):
        raise NotImplementedError()

    def full(self, X, y, **kwargs):
        pass
