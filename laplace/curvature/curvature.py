import torch
from torch.nn import MSELoss, CrossEntropyLoss


class CurvatureInterface:
    """Interface to access curvature for a model and corresponding likelihood.
    A `CurvatureInterface` must inherit from this baseclass and implement the
    necessary functions `jacobians`, `full`, `kron`, and `diag`.
    The interface might be extended in the future to account for other curvature
    structures, for example, a block-diagonal one.

    Parameters
    ----------
    model : torch.nn.Module or `laplace.feature_extractor.FeatureExtractor`
        torch model (neural network)
    likelihood : {'classification', 'regression'}
    last_layer : bool, default=False
        only consider curvature of last layer

    Attributes
    ----------
    lossfunc : torch.nn.MSELoss or torch.nn.CrossEntropyLoss
    factor : float
        conversion factor between torch losses and base likelihoods
        For example, \\(\\frac{1}{2}\\) to get to \\(\\mathcal{N}(f, 1)\\) from MSELoss.
    """

    def __init__(self, model, likelihood, last_layer=False):
        assert likelihood in ["regression", "classification"]
        self.likelihood = likelihood
        self.model = model
        self.last_layer = last_layer
        if likelihood == "regression":
            self.lossfunc = MSELoss(reduction="sum")
            self.factor = 0.5
        else:
            self.lossfunc = CrossEntropyLoss(reduction="sum")
            self.factor = 1.0

    @property
    def _model(self):
        return self.model.last_layer if self.last_layer else self.model

    @staticmethod
    def jacobians(model, x):
        """Compute Jacobians \\(\\nabla_\\theta f(x;\\theta)\\) at current parameter \\(\\theta\\).

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
        raise NotImplementedError

    @staticmethod
    def last_layer_jacobians(model, x):
        """Compute Jacobians \\(\\nabla_{\\theta_\\textrm{last}} f(x;\\theta_\\textrm{last})\\)
        only at current last-layer parameter \\(\\theta_{\\textrm{last}}\\).

        Parameters
        ----------
        model : laplace.feature_extractor.FeatureExtractor
        x : torch.Tensor

        Returns
        -------
        Js : torch.Tensor
            Jacobians `(batch, last-layer-parameters, outputs)`
        f : torch.Tensor
            output function `(batch, outputs)`
        """
        f, phi = model.forward_with_features(x)
        bsize = phi.shape[0]
        output_size = f.shape[-1]

        # calculate Jacobians using the feature vector 'phi'
        identity = (
            torch.eye(output_size, device=x.device).unsqueeze(0).tile(bsize, 1, 1)
        )
        # Jacobians are batch x output x params
        Js = torch.einsum("kp,kij->kijp", phi, identity).reshape(bsize, output_size, -1)
        if model.last_layer.bias is not None:
            Js = torch.cat([Js, identity], dim=2)

        return Js, f.detach()

    def gradients(self, x, y):
        """Compute gradients \\(\\nabla_\\theta \\ell(f(x;\\theta, y)\\) at current parameter \\(\\theta\\).

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
        raise NotImplementedError

    def full(self, x, y, **kwargs):
        """Compute a dense curvature (approximation) in the form of a \\(P \\times P\\) matrix
        \\(H\\) with respect to parameters \\(\\theta \\in \\mathbb{R}^P\\).

        Parameters
        ----------
        x : torch.Tensor
            input data `(batch, input_shape)`
        y : torch.Tensor
            labels `(batch, label_shape)`

        Returns
        -------
        loss : torch.Tensor
        H : torch.Tensor
            Hessian approximation `(parameters, parameters)`
        """
        raise NotImplementedError

    def kron(self, x, y, **kwargs):
        """Compute a Kronecker factored curvature approximation (such as KFAC).
        The approximation to \\(H\\) takes the form of two Kronecker factors \\(Q, H\\),
        i.e., \\(H \\approx Q \\otimes H\\) for each Module in the neural network permitting
        such curvature.
        \\(Q\\) is quadratic in the input-dimension of a module \\(p_{in} \\times p_{in}\\)
        and \\(H\\) in the output-dimension \\(p_{out} \\times p_{out}\\).

        Parameters
        ----------
        x : torch.Tensor
            input data `(batch, input_shape)`
        y : torch.Tensor
            labels `(batch, label_shape)`

        Returns
        -------
        loss : torch.Tensor
        H : `laplace.matrix.Kron`
            Kronecker factored Hessian approximation.
        """
        raise NotImplementedError

    def diag(self, x, y, **kwargs):
        """Compute a diagonal Hessian approximation to \\(H\\) and is represented as a
        vector of the dimensionality of parameters \\(\\theta\\).

        Parameters
        ----------
        x : torch.Tensor
            input data `(batch, input_shape)`
        y : torch.Tensor
            labels `(batch, label_shape)`

        Returns
        -------
        loss : torch.Tensor
        H : torch.Tensor
            vector representing the diagonal of H
        """
        raise NotImplementedError


class GGNInterface(CurvatureInterface):
    """Generalized Gauss-Newton or Fisher Curvature Interface.
    The GGN is equal to the Fisher information for the available likelihoods.
    In addition to `CurvatureInterface`, methods for Jacobians are required by subclasses.

    Parameters
    ----------
    model : torch.nn.Module or `laplace.feature_extractor.FeatureExtractor`
        torch model (neural network)
    likelihood : {'classification', 'regression'}
    last_layer : bool, default=False
        only consider curvature of last layer
    stochastic : bool, default=False
        Fisher if stochastic else GGN
    """

    def __init__(self, model, likelihood, last_layer=False, stochastic=False):
        self.stochastic = stochastic
        super().__init__(model, likelihood, last_layer)

    def _get_full_ggn(self, Js, f, y):
        """Compute full GGN from Jacobians.

        Parameters
        ----------
        Js : torch.Tensor
            Jacobians `(batch, parameters, outputs)`
        f : torch.Tensor
            functions `(batch, outputs)`
        y : torch.Tensor
            labels compatible with loss

        Returns
        -------
        loss : torch.Tensor
        H_ggn : torch.Tensor
            full GGN approximation `(parameters, parameters)`
        """
        loss = self.factor * self.lossfunc(f, y)
        if self.likelihood == "regression":
            H_ggn = torch.einsum("mkp,mkq->pq", Js, Js)
        else:
            # second derivative of log lik is diag(p) - pp^T
            ps = torch.softmax(f, dim=-1)
            H_lik = torch.diag_embed(ps) - torch.einsum("mk,mc->mck", ps, ps)
            H_ggn = torch.einsum("mcp,mck,mkq->pq", Js, H_lik, Js)
        return loss.detach(), H_ggn

    def full(self, x, y, **kwargs):
        """Compute the full GGN \\(P \\times P\\) matrix as Hessian approximation
        \\(H_{ggn}\\) with respect to parameters \\(\\theta \\in \\mathbb{R}^P\\).
        For last-layer, reduced to \\(\\theta_{last}\\)

        Parameters
        ----------
        x : torch.Tensor
            input data `(batch, input_shape)`
        y : torch.Tensor
            labels `(batch, label_shape)`

        Returns
        -------
        loss : torch.Tensor
        H_ggn : torch.Tensor
            GGN `(parameters, parameters)`
        """
        if self.stochastic:
            raise ValueError("Stochastic approximation not implemented for full GGN.")

        if self.last_layer:
            Js, f = self.last_layer_jacobians(self.model, x)
        else:
            Js, f = self.jacobians(self.model, x)
        loss, H_ggn = self._get_full_ggn(Js, f, y)

        return loss, H_ggn


class EFInterface(CurvatureInterface):
    """Interface for Empirical Fisher as Hessian approximation.
    In addition to `CurvatureInterface`, methods for gradients are required by subclasses.

    Parameters
    ----------
    model : torch.nn.Module or `laplace.feature_extractor.FeatureExtractor`
        torch model (neural network)
    likelihood : {'classification', 'regression'}
    last_layer : bool, default=False
        only consider curvature of last layer

    Attributes
    ----------
    lossfunc : torch.nn.MSELoss or torch.nn.CrossEntropyLoss
    factor : float
        conversion factor between torch losses and base likelihoods
        For example, \\(\\frac{1}{2}\\) to get to \\(\\mathcal{N}(f, 1)\\) from MSELoss.
    """

    def full(self, x, y, **kwargs):
        """Compute the full EF \\(P \\times P\\) matrix as Hessian approximation
        \\(H_{ef}\\) with respect to parameters \\(\\theta \\in \\mathbb{R}^P\\).
        For last-layer, reduced to \\(\\theta_{last}\\)

        Parameters
        ----------
        x : torch.Tensor
            input data `(batch, input_shape)`
        y : torch.Tensor
            labels `(batch, label_shape)`

        Returns
        -------
        loss : torch.Tensor
        H_ef : torch.Tensor
            EF `(parameters, parameters)`
        """
        Gs, loss = self.gradients(x, y)
        H_ef = Gs.T @ Gs
        return self.factor * loss.detach(), self.factor * H_ef
