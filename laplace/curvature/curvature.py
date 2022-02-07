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
    model : torch.nn.Module or `laplace.utils.feature_extractor.FeatureExtractor`
        torch model (neural network)
    likelihood : {'classification', 'regression'}
    last_layer : bool, default=False
        only consider curvature of last layer
    subnetwork_indices : torch.Tensor, default=None
        indices of the vectorized model parameters that define the subnetwork
        to apply the Laplace approximation over

    Attributes
    ----------
    lossfunc : torch.nn.MSELoss or torch.nn.CrossEntropyLoss
    factor : float
        conversion factor between torch losses and base likelihoods
        For example, \\(\\frac{1}{2}\\) to get to \\(\\mathcal{N}(f, 1)\\) from MSELoss.
    """
    def __init__(self, model, likelihood, last_layer=False, subnetwork_indices=None):
        assert likelihood in ['regression', 'classification']
        self.likelihood = likelihood
        self.model = model
        self.last_layer = last_layer
        self.subnetwork_indices = subnetwork_indices
        if likelihood == 'regression':
            self.lossfunc = MSELoss(reduction='sum')
            self.factor = 0.5
        else:
            self.lossfunc = CrossEntropyLoss(reduction='sum')
            self.factor = 1.

    @property
    def _model(self):
        return self.model.last_layer if self.last_layer else self.model

    def jacobians(self, x):
        """Compute Jacobians \\(\\nabla_\\theta f(x;\\theta)\\) at current parameter \\(\\theta\\).

        Parameters
        ----------
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

    def last_layer_jacobians(self, x):
        """Compute Jacobians \\(\\nabla_{\\theta_\\textrm{last}} f(x;\\theta_\\textrm{last})\\) 
        only at current last-layer parameter \\(\\theta_{\\textrm{last}}\\).

        Parameters
        ----------
        x : torch.Tensor

        Returns
        -------
        Js : torch.Tensor
            Jacobians `(batch, last-layer-parameters, outputs)`
        f : torch.Tensor
            output function `(batch, outputs)`
        """
        f, phi = self.model.forward_with_features(x)
        bsize = phi.shape[0]
        output_size = f.shape[-1]

        # calculate Jacobians using the feature vector 'phi'
        identity = torch.eye(output_size, device=x.device).unsqueeze(0).tile(bsize, 1, 1)
        # Jacobians are batch x output x params
        Js = torch.einsum('kp,kij->kijp', phi, identity).reshape(bsize, output_size, -1)
        if self.model.last_layer.bias is not None:
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
        H : `laplace.utils.matrix.Kron`
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
    model : torch.nn.Module or `laplace.utils.feature_extractor.FeatureExtractor`
        torch model (neural network)
    likelihood : {'classification', 'regression'}
    last_layer : bool, default=False
        only consider curvature of last layer
    subnetwork_indices : torch.Tensor, default=None
        indices of the vectorized model parameters that define the subnetwork
        to apply the Laplace approximation over
    stochastic : bool, default=False
        Fisher if stochastic else GGN
    """
    def __init__(self, model, likelihood, last_layer=False, subnetwork_indices=None, stochastic=False):
        self.stochastic = stochastic
        super().__init__(model, likelihood, last_layer, subnetwork_indices)

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
        if self.likelihood == 'regression':
            H_ggn = torch.einsum('mkp,mkq->pq', Js, Js)
        else:
            H_lik = self.H_log_likelihood(f)
            H_ggn = torch.einsum('mcp,mck,mkq->pq', Js, H_lik, Js)
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
            raise ValueError('Stochastic approximation not implemented for full GGN.')

        if self.last_layer:
            Js, f = self.last_layer_jacobians(x)
        else:
            Js, f = self.jacobians(x)
        loss, H_ggn = self._get_full_ggn(Js, f, y)

        return loss, H_ggn

    def H_log_likelihood(self, f, sigma_factor=None):
        """
        Second derivative (Hessian) of log-likelihood w.r.t. the output of NN \\(f\\)

        Parameters
        ----------
        f: torch.Tensor
            Output of the last layer of NN (before softmax layer)
        sigma_factor: torch.Tensor
            Precision (scaled with temperature) in regression likelihood. See _H_factor property in BaseLaplace

        Returns
        -------
        H_lik: torch.Tensor
              Hessian of \\(p(y|f)\\) w.r.t. \\(f\\) (batch, output_shape, output_shape)
        """
        if self.likelihood == 'regression':
            assert sigma_factor is not None, 'sigma_factor should be provided for regression'
            # second derivative is (1 / sigma^2) * I_{C}
            b, C = f.shape
            H_lik = sigma_factor * torch.unsqueeze(torch.eye(C), 0).repeat(b, 1, 1)
        else:
            # second derivative of log lik is diag(p) - pp^T
            ps = torch.softmax(f, dim=-1)
            H_lik = torch.diag_embed(ps) - torch.einsum('mk,mc->mck', ps, ps)
        return H_lik

    def gp_quantities(self, X, y, sigma_factor):
        """
         Parameters
        ----------
        x : torch.Tensor
            input data `(batch, input_shape)`
        y : torch.Tensor
            labels `(batch, output_shape)`
        sigma_factor: inverse of (scaled) likelihood noise

        Returns
        -------
        loss : torch.tensor
        Js : torch.tensor
              Jacobians (batch, output_shape, parameters)
        f : torch.tensor
              NN output (batch, output_shape)
        lambdas: torch.tensor
              Hessian of \\( p(y|f) \\) w.r.t. \\(f\\) (batch, output_shape, output_shape)
        """
        if self.last_layer:
            Js, f = self.last_layer_jacobians(X)
        else:
            Js, f = self.jacobians(X)
        lambdas = self.H_log_likelihood(f, sigma_factor)
        loss = self.factor * self.lossfunc(f, y)
        return loss.detach(), Js, f, lambdas


class EFInterface(CurvatureInterface):
    """Interface for Empirical Fisher as Hessian approximation.
    In addition to `CurvatureInterface`, methods for gradients are required by subclasses.

    Parameters
    ----------
    model : torch.nn.Module or `laplace.utils.feature_extractor.FeatureExtractor`
        torch model (neural network)
    likelihood : {'classification', 'regression'}
    last_layer : bool, default=False
        only consider curvature of last layer
    subnetwork_indices : torch.Tensor, default=None
        indices of the vectorized model parameters that define the subnetwork
        to apply the Laplace approximation over

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
