from math import sqrt, pi
import numpy as np
import torch
import warnings
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.distributions import MultivariateNormal, Dirichlet, Normal
from torch.utils.data import DataLoader

from laplace.utils import parameters_per_layer, invsqrt_precision, get_nll, validate, SoDSampler
from laplace.matrix import Kron
from laplace.curvature import BackPackGGN, BackPackEF, BackPackInterface, AsdlGGN, AsdlEF, AsdlInterface


__all__ = ['BaseLaplace', 'FullLaplace', 'KronLaplace', 'DiagLaplace', 'ParametricLaplace']


class BaseLaplace:
    """Baseclass for all Laplace approximations in this library.

    Parameters
    ----------
    model : torch.nn.Module
    likelihood : {'classification', 'regression'}
        determines the log likelihood Hessian approximation
    sigma_noise : torch.Tensor or float, default=1
        observation noise for the regression setting; must be 1 for classification
    prior_precision : torch.Tensor or float, default=1
        prior precision of a Gaussian prior (= weight decay);
        can be scalar, per-layer, or diagonal in the most general case
    prior_mean : torch.Tensor or float, default=0
        prior mean of a Gaussian prior, useful for continual learning
    temperature : float, default=1
        temperature of the likelihood; lower temperature leads to more
        concentrated posterior and vice versa.
    backend : subclasses of `laplace.curvature.CurvatureInterface`
        backend for access to curvature/Hessian approximations
    backend_kwargs : dict, default=None
        arguments passed to the backend on initialization, for example to
        set the number of MC samples for stochastic approximations.
    """
    def __init__(self, model, likelihood, sigma_noise=1., prior_precision=1.,
                 prior_mean=0., temperature=1., backend=BackPackGGN, backend_kwargs=None):
        if likelihood not in ['classification', 'regression']:
            raise ValueError(f'Invalid likelihood type {likelihood}')

        self.model = model
        self._device = next(model.parameters()).device

        self.n_params = len(parameters_to_vector(self.model.parameters()).detach())
        self.n_layers = len(list(self.model.parameters()))
        self.prior_precision = prior_precision
        self.prior_mean = prior_mean
        if sigma_noise != 1 and likelihood != 'regression':
            raise ValueError('Sigma noise != 1 only available for regression.')
        self.likelihood = likelihood
        self.sigma_noise = sigma_noise
        self.temperature = temperature
        self._backend = None
        self._backend_cls = backend
        self._backend_kwargs = dict() if backend_kwargs is None else backend_kwargs

        # log likelihood = g(loss)
        self.loss = 0.
        self.n_outputs = None
        self.n_data = None

        # MAP estimate of parameters (for ParametricLaplace this corresponds to the posterior mean)
        self.map_estimate = parameters_to_vector(self.model.parameters()).detach()

    @property
    def backend(self):
        if self._backend is None:
            self._backend = self._backend_cls(self.model, self.likelihood,
                                              **self._backend_kwargs)
        return self._backend

    def _curv_closure(self, X, y, N):
        raise NotImplementedError

    def _check_fit(self):
        raise NotImplementedError

    def fit(self, train_loader):
        raise NotImplementedError

    def log_marginal_likelihood(self, prior_precision=None, sigma_noise=None):
        """Compute the approximation to the log marginal likelihood subject
        to specific Laplace or GP approximations .
        Requires that the Laplace approximation has been fit before.
        The resulting torch.Tensor is differentiable in `prior_precision` and
        `sigma_noise` if these have gradients enabled.
        By passing `prior_precision` or `sigma_noise`, the current value is
        overwritten. This is useful for iterating on the log marginal likelihood.

        Parameters
        ----------
        prior_precision : torch.Tensor, optional
            prior precision if should be changed from current `prior_precision` value
        sigma_noise : [type], optional
            observation noise standard deviation if should be changed

        Returns
        -------
        log_marglik : torch.Tensor
        """
        # make sure we can differentiate wrt prior and sigma_noise for regression
        self._check_fit()

        # update prior precision (useful when iterating on marglik)
        if prior_precision is not None:
            self.prior_precision = prior_precision

        # update sigma_noise (useful when iterating on marglik)
        if sigma_noise is not None:
            if self.likelihood != 'regression':
                raise ValueError('Can only change sigma_noise for regression.')
            self.sigma_noise = sigma_noise

        return self._log_marginal_likelihood()

    def _log_marginal_likelihood(self):
        raise NotImplementedError

    @property
    def log_likelihood(self):
        """Compute log likelihood on the training data after `.fit()` has been called.
        The log likelihood is computed on-demand based on the loss and, for example,
        the observation noise which makes it differentiable in the latter for
        iterative updates.

        Returns
        -------
        log_likelihood : torch.Tensor
        """
        self._check_fit()

        factor = - self._H_factor
        if self.likelihood == 'regression':
            # loss used is just MSE, need to add normalizer for gaussian likelihood
            c = self.n_data * self.n_outputs * torch.log(self.sigma_noise * sqrt(2 * pi))
            return factor * self.loss - c
        else:
            # for classification Xent == log Cat
            return factor * self.loss

    def __call__(self, x, pred_type, link_approx, n_samples):
        """Compute the posterior predictive on input data `X`.

        Parameters
        ----------
        x : torch.Tensor
            `(batch_size, input_shape)`

        pred_type : {'glm', 'nn', 'gp'},
            type of posterior predictive, linearized GLM predictive or neural
            network (nn) sampling predictive or Gaussian Process (GP) predictive.
            For ParametricLaplace subclasses, the GLM predictive is consistent with
            the curvature approximations used there.

        link_approx : {'mc', 'probit', 'bridge'}
            how to approximate the classification link function for the `'glm'`.
            For `pred_type='nn'`, only 'mc' is possible.

        n_samples : int
            number of samples for `link_approx='mc'`.

        Returns
        -------
        predictive: torch.Tensor or Tuple[torch.Tensor]
            For `likelihood='classification'`, a torch.Tensor is returned with
            a distribution over classes (similar to a Softmax).
            For `likelihood='regression'`, a tuple of torch.Tensor is returned
            with the mean and the predictive variance.
        """
        raise NotImplementedError

    @staticmethod
    def _classification_predictive(f_mu, f_var, link_approx, n_samples):
        """
        """

        if link_approx not in ['mc', 'probit', 'bridge']:
            raise ValueError(f'Unsupported link approximation {link_approx}.')

        if link_approx == 'mc':
            try:
                dist = MultivariateNormal(f_mu, f_var)
            except:
                dist = Normal(f_mu, torch.diagonal(f_var, dim1=1, dim2=2).sqrt())
            return torch.softmax(dist.sample((n_samples,)), dim=-1).mean(dim=0)
        elif link_approx == 'probit':
            kappa = 1 / torch.sqrt(1. + np.pi / 8 * f_var.diagonal(dim1=1, dim2=2))
            return torch.softmax(kappa * f_mu, dim=-1)
        elif link_approx == 'bridge':
            _, K = f_mu.size(0), f_mu.size(-1)
            f_var_diag = torch.diagonal(f_var, dim1=1, dim2=2)
            sum_exp = torch.sum(torch.exp(-f_mu), dim=1).unsqueeze(-1)
            alpha = 1 / f_var_diag * (1 - 2 / K + torch.exp(f_mu) / (K ** 2) * sum_exp)
            dist = Dirichlet(alpha)
            return torch.nan_to_num(dist.mean, nan=1.0)

    def predictive(self, x, pred_type, link_approx, n_samples):
        return self(x, pred_type, link_approx, n_samples)

    def _check_jacobians(self, Js):
        if not isinstance(Js, torch.Tensor):
            raise ValueError('Jacobians have to be torch.Tensor.')
        if not Js.device == self._device:
            raise ValueError('Jacobians need to be on the same device as Laplace.')
        m, k, p = Js.size()
        if p != self.n_params:
            raise ValueError('Invalid Jacobians shape for Laplace posterior approx.')

    @property
    def prior_precision_diag(self):
        """Obtain the diagonal prior precision \\(p_0\\) constructed from either
        a scalar, layer-wise, or diagonal prior precision.

        Returns
        -------
        prior_precision_diag : torch.Tensor
        """
        if len(self.prior_precision) == 1:  # scalar
            return self.prior_precision * torch.ones(self.n_params, device=self._device)

        elif len(self.prior_precision) == self.n_params:  # diagonal
            return self.prior_precision

        elif len(self.prior_precision) == self.n_layers:  # per layer
            n_params_per_layer = parameters_per_layer(self.model)
            return torch.cat([prior * torch.ones(n_params, device=self._device) for prior, n_params
                              in zip(self.prior_precision, n_params_per_layer)])

        else:
            raise ValueError('Mismatch of prior and model. Diagonal, scalar, or per-layer prior.')

    @property
    def prior_mean(self):
        return self._prior_mean

    @prior_mean.setter
    def prior_mean(self, prior_mean):
        if np.isscalar(prior_mean) and np.isreal(prior_mean):
            self._prior_mean = torch.tensor(prior_mean, device=self._device)
        elif torch.is_tensor(prior_mean):
            if prior_mean.ndim == 0:
                self._prior_mean = prior_mean.reshape(-1).to(self._device)
            elif prior_mean.ndim == 1:
                if not len(prior_mean) in [1, self.n_params]:
                    raise ValueError('Invalid length of prior mean.')
                self._prior_mean = prior_mean
            else:
                raise ValueError('Prior mean has too many dimensions!')
        else:
            raise ValueError('Invalid argument type of prior mean.')

    @property
    def prior_precision(self):
        return self._prior_precision

    @prior_precision.setter
    def prior_precision(self, prior_precision):
        self._posterior_scale = None
        if np.isscalar(prior_precision) and np.isreal(prior_precision):
            self._prior_precision = torch.tensor([prior_precision], device=self._device)
        elif torch.is_tensor(prior_precision):
            if prior_precision.ndim == 0:
                # make dimensional
                self._prior_precision = prior_precision.reshape(-1).to(self._device)
            elif prior_precision.ndim == 1:
                if len(prior_precision) not in [1, self.n_layers, self.n_params]:
                    raise ValueError('Length of prior precision does not align with architecture.')
                self._prior_precision = prior_precision.to(self._device)
            else:
                raise ValueError('Prior precision needs to be at most one-dimensional tensor.')
        else:
            raise ValueError('Prior precision either scalar or torch.Tensor up to 1-dim.')

    def optimize_prior_precision_base(self, pred_type, method='marglik', n_steps=100, lr=1e-1,
                                      init_prior_prec=1., val_loader=None, loss=get_nll,
                                      log_prior_prec_min=-4, log_prior_prec_max=4, grid_size=100,
                                      link_approx='probit', n_samples=100,
                                      verbose=False):
        """Optimize the prior precision post-hoc using the `method`
        specified by the user.

        Parameters
        ----------
        pred_type : {'glm', 'nn', 'gp'}, default='glm'
            type of posterior predictive, linearized GLM predictive or neural
            network sampling predictive or Gaussian Process (GP) inference.
            The GLM predictive is consistent with the curvature approximations used here.
        method : {'marglik', 'CV'}, default='marglik'
            specifies how the prior precision should be optimized.
        n_steps : int, default=100
            the number of gradient descent steps to take.
        lr : float, default=1e-1
            the learning rate to use for gradient descent.
        init_prior_prec : float, default=1.0
            initial prior precision before the first optimization step.
        val_loader : torch.data.utils.DataLoader, default=None
            DataLoader for the validation set; each iterate is a training batch (X, y).
        loss : callable, default=get_nll
            loss function to use for CV.
        log_prior_prec_min : float, default=-4
            lower bound of gridsearch interval for CV.
        log_prior_prec_max : float, default=4
            upper bound of gridsearch interval for CV.
        grid_size : int, default=100
            number of values to consider inside the gridsearch interval for CV.
        link_approx : {'mc', 'probit', 'bridge'}, default='probit'
            how to approximate the classification link function for the `'glm'`.
            For `pred_type='nn'`, only `'mc'` is possible.
        n_samples : int, default=100
            number of samples for `link_approx='mc'`.
        verbose : bool, default=False
            if true, the optimized prior precision will be printed
            (can be a large tensor if the prior has a diagonal covariance).
        """
        if method == 'marglik':
            self.prior_precision = init_prior_prec
            log_prior_prec = self.prior_precision.log()
            log_prior_prec.requires_grad = True
            optimizer = torch.optim.Adam([log_prior_prec], lr=lr)
            for _ in range(n_steps):
                optimizer.zero_grad()
                prior_prec = log_prior_prec.exp()
                neg_log_marglik = -self.log_marginal_likelihood(prior_precision=prior_prec)
                neg_log_marglik.backward()
                optimizer.step()
            self.prior_precision = log_prior_prec.detach().exp()
        elif method == 'CV':
            if val_loader is None:
                raise ValueError('CV requires a validation set DataLoader')
            interval = torch.logspace(
                log_prior_prec_min, log_prior_prec_max, grid_size
            )
            self.prior_precision = self._gridsearch(
                loss, interval, val_loader, pred_type=pred_type,
                link_approx=link_approx, n_samples=n_samples
            )
        else:
            raise ValueError('For now only marglik and CV is implemented.')
        if verbose:
            print(f'Optimized prior precision is {self.prior_precision}.')

    def _gridsearch(self, loss, interval, val_loader, pred_type,
                    link_approx='probit', n_samples=100):
        results = list()
        prior_precs = list()
        for prior_prec in interval:
            self.prior_precision = prior_prec
            try:
                out_dist, targets = validate(
                    self, val_loader, pred_type=pred_type,
                    link_approx=link_approx, n_samples=n_samples
                )
                result = loss(out_dist, targets)
            except RuntimeError:
                result = np.inf
            results.append(result)
            prior_precs.append(prior_prec)
        return prior_precs[np.argmin(results)]

    @property
    def sigma_noise(self):
        return self._sigma_noise

    @sigma_noise.setter
    def sigma_noise(self, sigma_noise):
        self._posterior_scale = None
        if np.isscalar(sigma_noise) and np.isreal(sigma_noise):
            self._sigma_noise = torch.tensor(sigma_noise, device=self._device)
        elif torch.is_tensor(sigma_noise):
            if sigma_noise.ndim == 0:
                self._sigma_noise = sigma_noise.to(self._device)
            elif sigma_noise.ndim == 1:
                if len(sigma_noise) > 1:
                    raise ValueError('Only homoscedastic output noise supported.')
                self._sigma_noise = sigma_noise[0].to(self._device)
            else:
                raise ValueError('Sigma noise needs to be scalar or 1-dimensional.')
        else:
            raise ValueError('Invalid type: sigma noise needs to be torch.Tensor or scalar.')

    @property
    def _H_factor(self):
        sigma2 = self.sigma_noise.square()
        return 1 / sigma2 / self.temperature


class ParametricLaplace(BaseLaplace):
    """
    Parametric Laplace class.

    Subclasses need to specify how the Hessian approximation is initialized,
    how to add up curvature over training data, how to sample from the
    Laplace approximation, and how to compute the functional variance.

    A Laplace approximation is represented by a MAP which is given by the
    `model` parameter and a posterior precision or covariance specifying
    a Gaussian distribution \\(\\mathcal{N}(\\theta_{MAP}, P^{-1})\\).
    The goal of this class is to compute the posterior precision \\(P\\)
    which sums as
    \\[
        P = \\sum_{n=1}^N \\nabla^2_\\theta \\log p(\\mathcal{D}_n \\mid \\theta)
        \\vert_{\\theta_{MAP}} + \\nabla^2_\\theta \\log p(\\theta) \\vert_{\\theta_{MAP}}.
    \\]
    Every subclass implements different approximations to the log likelihood Hessians,
    for example, a diagonal one. The prior is assumed to be Gaussian and therefore we have
    a simple form for \\(\\nabla^2_\\theta \\log p(\\theta) \\vert_{\\theta_{MAP}} = P_0 \\).
    In particular, we assume a scalar, layer-wise, or diagonal prior precision so that in
    all cases \\(P_0 = \\textrm{diag}(p_0)\\) and the structure of \\(p_0\\) can be varied.
    """

    def __init__(self, model, likelihood, sigma_noise=1., prior_precision=1.,
                 prior_mean=0., temperature=1., backend=BackPackGGN, backend_kwargs=None):
        assert backend in [BackPackGGN, BackPackEF, AsdlGGN, AsdlEF], \
            'GGN or EF backends required in ParametricLaplace.'
        super().__init__(model, likelihood, sigma_noise, prior_precision,
                         prior_mean, temperature, backend, backend_kwargs)

        self.H = None

    def _init_H(self):
        raise NotImplementedError

    def _check_fit(self):
        if self.H is None:
            raise AttributeError('ParametricLaplace not fitted. Run fit() first.')

    def fit(self, train_loader):
        """Fit the local Laplace approximation at the parameters of the model.

        Parameters
        ----------
        train_loader : torch.data.utils.DataLoader
            each iterate is a training batch (X, y);
            `train_loader.dataset` needs to be set to access \\(N\\), size of the data set
        """
        if self.H is not None:
            raise ValueError('Already fit.')

        self._init_H()

        self.model.eval()

        X, _ = next(iter(train_loader))
        with torch.no_grad():
            self.n_outputs = self.model(X[:1].to(self._device)).shape[-1]
        setattr(self.model, 'output_size', self.n_outputs)

        N = len(train_loader.dataset)
        for X, y in train_loader:
            self.model.zero_grad()
            X, y = X.to(self._device), y.to(self._device)
            loss_batch, H_batch = self._curv_closure(X, y, N)
            self.loss += loss_batch
            self.H += H_batch

        self.n_data = N

    @property
    def scatter(self):
        """Computes the _scatter_, a term of the log marginal likelihood that
        corresponds to L-2 regularization:
        `scatter` = \\((\\theta_{MAP} - \\mu_0)^{T} P_0 (\\theta_{MAP} - \\mu_0) \\).

        Returns
        -------
        [type]
            [description]
        """
        delta = (self.map_estimate - self.prior_mean)
        return (delta * self.prior_precision_diag) @ delta

    @property
    def log_det_prior_precision(self):
        """Compute log determinant of the prior precision
        \\(\\log \\det P_0\\)

        Returns
        -------
        log_det : torch.Tensor
        """
        return self.prior_precision_diag.log().sum()

    @property
    def log_det_posterior_precision(self):
        """Compute log determinant of the posterior precision
        \\(\\log \\det P\\) which depends on the subclasses structure
        used for the Hessian approximation.

        Returns
        -------
        log_det : torch.Tensor
        """
        raise NotImplementedError

    @property
    def log_det_ratio(self):
        """Compute the log determinant ratio, a part of the log marginal likelihood.
        \\[
            \\log \\frac{\\det P}{\\det P_0} = \\log \\det P - \\log \\det P_0
        \\]

        Returns
        -------
        log_det_ratio : torch.Tensor
        """
        return self.log_det_posterior_precision - self.log_det_prior_precision

    def _log_marginal_likelihood(self):
        return self.log_likelihood - 0.5 * (self.log_det_ratio + self.scatter)

    def __call__(self, x, pred_type='glm', link_approx='probit', n_samples=100):
        self._check_fit()

        if pred_type not in ['glm', 'nn']:
            raise ValueError('Only glm and nn supported as prediction types.')

        if pred_type == 'glm':
            f_mu, f_var = self._glm_predictive_distribution(x)
            # regression
            if self.likelihood == 'regression':
                return f_mu, f_var
            # classification
            return self._classification_predictive(f_mu, f_var, link_approx, n_samples)
        else:
            samples = self._nn_predictive_samples(x, n_samples)
            if self.likelihood == 'regression':
                return samples.mean(dim=0), samples.var(dim=0)
            return samples.mean(dim=0)

    def predictive_samples(self, x, pred_type='glm', n_samples=100):
        """Sample from the posterior predictive on input data `x`.
        Can be used, for example, for Thompson sampling.

        Parameters
        ----------
        x : torch.Tensor
            input data `(batch_size, input_shape)`

        pred_type : {'glm', 'nn'}, default='glm'
            type of posterior predictive, linearized GLM predictive or neural
            network sampling predictive. The GLM predictive is consistent with
            the curvature approximations used here.

        n_samples : int
            number of samples

        Returns
        -------
        samples : torch.Tensor
            samples `(n_samples, batch_size, output_shape)`
        """
        self._check_fit()

        if pred_type not in ['glm', 'nn']:
            raise ValueError('Only glm and nn supported as prediction types.')

        if pred_type == 'glm':
            f_mu, f_var = self._glm_predictive_distribution(x)
            assert f_var.shape == torch.Size([f_mu.shape[0], f_mu.shape[1], f_mu.shape[1]])
            dist = MultivariateNormal(f_mu, f_var)
            samples = dist.sample((n_samples,))
            if self.likelihood == 'regression':
                return samples
            return torch.softmax(samples, dim=-1)

        else:  # 'nn'
            return self._nn_predictive_samples(x, n_samples)

    @torch.enable_grad()
    def _glm_predictive_distribution(self, X):
        Js, f_mu = self.backend.jacobians(self.model, X)
        f_var = self.functional_variance(Js)
        return f_mu.detach(), f_var.detach()

    def _nn_predictive_samples(self, X, n_samples=100):
        fs = list()
        for sample in self.sample(n_samples):
            vector_to_parameters(sample, self.model.parameters())
            fs.append(self.model(X.to(self._device)).detach())
        vector_to_parameters(self.map_estimate, self.model.parameters())
        fs = torch.stack(fs)
        if self.likelihood == 'classification':
            fs = torch.softmax(fs, dim=-1)
        return fs

    def functional_variance(self, Jacs):
        """Compute functional variance for the `'glm'` predictive:
        `f_var[i] = Jacs[i] @ P.inv() @ Jacs[i].T`, which is a output x output
        predictive covariance matrix.
        Mathematically, we have for a single Jacobian
        \\(\\mathcal{J} = \\nabla_\\theta f(x;\\theta)\\vert_{\\theta_{MAP}}\\)
        the output covariance matrix
        \\( \\mathcal{J} P^{-1} \\mathcal{J}^T \\).

        Parameters
        ----------
        Jacs : torch.Tensor
            Jacobians of model output wrt parameters
            `(batch, outputs, parameters)`

        Returns
        -------
        f_var : torch.Tensor
            output covariance `(batch, outputs, outputs)`
        """
        raise NotImplementedError

    def sample(self, n_samples=100):
        """Sample from the Laplace posterior approximation, i.e.,
        \\( \\theta \\sim \\mathcal{N}(\\theta_{MAP}, P^{-1})\\).

        Parameters
        ----------
        n_samples : int, default=100
            number of samples
        """
        raise NotImplementedError

    def optimize_prior_precision(self, pred_type='glm', method='marglik', n_steps=100, lr=1e-1,
                                 init_prior_prec=1., val_loader=None, loss=get_nll,
                                 log_prior_prec_min=-4, log_prior_prec_max=4, grid_size=100,
                                 link_approx='probit', n_samples=100,
                                 verbose=False):
        """
        `optimize_prior_precision_base` from `BaseLaplace` with `pred_type` in `{'glm', 'nn'}`
        """
        assert pred_type in ['glm', 'nn']
        self.optimize_prior_precision_base(pred_type, method, n_steps, lr,
                                           init_prior_prec, val_loader, loss,
                                           log_prior_prec_min, log_prior_prec_max,
                                           grid_size, link_approx, n_samples,
                                           verbose)

    @property
    def posterior_precision(self):
        """Compute or return the posterior precision \\(P\\).

        Returns
        -------
        posterior_prec : torch.Tensor
        """
        raise NotImplementedError


class FullLaplace(ParametricLaplace):
    """Laplace approximation with full, i.e., dense, log likelihood Hessian approximation
    and hence posterior precision. Based on the chosen `backend` parameter, the full
    approximation can be, for example, a generalized Gauss-Newton matrix.
    Mathematically, we have \\(P \\in \\mathbb{R}^{P \\times P}\\).
    See `BaseLaplace` for the full interface.
    """
    # key to map to correct subclass of BaseLaplace, (subset of weights, Hessian structure)
    _key = ('all', 'full')

    def __init__(self, model, likelihood, sigma_noise=1., prior_precision=1.,
                 prior_mean=0., temperature=1., backend=BackPackGGN, backend_kwargs=None):
        super().__init__(model, likelihood, sigma_noise, prior_precision,
                         prior_mean, temperature, backend, backend_kwargs)
        self._posterior_scale = None

    def _init_H(self):
        self.H = torch.zeros(self.n_params, self.n_params, device=self._device)

    def _curv_closure(self, X, y, N):
        return self.backend.full(X, y, N=N)

    def _compute_scale(self):
        self._posterior_scale = invsqrt_precision(self.posterior_precision)

    @property
    def posterior_scale(self):
        """Posterior scale (square root of the covariance), i.e.,
        \\(P^{-\\frac{1}{2}}\\).

        Returns
        -------
        scale : torch.tensor
            `(parameters, parameters)`
        """
        if self._posterior_scale is None:
            self._compute_scale()
        return self._posterior_scale

    @property
    def posterior_covariance(self):
        """Posterior covariance, i.e., \\(P^{-1}\\).

        Returns
        -------
        covariance : torch.tensor
            `(parameters, parameters)`
        """
        scale = self.posterior_scale
        return scale @ scale.T

    @property
    def posterior_precision(self):
        """Posterior precision \\(P\\).

        Returns
        -------
        precision : torch.tensor
            `(parameters, parameters)`
        """
        self._check_fit()
        return self._H_factor * self.H + torch.diag(self.prior_precision_diag)

    @property
    def log_det_posterior_precision(self):
        return self.posterior_precision.logdet()

    def functional_variance(self, Js):
        return torch.einsum('ncp,pq,nkq->nck', Js, self.posterior_covariance, Js)

    def sample(self, n_samples=100):
        dist = MultivariateNormal(loc=self.map_estimate, scale_tril=self.posterior_scale)
        return dist.sample((n_samples,))


class KronLaplace(ParametricLaplace):
    """Laplace approximation with Kronecker factored log likelihood Hessian approximation
    and hence posterior precision.
    Mathematically, we have for each parameter group, e.g., torch.nn.Module,
    that \\P\\approx Q \\otimes H\\.
    See `BaseLaplace` for the full interface and see
    `laplace.matrix.Kron` and `laplace.matrix.KronDecomposed` for the structure of
    the Kronecker factors. `Kron` is used to aggregate factors by summing up and
    `KronDecomposed` is used to add the prior, a Hessian factor (e.g. temperature),
    and computing posterior covariances, marginal likelihood, etc.
    Damping can be enabled by setting `damping=True`.
    """
    # key to map to correct subclass of BaseLaplace, (subset of weights, Hessian structure)
    _key = ('all', 'kron')

    def __init__(self, model, likelihood, sigma_noise=1., prior_precision=1.,
                 prior_mean=0., temperature=1., backend=BackPackGGN, damping=False,
                 **backend_kwargs):
        self.damping = damping
        super().__init__(model, likelihood, sigma_noise, prior_precision,
                         prior_mean, temperature, backend, **backend_kwargs)

    def _init_H(self):
        self.H = Kron.init_from_model(self.model, self._device)

    def _curv_closure(self, X, y, N):
        return self.backend.kron(X, y, N=N)

    def fit(self, train_loader, keep_factors=False):
        super().fit(train_loader)
        # Kron requires postprocessing as all quantities depend on the decomposition.
        if keep_factors:
            self.H_facs = self.H
        self.H = self.H.decompose(damping=self.damping)

    @property
    def posterior_precision(self):
        """Kronecker factored Posterior precision \\(P\\).

        Returns
        -------
        precision : `laplace.matrix.KronDecomposed`
        """
        self._check_fit()
        return self.H * self._H_factor + self.prior_precision

    @property
    def log_det_posterior_precision(self):
        return self.posterior_precision.logdet()

    def functional_variance(self, Js):
        return self.posterior_precision.inv_square_form(Js)

    def sample(self, n_samples=100):
        samples = torch.randn(n_samples, self.n_params, device=self._device)
        samples = self.posterior_precision.bmm(samples, exponent=-0.5)
        return self.map_estimate.reshape(1, self.n_params) + samples.reshape(n_samples, self.n_params)

    @BaseLaplace.prior_precision.setter
    def prior_precision(self, prior_precision):
        # Extend setter from Laplace to restrict prior precision structure.
        super(KronLaplace, type(self)).prior_precision.fset(self, prior_precision)
        if len(self.prior_precision) not in [1, self.n_layers]:
            raise ValueError('Prior precision for Kron either scalar or per-layer.')


class DiagLaplace(ParametricLaplace):
    """Laplace approximation with diagonal log likelihood Hessian approximation
    and hence posterior precision.
    Mathematically, we have \\(P \\approx \\textrm{diag}(P)\\).
    See `BaseLaplace` for the full interface.
    """
    # key to map to correct subclass of BaseLaplace, (subset of weights, Hessian structure)
    _key = ('all', 'diag')

    def _init_H(self):
        self.H = torch.zeros(self.n_params, device=self._device)

    def _curv_closure(self, X, y, N):
        return self.backend.diag(X, y, N=N)

    @property
    def posterior_precision(self):
        """Diagonal posterior precision \\(p\\).

        Returns
        -------
        precision : torch.tensor
            `(parameters)`
        """
        self._check_fit()
        return self._H_factor * self.H + self.prior_precision_diag

    @property
    def posterior_scale(self):
        """Diagonal posterior scale \\(\\sqrt{p^{-1}}\\).

        Returns
        -------
        precision : torch.tensor
            `(parameters)`
        """
        return 1 / self.posterior_precision.sqrt()

    @property
    def posterior_variance(self):
        """Diagonal posterior variance \\(p^{-1}\\).

        Returns
        -------
        precision : torch.tensor
            `(parameters)`
        """
        return 1 / self.posterior_precision

    @property
    def log_det_posterior_precision(self):
        return self.posterior_precision.log().sum()

    def functional_variance(self, Js: torch.Tensor) -> torch.Tensor:
        self._check_jacobians(Js)
        return torch.einsum('ncp,p,nkp->nck', Js, self.posterior_variance, Js)

    def sample(self, n_samples=100):
        samples = torch.randn(n_samples, self.n_params, device=self._device)
        samples = samples * self.posterior_scale.reshape(1, self.n_params)
        return self.map_estimate.reshape(1, self.n_params) + samples


class FunctionalLaplace(BaseLaplace):
    """
    Applying the GGN (General Gauss Newton) approximation for the Hessian in the Laplace approximation of the posterior
    turns the underlying probabilistic model from a BNN into a GLM (generalized linear model).
    This GLM (in the weight space) is equivalent to a GP (in the function space), see
    "Approximate Inference Turns Deep Networks into Gaussian Processes (Khan et al., 2019)"

    This class implements the (approximate) GP inference through which
    we obtain the desired quantities (a posterior predictive, a marginal log-likelihood).
    See "Improving predictions of Bayesian neural nets via local linearization (Immer et al., 2021)" for more details.

    Parameters
    ----------
    M : int
        number of data points for Subset-of-Data (SOD) approximate GP inference.
        By default (`M=None`), all data points from train dataset are used
    diagonal_kernel : bool
        GP kernel here is product of Jacobians, which results in a \\( C \\times C\\) matrix where \\(C\\) is the output
        dimension. If `diagonal_kernel=True`, only a diagonal of a GP kernel is used. This is (somewhat) equivalent to
        assuming independent GPs across output channels.
    diagonal_L : bool
        approximate \\( L_{MM} \\) with \\( diag(L_{MM}) \\). \\( L_{MM} \\) is a block-diagonal matrix, where blocks
        represent Hessians of per-data-point log-likelihood w.r.t. NN output \\( f \\). See Appendix A.2.1 in
        "Improving predictions of Bayesian neural nets via local linearization (Immer et al., 2021)"
        for exact definition.
        If False and `likelihood='regression'`, then nothing changes
            because \\( L_{MM} \\)  is anyways diagonal as long as we assume diagonal noise in the likelihood.
        If False and `likelihood='classification'`, then the algorithm from Chapter 3.5 from R&W 2006 GP book
            is used.


    See `BaseLaplace` class for the full interface.
    """
    # key to map to correct subclass of BaseLaplace, (subset of weights, Hessian structure)
    _key = ('all', 'GP')

    def __init__(self, model, likelihood, M=None, sigma_noise=1., prior_precision=1.,
                 prior_mean=0., temperature=1., backend=BackPackInterface, backend_kwargs=None,
                 diagonal_kernel=False, diagonal_L=True):
        assert backend in [BackPackInterface], 'Only BackPack backend is supported in FunctionalLaplace (for now)'
        super().__init__(model, likelihood, sigma_noise, prior_precision,
                         prior_mean, temperature, backend, backend_kwargs)

        self.M = M
        self.diagonal_kernel = diagonal_kernel
        self.diagonal_L = diagonal_L
        if not diagonal_kernel and likelihood == 'classification':
            assert diagonal_L, \
                'GP inference without independence for classification necessitates diagonal approximation of L!'

        self.K_MM = None
        self.Sigma_inv = None  # (K_{MM} + L_MM_inv)^{-1}
        self.train_loader = None  # needed in functional variance and marginal log likelihood
        self.batch_size = None
        self.prior_factor_sod = None
        self.mu = None  # mean of the log marginal likelihood

    def _check_fit(self):
        if (self.K_MM is None) or (self.Sigma_inv is None) or (self.train_loader is None):
            raise AttributeError('Laplace not fitted. Run fit() first.')

    def _init_K_MM(self):
        if self.diagonal_kernel:
            self.K_MM = [torch.zeros(size=(self.M, self.M), device=self._device) for _ in range(self.n_outputs)]
        else:
            self.K_MM = torch.zeros(size=(self.M * self.n_outputs, self.M * self.n_outputs), device=self._device)

    def _init_Sigma_inv(self):
        if self.diagonal_kernel:
            self.Sigma_inv = [torch.zeros(size=(self.M, self.M), device=self._device) for _ in range(self.n_outputs)]
        else:
            self.Sigma_inv = torch.zeros(size=(self.M * self.n_outputs, self.M * self.n_outputs), device=self._device)

    def _curv_closure(self, X, y):
        return self.backend.gp_quantities(X, y, self._H_factor)

    def _store_K_batch(self, K_batch, i, j):
        if self.diagonal_kernel:
            for c in range(self.n_outputs):
                self.K_MM[c][i * self.batch_size:min((i + 1) * self.batch_size, self.M),
                             j * self.batch_size:min((j + 1) * self.batch_size, self.M)] = K_batch[:, :, c]
                if i != j:
                    self.K_MM[c][j * self.batch_size:min((j + 1) * self.batch_size, self.M),
                                 i * self.batch_size:min((i + 1) * self.batch_size, self.M)] = torch.transpose(K_batch[:, :, c], 0, 1)
        else:
            bC = self.batch_size * self.n_outputs
            MC = self.M * self.n_outputs
            self.K_MM[i * bC:min((i + 1) * bC, MC), j * bC:min((j + 1) * bC, MC)] = K_batch
            if i != j:
                self.K_MM[j * bC:min((j + 1) * bC, MC), i * bC:min((i + 1) * bC, MC)] = torch.transpose(K_batch, 0, 1)

    def _build_L_inv(self, lambdas):
        if self.diagonal_kernel:
            if self.diagonal_L or self.likelihood == "regression":
                L_diag = torch.diagonal(torch.cat(lambdas, dim=0), dim1=-2, dim2=-1).reshape(-1)
                # rearrange and take the inverse for each MxM matrix separately
                return [torch.diag(1. / L_diag[i::self.n_outputs]) for i in range(self.n_outputs)]
            else:
                # TODO: double check the correctness of the algorithm in R&W before implementing
                # R&W 2006 algorithm
                raise NotImplementedError
        else:
            # in case of self.likelihood == "classificiation" we approximate L_MM with diagonal here
            # in case of self.likelihood == "regression" L_MM is anyways diagonal
            diag = torch.diagonal(torch.cat(lambdas, dim=0), dim1=-2, dim2=-1).reshape(-1)
            return torch.diag(1. / diag)

    def _build_Sigma_inv(self, lambdas):
        if self.diagonal_kernel:
            lambdas = self._build_L_inv(lambdas)
            return [torch.linalg.cholesky(self.K_MM[c] + lambdas[c]) for c in range(self.n_outputs)]
        else:
            return torch.linalg.cholesky(self.K_MM + self._build_L_inv(lambdas))

    def _get_SoD_data_loader(self, train_loader: DataLoader, seed: int = 0) -> DataLoader:
        """
        Subset-of-Datapoints data loader
        """
        np.random.seed(seed)
        return DataLoader(dataset=train_loader.dataset, batch_size=train_loader.batch_size,
                          sampler=SoDSampler(N=len(train_loader.dataset), M=self.M), shuffle=False)

    def fit(self, train_loader):
        """Fit the Laplace approximation of a GP posterior.

        Parameters
        ----------
        train_loader : torch.data.utils.DataLoader
            `train_loader.dataset` needs to be set to access \\(N\\), size of the data set
            `train_loader.batch_size` needs to be set to access \\(b\\) batch_size
        """

        X, _ = next(iter(train_loader))
        with torch.no_grad():
            self.n_outputs = self.model(X[:1].to(self._device)).shape[-1]
        setattr(self.model, 'output_size', self.n_outputs)
        self.batch_size = train_loader.batch_size

        if self.likelihood == 'regression' and self.n_outputs > 1 and self.diagonal_kernel:
            warnings.warn('Using FunctionalLaplace with the diagonal approximation of a GP kernel is not recommended '
                          'in the case of multivariate regression. Predictive variance will likely be overestimated.')

        self.model.eval()

        N = len(train_loader.dataset)
        self.n_data = N
        if self.M is None:  # by default, all training data points are used for GP inference
            self.M = N
        train_loader = self._get_SoD_data_loader(train_loader)
        self.train_loader = train_loader
        self.prior_factor_sod = self.M / self.n_data

        self._init_K_MM()
        self._init_Sigma_inv()
        diff_mu = self.prior_mean - self.map_estimate

        f, lambdas, mu = [], [], []
        for i, batch in enumerate(train_loader):
            X, y = batch
            X, y = X.to(self._device), y.to(self._device)
            loss_batch, Js_batch, f_batch, lambdas_batch = self._curv_closure(X, y)
            self.loss += loss_batch
            lambdas.append(lambdas_batch)
            f.append(f_batch)
            if self.likelihood == 'regression':
                mu_batch = y - (f_batch + torch.einsum('bcp,p->bc', Js_batch, diff_mu))
                mu.append(mu_batch)
            for j, batch_2 in enumerate(train_loader):
                if j >= i:
                    X2, _ = batch_2
                    X2 = X2.to(self._device)
                    K_batch = self._kernel_batch(Js_batch, X2)
                    self._store_K_batch(K_batch, i, j)

        self.Sigma_inv = self._build_Sigma_inv(lambdas)
        if self.likelihood == "regression":
            self.mu = torch.cat(mu, dim=0)

    def __call__(self, x, pred_type='gp', link_approx='probit', n_samples=100):
        if pred_type not in ['gp']:
            raise ValueError('Only gp supported as prediction type.')

        self._check_fit()

        f_mu, f_var = self.gp_posterior(x)
        # regression
        if self.likelihood == 'regression':
            return f_mu, f_var
        # classification
        return self._classification_predictive(f_mu, f_var, link_approx, n_samples)

    def predictive_samples(self, x, n_samples=100):
        """Sample from the posterior predictive on input data `x`.

        Parameters
        ----------
        x : torch.Tensor
            input data `(batch_size, input_shape)`

        n_samples : int
            number of samples

        Returns
        -------
        samples : torch.Tensor
            samples `(n_samples, batch_size, output_shape)`
        """
        self._check_fit()

        f_mu, f_var = self.gp_posterior(x)
        assert f_var.shape == torch.Size([f_mu.shape[0], f_mu.shape[1], f_mu.shape[1]])
        dist = MultivariateNormal(f_mu, f_var)
        samples = dist.sample((n_samples,))
        if self.likelihood == 'regression':
            return samples
        return torch.softmax(samples, dim=-1)

    def gp_posterior(self, X_star):
        """
        \\(q(f_* | x_*, \mathcal{D}) = \mathcal{N} (f_*, \Sigma_*) \\), where
        \\(\Sigma_* =  K_{**} - K_{*M} (K_{MM}+ L_{MM}^{-1})^{-1} K_{M*}\\)

        See eq. A.6 in "Improving predictions of Bayesian neural nets via local linearization (Immer et al., 2021)"

        Parameters
        ----------
        X : torch.Tensor
            test data points \\(X_* \in \mathbb{R}^{N_{test} \\times C} \\)

        Returns
        -------
        f_mu : torch.Tensor
            mean of the GP posterior distribution
        f_var: torch.Tensor
            variance of the GP posterior distribution


        """
        Js, f_mu = self._jacobians(X_star)
        f_var = self._gp_posterior_variance(Js, X_star)
        if self.diagonal_kernel:
            f_var = torch.diag_embed(f_var)
        return f_mu.detach(), f_var.detach()

    def _gp_posterior_variance(self, Js_star, X_star):
        """
        GP posterior variance: \\( k_{**} - K_{*M} (K_{MM}+ L_{MM}^{-1})^{-1} K_{M*}\\)

        Parameters
        ----------
        Js : torch.Tensor
            Jacobians of test data points
        X : torch.Tensor
            test data points \\(X \in \mathbb{R}^{N_{test} \\times C} \\)
        """

        self._check_fit()

        K_star = self._kernel_star(Js_star, X_star)

        K_M_star = []
        for X_batch, _ in self.train_loader:
            K_M_star_batch = self._kernel_batch_star(Js_star, X_batch)
            K_M_star.append(K_M_star_batch)

        f_var = K_star - self._build_K_star_M(K_M_star)
        return f_var

    def _build_K_star_M(self, K_M_star):
        K_M_star = torch.cat(K_M_star, dim=1)
        if self.diagonal_kernel:
            prods = []
            for c in range(self.n_outputs):
                v = torch.squeeze(torch.linalg.solve(self.Sigma_inv[c], K_M_star[:, :, c].unsqueeze(2)), 2)
                prod = torch.einsum('bm,bm->b', v, v)
                prods.append(prod.unsqueeze(1))
            prods = torch.cat(prods, dim=1)
            return prods

        else:
            # in the reshape below we go from (N_test, M, C, C) to (N_test, M*C, C)
            K_M_star = K_M_star.reshape(K_M_star.shape[0], -1, K_M_star.shape[-1])
            v = torch.linalg.solve(self.Sigma_inv, K_M_star)
            return torch.einsum('bcm,bcn->bmn', v, v)

    def _log_marginal_likelihood(self):
        self.fit(self.train_loader)
        if self.likelihood == 'classification':
            raise NotImplementedError
        elif self.likelihood == 'regression':
            return - 0.5 * (self.log_det_K + self.scatter_lml + self.M * self.n_outputs * np.log(2 * np.pi))

    @property
    def log_det_K(self):
        """
        Computes log determinant term in GP marginal likelihood
        """
        if self.diagonal_kernel:
            log_det = 0.
            for c in range(self.n_outputs):
                log_det += torch.logdet(self.K_MM[c] + torch.eye(n=self.K_MM[c].shape[0]) * self.sigma_noise.square())
            return log_det
        else:
            return torch.logdet(self.K_MM + torch.eye(n=self.K_MM.shape[0]) * self.sigma_noise.square())

    @property
    def scatter_lml(self):
        """
        Compute scatter term in GP log marginal likelihood
        """

        if self.diagonal_kernel:
            scatter = 0.
            for c in range(self.n_outputs):
                K_inv = torch.inverse(self.K_MM[c] + torch.eye(n=self.K_MM[c].shape[0]) * self.sigma_noise.square())
                scatter += torch.dot(self.mu[:, c], torch.matmul(K_inv, self.mu[:, c]))
        else:
            K_inv = torch.inverse(self.K_MM + torch.eye(n=self.K_MM.shape[0]) * self.sigma_noise.square())
            scatter = torch.dot(self.mu.reshape(-1), torch.matmul(K_inv, self.mu.reshape(-1)))
        return scatter
            
    def optimize_prior_precision(self, method='marglik', n_steps=100, lr=1e-1,
                                 init_prior_prec=1., val_loader=None, loss=get_nll,
                                 log_prior_prec_min=-4, log_prior_prec_max=4, grid_size=100,
                                 pred_type='gp', link_approx='probit', n_samples=100,
                                 verbose=False):
        """
        `optimize_prior_precision_base` from `BaseLaplace` with `pred_type='GP'`
        """
        assert pred_type == 'gp'
        self.optimize_prior_precision_base(pred_type, method, n_steps, lr,
                                           init_prior_prec, val_loader, loss,
                                           log_prior_prec_min, log_prior_prec_max,
                                           grid_size, link_approx, n_samples,
                                           verbose)

    def _kernel_batch(self, jacobians, batch):
        """
        Compute K_bb, which is part of K_MM kernel matrix.

        Parameters
        ----------
        jacobians : torch.Tensor (b, C, P)
        batch : torch.Tensor (b, C)

        Returns
        -------
        kernel : torch.tensor
            K_bb with shape (b * C, b * C)
        """
        if isinstance(self.backend, BackPackInterface):
            jacobians_2, _ = self._jacobians(batch)
            P = jacobians.shape[-1]  # nr model params
            prior = self.prior_factor_sod / self.prior_precision_diag
            if self.diagonal_kernel:
                kernel = torch.einsum('bcp,ecp->bec', jacobians, jacobians_2 * prior)
            else:
                kernel = torch.einsum('ap,p,bp->ab', jacobians.reshape(-1, P), prior, jacobians_2.reshape(-1, P))
            return kernel
        elif isinstance(self.backend, AsdlInterface):
            raise NotImplementedError

    def _kernel_star(self, jacobians, batch):
        """
        Compute K_star_star kernel matrix.

        Parameters
        ----------
        jacobians : torch.Tensor (b, C, P)
        batch : torch.Tensor (b, C)

        Returns
        -------
        kernel : torch.tensor
            K_star with shape (b, C, C)

        """
        if isinstance(self.backend, BackPackInterface):
            jacobians_2, _ = self._jacobians(batch)
            prior = self.prior_factor_sod / self.prior_precision_diag
            if self.diagonal_kernel:
                kernel = torch.einsum('bcp,bcp->bc', jacobians, jacobians_2 * prior)
            else:
                kernel = torch.einsum('bcp,p,bep->bce', jacobians, prior, jacobians_2)
            return kernel
        elif isinstance(self.backend, AsdlInterface):
            raise NotImplementedError

    def _kernel_batch_star(self, jacobians, batch):
        """
        Compute K_b_star, which is a part of K_M_star kernel matrix.

        Parameters
        ----------
        jacobians : torch.Tensor (b1, C, P)
        batch : torch.Tensor (b2, C)

        Returns
        -------
        kernel : torch.tensor
            K_batch_star with shape (b1, b2, C, C)
        """
        if isinstance(self.backend, BackPackInterface):
            jacobians_2, _ = self._jacobians(batch)
            prior = self.prior_factor_sod / self.prior_precision_diag
            if self.diagonal_kernel:
                kernel = torch.einsum('bcp,ecp->bec', jacobians, jacobians_2 * prior)
            else:
                kernel = torch.einsum('bcp,p,dep->bdce', jacobians, prior, jacobians_2)
            return kernel
        elif isinstance(self.backend, AsdlInterface):
            raise NotImplementedError

    def _jacobians(self, X):
        """
        A wrapper function to compute jacobians - this enables reusing same kernel methods (kernel_batch etc.)
        in FunctionalLaplace and FunctionalLLLaplace by simply overwriting this method instead of all kernel methods.
        """
        return self.backend.jacobians(self.model, X)
