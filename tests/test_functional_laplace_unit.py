import pytest
import torch
from laplace.baselaplace import FunctionalLaplace
from torch import nn
from torch.nn.utils import parameters_to_vector
from torch.utils.data import DataLoader, TensorDataset


@pytest.fixture
def reg_loader():
    X = torch.randn(10, 3)
    y = torch.randn(10, 2)
    return DataLoader(TensorDataset(X, y), batch_size=3)


@pytest.fixture
def model():
    model = torch.nn.Sequential(nn.Linear(3, 20), nn.Linear(20, 2))
    setattr(model, 'output_size', 2)
    model_params = list(model.parameters())
    setattr(model, 'n_layers', len(model_params))  # number of parameter groups
    setattr(model, 'n_params', len(parameters_to_vector(model_params)))
    return model


@pytest.fixture
def reg_Xy():
    torch.manual_seed(711)
    X = torch.randn(10, 3)
    y = torch.randn(10, 2)
    return X, y


def test_sod_data_loader(reg_loader, model):
    M = 5
    func_la = FunctionalLaplace(model, 'regression', M)
    sod_data_loader = func_la._get_SoD_data_loader(reg_loader)

    first_iter = []
    for x, _ in sod_data_loader:
        first_iter.append(x)
    second_iter = []
    for x, y in sod_data_loader:
        second_iter.append(x)

    first_iter = torch.cat(first_iter, dim=0)
    second_iter = torch.cat(second_iter, dim=0)

    assert torch.allclose(first_iter, second_iter)
    assert len(first_iter) == M


def test_store_K_batch_full_kernel(reg_loader, model, M=3, batch_size=2):
    C = model.output_size
    func_la = FunctionalLaplace(model, 'regression', M, diagonal_kernel=False)
    func_la.n_outputs = C
    func_la.batch_size = batch_size

    func_la._init_K_MM()
    assert torch.equal(torch.zeros(size=(M*C, M*C)), func_la.K_MM)

    func_la._store_K_batch(torch.ones(size=(4, 4)), 0, 0)
    assert torch.equal(torch.tensor([[1., 1., 1., 1., 0., 0.],
                                     [1., 1., 1., 1., 0., 0.],
                                     [1., 1., 1., 1., 0., 0.],
                                     [1., 1., 1., 1., 0., 0.],
                                     [0., 0., 0., 0., 0., 0.],
                                     [0., 0., 0., 0., 0., 0.]]),
                       func_la.K_MM)

    func_la._store_K_batch(torch.ones(size=(4, 2)), 0, 1)
    assert torch.equal(torch.tensor([[1., 1., 1., 1., 1., 1.],
                                     [1., 1., 1., 1., 1., 1.],
                                     [1., 1., 1., 1., 1., 1.],
                                     [1., 1., 1., 1., 1., 1.],
                                     [1., 1., 1., 1., 0., 0.],
                                     [1., 1., 1., 1., 0., 0.]]),
                       func_la.K_MM)

    func_la._store_K_batch(torch.ones(size=(2, 2)), 1, 1)
    assert torch.equal(torch.tensor([[1., 1., 1., 1., 1., 1.],
                                     [1., 1., 1., 1., 1., 1.],
                                     [1., 1., 1., 1., 1., 1.],
                                     [1., 1., 1., 1., 1., 1.],
                                     [1., 1., 1., 1., 1., 1.],
                                     [1., 1., 1., 1., 1., 1.]]),
                       func_la.K_MM)


def test_store_K_batch_block_diagonal_kernel(reg_loader, model, M=3, batch_size=2):
    C = model.output_size
    func_la = FunctionalLaplace(model, 'regression', M, diagonal_kernel=True)
    func_la.n_outputs = C
    func_la.batch_size = batch_size

    def _check(expected_K_MM):
        for c in range(C):
            assert torch.equal(expected_K_MM[c], func_la.K_MM[c])

    func_la._init_K_MM()
    expected = [torch.zeros(size=(M, M)) for _ in range(C)]
    _check(expected)

    func_la._store_K_batch(torch.cat([torch.ones(size=(2, 2, 1)), 2 * torch.ones(size=(2, 2, 1))], dim=2), 0, 0)
    expected = [(c + 1) * torch.tensor([[1., 1., 0.],
                                        [1., 1., 0.],
                                        [0., 0., 0.]]) for c in range(C)]
    _check(expected)

    func_la._store_K_batch(torch.cat([torch.ones(size=(2, 1, 1)), 2 * torch.ones(size=(2, 1, 1))], dim=2), 0, 1)
    expected = [(c + 1) * torch.tensor([[1., 1., 1.],
                                        [1., 1., 1.],
                                        [1., 1., 0.]]) for c in range(C)]
    _check(expected)

    func_la._store_K_batch(torch.cat([torch.ones(size=(1, 1, 1)), 2 * torch.ones(size=(1, 1, 1))], dim=2), 1, 1)
    expected = [(c + 1) * torch.tensor([[1., 1., 1.],
                                        [1., 1., 1.],
                                        [1., 1., 1.]]) for c in range(C)]
    _check(expected)


@pytest.mark.parametrize('kernel_type,jacobians,jacobians_2,expected_full_kernel,expected_block_diagonal_kernel',
                         [('kernel_batch',
                           torch.tensor([[[1., 1., 1.], [2., 2., 2.]],
                                         [[1., 1., 1.], [3., 3., 3.]]]),
                           torch.tensor([[[1., 1., 1.], [2., 2., 2.]],
                                         [[1., 1., 1.], [3., 3., 3.]]]),
                           torch.tensor([[3., 6., 3., 9.],
                                         [6., 12., 6., 18.],
                                         [3., 6., 3., 9.],
                                         [9., 18., 9., 27.]]),
                           torch.tensor([[[3., 12.],
                                         [3., 18.]],
                                        [[3., 18.],
                                         [3., 27.]]])
                           ),
                          ('kernel_star',
                           torch.tensor([[[1., 1., 1.], [1., 1., 1.]],
                                         [[1., 1., 1.], [2., 2., 2.]],
                                         [[1., 1., 1.], [3., 3., 3.]]]),
                           torch.tensor([[[1., 1., 1.], [1., 1., 1.]],
                                         [[1., 1., 1.], [2., 2., 2.]],
                                         [[1., 1., 1.], [3., 3., 3.]]]),
                           torch.tensor([[[3., 3.],
                                          [3., 3.]],
                                         [[3., 6.],
                                          [6., 12.]],
                                         [[3., 9.],
                                          [9., 27.]]]),
                          torch.tensor([[3., 3.],
                                        [3., 12.],
                                        [3., 27.]])
                           ),
                          ('kernel_batch_star',
                           torch.tensor([[[1., 1., 1.], [1., 1., 1.]],
                                         [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]]),
                           torch.tensor([[[1., 1., 1.], [2., 2., 2.]],
                                         [[1., 1., 1.], [3., 3., 3.]]]),
                           torch.tensor([[[[3., 6.], [3., 6.]], [[3., 9.], [3., 9.]]],
                                         [[[1.5, 3.], [1.5, 3.]], [[1.5, 4.5], [1.5, 4.5]]]]),
                           torch.tensor([[[3., 6.], [3., 9.]],
                                         [[1.5, 3.], [1.5, 4.5]]])
                           )])
def test_gp_kernel(mocker, reg_Xy, model, kernel_type, jacobians, jacobians_2,
                   expected_full_kernel, expected_block_diagonal_kernel):
    X, y = reg_Xy
    func_la = FunctionalLaplace(model, 'regression', prior_precision=1.0, diagonal_kernel=False)
    func_la.prior_factor_sod = 1.0
    if kernel_type == 'kernel_batch':
        kernel = func_la._kernel_batch
    elif kernel_type == 'kernel_star':
        kernel = func_la._kernel_star
    elif kernel_type == 'kernel_batch_star':
        kernel = func_la._kernel_batch_star
    else:
        raise ValueError

    # mocking jacobians
    def mock_jacobians(self, x):
        return jacobians_2, None
    mocker.patch('laplace.curvature.BackPackInterface.jacobians', mock_jacobians)

    #  mocking prior precision
    mocker.patch('laplace.baselaplace.FunctionalLaplace.prior_precision_diag', torch.ones(3))

    # X does not have an impact since we mock jacobians in mock_jacobians above
    full_kernel = kernel(jacobians, X)
    assert torch.equal(expected_full_kernel, full_kernel)

    func_la.diagonal_kernel = True
    block_diag_kernel = kernel(jacobians, X)
    assert torch.equal(expected_block_diagonal_kernel, block_diag_kernel)
