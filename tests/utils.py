import torch
from torch.utils.data import DataLoader, TensorDataset


def toy_regression_dataset_1d(sigma, n_train=150, n_test=500, batch_size=150):
    torch.manual_seed(711)
    # create simple sinusoid data set
    X_train = (torch.rand(n_train) * 8).unsqueeze(-1)
    y_train = torch.sin(X_train) + torch.randn_like(X_train) * sigma
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size)
    X_test = torch.linspace(-5, 13, n_test).unsqueeze(-1)  # +-5 on top of the training X-range

    return X_train, y_train, train_loader, X_test


def toy_multivariate_regression_dataset(sigma, d_input, n_train=150, n_test=500, batch_size=150):
    torch.manual_seed(711)
    # create simple sinusoid data set
    X_train = torch.rand(n_train, d_input) * 8
    y_train = torch.sin(X_train) + torch.randn_like(X_train) * sigma
    # y_train = torch.rand(n_train, d_input) * 8 + torch.randn_like(X_train) * sigma
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size)
    X_test = torch.rand(n_test, d_input) * 10
    return X_train, y_train, train_loader, X_test


def toy_classification_dataset(n_train=150, n_test=500, batch_size=150, in_dim=3, out_dim=2):
    torch.manual_seed(711)
    X_train = torch.randn(n_train, in_dim)
    y_train = torch.randint(out_dim, (n_train,))
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size)
    X_test = torch.randn(n_test, in_dim)
    return X_train, y_train, train_loader, X_test


def toy_model(train_loader: DataLoader, n_epochs=500, fit=True, in_dim=1, out_dim=1, regression=True):
    model = torch.nn.Sequential(torch.nn.Linear(in_dim, 50),
                                torch.nn.Tanh(),
                                torch.nn.Linear(50, out_dim))
    if fit:
        if regression:
            criterion = torch.nn.MSELoss()
        else:
            criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), weight_decay=5e-4, lr=1e-2)
        for i in range(n_epochs):
            for X, y in train_loader:
                optimizer.zero_grad()
                loss = criterion(model(X), y)
                loss.backward()
                optimizer.step()
    return model


def get_psd_matrix(dim):
    X = torch.randn(dim, dim*3)
    return X @ X.T / (dim * 3)


def grad(model):
    return torch.cat([p.grad.data.flatten() for p in model.parameters()]).detach()


def jacobians_naive(model, data):
    model.zero_grad()
    f = model(data)
    Jacs = list()
    for i in range(f.shape[0]):
        if len(f.shape) > 1:
            jacs = list()
            for j in range(f.shape[1]):
                rg = (i != (f.shape[0] - 1) or j != (f.shape[1] - 1))
                f[i, j].backward(retain_graph=rg)
                Jij = grad(model)
                jacs.append(Jij)
                model.zero_grad()
            jacs = torch.stack(jacs).t()
        else:
            rg = (i != (f.shape[0] - 1))
            f[i].backward(retain_graph=rg)
            jacs = grad(model)
            model.zero_grad()
        Jacs.append(jacs)
    Jacs = torch.stack(Jacs).transpose(1, 2)
    return Jacs.detach(), f.detach()
