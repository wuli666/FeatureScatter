import torch
from torch import nn


class MCR2Loss(nn.Module):
    def __init__(self, eps=1e-2, loss_weight=1.0, expd_weight=1.0, comp_weight=1.0, return_all=False):
        super(MCR2Loss, self).__init__()
        self.eps = eps
        self.loss_weight = loss_weight
        self.expd_weight = expd_weight
        self.comp_weight = comp_weight
        self.return_all = return_all

    def forward(self, cls_score, label):
        _, loss_expd, loss_comp = mcr2_loss(cls_score, label, self.eps)
        # print("\nloss_comp\t", loss_comp.cpu().detach().numpy())
        # print("loss_expd\t", loss_expd.cpu().detach().numpy())
        # print("loss_comp/loss_expd\t", loss_comp.cpu().detach().numpy()/loss_expd.cpu().detach().numpy())
        loss_mcr2 = loss_comp * self.comp_weight - loss_expd * self.expd_weight
        if self.return_all:
            return loss_mcr2, loss_expd * self.expd_weight, loss_comp * self.comp_weight
        else:
            return loss_mcr2


def mcr2_loss(Z, y, eps):
    if len(Z.shape) == 2:
        loss_func = compute_loss_vec
    elif len(Z.shape) == 3:
        loss_func = compute_loss_1d
    elif len(Z.shape) == 4:
        loss_func = compute_loss_2d
    else:
        raise NotImplemented
    return loss_func(Z, y, eps)


def compute_loss_vec(Z, y, eps):
    m, d = Z.shape
    c = d / (m * eps)
    loss_expd = logdet(c * covariance(Z)) / 2.
    loss_comp = 0.
    for j in y.unique():
        Z_j = Z[(y == int(j))]
        m_j = Z_j.shape[0]
        c_j = d / (m_j * eps)
        logdet_j = logdet(c_j * Z_j.T @ Z_j)
        loss_comp += logdet_j * m_j / (2 * m)
    loss_expd, loss_comp = loss_expd, loss_comp
    return loss_expd - loss_comp, loss_expd, loss_comp


def compute_loss_1d(V, y, eps):
    m, C, T = V.shape
    alpha = C / (m * eps)
    cov = alpha * covariance(V)
    loss_expd = logdet(cov.permute(2, 0, 1)).sum() / (2 * T)
    loss_comp = 0.
    for j in y.unique():
        V_j = V[y == int(j)]
        m_j = V_j.shape[0]
        alpha_j = C / (m_j * eps)
        cov_j = alpha_j * covariance(V_j)
        loss_comp += m_j / m * logdet(cov_j.permute(2, 0, 1)).sum() / (2 * T)
    loss_expd, loss_comp = loss_expd.real, loss_comp.real
    return loss_expd - loss_comp, loss_expd, loss_comp


def compute_loss_2d(V, y, eps):
    m, C, H, W = V.shape
    alpha = C / (m * eps)
    cov = alpha * covariance(V)
    loss_expd = logdet(cov.permute(2, 3, 0, 1)).sum() / (2 * H * W)
    loss_comp = 0.
    for j in y.unique():
        # print(y == int(j))
        V_j = V[(y == int(j))]
        # V_j = V[(y == int(j))[:, 0]]
        m_j = V_j.shape[0]
        alpha_j = C / (m_j * eps)
        cov_j = alpha_j * covariance(V_j)
        loss_comp += m_j / m * logdet(cov_j.permute(2, 3, 0, 1)).sum() / (2 * H * W)

    loss_expd, loss_comp = loss_expd, loss_comp
    return loss_expd - loss_comp, loss_expd, loss_comp


def covariance(X):
    return torch.einsum('ji...,jk...->ik...', X, X.conj())


def logdet(X):
    sgn, logdet = torch.slogdet(X)
    return sgn * logdet
