"""Analytic graph geometry and identifying reconstruction losses.

The coefficients use the archived cubic ordering: degree increases first,
then the exponent of u.  A chart is
    p(u,v) = (u,v,h(u,v)), x = p + tau*n.
Physical position is origin + radius * frame @ p.  No derivative is taken
through attention: all spatial derivatives differentiate the polynomial.
"""
from __future__ import annotations

import math
import torch


def powers(degree: int = 3):
    return [(i, d - i) for d in range(degree + 1) for i in range(d + 1)]


def basis(uv: torch.Tensor, degree: int = 3, du: int = 0, dv: int = 0):
    """Return monomial derivatives for uv[...,2], retaining dtype/device."""
    u, v = uv.unbind(-1)
    values = []
    for i, j in powers(degree):
        if i < du or j < dv:
            values.append(torch.zeros_like(u))
        else:
            factor = math.factorial(i) / math.factorial(i - du)
            factor *= math.factorial(j) / math.factorial(j - dv)
            values.append(factor * u.pow(i - du) * v.pow(j - dv))
    return torch.stack(values, -1)


def graph_geometry(coeff: torch.Tensor, uv: torch.Tensor, tau: torch.Tensor):
    """Evaluate coefficients [...,10] at uv[...,Q,2], tau[...,Q].

    Arithmetic is float64.  Invalid normal-coordinate charts raise before
    matrix inversion; no identity substitution or eigenvalue clipping can
    turn an invalid chart into a passing operator metric.  Offsets must
    already satisfy the reference chart's documented domain restrictions.
    This is a local chart calculation, not a global nearest-point certificate.
    """
    if coeff.shape[-1] != 10 or uv.shape[-1] != 2:
        raise ValueError("Expected cubic coefficients [...,10] and uv[...,Q,2]")
    if coeff.shape[:-1] != uv.shape[:-2] or tau.shape != uv.shape[:-1]:
        raise ValueError("Coefficient, chart-query and offset batch dimensions differ")
    coeff, uv, tau = coeff.double(), uv.double(), tau.double()
    if not all(bool(torch.isfinite(x).all()) for x in (coeff, uv, tau)):
        raise ValueError("Nonfinite chart coefficients or queries")

    def derivative(i, j):
        return (basis(uv, du=i, dv=j) * coeff.unsqueeze(-2)).sum(-1)

    h, hu, hv = derivative(0, 0), derivative(1, 0), derivative(0, 1)
    huu, huv, hvv = derivative(2, 0), derivative(1, 1), derivative(0, 2)
    zero, one = torch.zeros_like(h), torch.ones_like(h)
    q = torch.stack((-hu, -hv, one), -1)
    length = torch.linalg.vector_norm(q, dim=-1, keepdim=True)
    normal = q / length
    dq = torch.stack((torch.stack((-huu, -huv, zero), -1),
                      torch.stack((-huv, -hvv, zero), -1)), -1)
    dn = (dq - normal[..., :, None] *
          (normal[..., :, None] * dq).sum(-2, keepdim=True)) / length[..., None]
    tangent = torch.stack((torch.stack((one, zero, hu), -1),
                           torch.stack((zero, one, hv), -1)), -1)
    J = torch.cat((tangent + tau[..., None, None] * dn, normal[..., :, None]), -1)
    determinant = torch.linalg.det(J)
    g_surface = tangent.transpose(-1, -2) @ tangent
    stretch = g_surface + tau[..., None, None] * (tangent.transpose(-1, -2) @ dn)
    # Positive determinant alone permits crossing two focal singularities.
    # Test the connected positive-stretch branch as well.
    minimum_stretch = torch.linalg.eigvalsh((stretch + stretch.transpose(-1, -2)) / 2)[..., 0]
    valid = (determinant > 1e-7) & (minimum_stretch > 1e-7) & torch.isfinite(J).all(dim=(-1, -2))
    if not bool(valid.all()):
        bad = int((~valid).sum().detach().cpu())
        raise ValueError(f"{bad}/{valid.numel()} invalid normal-coordinate charts; operator evaluation rejected")
    g = J.transpose(-1, -2) @ J
    A = determinant[..., None, None] * torch.linalg.inv(g)
    return dict(p=torch.stack((uv[..., 0], uv[..., 1], h), -1), n=normal,
                A=A, J=J, area=length[..., 0], valid=valid, det=determinant)


def _mean(values, area, patch_weights=None):
    """Reference-area mean within each patch, then mean over objects."""
    if values.shape != area.shape or values.ndim != 3:
        raise ValueError("Expected values and quadrature areas [objects, patches, queries]")
    per_patch = (values * area).sum(-1) / area.sum(-1).clamp_min(1e-30)
    if patch_weights is None:
        return per_patch.mean()
    weights = patch_weights.to(dtype=per_patch.dtype, device=per_patch.device)
    if weights.shape != per_patch.shape or not bool(torch.isfinite(weights).all()):
        raise ValueError("Invalid patch quadrature weights")
    if bool((weights < 0).any()) or bool((weights.sum(-1) <= 0).any()):
        raise ValueError("Patch weights must be nonnegative with positive object mass")
    return ((per_patch * weights).sum(-1) / weights.sum(-1)).mean()


def operator_losses(predicted_coeff, target_coeff, uv, tau, patch_weights=None):
    """Continuous local height + normal-projector + relative-operator losses.

    Each term has coefficient one before the caller's common auxiliary weight.
    Height is measured in patch-radius units; there is no inactive tolerance
    interval.  The first two terms retain the modes invisible or weak in A.
    The reference area fixes quadrature weights independently of predictions.
    """
    with torch.no_grad():
        reference = graph_geometry(target_coeff, uv, tau)
        ev, U = torch.linalg.eigh(reference['A'])
        inverse_root = (U * ev.rsqrt().unsqueeze(-2)) @ U.transpose(-1, -2)
    learned = graph_geometry(predicted_coeff, uv, tau)
    area = reference['area'].detach()
    position_squared = (learned['p'] - reference['p']).square().sum(-1)
    P = learned['n'][..., :, None] * learned['n'][..., None, :]
    target_P = reference['n'][..., :, None] * reference['n'][..., None, :]
    normal_squared = (P - target_P).square().sum((-1, -2))
    identity = torch.eye(3, dtype=learned['A'].dtype, device=learned['A'].device)
    relative = inverse_root @ learned['A'] @ inverse_root - identity
    relative_squared = relative.square().sum((-1, -2))
    L_p = _mean(position_squared, area, patch_weights)
    L_P = _mean(normal_squared, area, patch_weights)
    L_A = _mean(relative_squared, area, patch_weights)
    with torch.no_grad():
        eigen_error = torch.linalg.eigvalsh(relative).abs().amax(-1)
        cosine = (learned['n'] * reference['n']).sum(-1).abs().clamp(0, 1)
        metrics = dict(
            patch_position_rms=L_p.sqrt(),
            patch_normal_degrees=_mean(torch.rad2deg(torch.acos(cosine)), area, patch_weights),
            relative_A_rms=_mean(eigen_error.square(), area, patch_weights).sqrt(),
            relative_A_max=eigen_error.max(),
            invalid_charts=relative.new_zeros(()),
        )
    return L_p + L_P + L_A, dict(patch_position=L_p, patch_normal=L_P, operator=L_A, **metrics)


def symmetric_frobenius_squared(packed_error):
    """Raw symmetric order xx,yy,zz,xy,xz,yz; off-diagonals count twice."""
    if packed_error.shape[-1] != 6:
        raise ValueError("Expected six raw symmetric components")
    return (packed_error.square() * packed_error.new_tensor([1, 1, 1, 2, 2, 2])).sum(-1)
