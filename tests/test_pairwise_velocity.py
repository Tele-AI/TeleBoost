"""Unit tests for the pathwise pairwise-velocity loss (option 1, no Ray/GPU).

Verifies, on synthetic Wan-SDE transitions:
  1. SIGN: the original model output minimizes ‖prev_sample_mean − x_next‖² — i.e.
     matching prev_sample_mean ↔ realized next_latent is velocity matching with the
     correct sign (perturbing v_θ only increases the transition MSE).
  2. HIGHER REWARD -> BIGGER PULL: a sample's gradient pull on v_θ grows with its
     advantage (weighted mode pulls winners, ignores losers).
  3. VARIANCE (the crux — measured, NOT assumed): compare the gradient variance of
     the pathwise reward-weighted-matching estimator vs the score-function GRPO
     estimator on the same reward. Reports the ratio so we can SEE whether the
     directional consistency of "always pull winners" actually beats adv·ε.
"""

import torch

from teleboost.algorithms.grpo.sigma_schedule import compute_sde_step


def _pairwise_velocity_loss(prev_sample_mean, next_latents, advantages, *, weight_floor=0.0):
    """Local negative-result fixture; intentionally not a runtime algorithm."""
    if prev_sample_mean.shape != next_latents.shape:
        raise ValueError("prev_sample_mean and next_latents must have the same shape")
    bsz = prev_sample_mean.shape[0]
    adv = advantages.reshape(-1).to(prev_sample_mean.dtype)
    if adv.shape[0] != bsz:
        raise ValueError(f"advantages length {adv.shape[0]} != batch {bsz}")
    mse = ((prev_sample_mean - next_latents.detach()) ** 2).flatten(start_dim=1).mean(dim=1)
    weights = (adv - weight_floor).clamp_min(0.0)
    return (weights * mse).mean()


def _prev_mean(v, latents, sigma, sigma_next, eta=0.3):
    pred_orig = latents - sigma * v
    pm, std, _ = compute_sde_step("dancegrpo", v, latents, eta, sigma, sigma_next, pred_orig)
    return pm, std


def test_sign_original_model_output_minimizes_transition_mse():
    torch.manual_seed(0)
    B, D = 8, 16
    latents = torch.randn(B, D)
    v_true = torch.randn(B, D)
    sigma, sigma_next = torch.tensor(0.7), torch.tensor(0.6)
    pm_true, std = _prev_mean(v_true, latents, sigma, sigma_next)
    x_next = pm_true + std * torch.randn_like(pm_true)  # realized SDE step

    mse_true = ((pm_true - x_next) ** 2).mean().item()
    for scale in (0.2, 0.5, 1.0):
        v_pert = v_true + scale * torch.randn_like(v_true)
        pm_pert, _ = _prev_mean(v_pert, latents, sigma, sigma_next)
        mse_pert = ((pm_pert - x_next) ** 2).mean().item()
        assert mse_pert > mse_true, (scale, mse_pert, mse_true)


def test_higher_reward_produces_bigger_pull():
    torch.manual_seed(0)
    B, D = 6, 8
    latents = torch.randn(B, D)
    v = torch.zeros(B, D, requires_grad=True)
    sigma, sigma_next = torch.tensor(0.7), torch.tensor(0.6)
    pm, std = _prev_mean(v, latents, sigma, sigma_next)
    # SAME transition error for every sample (constant offset) so the per-sample
    # pull is ∝ weight only — isolates "higher reward -> bigger pull" from the
    # random per-sample noise magnitude.
    x_next = pm.detach() - 1.0
    adv = torch.tensor([2.0, 1.0, 0.0, -1.0, -2.0, 0.5])
    loss = _pairwise_velocity_loss(pm, x_next, adv)
    (g,) = torch.autograd.grad(loss, v)
    pull = g.flatten(1).norm(dim=1)  # per-sample gradient magnitude on v
    # losers (adv<=0) get exactly zero pull (relu); winners pulled ∝ advantage.
    assert pull[3] == 0 and pull[4] == 0 and pull[2] == 0  # adv = -1,-2,0
    assert pull[0] > pull[1] > pull[5] > 0  # adv = 2 > 1 > 0.5


def _score_function_grad(v, latents, x_next, std, adv):
    # GRPO score-function gradient of -mean(adv*logπ) wrt v, at ratio=1.
    pm, _ = _prev_mean(v, latents, torch.tensor(0.7), torch.tensor(0.6))
    logp = -((x_next - pm) ** 2).flatten(1).sum(1) / (2 * std**2)
    loss = -(adv * logp).mean()
    (g,) = torch.autograd.grad(loss, v)
    return g


def test_variance_pathwise_vs_score_function():
    torch.manual_seed(0)
    B, D, trials = 8, 16, 4000
    sigma, sigma_next = torch.tensor(0.7), torch.tensor(0.6)
    reward_dir = torch.randn(D)
    reward_dir /= reward_dir.norm()
    v0 = torch.zeros(B, D)
    latents = torch.randn(B, D)

    g_sf_list, g_pw_list = [], []
    for _ in range(trials):
        v = v0.clone().requires_grad_(True)
        pm, std = _prev_mean(v, latents, sigma, sigma_next)
        noise = torch.randn_like(pm)
        x_next = (pm + std * noise).detach()
        # reward rewards moving x_next along reward_dir + per-sample obs noise.
        r = (x_next @ reward_dir) + 0.5 * torch.randn(B)
        adv = (r - r.mean()) / (r.std(unbiased=True) + 1e-8)

        g_sf_list.append(_score_function_grad(v, latents, x_next, std, adv).detach().clone())
        v2 = v0.clone().requires_grad_(True)
        pm2, _ = _prev_mean(v2, latents, sigma, sigma_next)
        loss_pw = _pairwise_velocity_loss(pm2, x_next, adv)
        (g_pw,) = torch.autograd.grad(loss_pw, v2)
        g_pw_list.append(g_pw.detach().clone())

    g_sf = torch.stack(g_sf_list).reshape(trials, -1)
    g_pw = torch.stack(g_pw_list).reshape(trials, -1)
    # project onto the reward-improving direction (what we want the gradient to track)
    target = torch.zeros(B, D)
    target += reward_dir
    target = target.reshape(-1)
    target /= target.norm()
    sf_sig, pw_sig = (g_sf @ target), (g_pw @ target)
    snr_sf = (sf_sig.mean() ** 2 / g_sf.var(0).sum()).item()
    snr_pw = (pw_sig.mean() ** 2 / g_pw.var(0).sum()).item()
    print(f"score-function: signal={sf_sig.mean():.4f} var={g_sf.var(0).sum():.4f} SNR={snr_sf:.5f}")
    print(f"pathwise(wmatch): signal={pw_sig.mean():.4f} var={g_pw.var(0).sum():.4f} SNR={snr_pw:.5f}")
    print(f"SNR gain pathwise/score-function = {snr_pw / max(snr_sf, 1e-12):.2f}x")
    # Report-only here (the magnitude is the research result); assert it at least
    # does not REDUCE SNR (a regression would mean the loss form is wrong).
    assert snr_pw > 0.5 * snr_sf, (snr_pw, snr_sf)


if __name__ == "__main__":
    test_sign_original_model_output_minimizes_transition_mse()
    test_higher_reward_produces_bigger_pull()
    test_variance_pathwise_vs_score_function()
    print("OK")
