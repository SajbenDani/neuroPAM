"""latents.py -- Encoding / decoding helpers around the trained FC-VAE."""

from __future__ import annotations

import numpy as np
import torch


def extract_latents(model, dataset, device, batch_size: int = 256):
    """
    Encode all subjects.  Returns ``Z_mu, Z_logvar, y_site, y_diag, X_hat``
    (numpy, CPU); ``X_hat`` is the deterministic decode of ``mu``.
    """
    model.eval()
    mus, logvars, sites, diags, xh = [], [], [], [], []
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for x, site, diag in loader:
            x = x.to(device)
            mu, logvar = model.encoder(x)
            mus.append(mu.cpu().numpy())
            logvars.append(logvar.cpu().numpy())
            xh.append(model.decoder(mu).cpu().numpy())
            sites.append(site.numpy())
            diags.append(diag.numpy())
    return (np.concatenate(mus), np.concatenate(logvars), np.concatenate(sites),
            np.concatenate(diags).astype(np.int64), np.concatenate(xh))


def encode_features(model, X_pca: np.ndarray, device, batch_size: int = 256) -> np.ndarray:
    """Encode an arbitrary (N, pca_dim) matrix to posterior means."""
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X_pca), batch_size):
            xb = torch.as_tensor(X_pca[i:i + batch_size], dtype=torch.float32, device=device)
            out.append(model.encoder(xb)[0].cpu().numpy())
    return np.concatenate(out)


def decode_latents(model, Z: np.ndarray, device, batch_size: int = 256) -> np.ndarray:
    """Decode latents to the PCA feature space."""
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(Z), batch_size):
            zb = torch.as_tensor(Z[i:i + batch_size], dtype=torch.float32, device=device)
            out.append(model.decoder(zb).cpu().numpy())
    return np.concatenate(out)


def decode_to_fc(model, Z: np.ndarray, dataset, device) -> np.ndarray:
    """Decode latents all the way back to the Fisher-z FC vector space."""
    return dataset.inverse_to_raw(decode_latents(model, Z, device))
