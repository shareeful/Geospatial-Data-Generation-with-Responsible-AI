import torch
import torch.nn as nn
import numpy as np
from collections import deque


class Generator(nn.Module):
    def __init__(self, latent_dim, n_classes, out_dim, hidden):
        super().__init__()
        layers = []
        prev = latent_dim + n_classes
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.LeakyReLU(0.2)]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)
        self.n_classes = n_classes

    def forward(self, z, c):
        oh = torch.zeros(c.size(0), self.n_classes, device=z.device)
        oh.scatter_(1, c.unsqueeze(1).long(), 1.0)
        return self.net(torch.cat([z, oh], 1))


class Discriminator(nn.Module):
    def __init__(self, in_dim, n_classes, hidden, dropout=0.3):
        super().__init__()
        layers = []
        prev = in_dim + n_classes
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.LeakyReLU(0.2), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, 1), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)
        self.n_classes = n_classes

    def forward(self, x, c):
        oh = torch.zeros(c.size(0), self.n_classes, device=x.device)
        oh.scatter_(1, c.unsqueeze(1).long(), 1.0)
        return self.net(torch.cat([x, oh], 1))


class RDPAccountant:
    def __init__(self, sample_rate, noise_mult, target_eps, delta):
        self.sample_rate = sample_rate
        self.noise_mult = noise_mult
        self.target_eps = target_eps
        self.delta = delta
        self.orders = np.arange(2, 128, 0.5)
        self.rdp = np.zeros_like(self.orders)

    def step(self):
        q, sigma = self.sample_rate, self.noise_mult
        for i, alpha in enumerate(self.orders):
            self.rdp[i] += (alpha * q * q) / (2 * sigma * sigma)

    def epsilon(self):
        return float(np.min(self.rdp - np.log(self.delta) / (self.orders - 1)))

    def budget_ok(self):
        return self.epsilon() <= self.target_eps


def dp_step(model, max_norm, noise_mult):
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.data.norm(2).item() ** 2
    total = total ** 0.5
    clip = max_norm / (total + 1e-6)
    if clip < 1.0:
        for p in model.parameters():
            if p.grad is not None:
                p.grad.data.mul_(clip)
    for p in model.parameters():
        if p.grad is not None:
            p.grad.data.add_(torch.normal(0, noise_mult * max_norm, size=p.grad.shape, device=p.grad.device))


def compute_seod_penalty(D, x_real, labels, zones, device):
    with torch.no_grad():
        preds = D(x_real, labels)
    uz = torch.unique(zones)
    if len(uz) < 2:
        return torch.tensor(0.0, device=device)
    tprs = []
    for z in uz:
        m = zones == z
        tprs.append((preds[m] > 0.5).float().mean())
    t = torch.stack(tprs)
    return t.max() - t.min()


def train_cgan(X, y, zones, cfg, use_dp=True, use_seod=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cc = cfg["cgan"]
    pc = cfg["privacy"]
    n_classes = len(np.unique(y))
    feat_dim = X.shape[1]

    G = Generator(cc["latent_dim"], n_classes, feat_dim, cc["hidden"]).to(device)
    D = Discriminator(feat_dim, n_classes, cc["hidden"], cc["dropout"]).to(device)
    opt_g = torch.optim.Adam(G.parameters(), lr=cc["lr"], betas=(cc["beta1"], cc["beta2"]))
    opt_d = torch.optim.Adam(D.parameters(), lr=cc["lr"], betas=(cc["beta1"], cc["beta2"]))
    bce = nn.BCELoss()

    Xt = torch.FloatTensor(X).to(device)
    yt = torch.LongTensor(y).to(device)
    zt = torch.LongTensor(zones).to(device)

    counts = np.bincount(y, minlength=n_classes).astype(float)
    weights = 1.0 / (counts + 1e-6)
    sw = torch.DoubleTensor(weights[y])
    sampler = torch.utils.data.WeightedRandomSampler(sw, len(X), replacement=True)
    ds = torch.utils.data.TensorDataset(Xt, yt, zt)
    dl = torch.utils.data.DataLoader(ds, batch_size=cc["batch_size"], sampler=sampler, drop_last=True)

    accountant = None
    if use_dp:
        accountant = RDPAccountant(cc["batch_size"] / len(X), pc["noise_multiplier"], pc["epsilon"], pc["delta"])

    loss_window = deque(maxlen=cc["early_stop_window"])
    history = []

    for epoch in range(cc["max_epochs"]):
        ep_dl, ep_gl, ep_seod, nb = 0, 0, 0, 0
        G.train(); D.train()

        for bx, by, bz in dl:
            bs = bx.size(0)
            ones = torch.ones(bs, 1, device=device)
            zeros = torch.zeros(bs, 1, device=device)

            opt_d.zero_grad()
            d_real = D(bx, by)
            loss_real = bce(d_real, ones)

            noise = torch.randn(bs, cc["latent_dim"], device=device)
            fake = G(noise, by)
            d_fake = D(fake.detach(), by)
            loss_fake = bce(d_fake, zeros)

            d_loss = loss_real + loss_fake
            seod_val = 0.0
            if use_seod:
                s = compute_seod_penalty(D, bx, by, bz, device)
                d_loss = d_loss + cc["lambda_seod"] * s
                seod_val = s.item()

            d_loss.backward()
            if use_dp:
                dp_step(D, pc["max_grad_norm"], pc["noise_multiplier"])
                accountant.step()
            opt_d.step()

            opt_g.zero_grad()
            noise = torch.randn(bs, cc["latent_dim"], device=device)
            fake = G(noise, by)
            g_loss = bce(D(fake, by), ones)
            g_loss.backward()
            opt_g.step()

            ep_dl += d_loss.item()
            ep_gl += g_loss.item()
            ep_seod += seod_val
            nb += 1

        avg_dl = ep_dl / max(nb, 1)
        avg_seod = ep_seod / max(nb, 1)
        loss_window.append(avg_dl)
        eps_spent = accountant.epsilon() if accountant else 0.0

        history.append({"epoch": epoch, "d_loss": avg_dl, "g_loss": ep_gl / max(nb, 1),
                         "seod": avg_seod, "eps": eps_spent})

        if epoch % 100 == 0:
            print(f"  epoch {epoch:4d}  D={avg_dl:.4f}  G={ep_gl/max(nb,1):.4f}  sEOD={avg_seod:.4f}  ε={eps_spent:.2f}")

        if use_dp and not accountant.budget_ok():
            print(f"  privacy budget exhausted at epoch {epoch}")
            break

        if len(loss_window) == cc["early_stop_window"]:
            if max(loss_window) - min(loss_window) < cc["early_stop_tol"]:
                print(f"  converged at epoch {epoch}")
                break

    eps_final = accountant.epsilon() if accountant else 0.0
    return G, D, history, eps_final


def generate_balanced(G, n_per_class, n_classes, latent_dim, device=None):
    if device is None:
        device = next(G.parameters()).device
    G.eval()
    Xs, ys = [], []
    with torch.no_grad():
        for c in range(n_classes):
            z = torch.randn(n_per_class, latent_dim, device=device)
            labels = torch.full((n_per_class,), c, dtype=torch.long, device=device)
            Xs.append(G(z, labels).cpu().numpy())
            ys.append(np.full(n_per_class, c))
    X = np.vstack(Xs)
    y = np.concatenate(ys)
    idx = np.random.permutation(len(X))
    return X[idx], y[idx]
