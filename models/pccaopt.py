"""=============================================================================
Probabilistic canonical correlation analysis. For references in comments:

    A Probabilistic Interpretation of Canonical Correlation Analysis.
    Bach, Jordan (2006).

    The EM algorithm for mixtures of factor analyzers.
    Ghahramani, Hinton (1996).

    Machine Learning: A Probabilistic Perspective
    Murphy (2006)

This version is vectorized and optimized.
============================================================================="""

import random
import torch
from   torch import nn
from   torch.distributions.multivariate_normal import MultivariateNormal as MVN

import cuda
import linalg as LA

# ------------------------------------------------------------------------------

outer = torch.ger
diag  = torch.diag
det   = torch.det
log   = torch.log
tr    = torch.trace
inv   = torch.inverse

# ------------------------------------------------------------------------------

class PCCAOpt(nn.Module):

    def __init__(self, latent_dim, dims, n_iters, private_z):
        """Initialize the probabilistic CCA model.

        :param latent_dim: The latent variable's dimension.
        :param dims:       Each modality's dimension.
        :param n_iters:    The number of EM iterations.
        """
        super(PCCAOpt, self).__init__()

        self.latent_dim = latent_dim
        self.n_iters    = n_iters
        self.private_z  = private_z

        p1, p2  = dims
        self.p1 = p1
        self.p2 = p2

        Lambda1, Lambda2, B1, B2, Psi1_diag, Psi2_diag = self.init_params()

        self.Lambda1   = nn.Parameter(Lambda1)
        self.Lambda2   = nn.Parameter(Lambda2)
        self.Psi1_diag = nn.Parameter(Psi1_diag)
        self.Psi2_diag = nn.Parameter(Psi2_diag)

        # For a description of PCCA with modality-specific private latent
        # variables, see Murphy (2006), p. 409.
        if self.private_z:
            self.B1 = nn.Parameter(B1)
            self.B2 = nn.Parameter(B2)

# ------------------------------------------------------------------------------

    def forward(self, y):
        """Fit the probabilistic CCA model using Expectation-Maximization.

        :param y: Observations of shape (n_dimensions, n_features).
        """
        Lambda_new, Psi_diag_new, nlls = self.em(y)

        params = self.untile_params(Lambda_new, Psi_diag_new)

        if self.private_z:
            Lambda1, Lambda2, B1, B2, Psi1_diag, Psi2_diag = params
            self.Lambda1.data = Lambda1
            self.Lambda2.data = Lambda2
            self.B1.data = B1
            self.B2.data = B2
            self.Psi1_diag.data = Psi1_diag
            self.Psi2_diag.data = Psi2_diag
            self.nlls = nlls
        else:
            Lambda1, Lambda2, Psi1_diag, Psi2_diag = params
            self.Lambda1.data = Lambda1
            self.Lambda2.data = Lambda2
            self.Psi1_diag.data = Psi1_diag
            self.Psi2_diag.data = Psi2_diag
            self.nlls = nlls

# ------------------------------------------------------------------------------

    def em(self, y):
        """Perform Expectation-Maximization.

        :param y: Observations of shape (n_dimensions, n_features).
        :return:  Parameters that maximize the likelihood given the data.
        """
        Lambda, Psi_diag = self.tile_params()
        neg_log_likes = []

        for _ in range(self.n_iters):
            Lambda, Psi_diag = self.em_step(y, Lambda, Psi_diag)
            nll = self.neg_log_likelihood(y, Lambda, Psi_diag)
            neg_log_likes.append(nll)

        return Lambda, Psi_diag, neg_log_likes

# ------------------------------------------------------------------------------

    def em_step(self, y, Lambda, Psi_diag):
        """Equations 5 and 6 in Ghahramani and Hinton (1996).

        :param y:        Observations of shape (n_dimensions, n_features).
        :param Lambda:   Current Lambda.
        :param Psi_diag: Current Psi_diag.
        :return:         Maximum likelihood parameters for the current step.
        """
        k = 3 * self.latent_dim if self.private_z else self.latent_dim
        p, n = y.shape

        PLL_inv = LA.woodbury_inv(Psi_diag, Lambda, Lambda.t(), k)

        # E-step: compute expected moments for latent variable z.
        # -------------------------------------------------------
        Ez  = self.E_z_given_y(Lambda, PLL_inv, y)
        Ezz = self.E_zzT_given_y(Lambda, PLL_inv, y, k)

        # M-step: compute optimal Lambda and Psi.
        # ---------------------------------------

        # Compute Lambda_new (Equation 5, G&H 1996).
        Lambda_lterm = LA.sum_outers(y, Ez)
        Lambda_rterm = Ezz
        Lambda_new   = Lambda_lterm @ inv(Lambda_rterm)

        # Compute Psi_diag_new (Equation 6, G&H 1996). Must use Lambda_new!
        Psi_rterm    = LA.sum_outers(y, y) - LA.sum_outers(Lambda_new @ Ez, y)
        Psi_diag_new = 1./n * diag(Psi_rterm)

        return Lambda_new, Psi_diag_new

# ------------------------------------------------------------------------------

    def E_z_given_y(self, L, PLL_inv, y):
        """Equation 2 in Ghahramani and Hinton (1996).
        """
        assert len(PLL_inv.shape) == 2
        beta = L.t() @ PLL_inv
        return beta @ y

# ------------------------------------------------------------------------------

    def E_zzT_given_y(self, L, PLL_inv, y, k):
        """Equation 4 in Ghahramani and Hinton (1996).
        """
        _, n = y.shape
        assert len(PLL_inv.shape) == 2
        beta = L.t() @ PLL_inv
        I    = torch.eye(k)
        bL   = beta @ L
        by   = beta @ y
        byyb = torch.einsum('ib,ob->io', [by, by])
        return n * (I - bL) + byyb

# ------------------------------------------------------------------------------

    def sample(self, y, n_samples):
        """Sample from the fitted probabilistic CCA model.

        :param n: The number of samples.
        :return:  Two views of n samples each.
        """
        k = 3 * self.latent_dim if self.private_z else self.latent_dim

        Lambda, Psi_diag = self.tile_params()

        PLL_inv = LA.woodbury_inv(Psi_diag, Lambda, Lambda.t(), k)
        z = self.E_z_given_y(Lambda, PLL_inv, y)

        m = Lambda @ z
        m1 = m[:self.p1]
        m2 = m[self.p1:]

        y1 = torch.empty(self.p1, n_samples)
        y2 = torch.empty(self.p2, n_samples)

        for i in range(n_samples):
            # Randomly select a latent variable.
            r = random.randint(0, z.shape[1])
            # Sample y using the mean for the chosen latent variable.
            y1[:, i] = MVN(m1[:, r], diag(self.Psi1_diag)).sample()
            y2[:, i] = MVN(m2[:, r], diag(self.Psi2_diag)).sample()

        return y1.t(), y2.t()

# ------------------------------------------------------------------------------

    def neg_log_likelihood(self, y, Lambda=None, Psi_diag=None):
        """Appendix A (p. 5) in Ghahramani and Hinton (1996).

        :param y:        (n x p)-dimensional observations.
        :param Lambda:   Current value for Lambda parameter.
        :param Psi_diag: Current value for Psi parameter.
        :return:         The negative log likelihood of the parameters given y.
        """
        p, n = y.shape
        k = 3 * self.latent_dim if self.private_z else self.latent_dim

        if Lambda is None and Psi_diag is None:
            Lambda, Psi_diag = self.tile_params()

        PLL_inv = LA.woodbury_inv(Psi_diag, Lambda, Lambda.t(), k)
        Ez  = self.E_z_given_y(Lambda, PLL_inv, y).t()
        Ezz = self.E_zzT_given_y(Lambda, PLL_inv, y, k).t()

        inv_Psi = diag(LA.diag_inv(Psi_diag))

        A = 1/2. * diag(y.t() @ inv_Psi @ y)
        B = diag(y.t() @ inv_Psi @ Lambda @ Ez.t())
        C = 1/2. * tr(Lambda.t() @ inv_Psi @ Lambda @ Ezz)
        rterm_sum = (A - B).sum() + C

        logdet = -n/2. * log(det(diag(Psi_diag)))

        ll = (logdet - rterm_sum).item()
        nll = -ll
        return nll

# ------------------------------------------------------------------------------

    def init_params(self):
        """Create model parameters and move them to appropriate device.

        :return: Model parameters.
        """
        p1 = self.p1
        p2 = self.p2
        k  = self.latent_dim
        device = cuda.device()

        Lambda1 = torch.randn(p1, k).to(device)
        Lambda2 = torch.randn(p2, k).to(device)

        # Create these modality-specific parameters regardless. We will discard
        # them if the user did not set `private_z=True`.
        B1 = torch.randn(p1, k).to(device)
        B2 = torch.randn(p2, k).to(device)

        Psi1_diag = torch.ones(p1).to(device)
        Psi2_diag = torch.ones(p2).to(device)

        return Lambda1, Lambda2, B1, B2, Psi1_diag, Psi2_diag

# ------------------------------------------------------------------------------

    def tile_params(self):
        """Tile parameters so we can use factor analysis updates for PCCA.

        :return: Model parameters concatenated appropriately.
        """
        p1 = self.p1
        p2 = self.p2
        k  = self.latent_dim
        device = cuda.device()

        B12 = torch.zeros(p1, k).to(device)
        B21 = torch.zeros(p2, k).to(device)

        if self.private_z:
            Lambda = torch.cat([
                torch.cat([self.Lambda1, self.B1, B12], dim=1),
                torch.cat([self.Lambda2, B21, self.B2], dim=1)
            ], dim=0)
        else:
            Lambda = torch.cat([self.Lambda1, self.Lambda2], dim=0)

        Psi_diag = torch.cat([self.Psi1_diag, self.Psi2_diag])

        return Lambda, Psi_diag

# ------------------------------------------------------------------------------

    def untile_params(self, Lambda, Psi_diag):
        """Takes tiled parameters and untiles them; reverse of `tile_params()`.

        :return: Model parameters unrolled or un-concatenated appropriately.
        """
        k = self.latent_dim
        Lambda1 = Lambda[:self.p1, :k]
        Lambda2 = Lambda[self.p1:, :k]
        Psi1_diag = Psi_diag[:self.p1]
        Psi2_diag = Psi_diag[self.p1:]

        if self.private_z:
            B1 = Lambda[:self.p1, k:2 * k]
            B2 = Lambda[self.p1:, 2 * k:]
            return Lambda1, Lambda2, B1, B2, Psi1_diag, Psi2_diag
        else:
            return Lambda1, Lambda2, Psi1_diag, Psi2_diag
