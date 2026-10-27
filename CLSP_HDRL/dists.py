import torch
import torch.nn.functional as F

class MaskedBernoulli:
    """Masked multivariate Bernoulli distribution for the HIGH-level agent."""

    def __init__(self, logits, mask):
        # Extremely negative value compatible with logits dtype
        very_neg = torch.finfo(logits.dtype).min / 2

        # Align mask dtype with logits for safe broadcasting
        mask = mask.to(dtype=logits.dtype)

         # Where mask == 0 -> assign very_neg to suppress the action component
        self.logits = torch.where(mask > 0, logits, torch.full_like(logits, very_neg))

        # Component-wise Bernoulli probabilities
        self.probs = torch.sigmoid(self.logits)

    def sample(self):
        """Sample binary actions {0, 1} from the independent Bernoulli distribution.

        :return: Sampled binary actions with the same shape as ``self.logits``.
        """
        return torch.bernoulli(self.probs)

    def log_prob(self, actions):
        """Compute the joint log-probability of a binary action vector.

        :param actions: Tensor of binary actions {0, 1} with the same shape as ``self.logits``.
        :return: Per-sample log-probabilities summed across the action dimension (shape: ``(B,)``).
        """
        actions = actions.to(dtype=self.logits.dtype)

        # log(sigmoid(x)) = -softplus(-x)
        logp1 = -F.softplus(-self.logits)          # log P(a=1)
        # log(1 - sigmoid(x)) = -x + log(sigmoid(x))
        logp0 = -self.logits + logp1               # log P(a=0)

        logp = actions * logp1 + (1 - actions) * logp0
        return logp.sum(dim=-1)


class MaskedCategorical:
    """Masked categorical distribution for the LOW-level agent."""
    def __init__(self, logits, mask):
        # Extremely negative value compatible with logits dtype
        very_neg = torch.finfo(logits.dtype).min / 2

        # Align mask dtype with logits for safe broadcasting
        mask = mask.to(dtype=logits.dtype)

        # Apply masking: logits -> very_neg where mask == 0
        masked_logits = torch.where(mask > 0, logits, torch.full_like(logits, very_neg))

        # If a row has no valid actions, enable action 0 to keep the distribution well-defined
        no_valid = (mask <= 0).all(dim=-1)  # shape: (B,)
        if no_valid.any():
            fix = masked_logits.new_full(masked_logits.shape, very_neg)
            fix[:, 0] = 0.0
            masked_logits = torch.where(no_valid.unsqueeze(-1), fix, masked_logits)

        self.logits = masked_logits
        self.probs = torch.softmax(self.logits, dim=-1)

    def sample(self):
        """Sample a valid discrete action from the masked categorical distribution.

        :return: Integer action indices (shape: ``(B,)``).
        """
        return torch.distributions.Categorical(logits=self.logits).sample()

    def log_prob(self, actions):
        """Compute the log-probability of the given discrete actions.

        :param actions: Tensor of chosen action indices (shape: ``(B,)``).
        :return: Log-probabilities of the selected actions (shape: ``(B,)``).
        """
        return torch.distributions.Categorical(logits=self.logits).log_prob(actions)


class BetaDist:
    """Beta(alpha, beta) distribution for the LOW-level continuous action."""

    _EPS = 1e-6

    def __init__(self, alpha_raw, beta_raw):
        self.alpha = F.softplus(alpha_raw) + 1.0
        self.beta = F.softplus(beta_raw) + 1.0
        self._dist = torch.distributions.Beta(self.alpha, self.beta)

    def sample(self):
        """Sample a continuous action in (0, 1) from the Beta distribution.

        :return: Sampled actions, same shape as ``self.alpha``.
        """
        a = self._dist.sample()
        return a.clamp(self._EPS, 1.0 - self._EPS)

    def log_prob(self, actions):
        """Compute the log-density of the given continuous actions.

        :param actions: Tensor of actions in [0, 1], same shape as ``self.alpha``.
        :return: Log-density of the given actions under the Beta distribution.
        """
        a = actions.clamp(self._EPS, 1.0 - self._EPS)
        return self._dist.log_prob(a)

    def entropy(self):
        """Differential entropy of the Beta distribution.

        :return: Entropy per batch element, same shape as ``self.alpha``.
        """
        return self._dist.entropy()

    @property
    def mean(self):
        """Deterministic ("greedy") action: the distribution mean.

        :return: ``alpha / (alpha + beta)``, same shape as ``self.alpha``.
        """
        return self.alpha / (self.alpha + self.beta)


# Plackett-Luce over the HIGH logits (production order, S-CLSP-SD)
def plackett_luce_log_prob(logits, order):
    """Log-probability of an ordering under a Plackett-Luce model whose scores are ``logits``:

    :param logits: (B, K) scores (the same HIGH logits used for selection).
    :param order: (B, K) long tensor with the ordered product indices in the first n positions of each row
        and -1 as padding.
    :return: (B,) log-probabilities (0 for rows with fewer than 2 products).
    """
    valid = order >= 0
    lo = torch.gather(logits, 1, order.clamp(min=0))
    lo = torch.where(valid, lo, torch.full_like(lo, float("-inf")))
    # log-sum-exp of the scores still available at each position: suffix logsumexp
    suffix_lse = torch.flip(torch.logcumsumexp(torch.flip(lo, dims=[1]), dim=1), dims=[1])
    terms = torch.where(valid, lo - suffix_lse, torch.zeros_like(lo))
    return terms.sum(dim=-1)


def plackett_luce_sample(logits_sel, rng):
    """Sample an ordering of the selected products with the Gumbel trick: adding i.i.d. Gumbel(0, 1) noise
    to the scores and sorting in descending order is an exact Plackett-Luce sample.

    :param logits_sel: 1-D numpy array with the scores of the selected products.
    :param rng: numpy Generator.
    :return: permutation (indices into ``logits_sel``) in sampled order.
    """
    import numpy as _np
    g = -_np.log(-_np.log(rng.uniform(1e-12, 1.0, size=len(logits_sel))))
    return _np.argsort(-(_np.asarray(logits_sel, dtype=_np.float64) + g), kind="stable")
