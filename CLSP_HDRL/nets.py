import torch
import torch.nn as nn


# Utilities: MLP construction and orthogonal initialization

def mlp(sizes, act=nn.Tanh):
    """Build a fully connected multi-layer perceptron (MLP).

    :param sizes: Layer widths in order: [input_dim, hidden_1, ..., hidden_n, output_dim].
    :param act: Activation class applied between hidden layers. Defaults to ``nn.Tanh``.
    :return: A sequential neural network composed of linear layers and intermediate activations.
    """
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
    return nn.Sequential(*layers)


def orthogonal_init(module, gain=1.0):
    """Apply orthogonal initialization to linear layers.

    :param module: PyTorch module to initialize.
    :param gain: Scaling factor applied to the orthogonal weight matrix.
    :return: The same module, potentially initialized in-place.
    """
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        nn.init.constant_(module.bias, 0.0)
    return module


def init_actor_critic_nets(actor, critic, act_hidden_act=nn.Tanh):
    """Orthogonally initialize actor and critic networks with head-specific gains.

    :param actor: Actor network (typically an ``nn.Sequential``).
    :param critic: Critic network (typically an ``nn.Sequential``).
    :param act_hidden_act: Activation class used in hidden layers (used to compute the appropriate gain).
    :return: None. Networks are modified in-place.
    """
    gain_hidden = nn.init.calculate_gain(
        "tanh" if act_hidden_act is nn.Tanh else "linear"
    )

    # Actor hidden layers
    for m in actor[:-1]:
        if isinstance(m, nn.Linear):
            orthogonal_init(m, gain=gain_hidden)
    # Actor output head (logits)
    if isinstance(actor[-1], nn.Linear):
        orthogonal_init(actor[-1], gain=0.01)

    # Critic hidden layers
    for m in critic[:-1]:
        if isinstance(m, nn.Linear):
            orthogonal_init(m, gain=gain_hidden)
    # Critic output head (V(s))
    if isinstance(critic[-1], nn.Linear):
        orthogonal_init(critic[-1], gain=1.0)


# Hierarchical policy networks
class HighPolicy(nn.Module):
    """High-level (HIGH) actor-critic network.

    :param obs_dim: Dimension of the high-level observation (typically 2K).
    :param K: Number of products (dimension of the MultiBinary action).
    :param hidden: Hidden layer widths for the MLP. Defaults to (256, 256).
    :param act: Activation class used between hidden layers. Defaults to ``nn.Tanh``.
    """
    def __init__(self, obs_dim, K, hidden=(256, 256), act=nn.Tanh):
        super().__init__()

        # Actor: logits for MultiBinary(K)
        self.actor = mlp([obs_dim, *hidden, K], act=act)

        # Critic: scalar V(s)
        self.critic = mlp([obs_dim, *hidden, 1], act=act)

        # Consistent orthogonal initialization for actor and critic
        init_actor_critic_nets(self.actor, self.critic, act_hidden_act=act)

    def forward(self, obs):
        """Compute policy logits and state value V(s).

        :param obs: Batch of high-level observations.
        :return: (logits, value)
        """
        logits = self.actor(obs)              # [B, K]
        value = self.critic(obs).squeeze(-1)  # [B]
        return logits, value


class LowPolicy(nn.Module):
    """Low-level (LOW) actor-critic network.

    :param obs_dim: Dimension of the low-level observation.
    :param hidden: Hidden layer widths for the MLP. Defaults to (256, 256).
    :param act: Activation class used between hidden layers. Defaults to ``nn.Tanh``.
    """
    def __init__(self, obs_dim, hidden=(256, 256), act=nn.Tanh):
        super().__init__()

        # Actor: raw (alpha_raw, beta_raw) parameters for a Beta(alpha, beta) action distribution over [0, 1]
        self.actor = mlp([obs_dim, *hidden, 2], act=act)

        # Critic: scalar V(s)
        self.critic = mlp([obs_dim, *hidden, 1], act=act)

        # Orthogonal init aligned with high-level setup
        init_actor_critic_nets(self.actor, self.critic, act_hidden_act=act)

    def forward(self, obs):
        """Compute LOW policy raw Beta parameters and associated state value V(s).

        :param obs: Batch of low-level observations.
        :return: (raw_params, value)
        """
        raw_params = self.actor(obs)          # [B, 2]
        value = self.critic(obs).squeeze(-1)  # [B]
        return raw_params, value

