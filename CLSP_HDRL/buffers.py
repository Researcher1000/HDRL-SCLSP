import numpy as np
import torch


# Utility: discount_cumsum
def discount_cumsum(x, discount):
    """Compute the discounted cumulative sum of a time sequence.

    :param x: Time sequence of values (e.g., rewards, deltas, or advantages).
    :param discount: Discount factor γ in (0, 1].
    :return: Discounted cumulative sums with the same shape as ``x``.
    """
    x = np.asarray(x, dtype=np.float32)
    y = np.empty_like(x)
    acc = 0.0

    # Traverse backwards and accumulate discounted future values
    for t in range(len(x) - 1, -1, -1):
        acc = x[t] + discount * acc
        y[t] = acc

    return y


# Utility: per-product window return for LOW substeps
def compute_low_window_returns(hb, su, chosen_list, gamma, H, scale=1.0, H_succ=0):
    """Monte-Carlo, per-product, finite-window return used to credit each LOW sub-step (see
    :class:`LowLevelBuffer`).

    :param hb: Holding + backorder cost per period and product, shape (T, K).
    :param su: Setup cost per period and product, shape (T, K).
    :param chosen_list: For each period, the sequence of visited products in visiting order (may be empty).
    :param gamma: Discount factor applied to the future window (gamma_low).
    :param H: Window length in periods (H >= 0).
    :param scale: Constant the returns are divided by.
    :param H_succ: Window (periods) of the successors' future-cost term (0 = disabled).
    :return: ``(G, n_valid)`` where ``G`` is a flat float64 array with one entry per sub-step of periods
        ``0..n_valid-1`` (period-major, then visit order) and ``n_valid = max(T - max(H, H_succ), 0)``.
    """
    hb = np.asarray(hb, dtype=np.float64)
    su = np.asarray(su, dtype=np.float64)
    T = hb.shape[0]
    H = int(H)
    H_succ = int(H_succ)
    n_valid = max(T - max(H, H_succ), 0)
    if n_valid == 0:
        return np.empty(0, dtype=np.float64), 0

    own = hb + su                                    # own cost of every product
    fut = np.zeros((n_valid, hb.shape[1]), dtype=np.float64)
    for s in range(1, H + 1):
        fut += (float(gamma) ** s) * own[s:s + n_valid]
    if H_succ > 0:
        fut_succ = np.zeros((n_valid, hb.shape[1]), dtype=np.float64)
        for s in range(1, H_succ + 1):
            fut_succ += (float(gamma) ** s) * own[s:s + n_valid]

    out = []
    for t in range(n_valid):
        idx = np.asarray(chosen_list[t], dtype=np.int64)
        if idx.size == 0:
            continue
        vals = hb[t, idx]
        suffix = np.cumsum(vals[::-1])[::-1]         # hb_i + sum of successors' hb
        tot = suffix + fut[t, idx]
        if H_succ > 0:
            sf = fut_succ[t, idx]
            tot = tot + (np.cumsum(sf[::-1])[::-1] - sf)   # successors' future costs
        out.append(-tot / float(scale))

    G = np.concatenate(out) if out else np.empty(0, dtype=np.float64)
    return G, n_valid


# HighLevelBuffer
class HighLevelBuffer:
    """PPO trajectory buffer for the HIGH-level agent.

    :param obs_dim: Dimension of the HIGH-level observation vector.
    :param act_dim: Dimension of the MultiBinary action (number of products).
    :param size: Maximum number of time steps stored in the buffer.
    :param gamma: Discount factor γ.
    :param lam: GAE parameter λ.
    :param device: Target device for returned tensors (e.g., "cpu" or "cuda").
    :param normalize_rewards: If ``True``, apply EMA-based reward normalization.
    :param normalize_advantages: If ``True``, normalize advantages when calling :meth:`get`.
    :param reward_norm_momentum: EMA momentum for reward mean/std updates. Typical values are in [0.9,
        0.99].
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95, device="cpu", normalize_rewards=False, 
                 normalize_advantages=True, reward_norm_momentum=0.99):
        # Data buffers
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.mask_buf = np.zeros((size, act_dim), dtype=np.float32)
        # Optional production order (Plackett-Luce sample): product indices, -1 padded.
        self.order_buf = np.full((size, act_dim), -1, dtype=np.int64)

        # Hyperparameters and internal state
        self.gamma = gamma
        self.lam = lam
        self.ptr = 0
        self.path_start_idx = 0
        self.max_size = size
        self.device = device

        # Normalization options
        self.normalize_rewards = normalize_rewards
        self.normalize_advantages = normalize_advantages
        self.reward_norm_momentum = reward_norm_momentum

        # Running reward statistics (EMA)
        self.r_mean = 0.0
        self.r_std = 1.0
        self._initialized_stats = False

    # Store one transition
    def store(self, obs, act, rew, val, logp, mask, order=None):
        """Store a single transition in the buffer.

        :param obs: Observation at the current time step.
        :param act: MultiBinary action taken at the current time step.
        :param rew: Immediate reward.
        :param val: Critic estimate V(s) for the current state.
        :param logp: Log-probability of the action under the behavior (old) policy.
        :param mask: Eligibility/feasibility mask associated with the action.
        :param order: Optional production order of the selected products (S-CLSP-SD, Plackett-Luce mode);
            stored padded with -1.
        """
        assert self.ptr < self.max_size, "Buffer is full; call get() before storing more."

        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.mask_buf[self.ptr] = mask
        self.order_buf[self.ptr] = -1
        if order is not None and len(order) > 0:
            self.order_buf[self.ptr, :len(order)] = np.asarray(order, dtype=np.int64)

        self.ptr += 1

    # Running reward statistics (EMA)
    def _update_running_reward_stats(self, batch_mean, batch_std):
        """Update running reward mean and standard deviation using an EMA."""
        batch_mean = float(batch_mean)
        batch_std = float(batch_std)
    
        # Prevent degenerate std
        batch_std = max(batch_std, 1e-8)
    
        if not self._initialized_stats:
            # Initialize from the first batch
            self.r_mean = np.float32(batch_mean)
            self.r_std = np.float32(batch_std)
            self._initialized_stats = True
            return
    
        m = self.reward_norm_momentum  

        self.r_mean = np.float32(m * self.r_mean + (1.0 - m) * batch_mean)
        self.r_std  = np.float32(m * self.r_std  + (1.0 - m) * batch_std)

        
    # Finish a trajectory segment: compute GAE and returns
    def finish_path(self, last_val=0.0):
        """Close the current trajectory segment and compute GAE(λ) advantages and returns.

        :param last_val: Bootstrap value estimate for the state following the final step of the segment.
        """
        ps = slice(self.path_start_idx, self.ptr)
    
        rews = self.rew_buf[ps].astype(np.float32, copy=False)
        vals = self.val_buf[ps].astype(np.float32, copy=False)
    
        # Optional reward normalization (EMA)
        if self.normalize_rewards and rews.size > 0:
            batch_mean = float(rews.mean())
            batch_std  = float(rews.std())
            self._update_running_reward_stats(batch_mean, batch_std)
    
            rews = (rews - self.r_mean) / (self.r_std + 1e-8)
            self.rew_buf[ps] = rews  # keep buffer consistent
    
        # Compute GAE(λ) advantages and discounted returns
        if rews.size > 0:
            next_vals = np.concatenate(
                [vals[1:], np.array([last_val], dtype=np.float32)]
            )
    
            deltas = rews + self.gamma * next_vals - vals
            adv = discount_cumsum(deltas, self.gamma * self.lam)
            ret = adv + vals
    
            self.adv_buf[ps] = adv.astype(np.float32)
            self.ret_buf[ps] = ret.astype(np.float32)
    
        self.path_start_idx = self.ptr

    
    # Retrieve training batch for PPO and reset buffer pointers
    def get(self):
        """Return the buffer contents as PyTorch tensors ready for PPO training, then reset internal
        pointers for the next rollout collection.

        :return: Dict with obs, act, ret, adv (normalized if enabled), logp, val, mask.
        """
        assert self.ptr == self.max_size, "Buffer is not full yet."

        adv = self.adv_buf
        if self.normalize_advantages and adv.size > 0:
            std = adv.std()
            if std > 1e-8:
                adv_norm = (adv - adv.mean()) / (std + 1e-8)
            else:
                adv_norm = adv - adv.mean()
        else:
            adv_norm = adv

        data = dict(
            obs=torch.as_tensor(self.obs_buf, device=self.device),
            act=torch.as_tensor(self.act_buf, device=self.device),
            ret=torch.as_tensor(self.ret_buf, device=self.device),
            adv=torch.as_tensor(adv_norm, device=self.device),
            logp=torch.as_tensor(self.logp_buf, device=self.device),
            val=torch.as_tensor(self.val_buf, device=self.device),
            mask=torch.as_tensor(self.mask_buf, device=self.device),
            order=torch.as_tensor(self.order_buf, device=self.device),
        )

        self.ptr = 0
        self.path_start_idx = 0

        return data


# LowLevelBuffer
class LowLevelBuffer:
    """Trajectory buffer for the LOW-level agent in the hierarchical PPO scheme.

    :param obs_dim: Dimension of the LOW-level observation vector.
    :param device: Device where returned tensors will be allocated ("cpu" or "cuda").
    :param gamma: Discount factor γ of the window return.
    :param normalize_advantages: If ``True``, normalize advantages when building the batch.
    :param window_H: Window length H (periods) of the LOW return.
    :param reward_scale: Constant dividing the LOW returns.
    :param succ_H: Window (periods) of the successors' future-cost term of the LOW return (0 = disabled),
        see :func:`compute_low_window_returns`.
    """

    def __init__(self, obs_dim, device="cpu", gamma=0.99, normalize_advantages=True,
                 window_H=30, reward_scale=100.0, succ_H=10):
        self.device = device
        self.gamma = gamma
        self.window_H = int(window_H)
        self.reward_scale = float(reward_scale)
        self.succ_H = int(succ_H)

        # Current episode (linear sequence of LOW sub-steps)
        self._obs = []
        self._act = []
        self._val = []
        self._logp = []
        # Per-period cost vectors and visit orders
        self._hb = []
        self._su = []                # setup cost per product, one (K,) array per period
        self._chosen = []            # visit order (list of product ids) per period
        self._period_ranges = []     # list of (start_idx, end_idx) for each period in the episode
        self._period_start = 0
        self.periods_in_batch = 0

        # Global batch accumulator (may contain multiple episodes)
        self._batch = dict(obs=[], act=[], val=[], logp=[], adv=[], ret=[])

        self.normalize_advantages = normalize_advantages

    # Period markers
    def start_period(self):
        """Mark the start of a new period within the current episode."""
        self._period_start = len(self._obs)

    def store_substep(self, obs, act, val, logp):
        """Store a LOW-level sub-step within the current period.

        :param obs: LOW observation at the sub-step.
        :param act: Continuous action in [0, 1] (fraction of the feasible capacity cap for the visited
            product).
        :param val: Critic estimate V(s) for the LOW-level state.
        :param logp: Log-probability (density) of the action under the old LOW policy.
        """
        self._obs.append(np.asarray(obs, dtype=np.float32))
        self._act.append(float(act))
        self._val.append(np.float32(val))
        self._logp.append(np.float32(logp))

    def end_period_assign_costs(self, hb_vec, su_vec, chosen):
        """Close the current period by recording its per-product cost vectors.

        :param hb_vec: Holding + backorder cost of every product this period (after the period's demand),
            shape (K,).
        :param su_vec: Setup cost of every product this period, shape (K,).
        :param chosen: Products visited this period, in the SAME order the substeps were stored via
            :meth:`store_substep`.
        """
        start = self._period_start
        end = len(self._obs)
        T = end - start

        chosen = [int(i) for i in chosen]
        assert len(chosen) == T, (
            f"end_period_assign_costs: expected {T} visited products "
            f"(one per substep stored this period), got {len(chosen)}."
        )

        self._hb.append(np.asarray(hb_vec, dtype=np.float64))
        self._su.append(np.asarray(su_vec, dtype=np.float64))
        self._chosen.append(chosen)

        self._period_ranges.append((start, end))
        self.periods_in_batch += 1

    # Close episode and accumulate into the global batch
    def finish_episode_and_collect(self):
        """Close the current episode: compute the per-product window return of every substep with a complete
        window, append it to the global batch and reset the episode buffers.
        """
        n_periods = len(self._hb)
        if len(self._obs) == 0 or n_periods == 0:
            self._reset_episode_buffers()
            return

        hb = np.stack(self._hb, axis=0)
        su = np.stack(self._su, axis=0)
        G, n_valid = compute_low_window_returns(
            hb, su, self._chosen, gamma=self.gamma, H=self.window_H, scale=self.reward_scale,
            H_succ=self.succ_H,
        )

        if n_valid > 0:
            n_keep = self._period_ranges[n_valid - 1][1]
            assert G.size == n_keep, (
                f"LowLevelBuffer: {G.size} returns but {n_keep} substeps kept."
            )
            if n_keep > 0:
                val = np.asarray(self._val[:n_keep], dtype=np.float64)
                G32 = G.astype(np.float32)
                adv = (G - val).astype(np.float32)

                self._batch["obs"].extend(self._obs[:n_keep])
                self._batch["act"].extend(self._act[:n_keep])
                self._batch["val"].extend(self._val[:n_keep])
                self._batch["logp"].extend(self._logp[:n_keep])
                self._batch["adv"].extend(adv.tolist())
                self._batch["ret"].extend(G32.tolist())

        self._reset_episode_buffers()

    def _reset_episode_buffers(self):
        """Clear internal buffers associated with the current episode."""
        self._obs, self._act, self._val, self._logp = [], [], [], []
        self._hb, self._su, self._chosen = [], [], []
        self._period_ranges = []
        self._period_start = 0

    # Batch readiness
    def is_batch_ready(self, target_periods):
        """Return whether the accumulated number of periods has reached the target.

        :param target_periods: Minimum number of periods required for the batch to be considered ready.
        :return: ``True`` if enough periods have been accumulated; otherwise ``False``.
        """
        return self.periods_in_batch >= int(target_periods)

    # Build tensors and reset accumulator
    def get(self):
        """Return the accumulated batch as PyTorch tensors, then reset the accumulator.

        :return: Dictionary with keys: ``obs``, ``act``, ``adv``, ``ret``, ``logp``, ``val``.
        """

        # Empty case
        if len(self._batch["obs"]) == 0:
            empty = torch.empty(0, device=self.device)
            self._batch = dict(obs=[], act=[], val=[], logp=[], adv=[], ret=[])
            self.periods_in_batch = 0
            return dict(obs=empty, act=empty, adv=empty, ret=empty,
                        logp=empty, val=empty)

        # Convert to tensors-
        obs  = torch.tensor(np.array(self._batch["obs"],  dtype=np.float32), device=self.device)
        act  = torch.tensor(np.array(self._batch["act"],  dtype=np.float32), device=self.device)
        val  = torch.tensor(np.array(self._batch["val"],  dtype=np.float32), device=self.device)
        logp = torch.tensor(np.array(self._batch["logp"], dtype=np.float32), device=self.device)
        adv  = torch.tensor(np.array(self._batch["adv"],  dtype=np.float32), device=self.device)
        ret  = torch.tensor(np.array(self._batch["ret"],  dtype=np.float32), device=self.device)

        # Optional advantage normalization
        if self.normalize_advantages and adv.numel() > 0:
            std = adv.std()
            if std > 1e-8:
                adv = (adv - adv.mean()) / (std + 1e-8)
            else:
                adv = adv - adv.mean()

        data = dict(obs=obs, act=act, adv=adv, ret=ret,
                    logp=logp, val=val)

        # Reset accumulator
        self._batch = dict(obs=[], act=[], val=[], logp=[], adv=[], ret=[])
        self.periods_in_batch = 0

        return data
