import numpy as np
import gymnasium as gym
from gymnasium import spaces

class CLSPHierEnv(gym.Env):
    """Hierarchical RL environment for the stochastic capacitated lot-sizing problem (S-CLSP).

    HIGH selects which products are eligible in the period; LOW then decides, one product at a time,
    how much of each selected product to produce.

    HIGH: action MultiBinary(K); observation (2K) = [coverage_norm(K), omega(K)], omega = active setup in
    {-1, +1}.
    LOW:  action in [0, 1] = fraction of the currently feasible capacity q_cap (after setup time, future
          reservation and EOQ cap), so every action is feasible; q = floor(action * q_cap), at least 1 unit
          if force_min_batch and q_cap >= 1. Batch size is 1. Observation (4K + 4) = [onehot, cap_norm,
          a_high_norm, a_high_bin, need_setup, coverage_norm, remaining_seq_norm, n_chosen_norm]; the
          K-vectors are indexed by decision rank (most confident product first), not by product index.
    Reward: r_t = -cost_t (holding + backorder + setup).

    :param mu: Mean demand per product. Scalar or array-like of length K.
    :param bs: Batch size (units per batch) for each product.
    :param h: Per-unit holding cost for each product.
    :param b: Per-unit backorder cost for each product.
    :param k: Setup cost for each product.
    :param st: Setup time for each product (expressed in capacity units / batches).
    :param CAP: Total capacity per period (in number of batches).
    :param cov: Demand coefficient(s) of variation per product.
    :param cov_tau: Coverage bound used to clip ``I / mu`` and rescale it to [-1, 1].
    :param reserve_future_min: If ``True``, the LOW-level reserves a minimum amount of capacity for products
        that have not yet been visited within the current period.
    :param force_min_batch: If ``True``, the LOW-level forces production of at least one batch when capacity
        is available.
    :param enable_Ilim: If ``True``, inventory is clipped to ``[-0.5 · I_lim, I_lim]`` where ``I_lim = 15 ·
        mu``.
    :param enforce_eoq_qmax: If ``True``, caps the per-product production quantity using an EOQ-based
        reference limit.
    :param eoq_slack_batches: Additional slack (in batches) added to the EOQ-based cap.
    :param enable_ip_eligibility: Enables a HIGH-level eligibility mask based on inventory and setup state.
    :param init_inventory: Optional deterministic initial inventory ``I_{i,0}`` for each product, used by
        :meth:`reset`.
    :param init_active_idx: Optional deterministic index of the product that starts the episode already set
        up on the machine (``omega``).
    :param horizon: Episode length in periods. If ``None``, the horizon is unbounded.
    :param seed: Base seed for the internal RNG.
    :param normalize_rewards_flag: Hint for the trainer; the environment does not normalize rewards
        internally.
    :param normalize_advantages_flag: Hint for the trainer; the environment does not normalize advantages
        internally.
    """

    metadata = {"render_modes": []}
        
    def __init__(
        self,
        # Problem parameters
        K, mu, bs, h, b, k, st, CAP,
        # Demand / normalization
        cov=None, cov_tau=5.0,
        # Demand correlation for new cases (optional)
        demand_corr_mode=None,
        demand_corr_pairs=None,
        demand_corr_groups=None,
        # LOW-level behavior
        reserve_future_min=True, force_min_batch=True,
        # Limits / toggles
        enable_Ilim=True,
        enforce_eoq_qmax=True,
        eoq_slack_batches=0,
        enable_ip_eligibility=True,
        # Deterministic initial state (optional)
        init_inventory=None,
        init_active_idx=None,
        # Episode
        horizon=None,
        # Seed and trainer flags
        seed=None,
        normalize_rewards_flag=True,
        normalize_advantages_flag=True,
    ):
        super().__init__()

        # 1) Base parameters
        self.K = K
        self.CAP = CAP

        # Vectorize with appropriate dtypes
        self.mu = np.asarray(mu, dtype=np.float32)
        self.bs = np.asarray(bs, dtype=np.int32)
        self.h  = np.asarray(h, dtype=np.float32)
        self.b  = np.asarray(b, dtype=np.float32)
        self.k  = np.asarray(k, dtype=np.float32)
        self.st = np.asarray(st, dtype=np.int32)

        # 2) Demand / normalization
        self.cov = np.asarray(cov, dtype=np.float32)
        self.cov_tau = cov_tau

        # Optional demand correlation: demand_corr_mode (None | "pair_plus_one"), demand_corr_pairs ((i, j) pairs).
        self.demand_corr_mode = demand_corr_mode
        self.demand_corr_pairs = demand_corr_pairs  
        self.demand_corr_groups = demand_corr_groups

        # 4) RNG and horizon
        self._seed = seed
        self.rng = np.random.default_rng(seed)
        self.horizon = horizon

        # 5) Internal state
        self.last_a_high = np.zeros(self.K, dtype=np.int8)

        # 6) Inventory limits
        self.enable_Ilim = enable_Ilim
        self.I_lim = 15.0 * self.mu
        
        # 7) LOW-level flags
        self.reserve_future_min = reserve_future_min
        self.force_min_batch_default = force_min_batch

        # 8) Optional EOQ-based q_max cap
        self.enforce_eoq_qmax = enforce_eoq_qmax
        self.eoq_slack_batches = eoq_slack_batches
        self._Q_eoq = None
        self._qmax_eoq = None
        
        if self.enforce_eoq_qmax:
            # Classical EOQ: Q_eoq = sqrt(2 * mu * k / h
            self._Q_eoq = np.sqrt(2.0 * self.mu * self.k / self.h)
        
            # Cap in batches: floor(Q_eoq / bs) + slack
            self._qmax_eoq = np.floor(self._Q_eoq / self.bs).astype(int) + self.eoq_slack_batches

        # 10) HIGH-level eligibility
        self.enable_ip_eligibility = enable_ip_eligibility

        # 10b) Deterministic initial state (optional)
        self.init_inventory = (
            None if init_inventory is None else np.asarray(init_inventory, dtype=np.int32)
        )
        self.init_active_idx = init_active_idx

        # 11) Trainer hints
        self.normalize_rewards_flag = normalize_rewards_flag
        self.normalize_advantages_flag = normalize_advantages_flag

        # 12) Gymnasium spaces
        self.observation_space = spaces.Box(
            low=-np.ones(2 * self.K, dtype=np.float32),
            high=np.ones(2 * self.K, dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.MultiBinary(self.K)
        self._low_action_n = self.CAP + 1
        
        # 13) Demand model setup and initial state
        self._setup_demand_params()
        self.reset()


    
    # Demand model: discrete uniform
    def _setup_demand_params(self):
        """Define the discrete uniform demand ranges for each product."""
        cov = self.cov.astype(np.float32, copy=False)
        mu  = self.mu.astype(np.float32, copy=False)
    
        K = self.K
        a = np.empty(K, dtype=np.int32)
        b = np.empty(K, dtype=np.int32)
    
        for i in range(K):
            if np.isclose(cov[i], 0.58, atol=1e-3):
                a[i] = 0
                b[i] = int(round(2 * mu[i]))
            else:
                # Low variability (e.g., cov ≈ 0.14)
                if np.isclose(mu[i], 4.0, atol=1e-3):
                    a[i], b[i] = 3, 5
                else:
                    a[i] = b[i] = int(round(mu[i]))
    
        self._unif_a = a
        self._unif_b = b


    
    
    # Demand sampling per period
    def _sample_demand(self):
        """Generate an integer-valued demand realization for the current period."""
        d = self.rng.integers(
            low=self._unif_a,
            high=self._unif_b + 1,  # numpy integers 'high' is exclusive
            size=self.K,
            dtype=np.int32,
        )
    
        mode = self.demand_corr_mode
        
        if mode == "pair_plus_one" and self.demand_corr_pairs is not None:
            for (i, j) in self.demand_corr_pairs:
                base = self.rng.integers(self._unif_a[i], self._unif_b[i] + 1, dtype=np.int32)
                d[i] = base
                d[j] = base + 1
                d[j] = np.clip(d[j], self._unif_a[j], self._unif_b[j])
        
        elif mode == "pair_same" and self.demand_corr_pairs is not None:
            for (i, j) in self.demand_corr_pairs:
                base = self.rng.integers(self._unif_a[i], self._unif_b[i] + 1, dtype=np.int32)
                d[i] = base
                d[j] = base
                d[j] = np.clip(d[j], self._unif_a[j], self._unif_b[j])  # safety
                d[i] = np.clip(d[i], self._unif_a[i], self._unif_b[i])
        
        elif mode == "group_same" and self.demand_corr_groups is not None:
            for grp in self.demand_corr_groups:
                grp = list(grp)
                if len(grp) == 0:
                    continue
                i0 = grp[0]
                base = self.rng.integers(self._unif_a[i0], self._unif_b[i0] + 1, dtype=np.int32)
                for j in grp:
                    d[j] = base
                    d[j] = np.clip(d[j], self._unif_a[j], self._unif_b[j])

    
        return d

        
    # HIGH-level observations
    def _coverage_norm_vec(self, I_view=None):
        """Compute the normalized inventory coverage vector."""
        I = self.I if I_view is None else np.asarray(I_view, dtype=np.float32)
    
        coverage = I.astype(np.float32) / self.mu
        cov_tau = np.float32(self.cov_tau)
    
        coverage = np.clip(coverage, -cov_tau, cov_tau) / cov_tau
    
        return coverage.astype(np.float32)

    
    
    def _obs_high(self):
        """Construct the observation vector for the HIGH-level agent."""
        cov_norm = self._coverage_norm_vec()
        omega_f = (2 * self.omega - 1).astype(np.float32)
    
        obs_high = np.concatenate((cov_norm, omega_f)).astype(np.float32)
        return obs_high


    # LOW state and mask
    def _low_state_mask_for(self, i, CAP_rem, a_high_scores, pos=None, chosen=None,
                            I_view=None, a_high_vec=None):

        """Build the LOW-level state vector and the feasible capacity cap for one product."""

        # Setup requirement and flags
        st_i = self.st[i]
        need_setup = (self.omega[i] == 0)

        # Optional future capacity reservation (if applicable)
        reserve_future = self._future_min_capacity(chosen, pos) if (chosen is not None and pos is not None) else 0
        cap_for_current = max(0, CAP_rem - reserve_future)

        # Natural q_max given remaining capacity and action-space definition
        qmax_natural = min(self._low_action_n - 1, cap_for_current)

        # Capacity-feasible q_max accounting for setup time (if needed)
        if need_setup:
            qmax_res = max(0, min(qmax_natural, cap_for_current - st_i))
        else:
            qmax_res = max(0, min(qmax_natural, cap_for_current))

        # Optional EOQ-based cap
        if self.enforce_eoq_qmax and (self._qmax_eoq is not None):
            qmax_res = min(qmax_res, max(0, int(self._qmax_eoq[i])))


        # Feasible production cap for this decision (int, in units).
        q_cap = int(max(0, qmax_res))

        # LOW state HIGH scores normalized to [0, 1] (already in [0, 1])
        scores = np.clip(np.asarray(a_high_scores, dtype=np.float32), 0.0, 1.0)

        # Binary HIGH selection vector (kept in [0, 1])
        if a_high_vec is None:
            a_high_vec_full = (scores > 0.5).astype(np.float32)
        else:
            a_high_vec_full = np.clip(np.asarray(a_high_vec, dtype=np.float32), 0.0, 1.0)

        # Full rank permutation for this period: `chosen` (LOW's visiting order, i.e.
        if chosen is not None:
            chosen_arr = np.asarray(chosen, dtype=np.int32).reshape(-1)
        else:
            chosen_arr = np.zeros(0, dtype=np.int32)
        not_chosen = np.setdiff1d(np.arange(self.K, dtype=np.int32), chosen_arr, assume_unique=True)
        if not_chosen.size > 0:
            nc_order = np.argsort(scores[not_chosen])[::-1]
            not_chosen_sorted = not_chosen[nc_order]
        else:
            not_chosen_sorted = not_chosen
        visit_order_full = np.concatenate((chosen_arr, not_chosen_sorted)).astype(np.int32)

        # One-hot encoding of THIS SUBSTEP'S RANK (``pos``) within `visit_order_full`
        onehot = np.zeros(self.K, dtype=np.float32)
        if pos is not None:
            onehot[int(pos)] = 1.0
        onehot_norm = onehot

        # Feasible capacity (q_cap, post-setup, post-EOQ) normalized by the total period capacity, in [0, 1].
        cap_norm = np.array([q_cap / max(1, self.CAP)], dtype=np.float32)

        # HIGH scores and binary selection, reindexed by rank (see docstring).
        a_high_norm = scores[visit_order_full].astype(np.float32)
        a_high_bin = a_high_vec_full[visit_order_full].astype(np.float32)

        # Setup requirement indicator in {0, 1}
        need_setup_norm = np.array([1.0 if need_setup else 0.0], dtype=np.float32)

        # Fraction of the selected products still pending in the LOW order.
        if chosen is not None and pos is not None:
            total_chosen = max(1, int(np.asarray(chosen).size))
            remaining_after = max(0, total_chosen - pos - 1)
            remaining_norm_val = remaining_after / total_chosen
            n_chosen_norm_val = total_chosen / max(1, self.K)
        else:
            remaining_norm_val = 0.0
            n_chosen_norm_val = 0.0
        remaining_norm = np.array([remaining_norm_val], dtype=np.float32)

        # How many products HIGH selected this period, normalized by K.
        n_chosen_norm = np.array([n_chosen_norm_val], dtype=np.float32)

        # Normalized inventory coverage (kept in [-1, 1]), reindexed by rank (see docstring).
        cov_norm_vec = self._coverage_norm_vec(I_view)[visit_order_full]

        # Final LOW state vector
        low_s = np.concatenate(
            (onehot_norm, cap_norm, a_high_norm, a_high_bin, need_setup_norm, cov_norm_vec,
             remaining_norm, n_chosen_norm),
            dtype=np.float32
        )

        return low_s, q_cap, bool(need_setup)


    # LOW continuous action -> production quantity
    def low_action_to_qty(self, action_frac, q_cap, force_min_batch=None):
        """Convert the LOW-level continuous action into an integer production quantity.

        :param action_frac: Raw action from the LOW policy, expected in [0, 1]. Values outside this range
            are clipped.
        :param q_cap: Feasible production cap (int, units) for the current product decision, as returned by
            `_low_state_mask_for`.
        :param force_min_batch: If True, forces at least 1 unit whenever ``q_cap >= 1``. If None, uses
            ``self.force_min_batch_default``.
        :return: Integer production quantity q in [0, q_cap].
        """
        fmb = self.force_min_batch_default if force_min_batch is None else force_min_batch
        a = float(np.clip(action_frac, 0.0, 1.0))
        q_cap = int(max(0, q_cap))

        q = int(np.floor(a * q_cap))
        if fmb and q_cap >= 1:
            q = max(1, q)

        return min(q, q_cap)


    # LOW-level visiting order (prob / paper)
    def plan_quantity_assignment_order(self, a_high_exec, a_high_scores):
        """Define the visit order for LOW-level quantity decisions within a period.

        :param a_high_exec: Binary vector indicating which products are activated by the HIGH-level policy
            (1 = selected, 0 = not selected).
        :param a_high_scores: Continuous scores or probabilities produced by the HIGH-level policy for each
            product.
        :return: Array of product indices in the order in which the LOW-level policy should assign
            production quantities.
        """
        a = np.asarray(a_high_exec, dtype=np.int8)
        chosen = np.nonzero(a)[0]
        if chosen.size == 0:
            return chosen.astype(np.int32)
    
        scores = np.clip(np.asarray(a_high_scores, dtype=np.float32), 0.0, 1.0)
        order = np.argsort(scores[chosen])[::-1]
        return chosen[order].astype(np.int32)

        
    def plan_production_order_paper(self, produced_mask, a_high_scores, I_view_final=None):
        """Define the production sequence used exclusively to update the setup carryover variable (omega) at
        the end of a period.

        :param produced_mask: Binary vector indicating which products were produced in the current period (1
            = produced, 0 = not produced).
        :param a_high_scores: Continuous scores or probabilities produced by the HIGH-level policy for each
            product.
        :param I_view_final: Optional pseudo inventory vector representing the within-period inventory state
            after LOW-level decisions.
        :return: Array of product indices in the order used to update the setup carryover variable omega,
            with guaranteed uniqueness.
        """
        produced_mask = np.asarray(produced_mask, dtype=np.int8)
        produced = np.nonzero(produced_mask)[0]
        if produced.size == 0:
            return produced.astype(np.int32)
    
        scores = np.clip(np.asarray(a_high_scores, dtype=np.float32), 0.0, 1.0)
    
        # HEAD = current setup item if produced
        omega_prod = self.omega[produced]
        head_idx = np.where(omega_prod == 1)[0]
        head = produced[head_idx[0]] if head_idx.size > 0 else None
    
        # Inventory reference for tail selection (pseudo inventory final)
        I_ref = self.I.astype(np.float32) if I_view_final is None else np.asarray(I_view_final, dtype=np.float32)
        cov_ref = I_ref / self.mu  # coverage prox
    
        # TAIL selection with constraint: head cannot be tail if produced.size > 1
        if head is not None and produced.size > 1:
            tail_candidates = produced[produced != head]
            tail = tail_candidates[np.argmin(cov_ref[tail_candidates])]
        else:
            tail = produced[np.argmin(cov_ref[produced])]
    
        # MID = remaining items excluding head and tail, sorted by prob desc
        mask_mid = (produced != tail)
        if head is not None:
            mask_mid &= (produced != head)
    
        mid = produced[mask_mid]
        if mid.size > 0:
            mid = mid[np.argsort(scores[mid])[::-1]]
    
        # Compose order: HEAD -> MID -> TAIL
        seq = []
        if head is not None:
            seq.append(int(head))
        seq.extend([int(x) for x in mid])
        seq.append(int(tail))
    
        # Uniq safety
        uniq = []
        seen = set()
        for x in seq:
            if x not in seen:
                seen.add(x)
                uniq.append(x)
    
        return np.asarray(uniq, dtype=np.int32)


    
    # Minimum future capacity reservation (LOW)
    def _future_min_capacity(self, chosen, pos):
        """Compute the minimum capacity that should be reserved for yet-unvisited products in the current
        period, following the logic described in the paper.
        """
        
        if not self.reserve_future_min:
            return 0
    
        # Convert chosen to an array (assumed valid when called)
        chosen = np.asarray(chosen, dtype=int)
    
        # If there are no future products, nothing needs to be reserved
        if pos >= chosen.size - 1:
            return 0
    
        # Products that remain to be visited
        fut_idxs = chosen[pos + 1:]
    
        # Reserve at least 1 batch per future product
        base = fut_idxs.size
    
        # Add setup time only for products without an active setup
        needs_setup = (self.omega[fut_idxs] == 0)
        setup_cost = self.st[fut_idxs][needs_setup].sum() if needs_setup.any() else 0
    
        return int(base + setup_cost)


    
    # HIGH-level eligibility mask
    def eligibility_mask_high(self):
        """Generate a binary mask indicating which products are eligible to be selected by the HIGH-level
        agent.

        :return: Binary vector of shape (K,), where: 1 = eligible product, 0 = blocked (ineligible) product.
        """
        if not self.enable_ip_eligibility:
            return np.ones(self.K, dtype=np.int32)
    
        # Ineligible: no active setup (omega == 0) and inventory above 5 * mu
        ineligible = (self.omega == 0) & (self.I > (5.0 * self.mu))
    
        # 1 = eligible, 0 = ineligible
        return (~ineligible).astype(np.int32)

    
    # RESET
    def reset(self, seed=None, options=None):
        """Reset the CLSP environment at the beginning of a new episode.

        :param seed: Optional seed for the random number generator.
        :param options: Standard Gymnasium argument (not used).
        :return: (obs, info): initial HIGH observation (float32) and an empty dict.
        """

        # If no seed is provided, use the stored one
        if seed is None:
            seed = self._seed

        # Gymnasium seeding only accepts int/None; Generator/SeedSequence seeds go directly to self.rng.
        if isinstance(seed, (np.random.Generator, np.random.SeedSequence)):
            super().reset(seed=None)
            self.rng = np.random.default_rng(seed)
        else:
            super().reset(seed=seed)
            self.rng = np.random.default_rng(seed)

        # Initial state
        if self.init_inventory is not None:
            self.I = np.array(self.init_inventory, dtype=np.int32, copy=True)
        else:
            self.I = np.zeros(self.K, dtype=np.int32)
        self.omega = np.zeros(self.K, dtype=np.int8)
        self.t = 0

        # Initial active setup: deterministic if requested, otherwise reproducibly drawn from self.rng
        if self.init_active_idx is not None:
            active_idx = int(self.init_active_idx)
        else:
            active_idx = self.rng.integers(0, self.K)
        self.omega[active_idx] = 1

        # Initial observation
        obs_init = self._obs_high()
        return obs_init, {}

        
    @property
    def low_obs_dim(self):
        """Dimension of the LOW-level observation vector.

        :return: Size of the observation vector used by the LOW-level agent.
        """
        return 4 * self.K + 4


    # STEP
    def step(self, a_high, low_actions=None, a_high_scores=None):
        """Execute one full period of the hierarchical CLSP environment.

        :param a_high: Binary vector {0,1}^K indicating which products are selected by HIGH.
        :param low_actions: LOW actions, fractions in [0, 1] of the feasible capacity q_cap of each visited product
            (see `low_action_to_qty`); every value is feasible by construction.
        :param a_high_scores: HIGH scores/probabilities per product (LOW observation features); if None, ``a_high`` is used.
        :return: (obs_next, reward, terminated, truncated, info); info holds the period metrics (costs, LOW sequence, ...).
        """
        
        # 1) HIGH action and validation
        a_high = np.asarray(a_high, dtype=np.int8)
        self.last_a_high = a_high.copy()
    
        # Scores for LOW features / ordering
        if a_high_scores is None:
            scores_vec = a_high.astype(np.float32)
        else:
            scores_vec = np.clip(np.asarray(a_high_scores, dtype=np.float32), 0.0, 1.0)
    
        # 2) No-production case
        active = np.nonzero(a_high)[0]
    
        if active.size == 0:
            # Demand only → inventory decreases
            d = self._sample_demand()
            self.I = self.I - d
    
            # Inventory clipping (if enabled)
            if self.enable_Ilim:
                hi = self.I_lim.astype(np.int32)
                lo = (-0.5 * self.I_lim).astype(np.int32)
                self.I = np.clip(self.I, lo, hi)
    
            # Costs
            holding = self.h * np.clip(self.I, 0, None)
            backord = self.b * np.clip(-self.I, 0, None)
            z = np.zeros(self.K, dtype=np.int8)
    
            cost_t = float(holding.sum() + backord.sum())
            reward = -cost_t

            # No product visited: all-zero setup vector.
            cost_per_product = (holding + backord).astype(np.float32)

            # Time advance
            self.t += 1
            terminated = (self.horizon is not None) and (self.t >= self.horizon)

            return (
                self._obs_high(),
                reward,
                terminated,
                False,
                {
                    "cost": cost_t,
                    "low_seq_used": [],
                    "q": np.zeros(self.K, dtype=np.int32),
                    "z": z,
                    "holding": holding.astype(np.float32),
                    "backord": backord.astype(np.float32),
                    "setup_cost": (self.k * z).astype(np.float32),
                    "cost_per_product": cost_per_product,
                    "chosen_order_used": np.array([], dtype=int),
                    "a_high_scores_used": scores_vec.copy(),
                    "normalize_rewards_flag": self.normalize_rewards_flag,
                    "normalize_advantages_flag": self.normalize_advantages_flag,
                },
            )

        # 3) Quantity-assignment order: highest prob decides first
        chosen = self.plan_quantity_assignment_order(a_high, scores_vec)
        
        # 4) Per-period state
        CAP_rem = self.CAP
        z = np.zeros(self.K, dtype=np.int8)      # setups performed
        q = np.zeros(self.K, dtype=np.int32)     # batches produced
        seq_used = []
        
        # Normalize low_actions to match the length of chosen
        if low_actions is None:
            low_actions = [0.0] * chosen.size
        else:
            low_actions = list(low_actions)
            if len(low_actions) < chosen.size:
                low_actions.extend([0] * (chosen.size - len(low_actions)))
            low_actions = low_actions[:chosen.size]
        
        I_view = self.I.copy()
        produced_count = 0

        # 5) LOW loop: decide quantities per product
        for pos, i in enumerate(chosen):
            low_s, q_cap, need_setup = self._low_state_mask_for(
                i,
                CAP_rem,
                a_high_scores=scores_vec,
                pos=pos,
                chosen=chosen,
                I_view=I_view,
                a_high_vec=a_high,  # current binary HIGH action
            )

            # low_actions[pos] is a continuous fraction in [0, 1] of q_cap itself
            a_frac = float(low_actions[pos])
            q_i = self.low_action_to_qty(a_frac, q_cap)

            # Update capacity and setups
            if q_i > 0 and need_setup:
                z[i] = 1
                CAP_rem -= self.st[i]

            q[i] = q_i
            CAP_rem = max(0, CAP_rem - q_i)

            # Within-period "visible" inventory update
            if q_i > 0:
                I_view[i] = I_view[i] + q_i * self.bs[i]
                produced_count += 1

            seq_used.append({
                "idx": int(i),
                "q_cap": int(q_cap),
                "action_frac": a_frac,
                "chosen_q": int(q_i),
                "CAP_rem_after": int(CAP_rem),
            })


        # 6) Apply production and demand
        if np.any(q):
            self.I = self.I + q * self.bs
        
        d = self._sample_demand()
        self.I = self.I - d
        
        if self.enable_Ilim:
            hi = self.I_lim.astype(np.int32)
            lo = (-0.5 * self.I_lim).astype(np.int32)
            self.I = np.clip(self.I, lo, hi)
        
        # 7) Costs and reward
        I_pos = np.clip(self.I, 0, None)
        I_neg = np.clip(-self.I, 0, None)
        
        holding = (self.h * I_pos).astype(np.float32)
        backord = (self.b * I_neg).astype(np.float32)
        setup_cost = (self.k * z).astype(np.float32)

        cost_t = float(holding.sum() + backord.sum() + setup_cost.sum())
        reward = -cost_t

        # Per-product cost (holding + backorder + setup), used for the LOW returns.
        cost_per_product = (holding + backord + setup_cost).astype(np.float32)

        # 8) Update setup carryover
        produced_mask = (q > 0).astype(np.int8)
        produced = np.nonzero(produced_mask)[0]
        if produced.size > 0:
            # Uses pseudo-inventory final (I_view already includes within-period production)
            prod_order = self.plan_production_order_paper(
                produced_mask=produced_mask,
                a_high_scores=scores_vec,
                I_view_final=I_view
            )
            last = int(prod_order[-1]) if prod_order.size > 0 else int(produced[-1])
            self.omega[:] = 0
            self.omega[last] = 1

        
        # 9) Time and termination condition
        self.t += 1
        terminated = (self.horizon is not None) and (self.t >= self.horizon)
        
        # 10) Output
        obs_next = self._obs_high().astype(np.float32)
        info = dict(
            cost=cost_t,
            low_seq_used=seq_used,
            q=q,
            z=z,
            holding=holding,
            backord=backord,
            setup_cost=setup_cost,
            cost_per_product=cost_per_product,
            chosen_qty_order=chosen.copy(),
            a_high_scores_used=scores_vec.copy(),
            normalize_rewards_flag=self.normalize_rewards_flag,
            normalize_advantages_flag=self.normalize_advantages_flag,
        )
        
        return obs_next, reward, terminated, False, info


def make_env_from_hyp(hyp, seed=None):
    """Create a CLSPHierEnv instance from a hyperparameter dictionary.

    :param hyp: Dictionary containing all environment and algorithm hyperparameters.
    :param seed: Optional random seed for the environment.
    :return: Instantiated CLSPHierEnv environment.
    """
    K  = hyp["K"]
    cf = hyp["capacity_factor"]
    mu, bs, h, b, k, st = hyp["mu"], hyp["bs"], hyp["h"], hyp["b"], hyp["k"], hyp["st"]
    CAP = int(np.ceil(round(float(cf * (mu.sum() / bs[0])), 6)))

    env = CLSPHierEnv(
        K, mu, bs, h, b, k, st, CAP,
        horizon=hyp["horizon"],
        cov=hyp["cov"],
        reserve_future_min=hyp.get("reserve_future_min", True),
        force_min_batch=hyp.get("force_min_batch", True),
        enable_Ilim=hyp.get("enable_Ilim", True),
        enforce_eoq_qmax=hyp.get("enforce_eoq_qmax", True),
        enable_ip_eligibility=hyp.get("enable_ip_eligibility", True),
        demand_corr_mode=hyp.get("demand_corr_mode", None),
        demand_corr_pairs=hyp.get("demand_corr_pairs", None),
        demand_corr_groups=hyp.get("demand_corr_groups", None),
        init_inventory=hyp.get("init_inventory", None),
        init_active_idx=hyp.get("init_active_idx", None),
        seed=seed,
    )
    return env

