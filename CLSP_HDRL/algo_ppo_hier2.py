import os
import random
from tqdm import trange

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
import matplotlib.pyplot as plt

from .envs2 import CLSPHierEnv2
from .nets import HighPolicy, LowPolicy
from .dists import MaskedBernoulli, BetaDist, plackett_luce_log_prob, plackett_luce_sample
from .buffers import HighLevelBuffer, LowLevelBuffer


class PPO_Hier2:
    """Joint hierarchical PPO training (HIGH + LOW) for CLSPHierEnv2 (S-CLSP-SD).

    Same algorithm as PPO_Hier. The production order of the selected products is sampled from a
    Plackett-Luce distribution over the HIGH logits (Gumbel trick); it is LOW's decision order and the
    production sequence, and its log-probability is added to HIGH's in the PPO ratio. Greedy evaluation
    uses the probability order. The capacity left for LOW is tracked with env.low_capacity_start /
    env.low_setup_charge.

    :param env: Configured environment.
    :param HYP: Hyperparameter dict. Main keys:
        lr_high, lr_low, lam (GAE lambda), clip_ratio, kl_target, batch_size, train_iters,
        steps_per_update, updates, print_every, out_dir,
        gamma (optionally gamma_high / gamma_low), entropy_coef (optionally entropy_coef_high / _low),
        low_return_horizon (30), low_succ_horizon (10, 0 disables), low_reward_scale (100).
    :param device: "cpu" or "cuda".
    :param seed: Optional seed.
    """

    def __init__(self, env, HYP, device="cpu", seed=None):
        self.env = env
        self.HYP = HYP
        self.device = device
        self.seed = seed

        # Dimensions derived from the environmen
        self.K = self.env.K
        self.high_obs_dim = self.env.observation_space.shape[0]  # 2K
        self.low_obs_dim = self.env.low_obs_dim                  # 4K + 4

        # HIGH and LOW policies; LOW outputs a Beta(alpha, beta) action in [0, 1] (fraction of q_cap).
        # MLP widths via HYP: hidden_sizes, hidden_high, hidden_low (default (256, 256)).
        hidden_high = tuple(HYP.get("hidden_high", HYP.get("hidden_sizes", (256, 256))))
        hidden_low  = tuple(HYP.get("hidden_low",  HYP.get("hidden_sizes", (256, 256))))

        self.high = HighPolicy(self.high_obs_dim, self.K, hidden=hidden_high).to(self.device)
        self.low = LowPolicy(self.low_obs_dim, hidden=hidden_low).to(self.device)

        # Optimizers LR schedule: start at level 0
        high_levels = HYP.get("lr_high_levels", [HYP["lr_high"]])
        low_levels  = HYP.get("lr_low_levels",  [HYP["lr_low"]])

        self.opt_high = Adam(self.high.parameters(), lr=float(high_levels[0]), eps=1e-5)
        self.opt_low  = Adam(self.low.parameters(),  lr=float(low_levels[0]),  eps=1e-5)


        # Per-level discount factors.
        self.gamma_high = float(HYP.get("gamma_high", HYP["gamma"]))
        self.gamma_low  = float(HYP.get("gamma_low",  HYP["gamma"]))

        # Per-level entropy bonus coefficients.
        self.entropy_coef_high = float(HYP.get("entropy_coef_high", HYP["entropy_coef"]))
        self.entropy_coef_low  = float(HYP.get("entropy_coef_low",  HYP["entropy_coef"]))

        # LOW return: per-product Monte-Carlo window return, see buffers.compute_low_window_returns.
        self.low_return_horizon = int(HYP.get("low_return_horizon", 30))
        self.low_reward_scale   = float(HYP.get("low_reward_scale", 100.0))
        # Window of the successors' future-cost term of the LOW return (0 = disabled).
        self.low_succ_horizon   = int(HYP.get("low_succ_horizon", 10))

        # RNG for the Plackett-Luce production order
        self.order_rng = np.random.default_rng(seed)

        # Experience buffers
        self.buf_high = HighLevelBuffer(
            obs_dim=self.high_obs_dim,
            act_dim=self.K,
            size=HYP["steps_per_update"],
            gamma=self.gamma_high,
            lam=HYP["lam"],
            device=self.device,
            normalize_rewards=True,
            normalize_advantages=True,
            reward_norm_momentum=0.99,
        )

        self.buf_low = LowLevelBuffer(
            obs_dim=self.low_obs_dim,
            device=self.device,
            gamma=self.gamma_low,
            normalize_advantages=True,
            window_H=self.low_return_horizon,
            reward_scale=self.low_reward_scale,
            succ_H=self.low_succ_horizon,
        )

        self.use_wandb = HYP.get("use_wandb", False)

        # Basic training logging
        self.curve_cost = []
        self.curve_real_reward = []

        # Validation metrics (new "best model" criterion)
        self.best_val_cost = None        # best validation cost (lower is better)
        self.best_val_step = None        # update index where best model was achieved
        self.val_cost_curve = []         # validation cost over time
        self.val_steps = []              # updates at which validation was performed

        os.makedirs(HYP["out_dir"], exist_ok=True)

        # Initial environment state
        self.obs, _ = self.env.reset(seed=self.seed)


    
    # PPO updates (HIGH / LOW)
    def _ppo_update_high(self, data):
        """Updates the HIGH network with PPO on the provided batch."""
        H = self.HYP
        self.high.train()

        N = data["obs"].shape[0]
        if N == 0:
            return 0.0, 0.0, 0.0

        idx = torch.randperm(N, device=self.device)
        last_kl = 0.0

        pi_losses_all = []
        v_losses_all = []

        for _ in range(H["train_iters"]):
            kls = []

            for j in range(0, N, H["batch_size"]):
                b = idx[j:j + H["batch_size"]]
                obs, act, adv, ret, logp_old, mask = (
                    data[k][b] for k in ["obs", "act", "adv", "ret", "logp", "mask"]
                )
                order_b = data["order"][b]

                # HIGH actor-critic forward pass
                logits, v = self.high(obs)

                # Masked distribution
                mask = mask.to(device=logits.device, dtype=logits.dtype)
                dist = MaskedBernoulli(logits, mask)
                # log_prob is the joint log-prob, summed over the K Bernoulli components (not averaged).
                logp = dist.log_prob(act)
                # Add the log-probability of the sampled production order
                logp = logp + plackett_luce_log_prob(dist.logits, order_b)
                probs = dist.probs

                # PPO: ratio and clipped policy loss
                ratio = torch.exp(logp - logp_old)
                clip_adv = torch.clamp(ratio, 1 - H["clip_ratio"], 1 + H["clip_ratio"]) * adv
                pi_loss = -torch.min(ratio * adv, clip_adv).mean()

                # Entropy summed over the K components.
                ent = -(probs * torch.log(probs + 1e-8)
                        + (1 - probs) * torch.log(1 - probs + 1e-8)).sum(dim=-1).mean()

                # Critic loss
                v_loss = F.mse_loss(v, ret)

                # Total loss
                loss = pi_loss + 0.5 * v_loss - self.entropy_coef_high * ent

                # Optimization step
                self.opt_high.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.high.parameters(), 10.0)
                self.opt_high.step()

                # Accumulate losses for logging
                pi_losses_all.append(float(pi_loss.item()))
                v_losses_all.append(float(v_loss.item()))

                # Approximate KL
                with torch.no_grad():
                    kl = (logp_old - logp).mean().clamp_min(0.0)
                    kls.append(float(kl))

            last_kl = float(np.mean(kls)) if kls else 0.0

            if last_kl > H["kl_target"]:
                break

        mean_pi_loss = float(np.mean(pi_losses_all)) if pi_losses_all else 0.0
        mean_v_loss = float(np.mean(v_losses_all)) if v_losses_all else 0.0

        return last_kl, mean_pi_loss, mean_v_loss
        

    def _ppo_update_low(self, data):
        """Updates the LOW network with PPO on the provided batch."""
        H = self.HYP
        self.low.train()

        N = data["obs"].shape[0]
        if N == 0:
            return 0.0, 0.0, 0.0

        idx = torch.randperm(N, device=self.device)
        last_kl = 0.0

        pi_losses_all = []
        v_losses_all = []

        for _ in range(H["train_iters"]):
            kls = []

            for j in range(0, N, H["batch_size"]):
                b = idx[j:j + H["batch_size"]]
                obs, act, adv, ret, logp_old = (
                    data[k][b] for k in ["obs", "act", "adv", "ret", "logp"]
                )

                # LOW actor-critic forward pass
                raw, v = self.low(obs)

                # Beta distribution over the continuous LOW action
                dist = BetaDist(raw[:, 0], raw[:, 1])
                logp = dist.log_prob(act)

                # PPO: ratio and clipped policy loss
                ratio = torch.exp(logp - logp_old)
                clip_adv = torch.clamp(ratio, 1 - H["clip_ratio"], 1 + H["clip_ratio"]) * adv
                pi_loss = -torch.min(ratio * adv, clip_adv).mean()

                # Differential entropy of the Beta distribution
                ent = dist.entropy().mean()

                # Critic loss
                v_loss = F.mse_loss(v, ret)

                # Total loss
                loss = pi_loss + 0.5 * v_loss - self.entropy_coef_low * ent

                # Optimization step
                self.opt_low.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.low.parameters(), 10.0)
                self.opt_low.step()

                # Accumulate losses for logging
                pi_losses_all.append(float(pi_loss.item()))
                v_losses_all.append(float(v_loss.item()))

                # Approximate KL
                with torch.no_grad():
                    kl = (logp_old - logp).mean().clamp_min(0.0)
                    kls.append(float(kl))

            last_kl = float(np.mean(kls)) if kls else 0.0

            if last_kl > H["kl_target"]:
                break

        mean_pi_loss = float(np.mean(pi_losses_all)) if pi_losses_all else 0.0
        mean_v_loss = float(np.mean(v_losses_all)) if v_losses_all else 0.0

        return last_kl, mean_pi_loss, mean_v_loss


    
    # training loop
    def train(self):
        """Run the joint HIGH + LOW PPO training loop: collect rollouts, update both policies, validate
        periodically and save the best checkpoints, curves and CSVs to HYP['out_dir'].
        """
        H = self.HYP
        print_every = H["print_every"]
        out_dir = H["out_dir"]
        episode_id = 0
        obs = self.obs  # initial state already reset

        # Evaluation frequency: 1% of total updates
        total_updates = H["updates"]
        eval_every = max(1, total_updates // 100)

        # Fixed base seed for evaluations (enables fair comparisons)
        eval_seed_base = H.get("eval_seed_base", 10000)

        # Early stopping based on validation
        early_stop_patience = H.get("early_stop_patience", 15)
        no_improve_evals = 0      # number of consecutive validations without improvement
        stop_training = False  
        
        # Step-wise LR schedule
        high_lr_levels = H.get("lr_high_levels", [H.get("lr_high", 1e-4)])
        low_lr_levels  = H.get("lr_low_levels",  [H.get("lr_low",  1e-4)])
        lr_drop_patience = int(H.get("lr_drop_patience_evals", 7))

        high_lr_stage = 0
        low_lr_stage  = 0
        

        for upd in trange(total_updates, desc="Train HIGH+LOW joint PPO"):
            total_costs = []

            # DATA COLLECTION
            while self.buf_high.ptr < self.buf_high.max_size or not self.buf_low.is_batch_ready(H["steps_per_update"]):
                obs_t = torch.as_tensor(obs[None, :], dtype=torch.float32, device=self.device)

                # HIGH: mask + action
                high_mask_np = self.env.eligibility_mask_high().astype(np.float32)[None, :]
                high_mask_t  = torch.as_tensor(high_mask_np, device=self.device, dtype=torch.float32)

                with torch.no_grad():
                    logits_H, vH = self.high(obs_t)
                    dist_H = MaskedBernoulli(logits_H, high_mask_t)
                
                    # original Bernoulli action
                    aH_t = dist_H.sample()
                    logpH_t = dist_H.log_prob(aH_t)
                
                    probs_H = dist_H.probs.squeeze(0).cpu().numpy().astype(np.float32)
                    logits_H_np = dist_H.logits.squeeze(0).cpu().numpy().astype(np.float64)
                
                # aH_raw: original Bernouill action (0/1)
                aH_raw = aH_t.squeeze(0).cpu().numpy().astype(np.int8)
                # Raw joint log-prob (summed over K), same scale as in _ppo_update_high.
                logpH_raw = float(logpH_t.item())
                
                # action executed in the environment: the sampled action restricted to the eligible products
                aH_env = aH_raw
                elig = self.env.eligibility_mask_high().astype(np.int8)
                aH_env = (aH_env & elig).astype(np.int8)

                # LOW order (definition of the sub-decision level)
                chosen = self.env.plan_quantity_assignment_order(aH_env, probs_H)

                # Plackett-Luce order of the selected products (Gumbel trick on the HIGH logits).
                logp_order = 0.0
                if chosen.size > 1:
                    sel = np.nonzero(aH_env)[0]
                    perm = plackett_luce_sample(logits_H_np[sel], self.order_rng)
                    chosen = sel[perm].astype(np.int32)
                    with torch.no_grad():
                        order_t = torch.full((1, self.K), -1, dtype=torch.long, device=self.device)
                        order_t[0, :chosen.size] = torch.as_tensor(chosen, dtype=torch.long, device=self.device)
                        logp_order = float(plackett_luce_log_prob(dist_H.logits, order_t).item())

                # LOW period
                self.buf_low.start_period()
                low_actions_q = []
                
                # Capacity left for production
                CAP_rem = self.env.low_capacity_start(aH_env)
                I_view = self.env.I.copy()  # virtual inventory within the period
                prev_idx = int(self.env.current_setup_idx())  # machine setup along the sequence
                
                for pos, i in enumerate(chosen):
                    # LOW state + feasible capacity cap given current CAP_rem
                    low_obs_np, q_cap, need_setup = self.env._low_state_mask_for(
                        i,
                        CAP_rem,
                        a_high_scores=probs_H,
                        pos=pos,
                        chosen=chosen,
                        I_view=I_view,
                        a_high_vec=aH_env,
                        prev_idx=prev_idx,
                    )

                    low_obs_t = torch.as_tensor(low_obs_np[None, :], dtype=torch.float32, device=self.device)

                    with torch.no_grad():
                        raw_L, vL = self.low(low_obs_t)
                        dist_L = BetaDist(raw_L[:, 0], raw_L[:, 1])
                        a_t = dist_L.sample()
                        logpL_t = dist_L.log_prob(a_t)

                    a_frac = float(a_t.item())
                    q_i = self.env.low_action_to_qty(a_frac, q_cap)

                    # Store LOW substep: act = fraction in [0, 1]
                    self.buf_low.store_substep(low_obs_np, a_frac, float(vL.item()), float(logpL_t.item()))
                    low_actions_q.append(a_frac)

                    # Updates CAP_rem with q_i (quantity)
                    if q_i > 0:
                        if need_setup:
                            CAP_rem -= self.env.low_setup_charge(prev_idx, i)
                        CAP_rem -= q_i
                        CAP_rem = max(0.0, CAP_rem)
                        I_view[i] = I_view[i] + q_i * self.env.bs[i]
                        prev_idx = int(i)


                # Environment step and global reward
                next_obs, _, term, trunc, info = self.env.step(
                    aH_env, low_actions=low_actions_q, a_high_scores=probs_H,
                    order=chosen,
                )
                cost_t = float(info["cost"])
                reward_t = -cost_t
                total_costs.append(cost_t)

                # Store transitions in buffers
                self.buf_high.store(
                    obs,
                    aH_raw,
                    reward_t,
                    float(vH.item()),
                    logpH_raw + logp_order,
                    mask=high_mask_np.squeeze(0),
                    order=chosen,
                )

                # LOW credit assignment: pass the per-product costs of this period (holding+backorder `hb`, setup `su`);
                # the window returns are built when the episode closes.
                hb_vec = (np.asarray(info["holding"], dtype=np.float64)
                          + np.asarray(info["backord"], dtype=np.float64))
                su_vec = np.asarray(info["cost_per_product"], dtype=np.float64) - hb_vec
                su_vec = np.where(su_vec > 1e-3, su_vec, 0.0)
                self.buf_low.end_period_assign_costs(hb_vec, su_vec, chosen)

                # Reset or advance
                if term or trunc:
                    self.buf_high.finish_path(last_val=0.0)
                    self.buf_low.finish_episode_and_collect()
                    episode_id += 1
                    obs, _ = self.env.reset(seed=self.seed + episode_id)
                else:
                    obs = next_obs

                # HIGH buffer fill condition
                if self.buf_high.ptr >= self.buf_high.max_size:
                    break

            # Close pending trajectories
            if self.buf_high.path_start_idx < self.buf_high.ptr:
                self.buf_high.finish_path(last_val=0.0)
            self.buf_low.finish_episode_and_collect()

            # Extract PPO batches
            dataH = self.buf_high.get()
            dataL = self.buf_low.get()

            # PPO UPDATE + LOSS METRICS
            kl_h, pi_loss_h, v_loss_h = self._ppo_update_high(dataH)
            kl_l, pi_loss_l, v_loss_l = self._ppo_update_low(dataL)

            # Training metrics
            mean_cost = float(np.mean(total_costs)) if total_costs else 0.0
            self.curve_cost.append(mean_cost)
            mean_real_reward = -mean_cost
            self.curve_real_reward.append(mean_real_reward)

            # Console logging
            if (upd + 1) % print_every == 0:
                print(
                    f"[Joint {upd+1}] mean RAW cost: {mean_cost:.3f} | "
                    f"mean REAL reward: {mean_real_reward:.3f} | "
                    f"KL_high={kl_h:.4f} | KL_low={kl_l:.4f}"
                )

            # WandB logging
            if self.use_wandb:
                import wandb

                log_dict = {
                    "update": upd + 1,
                    "mean_raw_cost": mean_cost,
                    "mean_real_reward": mean_real_reward,
                    "KL_high": kl_h,
                    "KL_low": kl_l,
                    "loss_actor_high": pi_loss_h,
                    "loss_critic_high": v_loss_h,
                    "loss_actor_low": pi_loss_l,
                    "loss_critic_low": v_loss_l,
                }

                wandb.log(log_dict, step=upd + 1)

            # PERIODIC VALIDATION (~every 1% of updates)
            do_eval = ((upd + 1) % eval_every == 0) or (upd == total_updates - 1)
            if do_eval:
                eval_stats = self.test(
                    n_runs=50,
                    horizon=228,
                    warmup=128,
                    greedy=True,
                    seed_base=eval_seed_base,
                    ckpt_high=None,
                    ckpt_low=None,
                    env_overrides={"horizon": 228, "enable_Ilim": False},
                )
                val_cost = float(eval_stats["mean_of_means"])

                # Store validation curves
                self.val_cost_curve.append(val_cost)
                self.val_steps.append(upd + 1)

                # Log validation cost to WandB
                if self.use_wandb:
                    import wandb
                    wandb.log(
                        {
                            "val_mean_cost": val_cost,
                        },
                        step=upd + 1,
                    )

                # Update best model according to validation + early stopping
                if (self.best_val_cost is None) or (val_cost < self.best_val_cost - 1e-9):
                    # Ha habido mejora
                    self.best_val_cost = val_cost
                    self.best_val_step = upd + 1
                    no_improve_evals = 0  

                    torch.save(self.high.state_dict(), os.path.join(out_dir, "high_joint_best.pt"))
                    torch.save(self.low.state_dict(),  os.path.join(out_dir, "low_joint_best.pt"))
                    print(f"[VALID] New best model: val_mean_cost={self.best_val_cost:.3f} @ update {self.best_val_step}")
                else:
                    no_improve_evals += 1
                    print(f"[VALID] No improvement in validation for {no_improve_evals} consecutive evaluations.")

                    # Step schedule: every N non-improving evaluations, drop LR one level (after update 1500)
                    if (upd + 1) >= 1500 and lr_drop_patience > 0 and (no_improve_evals % lr_drop_patience) == 0:

                        # HIGH: advance stage if possible
                        if high_lr_stage < (len(high_lr_levels) - 1):
                            high_lr_stage += 1
                            new_lr_h = float(high_lr_levels[high_lr_stage])
                            for pg in self.opt_high.param_groups:
                                pg["lr"] = new_lr_h

                        # LOW: advance stage if possible
                        if low_lr_stage < (len(low_lr_levels) - 1):
                            low_lr_stage += 1
                            new_lr_l = float(low_lr_levels[low_lr_stage])
                            for pg in self.opt_low.param_groups:
                                pg["lr"] = new_lr_l

                        lr_h_now = self.opt_high.param_groups[0]["lr"]
                        lr_l_now = self.opt_low.param_groups[0]["lr"]
                        print(f"[LR STEP] no_improve={no_improve_evals} | lr_high={lr_h_now:.2e} | lr_low={lr_l_now:.2e}")

                        if self.use_wandb:
                            import wandb
                            wandb.log({"lr_high": lr_h_now, "lr_low": lr_l_now}, step=upd + 1)

                    # Early stopping based on validation
                    if no_improve_evals > early_stop_patience:
                        print(
                            f"[EARLY STOP] No improvement in validation for {early_stop_patience} "
                            f"consecutive evaluations. Stopping training at update {upd+1}."
                        )
                        stop_training = True


            if stop_training:
                break


        # FINAL SAVES AND METRICS (best + final only)
        print("Joint training finished.")
        torch.save(self.high.state_dict(), os.path.join(out_dir, "high_joint_final.pt"))
        torch.save(self.low.state_dict(), os.path.join(out_dir, "low_joint_final.pt"))
        print("Saved models: high_joint_final.pt / low_joint_final.pt")
    
        if self.best_val_cost is not None:
            print(
                f"Best validation cost: {self.best_val_cost:.3f} "
                f"@ update {self.best_val_step} "
                f"(high_joint_best.pt / low_joint_best.pt)"
            )
    
        # Plots and CSVs
        try:
            plt.figure()
            plt.plot(self.curve_cost, label="Mean RAW cost per update")
            plt.xlabel("Update"); plt.ylabel("Mean raw cost"); plt.title("Joint – Cost")
            plt.grid(True); plt.legend(); plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "curve_joint_cost.png"), dpi=150)
    
            plt.figure()
            plt.plot(self.curve_real_reward, label="Mean REAL reward per update")
            plt.xlabel("Update"); plt.ylabel("Mean real reward"); plt.title("Joint – Real Reward")
            plt.grid(True); plt.legend(); plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "curve_joint_real_reward.png"), dpi=150)
    
            np.savetxt(os.path.join(out_dir, "curve_joint_cost.csv"), np.array(self.curve_cost), delimiter=",")
            np.savetxt(os.path.join(out_dir, "curve_joint_real_reward.csv"), np.array(self.curve_real_reward), delimiter=",")
    
            if len(self.val_cost_curve) > 0:
                np.savetxt(os.path.join(out_dir, "curve_val_cost.csv"), np.array(self.val_cost_curve), delimiter=",")
                np.savetxt(os.path.join(out_dir, "curve_val_steps.csv"), np.array(self.val_steps, dtype=np.int32), delimiter=",")
    
            print(
                "Saved plots/CSVs: curve_joint_cost.png, curve_joint_real_reward.png "
                "(+ validation curves if present)."
            )
        except Exception as e:
            print(f"(warning) could not generate/save plots or CSVs: {e}")


    # EVAL
    def _build_eval_env(self, env_overrides, seed):
        """Creates a fresh evaluation environment from the training environment by cloning its baseline
        configuration and applying optional overrides.
        """
        src = self.env
        ov = env_overrides or {}

        # Copy structural parameters from the original environment
        K   = src.K
        mu  = np.array(src.mu, copy=True)
        bs  = np.array(src.bs, copy=True)
        h   = np.array(src.h,  copy=True)
        b   = np.array(src.b,  copy=True)
        k   = np.array(src.k,  copy=True)
        st  = np.array(src.st, copy=True)
        CAP = src.CAP
        k_sd  = None if getattr(src, "k_sd", None)  is None else np.array(src.k_sd,  copy=True)
        st_sd = None if getattr(src, "st_sd", None) is None else np.array(src.st_sd, copy=True)

        # Inherited parameters with optional overrides
        kwargs = dict(
            horizon=ov.get("horizon", getattr(src, "horizon", 10_000) or 10_000),
            cov=np.array(getattr(src, "cov", np.ones(K, np.float32)), copy=True),
            seed=seed,
            enable_ip_eligibility=ov.get("enable_ip_eligibility", getattr(src, "enable_ip_eligibility", True)),
            enforce_eoq_qmax=ov.get("enforce_eoq_qmax", getattr(src, "enforce_eoq_qmax", True)),
            eoq_slack_batches=ov.get("eoq_slack_batches", getattr(src, "eoq_slack_batches", 0)),
            reserve_future_min=ov.get("reserve_future_min", getattr(src, "reserve_future_min", True)),
            force_min_batch=ov.get("force_min_batch", getattr(src, "force_min_batch_default", True)),
            enable_Ilim=ov.get("enable_Ilim", getattr(src, "enable_Ilim", True)),
            init_inventory=ov.get("init_inventory", getattr(src, "init_inventory", None)),
            init_active_idx=ov.get("init_active_idx", getattr(src, "init_active_idx", None)),
            k_sd=k_sd,
            st_sd=st_sd,
        )

        # New evaluation environment (independent from the training one)
        env_eval = CLSPHierEnv2(K, mu, bs, h, b, k, st, CAP, **kwargs)

        # Seed the spaces too (Gymnasium accepts int/None only)
        if isinstance(seed, int):
            env_eval.action_space.seed(seed)
            env_eval.observation_space.seed(seed)

        obs_init, _ = env_eval.reset(seed=seed)
        return env_eval, obs_init


    
    @staticmethod
    @torch.no_grad()
    def _act_high(env, high_model, obs_np, device, greedy=True):
        """Selects a HIGH-level action given the current observation and a trained HIGH policy."""
        obs_t = torch.as_tensor(obs_np[None, :], dtype=torch.float32, device=device)
        logits, v = high_model(obs_t)

        # HIGH eligibility mask
        mask_np = env.eligibility_mask_high().astype(np.float32)[None, :]
        mask_t = torch.as_tensor(mask_np, device=device, dtype=logits.dtype)

        # Masked Bernoulli distribution
        dist = MaskedBernoulli(logits, mask_t)
        probs = dist.probs[0].cpu().numpy().astype(np.float32)

        if greedy:
            # Deterministic threshold-based selection
            a = (probs >= 0.5).astype(np.int64)

            # Fallback if no product is selected
            if a.sum() == 0:
                omega_idx = int(np.argmax(env.omega))
                if mask_np[0, omega_idx] > 0:
                    a[omega_idx] = 1
                else:
                    elig = np.where(mask_np[0] > 0)[0]
                    if elig.size > 0:
                        a[elig[np.argmax(probs[elig])]] = 1
        else:
            # Stochastic sampling
            a = dist.sample()[0].cpu().numpy().astype(np.int64)

        # Enforce eligibility mask
        a = (a & mask_np.squeeze(0).astype(np.int64))

        # Log-probability of the selected action
        logp = dist.log_prob(
            torch.as_tensor(a[None, :], dtype=torch.float32, device=device)
        ).item()

        return a.astype(np.int64), float(logp), float(v.item()), mask_np.squeeze(0), probs


    @staticmethod
    @torch.no_grad()
    def _act_low_seq(env, low_model, a_high_env, a_high_scores=None, greedy=True,
                    device="cpu", force_min_batch=None, order_rng=None):
        """Generates the LOW-level action sequence (continuous fractions) for the products selected by the
        HIGH level within a single period.
        """
        fmb = force_min_batch if force_min_batch is not None else getattr(env, "force_min_batch_default", False)

        # LOW visit order induced by HIGH
        scores_vec = a_high_scores if a_high_scores is not None else a_high_env.astype(np.float32)
        chosen = env.plan_quantity_assignment_order(a_high_env, scores_vec)
        # Stochastic evaluation samples the Plackett-Luce order; greedy keeps the logit order.
        if order_rng is not None and not greedy and chosen.size > 1:
            sel = np.nonzero(a_high_env)[0]
            p = np.clip(np.asarray(scores_vec, dtype=np.float64)[sel], 1e-12, 1 - 1e-12)
            perm = plackett_luce_sample(np.log(p) - np.log1p(-p), order_rng)
            chosen = sel[perm].astype(np.int32)

        low_actions_frac = []

        CAP_rem_sim = env.low_capacity_start(a_high_env)
        prev_idx = int(env.current_setup_idx())  # machine setup along the sequence

        # Virtual within-period inventory (later LOW states see earlier production)
        I_view = env.I.copy()

        for pos, i in enumerate(chosen):
            low_obs_np, q_cap, need_setup = env._low_state_mask_for(
                i,
                CAP_rem_sim,
                a_high_scores=scores_vec,
                pos=pos,
                chosen=chosen,
                I_view=I_view,
                a_high_vec=a_high_env,
                prev_idx=prev_idx,
            )

            obs_t = torch.as_tensor(low_obs_np[None, :], dtype=torch.float32, device=device)
            raw, _ = low_model(obs_t)
            dist = BetaDist(raw[:, 0], raw[:, 1])

            if greedy:
                a_frac = float(dist.mean.item())
            else:
                a_frac = float(dist.sample().item())

            low_actions_frac.append(a_frac)

            q_i = env.low_action_to_qty(a_frac, q_cap, force_min_batch=fmb)

            # Update simulated capacity
            if q_i > 0 and need_setup:
                CAP_rem_sim -= env.low_setup_charge(prev_idx, i)
            CAP_rem_sim -= q_i
            CAP_rem_sim = max(0.0, CAP_rem_sim)

            # Update virtual inventory and machine setup for subsequent LOW states
            if q_i > 0:
                I_view[i] = I_view[i] + q_i * env.bs[i]
                prev_idx = int(i)

        return chosen, low_actions_frac


    
    def test(self, n_runs=10, horizon=10_000, warmup=1_000, greedy=True, seed_base=12345, 
             ckpt_high=None, ckpt_low=None, env_overrides=None):
        """Evaluates the HIGH+LOW pair in terms of **average cost per period** over multiple runs.

        :param n_runs: Number of independent evaluation runs.
        :param horizon: Maximum episode length (in steps) for each evaluation run.
        :param warmup: Number of initial steps to ignore when computing the mean cost (to discard the
            transient phase).
        :param greedy: If True, uses deterministic (greedy) decisions in both HIGH and LOW. If False, uses
            stochastic sampling.
        :param seed_base: Base seed used to spawn an independent per-run ``numpy.random.Generator`` via
            ``np.random.SeedSequence(seed_base, spawn_key=(r,))`` (NOT ``seed_base + r`` directly, so runs
            are statistically independent rather than merely offset). If None, determinism across runs is
            not enforced.
        :param ckpt_high: Optional path to a HIGH policy checkpoint to load before evaluation.
        :param ckpt_low: Optional path to a LOW policy checkpoint to load before evaluation.
        :param env_overrides: Optional dictionary overriding evaluation-environment parameters; they take
            precedence over the training environment.
        :return: Dict with n_runs, horizon, warmup, greedy, mean_costs_per_run, mean_of_means, std_of_means.
        """
        # Optional checkpoint loading
        if ckpt_high and os.path.isfile(ckpt_high):
            self.high.load_state_dict(torch.load(ckpt_high, map_location=self.device))
        if ckpt_low and os.path.isfile(ckpt_low):
            self.low.load_state_dict(torch.load(ckpt_low, map_location=self.device))
        self.high.eval(); self.low.eval()

        # Let env_overrides["horizon"] override the horizon argument
        env_overrides = dict(env_overrides or {})
        env_overrides.setdefault("horizon", horizon)

        mean_costs = []

        for r in trange(n_runs, desc="Test evaluation", leave=False):
            # One independent, reproducible demand stream per run: SeedSequence(seed_base, spawn_key=(r,)).
            if seed_base is not None:
                run_seed_seq = np.random.SeedSequence(seed_base, spawn_key=(r,))
                run_rng = np.random.default_rng(run_seed_seq)

                # Plain int from the same SeedSequence to reseed torch/random/numpy.
                run_seed_int = int(run_seed_seq.generate_state(1)[0])
                random.seed(run_seed_int)
                np.random.seed(run_seed_int)
                torch.manual_seed(run_seed_int)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(run_seed_int)
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
                torch.use_deterministic_algorithms(True)
            else:
                run_rng = None

            # Evaluation environment and initial state
            env_eval, obs = self._build_eval_env(env_overrides, seed=run_rng)

            costs = []
            for t in range(int(env_overrides["horizon"])):
                # HIGH action (deterministic or stochastic) + LOW sequence
                a_high_env, _, _, _, probs_H = self._act_high(env_eval, self.high, obs, device=self.device, greedy=greedy)
                chosen, low_actions_q = self._act_low_seq(
                    env_eval,
                    self.low,
                    a_high_env,
                    a_high_scores=probs_H,
                    greedy=greedy,
                    device=self.device,
                    force_min_batch=env_overrides.get("force_min_batch", None),
                    order_rng=self.order_rng,
                )

                # Step the environment with both decisions
                obs_next, _, terminated, truncated, info = env_eval.step(
                    a_high_env,
                    low_actions=low_actions_q,
                    a_high_scores=probs_H,
                    order=chosen,
                )

                # Accumulate cost only after warm-up
                if t >= int(warmup):
                    costs.append(float(info["cost"]))

                if terminated or truncated:
                    break

                obs = obs_next

            # Mean cost for this run
            mc = float(np.mean(costs)) if costs else float("nan")
            mean_costs.append(mc)

        # Aggregate across runs
        mean_of_means = float(np.mean(mean_costs)) if mean_costs else float("nan")
        std_of_means = float(np.std(mean_costs, ddof=1)) if len(mean_costs) > 1 else 0.0

        return {
            "n_runs": n_runs,
            "horizon": int(env_overrides["horizon"]),
            "warmup": int(warmup),
            "greedy": bool(greedy),
            "mean_costs_per_run": mean_costs,
            "mean_of_means": mean_of_means,
            "std_of_means": std_of_means,
        }

