\
\
   

import os
import json
from dataclasses import dataclass
from typing import Optional, Union

import draccus

from experiments.robot.libero.run_libero_eval import (
    GenerateConfig as BaseGenerateConfig,
    validate_config,
    setup_logging,
    log_message,
    load_initial_states,
    TaskSuite,
    TASK_MAX_STEPS,
    check_unnorm_key,
)
from experiments.robot.openvla_utils import (
    get_action_head,
    get_noisy_action_projector,
    get_processor,
    get_proprio_projector,
    resize_image_for_policy,
)
from experiments.robot.libero.libero_utils import (
    get_libero_env,
    get_libero_dummy_action,
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
)
from experiments.robot.robot_utils import (
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)

import numpy as np
import tqdm


@dataclass
class PruneV2GenerateConfig(BaseGenerateConfig):
    qk_config_json: Optional[str] = None
    qk_keep_enabled: bool = True
    qk_layer: int = 0
    qk_keep_ratio: float = 0.75
                                                                                                 
    qk_keep_split: Optional[str] = None
    qk_debug: bool = False
    qk_log_topk: int = 16

                          
    use_dynamic_visual_strategy: bool = True
    decision_method: str = "avg"                               
    adjacent_variant: str = "classic"                             
    adjacent_extrema_window: int = 3
    adjacent_lookback: int = 2
    adjacent_last_state: bool = False
    initial_state: int = 0                                       
    delta_method: str = "net"                                                                     
    L_eff: float = 0.15
    min_delta_pos: float = 0.0
    min_delta_rot: float = 0.0
    hysteresis_up: float = 0.0
    hysteresis_down: float = 0.0
    tol_equal: float = 0.0
                                  
    limit_consecutive_pruned_enabled: bool = True
    limit_max_consecutive_pruned: int = 3

                               


def _load_qk_config_from_json_if_any(cfg: PruneV2GenerateConfig) -> None:
    if cfg.qk_config_json and os.path.isfile(cfg.qk_config_json):
        try:
            with open(cfg.qk_config_json, "r") as f:
                data = json.load(f)
            cfg.qk_keep_enabled = bool(data.get("qk_keep_enabled", cfg.qk_keep_enabled))
            cfg.qk_layer = int(data.get("qk_layer", cfg.qk_layer))
            cfg.qk_keep_ratio = float(data.get("qk_keep_ratio", cfg.qk_keep_ratio))
            split = data.get("qk_keep_split", cfg.qk_keep_split)
            if isinstance(split, list):
                                                                                 
                split = ",".join(str(float(x)) for x in split)
            cfg.qk_keep_split = split
                                                        
            cfg.qk_debug = bool(data.get("qk_debug", cfg.qk_debug))
            cfg.qk_log_topk = int(data.get("qk_log_topk", cfg.qk_log_topk))
                              
            if "num_trials_per_task" in data:
                cfg.num_trials_per_task = int(data.get("num_trials_per_task", cfg.num_trials_per_task))
                                     
            cfg.use_dynamic_visual_strategy = bool(data.get("use_dynamic_visual_strategy", cfg.use_dynamic_visual_strategy))
            cfg.decision_method = str(data.get("decision_method", cfg.decision_method))
            cfg.adjacent_variant = str(data.get("adjacent_variant", cfg.adjacent_variant))
            cfg.adjacent_extrema_window = int(data.get("adjacent_extrema_window", cfg.adjacent_extrema_window))
            cfg.adjacent_last_state = bool(data.get("adjacent_last_state", cfg.adjacent_last_state))
            cfg.adjacent_lookback = int(data.get("adjacent_lookback", cfg.adjacent_lookback))
            cfg.initial_state = int(data.get("initial_state", cfg.initial_state))
            cfg.delta_method = str(data.get("delta_method", cfg.delta_method))
            cfg.L_eff = float(data.get("L_eff", cfg.L_eff))
            cfg.min_delta_pos = float(data.get("min_delta_pos", cfg.min_delta_pos))
            cfg.min_delta_rot = float(data.get("min_delta_rot", cfg.min_delta_rot))
            cfg.hysteresis_up = float(data.get("hysteresis_up", cfg.hysteresis_up))
            cfg.hysteresis_down = float(data.get("hysteresis_down", cfg.hysteresis_down))
            cfg.tol_equal = float(data.get("tol_equal", cfg.tol_equal))
                                      
            cfg.limit_consecutive_pruned_enabled = bool(data.get("limit_consecutive_pruned_enabled", getattr(cfg, "limit_consecutive_pruned_enabled", False)))
            cfg.limit_max_consecutive_pruned = int(data.get("limit_max_consecutive_pruned", getattr(cfg, "limit_max_consecutive_pruned", 3)))

        except Exception as e:
            print(f"[prunevla_v2] failed to load qk config json: {e}")


def _prepare_observation(obs, resize_size):
    img = get_libero_image(obs)
    wrist_img = get_libero_wrist_image(obs)
    img_resized = resize_image_for_policy(img, resize_size)
    wrist_img_resized = resize_image_for_policy(wrist_img, resize_size)
    observation = {
        "full_image": img_resized,
        "wrist_image": wrist_img_resized,
        "state": np.concatenate(
            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])                
        ),
    }
    return observation, img


def _process_action(action, model_family):
    action = normalize_gripper_action(action, binarize=True)
    if model_family == "openvla":
        action = invert_gripper_action(action)
    return action


def _initialize_components(cfg: PruneV2GenerateConfig):
    model = get_model(cfg)
    proprio_projector = get_proprio_projector(cfg, model.llm_dim, proprio_dim=8) if cfg.use_proprio else None
    action_head = get_action_head(cfg, model.llm_dim) if (cfg.use_l1_regression or cfg.use_diffusion) else None
    noisy_action_projector = get_noisy_action_projector(cfg, model.llm_dim) if cfg.use_diffusion else None
    processor = get_processor(cfg) if cfg.model_family == "openvla" else None
    return model, action_head, proprio_projector, noisy_action_projector, processor


def _run_one_episode(cfg: PruneV2GenerateConfig, env, task_description: str,
                     model, resize_size, processor, action_head, proprio_projector, noisy_action_projector,
                     initial_state=None, log_file=None):
    from collections import deque
    env.reset()
    obs = env.set_init_state(initial_state) if initial_state is not None else env.get_observation()

    action_queue = deque(maxlen=cfg.num_open_loop_steps)
    t = 0
    max_steps = TASK_MAX_STEPS[cfg.task_suite_name]

                        
                             
    class _OnlineStrategyController:
        def __init__(self):
            self.curr_state = int(getattr(cfg, "initial_state", 1))
            self.initial_state = int(getattr(cfg, "initial_state", 1))
            self.curr_window_start_frame = None
            self.frame_counter_global = 0
            self.pos_list = []
            self.aa_list = []
                                  
            self.ones_streak = 1 if (self.initial_state == 1 and getattr(cfg, "limit_consecutive_pruned_enabled", False)) else 0

        def _axis_angle_to_matrix(self, aa):
            theta = float(np.linalg.norm(aa))
            if theta < 1e-12:
                return np.eye(3)
            axis = aa / theta
            x, y, z = axis
            K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=np.float64)
            s = np.sin(theta)
            c = np.cos(theta)
            return np.eye(3) + s * K + (1.0 - c) * (K @ K)

        def _relative_rotation_angle(self, R_prev, R_curr):
            R_rel = R_prev.T @ R_curr
            tr = float(np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0))
            ang = float(np.arccos(tr))
            return 0.0 if abs(ang) < 1e-12 else ang

        def observe_step(self, eef_pos, eef_quat):
            aa = quat2axisangle(eef_quat)
            if self.curr_window_start_frame is None:
                self.curr_window_start_frame = self.frame_counter_global
            self.pos_list.append(np.array(eef_pos, dtype=np.float64))
            self.aa_list.append(np.array(aa, dtype=np.float64))
            self.frame_counter_global += 1

        def _compute_window_delta(self):
            positions = np.stack(self.pos_list, axis=0) if len(self.pos_list) > 0 else np.zeros((0, 3))
            axis_angles = np.stack(self.aa_list, axis=0) if len(self.aa_list) > 0 else np.zeros((0, 3))
            method = (getattr(cfg, "delta_method", "net") or "net").lower()
            if positions.shape[0] <= 1:
                return 0.0
            if method == "net":
                dp = positions[-1] - positions[0]
                return float(np.linalg.norm(dp))
                                                        
            if method in ("sum", "arc_sum", "hypot"):
                step_pos = np.linalg.norm(positions[1:] - positions[:-1], axis=1)
                step_pos = np.where(step_pos < getattr(cfg, "min_delta_pos", 0.0), 0.0, step_pos)
                return float(np.sum(step_pos))
                      
            dp = positions[-1] - positions[0]
            return float(np.linalg.norm(dp))

        def close_window_and_decide_next(self):
            d = self._compute_window_delta()
            next_state = self.curr_state
            dm = (getattr(cfg, "decision_method", "avg") or "avg").lower()
            if dm == "avg":
                                            
                                                
                if not hasattr(self, "_avg_d"):
                    self._avg_d = d
                else:
                    self._avg_d = 0.5 * self._avg_d + 0.5 * d
                next_state = 1 if (d >= self._avg_d) else 0
            else:
                                                               
                Lw = max(3, int(getattr(cfg, "adjacent_extrema_window", 3)))
                if not hasattr(self, "_history_d"):
                    self._history_d = []
                self._history_d.append(d)
                if len(self._history_d) < Lw:
                    next_state = int(getattr(cfg, "initial_state", 0))
                else:
                    prev_vals = [float(x) for x in self._history_d[-(Lw-1)-1:-1]] if len(self._history_d) >= Lw else []
                    if len(prev_vals) == 0:
                        next_state = int(getattr(cfg, "initial_state", 0))
                    else:
                        up_thr = max(prev_vals)
                        dn_thr = min(prev_vals)
                        if d >= up_thr:
                            next_state = 1
                        elif d < dn_thr:
                            next_state = 0
                        else:
                            if cfg.adjacent_last_state:
                                next_state = self.curr_state
                            else:
                                next_state = 1       
                                       
            ns = int(next_state)
            if bool(getattr(cfg, "limit_consecutive_pruned_enabled", False)):
                try:
                    streak = int(getattr(self, "ones_streak", 0))
                except Exception:
                    streak = 0
                max_ones = int(getattr(cfg, "limit_max_consecutive_pruned", 3))
                if ns == 1:
                    if streak >= max_ones:
                        ns = 0
                        streak = 0
                    else:
                        streak += 1
                else:
                    streak = 0
                self.ones_streak = streak
            self.curr_state = ns
            self.pos_list.clear()
            self.aa_list.clear()
            self.curr_window_start_frame = None

        def get_current_state(self):
            return int(self.curr_state)

    strategy = _OnlineStrategyController() if bool(getattr(cfg, "use_dynamic_visual_strategy", False)) else None

    def _to_rgb_ndarray(x) -> np.ndarray:
        try:
            import numpy as _np
            if hasattr(x, 'detach'):
                x = x.detach().cpu().numpy()
            arr = _np.array(x)
            if arr.ndim == 2:
                arr = _np.stack([arr, arr, arr], axis=-1)
            elif arr.ndim == 3:
                            
                if arr.shape[0] in (3, 4) and arr.shape[-1] not in (3, 4):
                    arr = _np.transpose(arr[:3], (1, 2, 0))
                                                   
                if arr.shape[-1] == 4:
                    arr = arr[..., :3]
                                                                
                if arr.shape[-1] > 4:
                    arr = arr[..., :3]
                                  
            if arr.dtype != _np.uint8:
                arr = arr.astype(_np.float32)
                arr = _np.clip(arr, 0, 255)
                arr = arr.astype(_np.uint8)
            return arr
        except Exception:
            try:
                from PIL import Image as _Image
                return _np.array(_Image.fromarray(x).convert('RGB'))
            except Exception:
                return _np.zeros((64, 64, 3), dtype=_np.uint8)

    success = False
    while t < max_steps + cfg.num_steps_wait:
        if t < cfg.num_steps_wait:
            obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
            t += 1
            continue

        observation, img = _prepare_observation(obs, resize_size)

        if len(action_queue) == 0:
            if strategy is not None:
                                          
                if strategy.curr_window_start_frame is None:
                    try:
                        eef_pos_now = np.array(obs.get("robot0_eef_pos", [0, 0, 0]), dtype=float)
                        eef_quat_now = np.array(obs.get("robot0_eef_quat", [0, 0, 0, 1]), dtype=float)
                        strategy.observe_step(eef_pos_now, eef_quat_now)
                    except Exception:
                        pass
                state = int(strategy.get_current_state())
                try:
                    cfg.qk_keep_enabled = (state == 1)
                except Exception:
                    pass
            else:
                state = 1 if bool(getattr(cfg, "qk_keep_enabled", False)) else 0
            actions = get_action(
                cfg,
                model,
                observation,
                task_description,
                processor=processor,
                action_head=action_head,
                proprio_projector=proprio_projector,
                noisy_action_projector=noisy_action_projector,
                use_film=cfg.use_film,
            )
            action_queue.extend(actions)

        action = action_queue.popleft()
        action = _process_action(action, cfg.model_family)
        obs, reward, done, info = env.step(action.tolist())
                              
        if strategy is not None:
            try:
                eef_pos_now = np.array(obs.get("robot0_eef_pos", [0, 0, 0]), dtype=float)
                eef_quat_now = np.array(obs.get("robot0_eef_quat", [0, 0, 0, 1]), dtype=float)
                strategy.observe_step(eef_pos_now, eef_quat_now)
            except Exception:
                pass
        if done:
            success = True
            break
        t += 1

                                    
        if strategy is not None and len(action_queue) == 0 and strategy.curr_window_start_frame is not None:
            strategy.close_window_and_decide_next()

    return success


@draccus.wrap()
def eval_libero_prune_v2(cfg: PruneV2GenerateConfig) -> float:
    validate_config(cfg)
    set_seed_everywhere(cfg.seed)
    _load_qk_config_from_json_if_any(cfg)

    model, action_head, proprio_projector, noisy_action_projector, processor = _initialize_components(cfg)

                                          
    if processor is not None:
        try:
            check_unnorm_key(cfg, model)
        except AssertionError:
                                                
            desired = str(getattr(cfg, "task_suite_name", "")).strip()
            keys = list(getattr(model, "norm_stats", {}).keys()) if hasattr(model, "norm_stats") else []
            resolved = None
            if not keys:
                raise
            if desired in keys:
                resolved = desired
            elif f"{desired}_no_noops" in keys:
                resolved = f"{desired}_no_noops"
            else:
                                          
                for k in keys:
                    if desired and desired in k:
                        resolved = k
                        break
                                  
                if resolved is None and len(keys) == 1:
                    resolved = keys[0]
            assert resolved is not None, f"Failed to resolve unnorm_key from norm_stats keys={keys}"
            cfg.unnorm_key = resolved
        try:
            print(f"[prunevla_v2] Resolved unnorm_key = {cfg.unnorm_key}")
            keys = list(getattr(model, "norm_stats", {}).keys())
            print(f"[prunevla_v2] Available norm_stats keys = {keys}")
        except Exception:
            pass

    resize_size = get_image_resize_size(cfg)
    log_file, local_log_filepath, run_id = setup_logging(cfg)

    from libero.libero import benchmark
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()

    log_message(f"Task suite: {cfg.task_suite_name}", log_file)

    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(task_suite.n_tasks)):
        task = task_suite.get_task(task_id)
        initial_states, all_initial_states = load_initial_states(cfg, task_suite, task_id, log_file)
        env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)

        for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
            if cfg.initial_states_path == "DEFAULT":
                initial_state = initial_states[episode_idx]
            else:
                initial_states_task_key = task_description.replace(" ", "_")
                episode_key = f"demo_{episode_idx}"
                if not all_initial_states[initial_states_task_key][episode_key]["success"]:
                    log_message(f"Skipping task {task_id} episode {episode_idx} due to failed expert demo!", log_file)
                    continue
                initial_state = np.array(all_initial_states[initial_states_task_key][episode_key]["initial_state"])
            log_message(f"\nTask: {task_description}", log_file)
            log_message(f"Starting episode {total_episodes + 1}...", log_file)

            success = _run_one_episode(
                cfg, env, task_description, model, resize_size,
                processor, action_head, proprio_projector, noisy_action_projector,
                initial_state, log_file,
            )

            total_episodes += 1
            if success:
                total_successes += 1

            log_message(f"Success: {success}", log_file)
            log_message(f"# episodes completed so far: {total_episodes}", log_file)
            log_message(f"# successes: {total_successes} ({(total_successes / total_episodes * 100):.1f}%)", log_file)

    final_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0.0

    log_message("Final results:", log_file)
    log_message(f"Total episodes: {total_episodes}", log_file)
    log_message(f"Total successes: {total_successes}", log_file)
    log_message(f"Overall success rate: {final_success_rate:.4f} ({final_success_rate * 100:.1f}%)", log_file)


    if log_file:
        log_file.close()

    return final_success_rate


if __name__ == "__main__":
    eval_libero_prune_v2()


