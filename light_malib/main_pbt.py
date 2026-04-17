# Copyright 2022 Digital Brain Laboratory, Yan Song and He jiang
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from light_malib.utils.logger import Logger
import re
import ray
import argparse
from light_malib.utils.cfg import load_cfg, convert_to_easydict
from light_malib.utils.random import set_random_seed
from light_malib.framework.pbt_runner import PBTRunner
import time
import os
import yaml
from omegaconf import OmegaConf

import pathlib

BASE_DIR = str(pathlib.Path(__file__).resolve().parent.parent)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to checkpoint directory to resume training from"
    )
    args = parser.parse_args()
    return args


def get_local_ip_address():
    import socket

    ip_address = socket.gethostbyname(socket.gethostname())
    return ip_address


def start_cluster():
    import os

    ray_temp = os.environ.get("RAY_TMPDIR")
    init_kwargs = {}
    if ray_temp:
        init_kwargs["_temp_dir"] = ray_temp

    # Do not use address="auto" on shared nodes: workers can attach to a foreign Ray that
    # later dies, or raylet sockets go stale → "Could not connect to socket .../raylet".
    Logger.warning("Starting a fresh local Ray cluster (isolated; RAY_TMPDIR if set).")
    cluster_start_info = ray.init(**init_kwargs)

    Logger.warning(
        "============== Cluster Info ==============\n{}".format(cluster_start_info)
    )
    Logger.warning("* cluster resources:\n{}".format(ray.cluster_resources()))
    Logger.warning(
        "this worker ip: {}".format(ray.get_runtime_context().worker.node_ip_address)
    )
    return cluster_start_info


def main():
    args = parse_args()
    cfg = load_cfg(args.config)

    set_random_seed(cfg.seed)

    assert cfg.distributed.nodes.master.ip is not None
    cluster_start_info = start_cluster()

    if cfg.distributed.nodes.master.ip == "auto":
        # ip = get_local_ip_address()
        ip = ray.get_runtime_context().worker.node_ip_address
        cfg.distributed.nodes.master.ip = ip
        Logger.warning("Automatically set master ip to local ip address: {}".format(ip))

    # check cfg
    # check gpu number here
    assert (
        cfg.training_manager.num_trainers <= ray.cluster_resources()["GPU"]
    ), "#trainers({}) should be <= #gpus({})".format(
        cfg.training_manager.num_trainers, ray.cluster_resources()["GPU"]
    )
    # check batch size here
    assert (
        cfg.training_manager.batch_size <= cfg.data_server.table_cfg.capacity
    ), "batch_size({}) should be <= capacity({})".format(
        cfg.training_manager.batch_size, cfg.data_server.table_cfg.capacity
    )
    # check sync_training
    if cfg.framework.sync_training and cfg.framework.get('on_policy', True):
        assert cfg.data_server.table_cfg.sample_max_usage==1
        assert cfg.training_manager.batch_size==cfg.rollout_manager.batch_size
        assert cfg.rollout_manager.worker.sample_length<=0

    # Handle checkpoint resumption
    if args.resume_from is not None:
        # Resume from existing checkpoint
        if not os.path.exists(args.resume_from):
            raise FileNotFoundError(f"Checkpoint directory not found: {args.resume_from}")
        cfg.expr_log_dir = os.path.abspath(args.resume_from)
        Logger.warning(f"Resuming training from checkpoint: {cfg.expr_log_dir}")
        resume_from_checkpoint = True
    else:
        # Create new training run with timestamp
        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        cfg.expr_log_dir = os.path.join(
            cfg.log_dir, cfg.expr_group, cfg.expr_name, timestamp
        )
        cfg.expr_log_dir = os.path.join(BASE_DIR, cfg.expr_log_dir)
        resume_from_checkpoint = False

    os.makedirs(cfg.expr_log_dir, exist_ok=True)

    # Only save config if not resuming (to avoid overwriting)
    if not resume_from_checkpoint:
        yaml_path = os.path.join(cfg.expr_log_dir, "config.yaml")
        with open(yaml_path, "w") as f:
            f.write(OmegaConf.to_yaml(cfg))
            # yaml.dump(OmegaConf.to_yaml(cfg), f, sort_keys=False)
    else:
        # Append resume info to log
        resume_log = os.path.join(cfg.expr_log_dir, "resume_log.txt")
        with open(resume_log, "a") as f:
            f.write(f"\n--- Resumed at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} ---\n")
        
        # Programmatically inject the initial_policies so PBTRunner actually loads the weights.
        # If no valid agent subfolders or checkpoints are found under args.resume_from,
        # we simply skip injection and treat this as fresh training.
        latest_checkpoint_path_for_resume = None
        for pop_idx, pop in enumerate(cfg.populations):
            if "algorithm" in pop and "policy_init_cfg" in pop["algorithm"]:
                policy_init_cfg = pop["algorithm"]["policy_init_cfg"]
                for agent_id, agent_cfg in policy_init_cfg.items():
                    # Construct path directly to the exact policy directory
                    # Structure: args.resume_from / agent_id / agent_id-population_id
                    # E.g., agent_0 / agent_0-default-1
                    # Assuming default population "default-1" per the provided structure
                    policy_parent_dir = os.path.join(
                        args.resume_from, agent_id, f"{agent_id}-default-1"
                    )
                    if not os.path.exists(policy_parent_dir):
                        # No checkpoint folder for this agent; skip and let it train from scratch.
                        continue

                    # Look inside the exact directory for epochs and .last/.best
                    policy_dirs = []
                    epoch_dirs = []

                    for d in os.listdir(policy_parent_dir):
                        dir_path = os.path.join(policy_parent_dir, d)
                        if not os.path.isdir(dir_path):
                            continue

                        if d.endswith(".last"):
                            policy_dirs.append(dir_path)
                        elif d.endswith(".best"):
                            policy_dirs.append(dir_path)
                        elif d.startswith("epoch_"):
                            try:
                                epoch_num = int(d.split("_")[1])
                                epoch_dirs.append((epoch_num, dir_path))
                            except ValueError:
                                pass

                    latest_policy_path = None
                    # Prioritize .last
                    last_dirs = [p for p in policy_dirs if p.endswith(".last")]
                    if last_dirs:
                        latest_policy_path = last_dirs[0]
                    # If no .last, try highest epoch numbers
                    elif epoch_dirs:
                        # Sort by highest epoch num
                        epoch_dirs.sort(key=lambda x: x[0], reverse=True)
                        latest_policy_path = epoch_dirs[0][1]
                    # Fallback to .best
                    elif policy_dirs:
                        latest_policy_path = policy_dirs[0]

                    if latest_policy_path:
                        latest_policy = os.path.basename(latest_policy_path)

                        # Create the configuration block to force loading this policy
                        new_policy_cfg = {
                            "policy_id": latest_policy,
                            "policy_dir": latest_policy_path,
                        }

                        # Set strategy to pretrained to ensure it loads from the dir
                        for init_cfg in agent_cfg.get("init_cfg", []):
                            init_cfg["strategy"] = "pretrained"
                            init_cfg["policy_id"] = latest_policy
                            init_cfg["policy_dir"] = latest_policy_path

                        # Override initial_policies to load our checkpoint
                        agent_cfg["initial_policies"] = [new_policy_cfg]
                        Logger.warning(
                            f"Injected resume config: agent {agent_id} will load {latest_policy_path}"
                        )

                        # Track a checkpoint path to potentially set resume_epoch from.
                        latest_checkpoint_path_for_resume = latest_policy_path

        # Extract epoch number from one of the loaded checkpoint folder names (if any)
        # Supports: "epoch_500", "500.last", "500.best"
        if latest_checkpoint_path_for_resume is not None:
            folder_name = os.path.basename(latest_checkpoint_path_for_resume)
            epoch_match = re.search(r"(\d+)", folder_name)
            if epoch_match:
                resume_epoch = int(epoch_match.group(1))
                cfg.rollout_manager.resume_epoch = resume_epoch
                Logger.warning(f"Rollout epoch will resume from {resume_epoch}")

    cfg = convert_to_easydict(cfg)

    from light_malib.monitor.monitor import Monitor
    from light_malib.utils.distributed import get_resources

    Monitor = ray.remote(**get_resources(cfg.monitor.distributed.resources))(Monitor)
    monitor = Monitor.options(name="Monitor", max_concurrency=5).remote(cfg)

    runner = PBTRunner(cfg)

    try:
        runner.run()
    except KeyboardInterrupt as e:
        Logger.warning(
            "Detected KeyboardInterrupt event, start background resources recycling threads ..."
        )
    finally:
        runner.close()
        ray.get(monitor.close.remote())
        ray.shutdown()


if __name__ == "__main__":
    main()
