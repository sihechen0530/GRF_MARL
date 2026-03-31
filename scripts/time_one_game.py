#!/usr/bin/env python3
import sys, time
sys.path.insert(0, '.')

import torch
torch.set_num_threads(1)

from light_malib.algorithm.mappo.policy import MAPPO
from light_malib.envs.gr_football.env import GRFootballEnv
from light_malib.utils.cfg import load_cfg
from light_malib.utils.episode import EpisodeKey

cfg = load_cfg('expr_configs/cooperative_MARL_benchmark/full_game/11_vs_11_hard/mappo.yaml')
cfg['rollout_manager']['worker']['envs'][0]['scenario_config']['render'] = False
env_cfg = cfg.rollout_manager.worker.envs[0]
rollout_length = cfg.rollout_manager.worker.eval_rollout_length
print(f"rollout_length: {rollout_length}", flush=True)

opponent = 'light_malib/trained_models/gr_football/11_vs_11/built_in'
print("Loading models...", flush=True)
p0 = MAPPO.load(opponent, env_agent_id='agent_0'); p0.eval()
p1 = MAPPO.load(opponent, env_agent_id='agent_1'); p1.eval()
print("Creating env...", flush=True)
env = GRFootballEnv(0, None, env_cfg)

custom_reset_config = {
    'feature_encoders': {'agent_0': p0.feature_encoder, 'agent_1': p1.feature_encoder},
    'main_agent_id': 'agent_0',
    'rollout_length': rollout_length,
}

N = 3
print(f"Running {N} games...", flush=True)
t = time.time()
for game in range(N):
    env_rets = env.reset(custom_reset_config)
    sd0 = {**env_rets['agent_0'], **p0.get_initial_state(env.num_players['agent_0'])}
    sd1 = {**env_rets['agent_1'], **p1.get_initial_state(env.num_players['agent_1'])}
    steps = 0
    for _ in range(rollout_length + 1):
        out0 = p0.compute_action(
            inference=True, explore=False, to_numpy=True, no_critic=True,
            **{EpisodeKey.CUR_OBS: sd0[EpisodeKey.NEXT_OBS],
               EpisodeKey.ACTION_MASK: sd0[EpisodeKey.ACTION_MASK],
               EpisodeKey.DONE: sd0[EpisodeKey.DONE],
               EpisodeKey.ACTOR_RNN_STATE: sd0[EpisodeKey.ACTOR_RNN_STATE],
               EpisodeKey.CRITIC_RNN_STATE: sd0[EpisodeKey.CRITIC_RNN_STATE]})
        out1 = p1.compute_action(
            inference=True, explore=False, to_numpy=True, no_critic=True,
            **{EpisodeKey.CUR_OBS: sd1[EpisodeKey.NEXT_OBS],
               EpisodeKey.ACTION_MASK: sd1[EpisodeKey.ACTION_MASK],
               EpisodeKey.DONE: sd1[EpisodeKey.DONE],
               EpisodeKey.ACTOR_RNN_STATE: sd1[EpisodeKey.ACTOR_RNN_STATE],
               EpisodeKey.CRITIC_RNN_STATE: sd1[EpisodeKey.CRITIC_RNN_STATE]})
        env_rets = env.step({
            'agent_0': {EpisodeKey.ACTION: out0[EpisodeKey.ACTION]},
            'agent_1': {EpisodeKey.ACTION: out1[EpisodeKey.ACTION]},
        })
        sd0 = {**env_rets['agent_0'],
               EpisodeKey.ACTOR_RNN_STATE: out0[EpisodeKey.ACTOR_RNN_STATE],
               EpisodeKey.CRITIC_RNN_STATE: sd0[EpisodeKey.CRITIC_RNN_STATE]}
        sd1 = {**env_rets['agent_1'],
               EpisodeKey.ACTOR_RNN_STATE: out1[EpisodeKey.ACTOR_RNN_STATE],
               EpisodeKey.CRITIC_RNN_STATE: sd1[EpisodeKey.CRITIC_RNN_STATE]}
        steps += 1
        if env.is_terminated():
            break
    print(f"  game {game+1}: {steps} steps", flush=True)

elapsed = time.time() - t
per_game = elapsed / N
wall_time = (100 / 16) * per_game

print(f"\nTotal for {N} games: {elapsed:.1f}s")
print(f"Per game:            {per_game:.1f}s")
print(f"Per step:            {per_game/rollout_length*1000:.2f}ms")
print(f"\n100 games / 16 workers ≈ {wall_time:.0f}s  ({wall_time/60:.1f} min)")
