# Generated Φ modules

`scripts/generate_phi_llm.py` writes LLM-produced `phi(state, role)` here by default.

**Import path** (with `PYTHONPATH` = repo root, as in `train.slurm`):

```text
generated_phi.phi_llm
```

Set in yaml (see `expr_configs/.../counterattack_easy/ippo_eureka_llm.yaml`):

```yaml
potential_shaping:
  phi_module: generated_phi.phi_llm
```

Do not commit API keys. Generated `.py` files may be gitignored by team policy.
