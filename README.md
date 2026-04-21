# LVAR

Minimal prototype of a controller-driven latent visual active reasoning loop on top of `Qwen/Qwen2-VL-2B-Instruct`.

## What is included

- A single main model file with the recurrent LVAR loop
- CLEVR CoGenT dataset utilities
- Reward helpers for correctness and delta reward
- Inference, debug, and custom GRPO-style training scripts
- Unit tests that exercise the core logic without downloading Qwen weights

## Quick start

1. Install dependencies from `requirements.txt`.
2. Review or edit `configs/qwen2vl_lvar.yaml`.
3. Run a single-example debug pass:

```bash
python scripts/debug_single.py --config configs/qwen2vl_lvar.yaml
```

4. Run inference and write JSONL results:

```bash
python scripts/infer_clevr.py --config configs/qwen2vl_lvar.yaml --limit 10
```

5. Run the unit tests:

```bash
python -m unittest discover -s tests -v
```

## Notes

- The Qwen backbone is frozen in v1.
- The trainable parameters are limited to the LVAR-specific additions.
- The final act-token drop keeps the earlier computation graph intact by slicing and concatenating existing tensors without detaching them.
