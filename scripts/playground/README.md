# Relighting composition demo

Run this command from the repository root after replacing the
`path/to/data` placeholders with local paths:

```bash
python scripts/playground/demo_compose_relighting.py \
  --base-checkpoint path/to/data/kitchen_ckpt_last.pt \
  --object-checkpoint path/to/data/chair_ckpt_last_scaled.pt \
  --dataset path/to/data/kitchen \
  --sequence scripts/playground/demo_seq \
  --out-dir path/to/data/demo_output \
  --composed-checkpoint path/to/data/demo_output/kitchen_chair_composed.pt \
  --spp 1024
```
