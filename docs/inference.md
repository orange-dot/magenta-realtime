# Inference

**JAX:**
```bash
# Generate 4 seconds of audio
mrt jax generate
```

**MLX:**
```bash
# Generate 4 seconds of audio
mrt mlx generate --bits=8
```

For Linux CPU-only MLX smoke tests, use `mrt2_small`, the raw checkpoint path,
and a short duration:

```bash
mrt models download mrt2_small
mrt checkpoints download mrt2_small.safetensors
mrt mlx generate \
  --device cpu \
  --no-mlxfn \
  --model=mrt2_small \
  --bits=8 \
  --warmup-steps=0 \
  --duration=0.04
```

Exported `.mlxfn` models are not supported on MLX CPU; use `--no-mlxfn`.

To print MusicCoCa tokens for a prompt directly without generating audio:

```python
from magenta_rt.musiccoca import MusicCoCa
m = MusicCoCa()
print(m.tokenize(m.embed('a jazz piano trio')).tolist())

# Get tokens from audio
from magenta_rt.audio import Waveform
wav = Waveform.from_file("jazz_piano_trio.wav")
print(m.tokenize(m.embed(wav)).tolist())
```

## Bulk generation

Bulk-generate 60s audio clips from MusicCoCa prompts for listener evaluation:

```bash
python scripts/bulk_generate.py --size=mrt2_base
```

Outputs are saved to `outputs/eval_audio/<size>/`.
