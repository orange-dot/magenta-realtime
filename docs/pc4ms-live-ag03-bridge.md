# PC4MS Live ADG To AG03 Preview

Status: local lab smoke path. PC4MS remains the live drum authority. Magenta RT
is a semantic preview renderer below PC4MS ADG.

## Audio Contract

- PC4MS publishes `pc4ms.magenta_live_chunk.v1` JSON datagrams when
  `PC4MS_MAGENTA_LIVE_SOCKET` is set.
- Magenta/JAX writes `s16_interleaved_stereo` frames to a file-backed ring.
- `mrt-alsa-host` reads S16 ring frames and converts them for Yamaha playback.
- Yamaha AG03/AG06 direct playback should use exact 48 kHz stereo with
  `--format auto`, which tries `S32_LE`, `S24_LE`, `S24_3LE`, then `S16_LE`.

## Processes

Start the Magenta sidecar:

```bash
mrt pc4ms live-daemon \
  --socket /tmp/pc4ms-magenta-live.sock \
  --ring /tmp/mrt2-pc4ms-live.ring \
  --chunk-frames 8 \
  --ring-backpressure \
  --metrics-log outputs/pc4ms-live-ag03/daemon.jsonl
```

Ring backpressure is enabled by default. Keep it enabled for live playback so a
fast JAX/CUDA producer waits for the ALSA host instead of overwriting queued
audio. Use `--no-ring-backpressure` only for producer-ceiling benchmarks.

Start the ALSA host:

```bash
cargo run --manifest-path tools/mrt-alsa-host/Cargo.toml -- \
  --ring /tmp/mrt2-pc4ms-live.ring \
  --device hw:CARD=AG06AG03,DEV=0 \
  --format auto \
  --start-threshold-frames 30720 \
  --duration-seconds 120
```

Start PC4MS live with publishing enabled:

```bash
PC4MS_MAGENTA_LIVE_SOCKET=/tmp/pc4ms-magenta-live.sock \
  cargo run --locked -p pc4ms-workbench
```

## Acceptance Notes

- `chunk=8` is the stable reference mode.
- `chunk=4` is only a low-latency candidate until a longer AG03 run has no
  post-priming ALSA xrun or ring underrun.
- Do not describe this as a production renderer: the AIG/PC4MS role here is the
  semantic bridge, and Magenta is only the local neural preview sink.
