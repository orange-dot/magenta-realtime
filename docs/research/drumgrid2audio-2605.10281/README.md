# DrumGrid2Audio Research Notes

Local note for arXiv:2605.10281, "Drum Synthesis from Expressive
Drum Grids via Neural Audio Codecs", and its cited papers. PDFs are in
`papers/`.

## Why This Matters Here

The paper is directly relevant to a narrow Magenta RealTime extension:
rendering an explicit, expressive drum grid into audio by predicting neural
codec tokens. MRT2 already has Python JAX/MLX inference paths and local Linux
CPU smoke support for `mrt2_small`; this research suggests a separate offline
drum-rendering lane rather than a replacement for the main prompt-conditioned
music model.

The important boundary: this is not preference ranking and not codec
reconstruction training. It is supervised conditional token prediction:

`MIDI-derived drum grid -> Transformer -> codec tokens -> fixed codec decoder`.

## Lab Integration Map

This repository is part of the wider local lab under
`/home/dev/work-base-20260421`. In that lab, the relevant current anchors are:

- `workspace/systems/aig-engine`: AIG/ADG semantic authority, `GesturePacket`,
  `AtomSpec`, deterministic reference rendering, sample-material rendering, and
  EnCodec boundary experiments.
- `workspace/systems/pc4-microkit-studio`: the PC4MS drum-engine source of live
  drummer intent, generated ADG bundles, Drum Studio runtime, and AIG export
  requests.
- `workspace/systems/audio-atom-model-lab`: Rust-first bounded atom codec lab
  for kick/snare/hat material and EnCodec comparison.
- `workspace/systems/encodec`: local EnCodec fork for AIG/offline codec work.
- `workspace/systems/magenta-realtime`: MRT2 inference, Linux MLX/JAX work, and
  the current lossy AIG-to-MRT2 conditioning bridge.

The fit is strongest if DrumGrid2Audio is treated as a phrase/window-level
neural drum renderer below AIG/ADG authority:

```text
PC4MS drum-engine / ADG bundle
  -> AIG import
  -> GesturePacket / resolved AIG / AtomSpec
  -> DrumGrid2Audio-style expressive grid cache
  -> EnCodec-token predictor
  -> fixed codec decoder
  -> drum phrase or stem WAV + metrics + trace
```

That lane complements, rather than replaces, the existing atom lane:

```text
GesturePacket / AtomSpec
  -> short atom material backend
  -> AIG scheduler and mixer
```

The atom lane is better for realtime-bounded material and per-hit traceability.
The DrumGrid2Audio lane is better for learning bar-level/phrase-level acoustic
context: cymbal wash continuity, kit identity, room/tail behavior, and the
interaction between nearby hits.

## Current Local Evidence And Gaps

The lab already has partial evidence around this shape:

- `aig-engine/docs/AIG-MAGENTA-GMD-EGMD-DRUM-LANE.md` already identifies
  GMD/E-GMD as the right open drum corpora and recommends a drum-first
  code model before full MRT2 fine-tuning.
- `magenta_rt/aig_bridge.py` converts AIG `GesturePacket` artifacts to MRT2's
  existing 25 Hz note/drum conditioning lanes and explicitly records lost ADG
  fields such as body, transient, density, protect flags, phrase role, tuplets,
  relationship identity, and state writes.
- PC4MS `crates/drum-engine/src/types.rs` exposes `AdgGesture` fields that map
  naturally into an expressive drum grid: role, kind, tick, strength, duration,
  body, transient, openness, density, micro-offset, velocity delta, protect
  flags, phrase role, variation seed, surface touch, tuplets, and reason.
- PC4MS `drum-studio-runtime` writes `pc4ms.aig_export_request.v1` with
  `rhythm = pc4ms-drum-engine`, `runtime = pc4ms-drum-studio-runtime`,
  `material = aig-engine`, and `codec_training = encodec-offline`.
- Existing runs under `runs/aig-magenta-jax-v0/` show MRT2 conditioning probes,
  including quantization errors and lossy-field reports.
- Existing runs under `runs/aig-phrase-e2e/` and
  `runs/aig-external-drum-corpus-v0/` show atom manifests, EnCodec roundtrip
  manifests, generated atom manifests, and phrase assembly reports.

The main gap is still training evidence. The current lab can export and compare
semantic drum artifacts, but it does not yet have a committed DrumGrid2Audio
dataset builder, token target cache, model training command, checkpoint, or
evaluation report.

## Where The Papers Land In The Lab

DrumGrid2Audio [Main] should drive the next drum-neural training design. It is
the closest published pattern to the desired local shape: explicit drum intent
in, codec-token audio out, objective onset/audio metrics, and a controlled
tokenizer comparison.

E-GMD [3] is the best public bootstrap corpus. It can seed the cache format,
lane mapping, metadata model, attribution discipline, and evaluation split. It
does not replace local PC4MS/ADG material, but it gives enough aligned
MIDI/audio to debug the learning problem before relying on small private runs.

GrooVAE/GMD [8] supports the symbolic side: humanization, velocity, timing,
Tap2Drum, and drummer identity. It belongs above the renderer, close to
PC4MS/AIG gesture generation.

EnCodec/SoundStream [4, 11] define the first token target. The primary paper's
results reinforce the current lab instinct to use EnCodec before DAC/X-Codec
for a first code-prediction model.

DAC/TRIA [6, 15] remain useful later. DAC gives high-fidelity 44.1 kHz tokens
and TRIA shows masked modeling can work for drum gestures and timbre prompts,
but DrumGrid2Audio found DAC harder to predict from grids. That makes DAC a
second-phase comparison, not the first target.

X-Codec [5] is conceptually important because it asks whether codec tokens are
good for language modeling, not just compression. In the primary drum-grid
paper, however, it underperforms EnCodec for this supervised drum renderer.

STAGE/DARC [13, 14] are product-adjacent references for drum/accompaniment
generation with rhythm control. They are less aligned with this lab's authority
model because their primary controls are audio context/rhythm prompts rather
than inspectable ADG source truth.

FAD/FAD-infinity [16, 17] should inform evaluation reporting, especially sample
size, reference set, embedding model, and outlier handling. The lab should never
use one FAD number as an audio-quality claim.

## Primary Paper Summary

DrumGrid2Audio trains a non-autoregressive Transformer to map 4-beat expressive
drum grids to audio codec token sequences. The waveform is produced by a fixed,
pretrained codec decoder. The authors compare three tokenizers under a common
training/evaluation setup:

- EnCodec: `facebook/encodec_32khz`, 4 RVQ codebooks, codebook size 2048.
- DAC: `descript/dac_44khz`, 9 RVQ codebooks, codebook size 1024.
- X-Codec: `hf-audio/xcodec-hubert-general`, semantic-augmented codec, 4
  codebooks at 2 kbps in their setup.

Training data comes from E-GMD: 444 hours of aligned drum MIDI/audio across 43
kits, with human timing and velocity annotations. They cut performances into
beat-aligned 4-beat windows, cache the aligned MIDI grid and codec targets, then
train one predictor per codec family.

The grid has two key per-drum-lane features:

- `drum_hit[d,t]`: Gaussian onset activity around each MIDI onset frame.
- `drum_vel[d,t]`: normalized MIDI velocity at onset frames.

The model also gets beat position modulo 4, log-scaled BPM, drummer ID, and
kit ID for all-kit experiments. The target is `tgt[c,t]`, the codec token index
for each codebook and frame.

## Model And Training Details

All models use the same bidirectional Transformer encoder and predict all frames
non-autoregressively. The loss is mean cross-entropy over codebooks and frames,
ignoring PAD tokens. Since beat-synchronous windows vary in seconds, batches are
padded and masked.

Capacity settings:

- Base: `d_model=768`, 6 layers, 8 heads, FF multiplier 4, dropout 0.1.
- Large: `d_model=1536`, 10 layers, 12 heads, FF multiplier 4, dropout 0.1.

Training uses AdamW, learning rate `6e-5`, gradient clipping at 1.0, validation
every 300 steps, early stopping after 5000 steps without validation NLL
improvement, batch size 24 for Base and 8 for Large. Reported training hardware
is one RTX 3080 with 10 GB VRAM.

## Results To Remember

EnCodec is the practical winner in this setup. It has the lowest token NLL/PPL,
highest token accuracy, best MR-STFT spectral convergence, strongest RMS-envelope
correlation, and best FAD in the main Base evaluations.

Base all-kits headline numbers:

| Codec | NLL | PPL | Acc | MR-STFT | Env corr | Onset F1 | FAD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| EnCodec | 2.153 | 11.6 | 43.4% | 0.827 | 0.710 | 70.6% | 0.193 |
| X-Codec | 4.429 | 104.9 | 12.5% | 1.669 | 0.568 | 68.8% | 0.277 |
| DAC | 6.153 | 521.8 | 4.7% | 1.034 | 0.602 | 69.3% | 0.405 |

Large models degrade across codecs. The paper treats this as optimization
instability, not evidence that larger capacity is impossible. For our purposes,
start smaller and make the data/cache/metrics boringly reliable before scaling.

The authors hypothesize that EnCodec is easier because its token stream is more
constrained and redundant on drum windows: lower effective entropy and slower
token turnover than DAC or X-Codec. That is a useful diagnostic to reproduce
locally before committing to a tokenizer.

Important caveat: the paper does not include a listening study or formal
statistical testing. Treat the metric differences as descriptive. For product or
musical claims, subjective listening remains required.

## Metrics Worth Reusing

A local prototype should not rely on one score. The paper's metric set is a good
starting gate:

- Token metrics: NLL, perplexity, token accuracy, PAD ignored.
- Audio metrics: RMSE, MAE, multi-resolution STFT spectral convergence,
  RMS-envelope correlation, transient-to-tail energy ratio error.
- Rhythm alignment: onset precision/recall/F1 against grid-derived onsets,
  with about 50 ms tolerance.
- Distribution/perceptual metric: FAD or FAD-infinity using a music-relevant
  embedding and a consistent reference set.
- Human check: listening comparisons after objective smoke passes.

The FAD references are especially relevant: plain FAD can be biased by sample
size, embedding choice, and reference set quality. If we use it locally, log the
embedding model, reference set, sample count, sample duration, and whether it is
plain FAD or FAD-infinity.

## Reference Map

All referenced papers were downloaded locally:

| Ref | Local PDF | Role |
| --- | --- | --- |
| [1] | `papers/2209.03143-audiolm.pdf` | Establishes audio generation as language modeling over semantic/acoustic tokens. |
| [2] | `papers/2301.11325-musiclm.pdf` | Text-to-music over hierarchical audio tokens; useful context for token-based music generation. |
| [3] | `papers/2004.00188-expanded-groove-midi-dataset.pdf` | E-GMD source: 444h drum MIDI/audio, 43 kits, velocity labels, perceptual warning about classifier metrics. |
| [4] | `papers/2210.13438-encodec.pdf` | Real-time neural audio codec; RVQ tokens and decoder used as the strongest target in DrumGrid2Audio. |
| [5] | `papers/2408.17175-xcodec-codec-does-matter.pdf` | Semantic-augmented codec; good reminder that compression-optimal tokens are not always LM-optimal. |
| [6] | `papers/2306.06546-dac-improved-rvqgan.pdf` | High-fidelity universal DAC codec; attractive reconstruction target but harder token space here. |
| [7] | `papers/2008.12073-drumgan.pdf` | One-shot drum timbre generation with perceptual feature control; not groove rendering. |
| [8] | `papers/1905.06118-groovae.pdf` | Symbolic drum humanization, infilling, Tap2Drum, and the original GMD lineage. |
| [9] | `papers/2206.05408-spectrogram-diffusion.pdf` | MIDI-to-spectrogram-to-audio neural synthesis; alternative to codec tokens. |
| [10] | `papers/2106.07431-crash.pdf` | Raw-audio diffusion for short drum sounds; useful for sound design, less direct for full grooves. |
| [11] | `papers/2107.03312-soundstream.pdf` | End-to-end RVQ neural audio codec lineage that EnCodec/MusicLM build on. |
| [12] | `papers/2507.08530-midi-valle.pdf` | Symbolic performance MIDI to audio via neural codec LM; close precedent outside drums. |
| [13] | `papers/2504.05690-stage.pdf` | Stem accompaniment using MusicGen/EnCodec prefix conditioning; rhythm conditioning via audio context. |
| [14] | `papers/2601.02357-darc.pdf` | Drum accompaniment with explicit rhythm prompts and musical context; useful contrast to grid conditioning. |
| [15] | `papers/2509.15625-tria.pdf` | Audio-prompted drums with DAC masked LM; strong for gesture-to-drum audio and timbre prompts. |
| [16] | `papers/1812.08466-fad.pdf` | Original FAD metric, reference-free distribution distance over audio embeddings. |
| [17] | `papers/2311.01616-adapting-fad.pdf` | FAD for generative music, sample-size bias, embedding/reference-set caveats, fadtk. |
| Main | `papers/2605.10281-drumgrid2audio.pdf` | Expressive drum grid to codec token renderer. |

## Relation To Prior Work

The primary paper sits between two older lines:

- Symbolic groove modeling: GrooVAE learns humanized MIDI timing and velocity,
  but does not synthesize waveform audio.
- Neural audio generation: AudioLM, MusicLM, MusicGen-like systems generate
  codec tokens, but often use text/audio prompts rather than explicit drum grids.

DrumGrid2Audio's useful contribution is the explicit supervised bridge from
expressive MIDI-like control to codec-token audio. It is narrower than full
music generation, but that narrowness is a strength for local engineering: data
alignment, metrics, and failure analysis are tractable.

TRIA and DARC are nearby but different. TRIA maps arbitrary rhythmic sound
gestures plus timbre prompts to drum audio using DAC masked modeling. DARC maps
musical context plus rhythm prompts to drum stems. DrumGrid2Audio is more
deterministic and editor-friendly: it starts from an inspectable grid with lanes,
velocity, microtiming, tempo, drummer, and kit metadata.

## Implications For Local Work

1. Start with EnCodec-like or existing repo codec paths before DAC/X-Codec.
   The paper's result suggests that "higher-fidelity codec" does not imply
   "easier conditional token modeling."

2. Keep the first prototype offline. The method is not proven as a real-time
   renderer, and the main repo's real-time guarantees are centered on MRT2
   streaming, not a new drum-grid renderer.

3. Build a fixed cache first. Every experiment should read the same window IDs,
   aligned audio slices, grid tensors, codec tokens, BPM, kit ID, drummer ID,
   and split labels. Without this, codec comparisons become noisy.

4. Reproduce token entropy/turnover diagnostics. Before training, compare token
   entropy, codebook usage, and frame-to-frame token changes for candidate
   tokenizers on drum windows.

5. Do not scale too early. A small or Base-like model with stable masking,
   padding, and validation NLL is more valuable than a larger model that
   collapses onset alignment or envelope correlation.

6. Treat objective metrics as gates, not proof. Onset F1, MR-STFT, envelope
   correlation, and FAD can catch obvious failures. They do not replace listening.

## Suggested Local Prototype Shape

For this repo, a conservative first slice would be:

1. Add a research-only cache builder that accepts both E-GMD/GMD examples and
   local PC4MS `*.adg-bundle` / AIG `GesturePacket` artifacts.
2. Export a DrumGrid2Audio-style grid from AIG/ADG: role lanes, onset Gaussian,
   velocity/strength-at-onset, beat phase, BPM, kit/material ID, and optional
   continuous lanes for body, transient, openness, and density.
3. Pick one token target first. Use EnCodec-compatible infrastructure before
   comparing DAC/X-Codec/SpectroStream.
4. Run no-training diagnostics: token entropy, codebook usage, frame-to-frame
   turnover, ADG-to-grid loss report, and comparison against the existing 25 Hz
   MRT2 conditioning quantization report.
5. Train a small offline Transformer token predictor, separate from MRT2
   prompt-generation code and separate from the AIG material ranker.
6. Decode generated tokens to WAV and emit a report with token metrics, onset
   F1 against the ADG/E-GMD grid, MR-STFT, envelope correlation, FAD config,
   existing AIG analyzer checks, and a short listening checklist.
7. Compare four baselines on the same source bundle: deterministic AIG render,
   sample-material/Virtuosity render, current lossy MRT2 bridge render, and
   DrumGrid2Audio-style codec-token render.
8. Only after the smoke lane works, test whether the phrase renderer should feed
   MRT2 workflows, stay as an AIG neural backend, or split into atom and phrase
   modes.

Do not label this as "quality render" until the AIG claims ledger has concrete
training, evaluation, and listening evidence. The correct early label is closer
to "experimental grid-to-codec drum renderer."

## Open Questions

- Can MRT2's existing SpectroStream resources serve as a practical tokenizer for
  this narrow drum-rendering task, or is EnCodec still the better first target?
- Do local drum assets have enough paired MIDI/audio with reliable microtiming
  and velocity to train anything beyond a toy renderer?
- How much kit identity should be explicit metadata versus inferred from audio
  prompt/context?
- Should the first useful tool be "render this explicit grid" or "humanize this
  quantized grid, then render"?
- What latency is acceptable? Offline rendering is easy to justify; real-time
  performance would need a separate benchmark.

## Source Links

- Primary paper: https://arxiv.org/abs/2605.10281
- Primary PDF: https://arxiv.org/pdf/2605.10281
- Project page mentioned by the paper: https://github.com/kostantinos-soiledis/midigroove_poc
