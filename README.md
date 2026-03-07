# New Voice

Local Coqui TTS fine-tuning workflow for a single speaker dataset built from multiple recordings and transcripts.

## What It Does

This repo does 3 things:

1. Builds a combined dataset from folders of recordings and transcripts
2. Resumes dataset prep if it stops halfway through
3. Fine-tunes a pretrained Coqui TTS model and generates speech from the result

## Inputs

Put your files in folders:

- `data/voices/`
- `data/transcripts/`

Each recording should have a matching transcript.

Supported recording formats:

- `.wav`
- `.mp3`
- `.mp4`
- `.m4a`
- `.flac`
- `.ogg`
- `.aac`

Supported transcript formats:

- `.txt`
- `.pdf`

If file stems match, setup pairs them by name.

Example:

- `data/voices/clip01.wav`
- `data/transcripts/clip01.txt`

If stems do not match, setup falls back to sorted-order pairing, but matching names are the safer option.

## Requirements

Install Python packages from [requirements.txt](/c:/Users/Femi.Badmus/Desktop/new-voice/requirements.txt).

You also need:

- Gentle forced aligner running at `http://localhost:8765`
- A pretrained Coqui model checkpoint and matching `config.json`
- `ffmpeg` only if you use compressed audio formats like `.mp3` or `.mp4`

Note: `TTS==0.22.0` is safer on Python 3.10 or 3.11 than on newer Python versions.

## Install

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Step 1: Prepare Dataset

Run:

```powershell
python setup.py
```

Or with explicit folders:

```powershell
python setup.py --voice-path data\voices --transcript-path data\transcripts
```

What setup does:

1. Verifies the number of recordings matches the number of transcripts
2. Pairs files by stem when possible
3. Processes each pair into `generated/items/<item_id>/`
4. Writes all clip segments to `generated/wavs/`
5. Rebuilds one combined `generated/dataset.csv`
6. Tracks progress in `generated/progress.json`

If setup stops halfway through, run it again without `--force` and it resumes from the next unfinished item.

If you want to start all over from scratch:

```powershell
python setup.py --force
```

That deletes the current `generated/` contents and rebuilds everything.

## Generated Layout

After setup runs, you will have:

- `generated/progress.json`
- `generated/dataset.csv`
- `generated/wavs/*.wav`
- `generated/items/<item_id>/recording.wav`
- `generated/items/<item_id>/extracted_text.txt`
- `generated/items/<item_id>/alignment.json`
- `generated/items/<item_id>/dataset.csv`

## Step 2: Fine-Tune

You need a pretrained model first:

- `config.json`
- `model.pth` or another valid checkpoint file

Then run:

```powershell
python train.py `
  --config-path C:\path\to\config.json `
  --restore-path C:\path\to\model.pth `
  --run-name my-voice
```

Useful optional flags:

- `--epochs 50`
- `--batch-size 8`
- `--learning-rate 1e-5`
- `--output-dir output`
- `--no-mixed-precision`

Training outputs go into `output/`.

## Step 3: Generate Speech

After training finishes:

```powershell
python main.py `
  --run-dir output\run-... `
  --text "This is a test of my fine tuned voice." `
  --output-path demo.wav
```

You can also point directly at a checkpoint:

```powershell
python main.py `
  --model-path C:\path\to\best_model.pth `
  --config-path C:\path\to\config.json `
  --text "Hello world"
```

## Current Limitation

The current preprocessing still relies on Gentle word alignment and slices clips from aligned words. The batch/resume flow is now better, but voice quality will still be limited if your dataset remains mostly word-level.

For stronger results, longer phrase-level or sentence-level segmentation is the next improvement.

## Git Notes

These paths are intentionally ignored and removed from history:

- `generated/`
- `output/`
- `__pycache__/`

