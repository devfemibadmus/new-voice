import argparse
from pathlib import Path

from TTS.api import TTS


MODEL_PATTERNS = (
    "best_model*.pth",
    "model*.pth",
    "checkpoint*.pth",
    "*.pth",
    "*.pt",
    "*.bin",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Synthesize speech from a local Coqui TTS checkpoint.")
    parser.add_argument("--text", default="Hello, this is my custom voice!", help="Text to synthesize.")
    parser.add_argument("--output-path", default="output.wav", help="Destination WAV file.")
    parser.add_argument("--output-dir", default="output", help="Root output directory that contains run folders.")
    parser.add_argument("--run-dir", help="Specific run directory to load from.")
    parser.add_argument("--model-path", help="Explicit checkpoint path. Overrides --run-dir discovery.")
    parser.add_argument("--config-path", help="Explicit config.json path. Overrides automatic discovery.")
    parser.add_argument("--vocoder-path", help="Optional local vocoder checkpoint path.")
    parser.add_argument("--vocoder-config-path", help="Optional local vocoder config path.")
    parser.add_argument("--use-cuda", action="store_true", help="Move the synthesizer to CUDA after loading.")
    return parser.parse_args()


def latest_run_dir(output_dir):
    root = Path(output_dir)
    if not root.exists():
        raise FileNotFoundError(f"Output directory not found: {root}")

    run_dirs = [path for path in root.iterdir() if path.is_dir() and path.name.startswith("run-")]
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found in {root}")
    return max(run_dirs, key=lambda path: path.stat().st_mtime)


def find_checkpoint(run_dir):
    candidates = []
    for pattern in MODEL_PATTERNS:
        candidates.extend(run_dir.glob(pattern))

    candidates = [path for path in candidates if path.is_file()]
    if not candidates:
        raise FileNotFoundError(f"No checkpoint file found in {run_dir}")

    preferred_names = ("best_model", "model", "checkpoint")
    candidates.sort(
        key=lambda path: (
            0 if any(path.name.startswith(prefix) for prefix in preferred_names) else 1,
            -path.stat().st_mtime,
        )
    )
    return candidates[0]


def find_config(model_path, explicit_config_path=None):
    if explicit_config_path:
        config_path = Path(explicit_config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        return config_path

    local_config = model_path.parent / "config.json"
    if local_config.exists():
        return local_config

    parent_config = model_path.parent.parent / "config.json"
    if parent_config.exists():
        return parent_config

    raise FileNotFoundError(f"Could not find config.json near {model_path}")


def resolve_paths(args):
    if args.model_path:
        model_path = Path(args.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    else:
        run_dir = Path(args.run_dir) if args.run_dir else latest_run_dir(args.output_dir)
        model_path = find_checkpoint(run_dir)

    config_path = find_config(model_path, args.config_path)
    return model_path, config_path


def main():
    args = parse_args()
    model_path, config_path = resolve_paths(args)

    tts = TTS(
        model_path=str(model_path),
        config_path=str(config_path),
        vocoder_path=args.vocoder_path,
        vocoder_config_path=args.vocoder_config_path,
    )
    if args.use_cuda:
        tts = tts.to("cuda")

    output_path = Path(args.output_path)
    tts.tts_to_file(text=args.text, file_path=str(output_path))
    print(f"Loaded model: {model_path}")
    print(f"Loaded config: {config_path}")
    print(f"Speech saved to {output_path}")


if __name__ == "__main__":
    main()
