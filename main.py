import argparse
from pathlib import Path

from TTS.api import TTS
from TTS.utils.generic_utils import get_user_data_dir
from TTS.utils.manage import ModelManager


MODEL_PATTERNS = (
    "best_model*.pth",
    "model*.pth",
    "checkpoint*.pth",
    "*.pth",
    "*.pt",
    "*.bin",
)
DEFAULT_TTS_MODEL_NAME = "tts_models/en/ljspeech/glow-tts"
DEFAULT_VOCODER_MODEL_NAME = "vocoder_models/en/ljspeech/hifigan_v2"


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
    parser.add_argument(
        "--tts-model-name",
        default=DEFAULT_TTS_MODEL_NAME,
        help=f"Default pretrained TTS model to use if no local run or explicit checkpoint is provided. Default: {DEFAULT_TTS_MODEL_NAME}",
    )
    parser.add_argument(
        "--vocoder-model-name",
        default=DEFAULT_VOCODER_MODEL_NAME,
        help=f"Default vocoder model to download if no local vocoder path is provided. Default: {DEFAULT_VOCODER_MODEL_NAME}",
    )
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


def download_model_dir(model_name):
    print(f"Downloading or locating model: {model_name}")
    ModelManager().download_model(model_name)
    model_dir = Path(get_user_data_dir("tts")) / model_name.replace("/", "--")
    if not model_dir.exists():
        raise FileNotFoundError(f"Downloaded model directory not found for {model_name}: {model_dir}")
    return model_dir


def resolve_paths(args):
    if args.model_path:
        model_path = Path(args.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    else:
        try:
            run_dir = Path(args.run_dir) if args.run_dir else latest_run_dir(args.output_dir)
            model_path = find_checkpoint(run_dir)
        except FileNotFoundError:
            model_dir = download_model_dir(args.tts_model_name)
            model_path = find_checkpoint(model_dir)

    config_path = find_config(model_path, args.config_path)
    return model_path, config_path


def resolve_vocoder_paths(args):
    if args.vocoder_path and args.vocoder_config_path:
        return args.vocoder_path, args.vocoder_config_path

    if args.vocoder_path or args.vocoder_config_path:
        raise FileNotFoundError("Provide both --vocoder-path and --vocoder-config-path, or neither.")

    model_dir = download_model_dir(args.vocoder_model_name)
    vocoder_path = find_checkpoint(model_dir)
    vocoder_config_path = find_config(vocoder_path)
    return str(vocoder_path), str(vocoder_config_path)


def main():
    args = parse_args()
    model_path, config_path = resolve_paths(args)
    vocoder_path, vocoder_config_path = resolve_vocoder_paths(args)

    tts = TTS(
        model_path=str(model_path),
        config_path=str(config_path),
        vocoder_path=vocoder_path,
        vocoder_config_path=vocoder_config_path,
    )
    if args.use_cuda:
        tts = tts.to("cuda")

    output_path = Path(args.output_path)
    tts.tts_to_file(text=args.text, file_path=str(output_path))
    print(f"Loaded model: {model_path}")
    print(f"Loaded config: {config_path}")
    print(f"Loaded vocoder: {vocoder_path}")
    print(f"Loaded vocoder config: {vocoder_config_path}")
    print(f"Speech saved to {output_path}")


if __name__ == "__main__":
    main()
