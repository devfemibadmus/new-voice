import argparse
import csv
import statistics
import wave
from pathlib import Path

from pydub import AudioSegment
from trainer import Trainer, TrainerArgs
from TTS.config import load_config
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models import setup_model as setup_tts_model
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor
from TTS.utils.generic_utils import get_user_data_dir
from TTS.utils.manage import ModelManager


DEFAULT_DATASET_DIR = Path("generated")
DEFAULT_PREPARED_DATASET_DIR = Path("generated_finetune")
DEFAULT_OUTPUT_DIR = Path("output")
DEFAULT_TTS_MODEL_NAME = "tts_models/en/ljspeech/glow-tts"
MODEL_PATTERNS = ("best_model*.pth", "model*.pth", "checkpoint*.pth", "*.pth", "*.pt", "*.bin")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune a pretrained Coqui TTS model on a local single-speaker dataset."
    )
    parser.add_argument("--config-path", help="Path to the pretrained model config.json.")
    parser.add_argument("--restore-path", help="Path to the pretrained model checkpoint.")
    parser.add_argument(
        "--model-name",
        default=DEFAULT_TTS_MODEL_NAME,
        help=f"Default pretrained TTS model to download when paths are not provided. Default: {DEFAULT_TTS_MODEL_NAME}",
    )
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR), help="Directory containing dataset.csv and wavs/.")
    parser.add_argument(
        "--metadata-source",
        default="dataset.csv",
        help="Source metadata file inside --dataset-dir. Supports id|speaker|text or LJSpeech rows.",
    )
    parser.add_argument(
        "--prepared-dataset-dir",
        default=str(DEFAULT_PREPARED_DATASET_DIR),
        help="Directory to write normalized wavs and LJSpeech metadata for fine-tuning.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for training outputs.")
    parser.add_argument("--run-name", default="custom-voice-finetune", help="Run name inside the output directory.")
    parser.add_argument("--language", default="en", help="Dataset language code.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of fine-tuning epochs.")
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size.")
    parser.add_argument("--eval-batch-size", type=int, default=8, help="Evaluation batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-5, help="Fine-tuning learning rate.")
    parser.add_argument("--save-step", type=int, default=500, help="Checkpoint save interval.")
    parser.add_argument("--print-step", type=int, default=25, help="Log interval.")
    parser.add_argument("--num-loader-workers", type=int, default=2, help="Training dataloader workers.")
    parser.add_argument("--num-eval-loader-workers", type=int, default=2, help="Eval dataloader workers.")
    parser.add_argument("--min-duration-ms", type=int, default=0, help="Skip clips shorter than this duration.")
    parser.add_argument("--min-text-length", type=int, default=1, help="Skip clips with text shorter than this.")
    parser.add_argument("--max-samples", type=int, default=0, help="Optional cap on prepared samples. 0 disables it.")
    parser.add_argument("--no-mixed-precision", action="store_true", help="Disable mixed precision.")
    return parser.parse_args()


def get_value(obj, key, default=None):
    if hasattr(obj, key):
        return getattr(obj, key)
    if isinstance(obj, dict):
        return obj.get(key, default)
    return default


def find_checkpoint(model_dir):
    candidates = []
    for pattern in MODEL_PATTERNS:
        candidates.extend(model_dir.glob(pattern))

    candidates = [path for path in candidates if path.is_file()]
    if not candidates:
        raise FileNotFoundError(f"No checkpoint file found in {model_dir}")

    preferred_names = ("best_model", "model", "checkpoint")
    candidates.sort(
        key=lambda path: (
            0 if any(path.name.startswith(prefix) for prefix in preferred_names) else 1,
            -path.stat().st_mtime,
        )
    )
    return candidates[0]


def download_model_dir(model_name):
    print(f"Downloading or locating pretrained model: {model_name}")
    ModelManager().download_model(model_name)
    model_dir = Path(get_user_data_dir("tts")) / model_name.replace("/", "--")
    if not model_dir.exists():
        raise FileNotFoundError(f"Downloaded model directory not found for {model_name}: {model_dir}")
    return model_dir


def resolve_pretrained_paths(args):
    if args.config_path and args.restore_path:
        return Path(args.config_path), Path(args.restore_path)

    model_dir = download_model_dir(args.model_name)
    config_path = Path(args.config_path) if args.config_path else model_dir / "config.json"
    restore_path = Path(args.restore_path) if args.restore_path else find_checkpoint(model_dir)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not restore_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {restore_path}")
    return config_path, restore_path


def get_audio_duration_ms(path):
    with wave.open(str(path), "rb") as wav_file:
        frame_rate = wav_file.getframerate()
        frame_count = wav_file.getnframes()
    if frame_rate == 0:
        return 0
    return int((frame_count / frame_rate) * 1000)


def normalize_text(text):
    cleaned = " ".join(text.strip().split())
    cleaned = cleaned.replace("\u2019", "'").replace("\u2018", "'")
    cleaned = cleaned.replace("\u201c", '"').replace("\u201d", '"')
    cleaned = cleaned.replace("\ufeff", "")
    return cleaned


def source_row_to_sample(row):
    if not row:
        return None

    if len(row) >= 3:
        clip_id = row[0].strip()
        text = normalize_text(row[-1])
    elif len(row) == 2:
        clip_id = row[0].strip()
        text = normalize_text(row[1])
    else:
        return None

    if not clip_id or not text:
        return None
    return clip_id, text


def prepare_dataset(args, target_sample_rate):
    dataset_dir = Path(args.dataset_dir)
    source_metadata = dataset_dir / args.metadata_source
    source_wavs_dir = dataset_dir / "wavs"
    prepared_dir = Path(args.prepared_dataset_dir)
    prepared_wavs_dir = prepared_dir / "wavs"
    prepared_metadata = prepared_dir / "metadata.csv"

    if not source_metadata.exists():
        raise FileNotFoundError(f"Metadata file not found: {source_metadata}")
    if not source_wavs_dir.exists():
        raise FileNotFoundError(f"WAV directory not found: {source_wavs_dir}")

    prepared_wavs_dir.mkdir(parents=True, exist_ok=True)

    durations = []
    kept_rows = []
    skipped_short = 0
    skipped_missing = 0
    skipped_invalid = 0

    with source_metadata.open("r", encoding="utf-8", newline="") as metadata_file:
        reader = csv.reader(metadata_file, delimiter="|")
        for row in reader:
            sample = source_row_to_sample(row)
            if sample is None:
                skipped_invalid += 1
                continue

            clip_id, text = sample
            if len(text) < args.min_text_length:
                skipped_short += 1
                continue

            source_wav = source_wavs_dir / f"{clip_id}.wav"
            if not source_wav.exists():
                skipped_missing += 1
                continue

            duration_ms = get_audio_duration_ms(source_wav)
            if duration_ms < args.min_duration_ms:
                skipped_short += 1
                continue

            prepared_wav = prepared_wavs_dir / f"{clip_id}.wav"
            segment = AudioSegment.from_wav(source_wav)
            segment = segment.set_channels(1)
            segment = segment.set_frame_rate(target_sample_rate)
            segment.export(prepared_wav, format="wav")

            kept_rows.append((clip_id, text, text))
            durations.append(duration_ms)

            if args.max_samples and len(kept_rows) >= args.max_samples:
                break

    if not kept_rows:
        raise RuntimeError(
            "No usable samples were prepared. Lower the filtering thresholds or rebuild the dataset with longer clips."
        )

    with prepared_metadata.open("w", encoding="utf-8", newline="") as metadata_file:
        writer = csv.writer(metadata_file, delimiter="|")
        writer.writerows(kept_rows)

    median_duration = int(statistics.median(durations))
    mean_duration = int(statistics.mean(durations))

    print(f"Prepared {len(kept_rows)} samples at {target_sample_rate} Hz into {prepared_dir}")
    print(
        f"Skipped {skipped_short} short, {skipped_missing} missing, and {skipped_invalid} invalid rows."
    )
    print(f"Mean clip duration: {mean_duration} ms | Median clip duration: {median_duration} ms")
    if median_duration < 700:
        print("Warning: this dataset is mostly word-level audio. The run can proceed, but voice quality will be weak.")
        print("Warning: rebuild the dataset with phrase-level or sentence-level segments for a usable cloned voice.")
    elif median_duration < 2000:
        print("Warning: clips are still short. Fine-tuning quality is usually much better with phrase-level segments.")

    return prepared_dir, prepared_metadata


def build_dataset_config(prepared_dir, metadata_path, language, dataset_name):
    return BaseDatasetConfig(
        formatter="ljspeech",
        dataset_name=dataset_name,
        path=str(prepared_dir),
        meta_file_train=metadata_path.name,
        meta_file_val=metadata_path.name,
        language=language,
    )


def main():
    args = parse_args()

    config_path, restore_path = resolve_pretrained_paths(args)
    config = load_config(str(config_path))
    target_sample_rate = int(get_value(config.audio, "sample_rate"))

    prepared_dir, prepared_metadata = prepare_dataset(args, target_sample_rate)
    dataset_config = build_dataset_config(prepared_dir, prepared_metadata, args.language, args.run_name)

    config.datasets = [dataset_config]
    config.output_path = str(Path(args.output_dir))
    config.run_name = args.run_name
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.eval_batch_size = args.eval_batch_size
    config.lr = args.learning_rate
    config.save_step = args.save_step
    config.print_step = args.print_step
    config.run_eval = True
    config.test_delay_epochs = -1
    config.num_loader_workers = args.num_loader_workers
    config.num_eval_loader_workers = args.num_eval_loader_workers
    config.mixed_precision = not args.no_mixed_precision

    ap = AudioProcessor.init_from_config(config)
    tokenizer, config = TTSTokenizer.init_from_config(config)

    train_samples, eval_samples = load_tts_samples(
        dataset_config,
        eval_split=True,
        eval_split_max_size=get_value(config, "eval_split_max_size"),
        eval_split_size=get_value(config, "eval_split_size"),
    )

    model = setup_tts_model(config)
    model.load_checkpoint(config, str(restore_path), eval=False)

    trainer = Trainer(
        TrainerArgs(),
        config,
        str(Path(args.output_dir)),
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
    trainer.fit()


if __name__ == "__main__":
    main()
