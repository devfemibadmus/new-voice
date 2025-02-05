import os
from TTS.tts.configs.tacotron2_config import Tacotron2Config
from TTS.utils.audio import AudioProcessor
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.tacotron2 import Tacotron2
from trainer import Trainer, TrainerArgs

from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent


output_path = "output"
if not os.path.exists(output_path):
    os.makedirs(output_path)

dataset_config = {
    "formatter": "ljspeech",
    "meta_file_train": "dataset.csv",
    "meta_file_val": "dataset.csv",
    "path": "generated",
    "dataset_name": "chatgpt_free_voice",
    "ignored_speakers": [],
    "language": "en",
    "meta_file_attn_mask": [],
    "meta_file_pitch_stats": [], 
    "meta_file_energy_stats": []
}

config = Tacotron2Config(
    batch_size=32,
    eval_batch_size=16,
    num_loader_workers=4,
    num_eval_loader_workers=4,
    run_eval=True,
    test_delay_epochs=-1,
    epochs=100,
    text_cleaner="english_cleaners",
    use_phonemes=False,
    print_step=25,
    print_eval=True,
    mixed_precision=True,
    output_path=output_path,
    datasets=[dataset_config],
    save_step=500,
)

ap = AudioProcessor.init_from_config(config)

tokenizer, config = TTSTokenizer.init_from_config(config)

train_samples, eval_samples = load_tts_samples(
    dataset_config,
    eval_split=True,
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=config.eval_split_size,
)

model = Tacotron2(config, ap, tokenizer, speaker_manager=None)

trainer = Trainer(
    TrainerArgs(),
    config,
    output_path,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
)

if __name__ == '__main__':
    trainer.fit()
