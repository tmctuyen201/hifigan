import os
import csv
from datasets import load_dataset
import torchaudio
import torchaudio.transforms as T
import numpy as np
import torch
import librosa

# Tải dataset từ Hugging Face
dataset = load_dataset("trinhtuyen201/suhuyn_audio_ver2", split="train")

# Thiết lập Mel-spectrogram extractor
mel_transform = T.MelSpectrogram(
    sample_rate=22050, n_mels=80, hop_length=256, n_fft=1024)


def extract_mel(waveform):
    mel_spectrogram = mel_transform(waveform)
    return mel_spectrogram.float()


# Tạo thư mục
os.makedirs("data/wavs", exist_ok=True)

# Tạo file metadata.csv
metadata_path = "data/metadata.csv"
# Tạo file metadata.csv
with open(metadata_path, mode="w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f, delimiter='|')

    for i, sample in enumerate(dataset):
        waveform = sample["audio"]["array"]
        text = sample["transcription"]
        if text == "♪" or text == "[Music]" or text == "*Music*":
            continue
        text = text.replace(";", "").replace("@", "").replace("/", "").replace("\"", "").replace("?", "").replace("!", "").replace("[", "").replace(")", "").replace("(", "").replace("*", "")
        if isinstance(waveform, np.ndarray):
            waveform = torch.tensor(waveform).float()
        if waveform.ndimension() == 1:
            waveform = waveform.unsqueeze(0)
        # Resample waveform from 22,050 Hz to 16,000 Hz
        original_sample_rate = 22050
        target_sample_rate = 16000

        # Convert waveform to numpy for resampling
        waveform_numpy = waveform.squeeze().numpy()

        # Perform the resampling
        waveform_resampled = librosa.resample(waveform_numpy, orig_sr=original_sample_rate, target_sr=target_sample_rate)

        # Convert the resampled waveform back to tensor
        waveform_resampled_tensor = torch.tensor(waveform_resampled).float()
        # Extract Mel
        # mel_spectrogram = extract_mel(waveform)
        waveform_resampled_tensor = waveform_resampled_tensor.unsqueeze(0)
        # Save mel
        # mel_file_path = f"data/mels/sample_{i}.npy"
        # np.save(mel_file_path, mel_spectrogram.numpy())

        # Save audio
        wav_file_name = f"sample_{i}.wav"
        wav_file_path = f"data/wavs/{wav_file_name}"
        torchaudio.save(wav_file_path, waveform_resampled_tensor, 16000)

        # Write metadata line
        writer.writerow([os.path.splitext(wav_file_name)[0], text])
