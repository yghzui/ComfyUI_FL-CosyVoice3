"""
Audio Utilities for FL CosyVoice3
Handles audio format conversions and processing
"""

import torch
import torchaudio
import soundfile as sf
import tempfile
import os
import numpy as np
from typing import Dict, Any, Tuple, Optional

try:
    from pydub import AudioSegment
except ImportError:
    print("pydub not installed, some features may not work")



def comfyui_audio_to_tensor(audio: Dict[str, Any]) -> Tuple[torch.Tensor, int]:
    """
    Convert ComfyUI AUDIO format to tensor and sample rate

    Args:
        audio: ComfyUI audio dict {"waveform": tensor, "sample_rate": int}

    Returns:
        Tuple of (waveform_tensor, sample_rate)
    """
    waveform = audio['waveform']
    sample_rate = audio['sample_rate']

    return waveform, sample_rate


def tensor_to_comfyui_audio(waveform: torch.Tensor, sample_rate: int) -> Dict[str, Any]:
    """
    Convert tensor to ComfyUI AUDIO format

    Args:
        waveform: Audio tensor
        sample_rate: Sample rate in Hz

    Returns:
        ComfyUI audio dict
    """
    # Ensure waveform is on CPU
    if waveform.device != torch.device('cpu'):
        waveform = waveform.cpu()

    # Ensure proper shape [batch, channels, samples]
    if waveform.ndim == 1:
        # Mono, no batch -> [1, 1, samples]
        waveform = waveform.unsqueeze(0).unsqueeze(0)
    elif waveform.ndim == 2:
        # Either [channels, samples] or [batch, samples]
        # Assume [channels, samples] and add batch dim
        waveform = waveform.unsqueeze(0)

    return {
        "waveform": waveform,
        "sample_rate": sample_rate
    }


def save_audio_to_tempfile(waveform: torch.Tensor, sample_rate: int, suffix: str = ".wav") -> str:
    """
    Save audio tensor to a temporary file

    Args:
        waveform: Audio tensor [channels, samples] or [batch, channels, samples]
        sample_rate: Sample rate in Hz
        suffix: File suffix

    Returns:
        Path to temporary file
    """
    # Ensure waveform is on CPU
    if waveform.device != torch.device('cpu'):
        waveform = waveform.cpu()

    # Remove batch dimension if present
    if waveform.ndim == 3:
        waveform = waveform.squeeze(0)

    # Create temp file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_path = temp_file.name
    temp_file.close()

    # Save audio using soundfile directly (avoids torchaudio's torchcodec requirement)
    # soundfile expects shape (samples, channels), so transpose from (channels, samples)
    audio_np = waveform.cpu().numpy()
    if audio_np.ndim == 2:
        audio_np = audio_np.T  # (channels, samples) -> (samples, channels)
    sf.write(temp_path, audio_np, sample_rate)

    return temp_path


def load_audio_from_path(audio_path: str, target_sample_rate: Optional[int] = None) -> Dict[str, Any]:
    """
    Load audio file from path into ComfyUI AUDIO format

    Args:
        audio_path: Path to audio file
        target_sample_rate: Target sample rate (None to keep original)

    Returns:
        ComfyUI audio dict
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Load audio
    waveform, sample_rate = torchaudio.load(audio_path)

    # Resample if needed
    if target_sample_rate is not None and target_sample_rate != sample_rate:
        resampler = torchaudio.transforms.Resample(sample_rate, target_sample_rate)
        waveform = resampler(waveform)
        sample_rate = target_sample_rate

    # Convert to ComfyUI format
    return tensor_to_comfyui_audio(waveform, sample_rate)


def resample_audio(waveform: torch.Tensor, orig_sample_rate: int, target_sample_rate: int) -> torch.Tensor:
    """
    Resample audio tensor to target sample rate

    Args:
        waveform: Audio tensor
        orig_sample_rate: Original sample rate
        target_sample_rate: Target sample rate

    Returns:
        Resampled audio tensor
    """
    if orig_sample_rate == target_sample_rate:
        return waveform

    resampler = torchaudio.transforms.Resample(orig_sample_rate, target_sample_rate)
    return resampler(waveform)


def ensure_mono(waveform: torch.Tensor) -> torch.Tensor:
    """
    Convert audio to mono by averaging channels

    Args:
        waveform: Audio tensor [..., channels, samples]

    Returns:
        Mono audio tensor [..., 1, samples]
    """
    if waveform.shape[-2] == 1:
        return waveform

    # Average across channels
    return waveform.mean(dim=-2, keepdim=True)


def ensure_stereo(waveform: torch.Tensor) -> torch.Tensor:
    """
    Convert audio to stereo

    Args:
        waveform: Audio tensor [..., channels, samples]

    Returns:
        Stereo audio tensor [..., 2, samples]
    """
    if waveform.shape[-2] == 2:
        return waveform

    if waveform.shape[-2] == 1:
        # Duplicate mono to stereo
        return waveform.repeat(*([1] * (waveform.ndim - 2)), 2, 1)

    # Multiple channels - take first two
    return waveform[..., :2, :]


def normalize_audio(waveform: torch.Tensor, target_peak: float = 0.95) -> torch.Tensor:
    """
    Normalize audio to target peak amplitude

    Args:
        waveform: Audio tensor
        target_peak: Target peak amplitude (0.0 - 1.0)

    Returns:
        Normalized audio tensor
    """
    current_peak = waveform.abs().max()

    if current_peak > 0:
        waveform = waveform * (target_peak / current_peak)

    return waveform


def prepare_audio_for_cosyvoice(
    audio: Dict[str, Any],
    target_sample_rate: int = 16000,
    mono: bool = True
) -> Tuple[torch.Tensor, int, Optional[str]]:
    """
    Prepare ComfyUI audio for CosyVoice inference

    Args:
        audio: ComfyUI audio dict
        target_sample_rate: Target sample rate for CosyVoice
        mono: Convert to mono

    Returns:
        Tuple of (waveform, sample_rate, temp_file_path)
    """
    waveform, sample_rate = comfyui_audio_to_tensor(audio)

    # Remove batch dimension if present
    if waveform.ndim == 3:
        waveform = waveform.squeeze(0)

    # Convert to mono if needed
    if mono and waveform.shape[0] > 1:
        waveform = ensure_mono(waveform)

    # Resample if needed
    if sample_rate != target_sample_rate:
        waveform = resample_audio(waveform, sample_rate, target_sample_rate)
        sample_rate = target_sample_rate

    # Save to temp file (CosyVoice may expect file paths)
    temp_path = save_audio_to_tempfile(waveform, sample_rate)

    return waveform, sample_rate, temp_path


def save_raw_audio_to_tempfile(audio: Dict[str, Any]) -> str:
    """
    Save ComfyUI audio to temp file WITHOUT any processing.

    CosyVoice's load_wav() handles mono conversion and resampling internally,
    so we should NOT preprocess the audio.

    Args:
        audio: ComfyUI audio dict {"waveform": tensor, "sample_rate": int}

    Returns:
        Path to temporary file
    """
    waveform = audio['waveform']
    sample_rate = audio['sample_rate']

    # Remove batch dim if present
    if waveform.ndim == 3:
        waveform = waveform.squeeze(0)

    # Ensure CPU
    if waveform.device != torch.device('cpu'):
        waveform = waveform.cpu()

    # Save directly without any processing
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    temp_path = temp_file.name
    temp_file.close()

    # soundfile expects (samples, channels) format
    audio_np = waveform.numpy()
    if audio_np.ndim == 2:
        audio_np = audio_np.T  # (channels, samples) -> (samples, channels)
    sf.write(temp_path, audio_np, sample_rate)

    return temp_path


def cleanup_temp_file(temp_path: Optional[str]):
    """
    Clean up temporary audio file

    Args:
        temp_path: Path to temporary file
    """
    if temp_path and os.path.exists(temp_path):
        try:
            os.unlink(temp_path)
        except:
            pass


def audiosegment_to_tensor(segment: 'AudioSegment') -> Tuple[torch.Tensor, int]:
    """
    Convert pydub AudioSegment to tensor and sample rate
    """
    channel_count = segment.channels
    sample_width = segment.sample_width
    frame_rate = segment.frame_rate
    
    # Get raw data
    raw_data = segment.raw_data
    
    # Convert to numpy array
    if sample_width == 2:
        dtype = np.int16
    elif sample_width == 4:
        dtype = np.int32
    elif sample_width == 1:
        dtype = np.int8
    else:
        raise ValueError(f"Unsupported sample width: {sample_width}")
        
    audio_array = np.frombuffer(raw_data, dtype=dtype)
    
    # Reshape if stereo
    if channel_count > 1:
        audio_array = audio_array.reshape((-1, channel_count)).T
    else:
        audio_array = audio_array.reshape((1, -1))
        
    # Normalize to float -1..1
    if sample_width == 2:
        audio_tensor = torch.from_numpy(audio_array).float() / 32768.0
    elif sample_width == 4:
        audio_tensor = torch.from_numpy(audio_array).float() / 2147483648.0
    elif sample_width == 1:
        audio_tensor = torch.from_numpy(audio_array).float() / 128.0
        
    # Add batch dimension [1, channels, samples]
    if audio_tensor.ndim == 2:
        audio_tensor = audio_tensor.unsqueeze(0)
        
    return audio_tensor, frame_rate


def tensor_to_audiosegment(waveform: torch.Tensor, sample_rate: int) -> 'AudioSegment':
    """
    Convert tensor to pydub AudioSegment
    """
    # Ensure CPU
    if waveform.device != torch.device('cpu'):
        waveform = waveform.cpu()
        
    # Remove batch dim
    if waveform.ndim == 3:
        waveform = waveform.squeeze(0)
        
    # Convert to numpy [channels, samples]
    audio_np = waveform.numpy()
    
    # Convert to int16 PCM
    audio_int16 = (audio_np * 32767).astype(np.int16)
    
    # Transpose to [samples, channels] for pydub
    if audio_int16.ndim == 2:
        audio_int16 = audio_int16.T
        
    # Create AudioSegment
    return AudioSegment(
        audio_int16.tobytes(),
        frame_rate=sample_rate,
        sample_width=2,
        channels=waveform.shape[0]
    )


def time_stretch(seg: 'AudioSegment', ratio: float) -> 'AudioSegment':
    """
    Time stretch an AudioSegment using ffmpeg atempo filter
    """
    if abs(ratio - 1.0) <= 0.001:
        return seg
        
    import uuid
    import subprocess
    
    unique_id = uuid.uuid4().hex
    temp_dir = tempfile.gettempdir()
    tin = os.path.join(temp_dir, f"ts_in_{unique_id}.wav")
    tout = os.path.join(temp_dir, f"ts_out_{unique_id}.wav")
    
    try:
        seg.export(tin, format="wav")
        
        # atempo filter supports 0.5 to 2.0
        filters = []
        r = ratio
        while r > 2.0:
            filters.append("atempo=2.0")
            r /= 2.0
        while r < 0.5:
            filters.append("atempo=0.5")
            r /= 0.5
        filters.append(f"atempo={r}")
        filter_str = ",".join(filters)
        
        cmd = ["ffmpeg", "-y", "-i", tin, "-filter:a", filter_str, tout]
        
        # Hide ffmpeg output on windows
        startupinfo = None
        if os.name == 'nt':
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, startupinfo=startupinfo)
        
        if os.path.exists(tout):
            result = AudioSegment.from_file(tout)
            return result
        else:
            return seg
            
    except Exception as e:
        print(f"[Audio Utils] Time stretch error: {e}")
        return seg
    finally:
        # clean up
        try:
            if os.path.exists(tin): os.remove(tin)
            if os.path.exists(tout): os.remove(tout)
        except: pass
