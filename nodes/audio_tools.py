"""
FL Audio Tools Nodes
Nodes for audio splitting and post-processing (alignment/merging)
"""

import torch
import os
import sys
import tempfile
import numpy as np
from typing import Tuple, Dict, Any, List

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from pydub import AudioSegment
except ImportError:
    print("pydub not installed, FL Audio Tools will not work correctly")

try:
    from ..utils.audio_utils import (
        tensor_to_audiosegment, 
        audiosegment_to_tensor, 
        save_audio_to_tempfile,
        cleanup_temp_file,
        time_stretch
    )
    # Import logic from audio_split.py
    from .audio_split import (
        split_audio_on_silence,
        split_audio_vad,
        split_audio_vad_f0,
        split_audio_whisperx
    )
except ImportError as e:
    # Try alternative relative import path
    try:
        from utils.audio_utils import (
            tensor_to_audiosegment, 
            audiosegment_to_tensor, 
            save_audio_to_tempfile,
            cleanup_temp_file,
            time_stretch
        )
        from nodes.audio_split import (
            split_audio_on_silence,
            split_audio_vad,
            split_audio_vad_f0,
            split_audio_whisperx
        )
    except ImportError as e2:
        print(f"Error importing audio tools: {e2}")

# Check for whisperx
try:
    import whisperx
    HAS_WHISPERX = True
except ImportError:
    HAS_WHISPERX = False

class FL_Audio_Split:
    """
    Split audio into segments based on silence or VAD
    """
    
    RETURN_TYPES = ("AUDIO", "INT")
    RETURN_NAMES = ("audio_list", "split_count")
    OUTPUT_IS_LIST = (True, False)
    FUNCTION = "split_audio"
    CATEGORY = "🔊FL CosyVoice3/Tools"

    @classmethod
    def INPUT_TYPES(cls):
        methods = ["Silence", "VAD", "VAD+F0"]
        if HAS_WHISPERX:
            methods.append("WhisperX")
            
        return {
            "required": {
                "audio": ("AUDIO", {
                    "description": "Input audio to split"
                }),
                "method": (methods, {
                    "default": "Silence",
                    "description": "Splitting method"
                }),
                "max_duration": ("FLOAT", {
                    "default": 30.0,
                    "min": 1.0,
                    "max": 300.0,
                    "step": 0.5,
                    "description": "Max segment duration (seconds)"
                }),
                "min_duration": ("FLOAT", {
                    "default": 5.0,
                    "min": 0.5,
                    "max": 60.0,
                    "step": 0.5,
                    "description": "Min segment duration (seconds)"
                }),
                "silence_thresh": ("INT", {
                    "default": -60,
                    "min": -100,
                    "max": 0,
                    "step": 1,
                    "description": "Silence threshold (dB) for Silence method"
                }),
                "vad_threshold": ("FLOAT", {
                    "default": 0.6,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "description": "VAD threshold for VAD methods"
                }),
                "f0_threshold": ("INT", {
                    "default": 80,
                    "min": 40,
                    "max": 1000,
                    "step": 10,
                    "description": "F0 threshold (Hz) for VAD+F0 method"
                }),
            }
        }

    def split_audio(
        self,
        audio: Dict[str, Any],
        method: str,
        max_duration: float,
        min_duration: float,
        silence_thresh: int,
        vad_threshold: float,
        f0_threshold: int
    ) -> Tuple[List[Dict[str, Any]], int]:
        
        print(f"\n{'='*60}")
        print(f"[FL Audio Split] Splitting audio...")
        print(f"[FL Audio Split] Method: {method}")
        print(f"[FL Audio Split] Max Duration: {max_duration}s")
        print(f"{'='*60}\n")
        
        # 1. Convert input audio to AudioSegment directly (in-memory)
        waveform = audio['waveform']
        sample_rate = audio['sample_rate']
        
        # Convert tensor to AudioSegment
        audio_segment = tensor_to_audiosegment(waveform, sample_rate)
        
        # 2. Call splitting function with AudioSegment input
        # Convert seconds to milliseconds
        max_ms = max_duration * 1000
        min_ms = min_duration * 1000
        
        segments = []
        
        if method == "Silence":
            segments = split_audio_on_silence(
                audio_segment, None,
                max_duration=max_ms, min_duration=min_ms,
                silence_thresh=silence_thresh,
                save=False
            )
        elif method == "VAD":
            segments = split_audio_vad(
                audio_segment, None,
                max_duration=max_ms, min_duration=min_ms,
                vad_threshold=vad_threshold,
                save=False
            )
        elif method == "VAD+F0":
            segments = split_audio_vad_f0(
                audio_segment, None,
                max_duration=max_ms, min_duration=min_ms,
                vad_threshold=vad_threshold,
                f0_threshold=float(f0_threshold),
                save=False
            )
        elif method == "WhisperX":
            try:
                # WhisperX still requires temp files internally in our modified function if input is AudioSegment
                # But the wrapper handles it.
                segments = split_audio_whisperx(
                    audio_segment, None,
                    max_duration=max_ms, min_duration=min_ms,
                    save=False
                )
            except Exception as e:
                print(f"[FL Audio Split] WhisperX failed, falling back to Silence: {e}")
                segments = split_audio_on_silence(
                    audio_segment, None,
                    max_duration=max_ms, min_duration=min_ms,
                    silence_thresh=silence_thresh,
                    save=False
                )
        
        # 3. Convert split segments back to tensor list
        result_list = []
        for segment in segments:
            try:
                seg_waveform, seg_sr = audiosegment_to_tensor(segment)
                
                # Convert to ComfyUI format
                # Ensure we have the right shape [batch, channels, samples]
                if seg_waveform.ndim == 2:
                    seg_waveform = seg_waveform.unsqueeze(0)
                    
                result_list.append({
                    "waveform": seg_waveform,
                    "sample_rate": seg_sr
                })
            except Exception as e:
                print(f"[FL Audio Split] Failed to convert segment: {e}")
        
        split_count = len(result_list)
        print(f"[FL Audio Split] Created {split_count} segments")
        return (result_list, split_count)


class FL_Audio_Align:
    """
    Align generated audio duration to original audio
    """
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("aligned_audio",)
    FUNCTION = "align_audio"
    CATEGORY = "🔊FL CosyVoice3/Tools"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_audio": ("AUDIO", {
                    "description": "Original audio (reference duration)"
                }),
                "generated_audio": ("AUDIO", {
                    "description": "Generated audio to align"
                }),
                "speed_factor": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.5,
                    "max": 2.0,
                    "step": 0.05,
                    "description": "Manual speed adjustment (multiplier)"
                }),
                "alignment": ("BOOLEAN", {
                    "default": True,
                    "description": "Force align duration to original (overrides speed_factor)"
                }),
            }
        }

    def align_audio(
        self,
        original_audio: Dict[str, Any],
        generated_audio: Dict[str, Any],
        speed_factor: float,
        alignment: bool
    ) -> Tuple[Dict[str, Any]]:
        
        print(f"\n{'='*60}")
        print(f"[FL Audio Align] Aligning audio...")
        print(f"[FL Audio Align] Force Alignment: {alignment}")
        if not alignment:
            print(f"[FL Audio Align] Speed Factor: {speed_factor}")
        print(f"{'='*60}\n")
        
        # Convert to AudioSegment
        orig_seg = tensor_to_audiosegment(original_audio['waveform'], original_audio['sample_rate'])
        gen_seg = tensor_to_audiosegment(generated_audio['waveform'], generated_audio['sample_rate'])
        
        processed_seg = gen_seg
        
        if len(orig_seg) > 0 and len(gen_seg) > 0:
            ratio = 1.0
            if alignment:
                # Calculate ratio to match original duration
                ratio = len(gen_seg) / len(orig_seg)
                print(f"[FL Audio Align] Auto-calculated ratio: {ratio:.4f}")
            else:
                # Use manual speed factor (inverse of stretch ratio)
                # If speed is 2x, duration is 0.5x, so stretch ratio is 0.5
                # But time_stretch implementation: ratio < 1 means faster (shorter), ratio > 1 means slower (longer)?
                # Let's check time_stretch implementation in utils/audio_utils.py (not visible here but usually ratio = new_duration / old_duration)
                # Wait, standard ffmpeg atempo filter: >1.0 slows down, <1.0 speeds up.
                # In my previous thought process/code reading: 
                # "ratio = current_duration / desired_duration" -> if current > desired (too long), ratio > 1, so we need to speed up?
                # Actually, let's re-read the time_stretch logic I saw earlier or recall it.
                # The shared code showed:
                # while r > 2.0: filters.append("atempo=2.0"); r /= 2.0
                # atempo filter: "atempo=2.0" means 2x speed (shorter duration).
                # So if ratio (passed to time_stretch) is 2.0, it speeds up by 2x.
                # If I want to align: ratio = len(gen_seg) / len(orig_seg).
                # Example: Gen=10s, Orig=5s. Ratio = 2.0. We need to speed up by 2x to get 5s.
                # So time_stretch(seg, 2.0) -> returns 5s audio. Correct.
                
                # If manual speed_factor is provided (e.g. 1.2x speed):
                # We should pass 1.2 directly to time_stretch.
                ratio = speed_factor

            if abs(ratio - 1.0) > 0.001:
                processed_seg = time_stretch(gen_seg, ratio)
        
        # Convert back to tensor
        wf, sr = audiosegment_to_tensor(processed_seg)
        if wf.ndim == 2:
            wf = wf.unsqueeze(0)
            
        result = {
            "waveform": wf,
            "sample_rate": sr
        }
        
        return (result,)


class FL_Audio_Merge:
    """
    Merge multiple audio segments into one
    """
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("merged_audio",)
    INPUT_IS_LIST = True
    FUNCTION = "merge_audio"
    CATEGORY = "🔊FL CosyVoice3/Tools"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_list": ("AUDIO", {
                    "description": "List of audio segments to merge"
                }),
                "crossfade_ms": ("INT", {
                    "default": 100,
                    "min": 0,
                    "max": 2000,
                    "step": 50,
                    "description": "Crossfade duration (ms). Warning: Reduces total duration!"
                }),
            }
        }

    def merge_audio(
        self,
        audio_list: List[Dict[str, Any]],
        crossfade_ms: List[int] # ComfyUI passes single values as list when INPUT_IS_LIST=True
    ) -> Tuple[Dict[str, Any]]:
        
        print(f"\n{'='*60}")
        print(f"[FL Audio Merge] Merging audio...")
        
        # Handle list inputs
        # Even single values are passed as lists when INPUT_IS_LIST is True for the main input
        # But for non-list inputs (like crossfade_ms), they might be passed as a list of 1 element
        # or a list of length equal to audio_list if they are linked?
        # Usually standard widgets (INT) are passed as a list with 1 element if they are not linked.
        # Let's safely get the first element.
        fade_ms = crossfade_ms[0] if isinstance(crossfade_ms, list) else crossfade_ms
        
        print(f"[FL Audio Merge] Input segments: {len(audio_list)}")
        print(f"[FL Audio Merge] Crossfade: {fade_ms}ms")
        print(f"{'='*60}\n")
        
        if not audio_list:
            # Return empty audio
            empty = AudioSegment.silent(duration=1000)
            wf, sr = audiosegment_to_tensor(empty)
            return ({"waveform": wf.unsqueeze(0), "sample_rate": sr},)
            
        combined_audio = AudioSegment.empty()
        
        for i, audio_dict in enumerate(audio_list):
            # Convert to AudioSegment
            seg = tensor_to_audiosegment(audio_dict['waveform'], audio_dict['sample_rate'])
            
            if i == 0:
                combined_audio = seg
            else:
                # Apply crossfade if specified
                # pydub append with crossfade
                # If crossfade is longer than either segment, pydub might raise an error or handle it.
                # Let's ensure crossfade is not too long.
                current_fade = fade_ms
                if current_fade > 0:
                    # Limit crossfade to half of the shorter segment to avoid issues
                    limit = min(len(combined_audio), len(seg)) / 2
                    if current_fade > limit:
                        current_fade = int(limit)
                        print(f"[FL Audio Merge] Warning: Crossfade reduced to {current_fade}ms for segment {i}")
                
                combined_audio = combined_audio.append(seg, crossfade=current_fade)
        
        # Convert combined audio to tensor
        merged_wf, merged_sr = audiosegment_to_tensor(combined_audio)
        if merged_wf.ndim == 2:
            merged_wf = merged_wf.unsqueeze(0)
            
        result = {
            "waveform": merged_wf,
            "sample_rate": merged_sr
        }
        
        return (result,)
