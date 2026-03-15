# -*- coding: utf-8 -*-
# Created time : 2024/12/13 23:54 
# Auther : ygh
# File   : audio_split.py
# Description : Audio splitting logic, updated to support in-memory processing

from pydub import AudioSegment
from pydub.silence import detect_silence
import os
import logging
import numpy as np
import importlib

def merge_audio_segments(segments, max_duration=30 * 1000, min_duration=5 * 1000):
    """
    Merge audio segments shorter than min_duration.
    
    :param segments: List[AudioSegment]
    :return: Merged list of AudioSegment
    """
    i = 0
    while i < len(segments):
        if len(segments[i]) < min_duration:  # Current segment < min_duration
            if i > 0 and len(segments[i]) + len(segments[i - 1]) <= max_duration:
                # Merge with previous
                segments[i - 1] = segments[i - 1] + segments[i]
                del segments[i]
                i -= 1  # Re-check from previous
            elif i < len(segments) - 1 and len(segments[i]) + len(segments[i + 1]) <= max_duration:
                # Merge with next
                segments[i + 1] += segments[i]
                del segments[i]
            else:
                # Cannot merge
                i += 1
        else:
            i += 1
    return segments

def _load_audio(input_data):
    """
    Helper to load audio from file path or AudioSegment object
    """
    if isinstance(input_data, str):
        if not os.path.exists(input_data):
             raise FileNotFoundError(f"Audio file not found: {input_data}")
        return AudioSegment.from_file(input_data), os.path.basename(input_data).split(".")[0]
    elif isinstance(input_data, AudioSegment):
        return input_data, "audio_segment"
    else:
        raise ValueError("Input must be a file path string or AudioSegment object")

def split_audio_on_silence(input_data, output_dir=None, max_duration=30 * 1000, min_duration=5 * 1000, silence_thresh=-60, min_silence_len=500, extra_ms=200, save=True):
    """
    Split audio based on silence.
    """
    audio, audio_name = _load_audio(input_data)
    
    if save and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    audio_length = len(audio)
    # Detect silence
    silent_ranges = detect_silence(audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh)
    silent_ranges = [(start, end) for start, end in silent_ranges]
    print(f"silent_ranges: {silent_ranges}")

    segments = []
    start = 0

    while start < audio_length:
        search_start = start + min_duration
        search_end = start + max_duration
        
        if search_start >= audio_length:
            if segments:
                if len(segments[-1]) + (audio_length - start) <= max_duration:
                    segments[-1] += audio[start:]
                else:
                    segments.append(audio[start:])
            else:
                segments.append(audio[start:])
            break

        current_max_end = min(search_end, audio_length)
        
        candidates = []
        for silence_start, silence_end in silent_ranges:
            overlap_start = max(silence_start, search_start)
            overlap_end = min(silence_end, current_max_end)
            if overlap_start < overlap_end:
                cp = (overlap_start + overlap_end) // 2
                candidates.append(cp)
        if candidates:
            best_cp = max(candidates)
            seg_end = min(best_cp + extra_ms, audio_length)
            segments.append(audio[start:seg_end])
            start = best_cp
        else:
            if current_max_end == audio_length:
                segments.append(audio[start:])
                break
            else:
                seg_end = min(current_max_end + extra_ms, audio_length)
                segments.append(audio[start:seg_end])
                start = current_max_end

    logging.info(f"split audio length: {len(segments)}")
    print(f"split audio length: {len(segments)}")

    total_length_before = sum(len(segment) for segment in segments)
    total_length_after = len(audio)
    if total_length_before != total_length_after:
        logging.error(f"Length mismatch: {total_length_before} != {total_length_after}")

    segments = merge_audio_segments(segments, max_duration, min_duration)
    
    if save and output_dir:
        output_path_list = []
        name_tag = f"{audio_name}_sil_{int(max_duration)}_{int(min_duration)}_{int(min_silence_len)}_{int(silence_thresh)}_{int(extra_ms)}"
        for i, segment in enumerate(segments):
            output_path = os.path.join(output_dir, f"{name_tag}_{i + 1}.wav")
            segment.export(output_path, format="wav")
            logging.info(f"save: {output_path}")
            output_path_list.append(os.path.abspath(output_path))
        return output_path_list
    else:
        return segments

def split_audio_vad(input_data, output_dir=None, max_duration=30 * 1000, min_duration=5 * 1000,
                    frame_ms=30, vad_threshold=0.6, min_quiet_ms=300, extra_ms=200, save=True):
    
    audio, audio_name = _load_audio(input_data)
    
    if save and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    audio_length = len(audio)
    frame_len = frame_ms
    frames = []
    i = 0
    while i < audio_length:
        end = min(i + frame_len, audio_length)
        frames.append(audio[i:end])
        i = end
    energies = [seg.rms for seg in frames]
    if len(energies) == 0:
        return []
    
    arr = np.array(energies, dtype=np.float32)
    base = np.percentile(arr, 20)
    eps = 1e-6
    probs = (arr - base) / (arr.max() - base + eps)
    probs = np.clip(probs, 0.0, 1.0)
    speech = probs >= vad_threshold
    hang_on = 3
    hang_off = 6
    smooth = []
    state = False
    on_count = 0
    off_count = 0
    for s in speech:
        if s:
            on_count += 1
            off_count = 0
            if not state and on_count >= hang_on:
                state = True
        else:
            off_count += 1
            on_count = 0
            if state and off_count >= hang_off:
                state = False
        smooth.append(state)
    quiet_span = max(1, min_quiet_ms // frame_len)
    segments = []
    start = 0
    while start < audio_length:
        search_start = start + min_duration
        search_end = min(start + max_duration, audio_length)
        if search_start >= audio_length:
            if segments:
                if len(segments[-1]) + (audio_length - start) <= max_duration:
                    segments[-1] += audio[start:]
                else:
                    segments.append(audio[start:])
            else:
                segments.append(audio[start:])
            break
        fs = search_start // frame_len
        fe = max(fs + 1, search_end // frame_len)
        run_len = 0
        run_start = fs
        candidates = []
        for idx in range(fs, fe):
            if not smooth[idx]:
                if run_len == 0:
                    run_start = idx
                run_len += 1
                if run_len >= quiet_span:
                    cp = (run_start + idx) // 2 * frame_len
                    cp = max(search_start, min(cp, search_end))
                    candidates.append(cp)
            else:
                run_len = 0
        if candidates:
            best_cp = max(candidates)
            seg_end = min(best_cp + extra_ms, audio_length)
            segments.append(audio[start:seg_end])
            start = best_cp
        else:
            seg_end = min(search_end + extra_ms, audio_length)
            segments.append(audio[start:seg_end])
            start = search_end
            
    total_length_before = sum(len(seg) for seg in segments)
    if total_length_before != audio_length:
        logging.warning(f"VAD length mismatch: {total_length_before} != {audio_length}")
    segments = merge_audio_segments(segments, max_duration, min_duration)
    
    if save and output_dir:
        output_path_list = []
        name_tag = f"{audio_name}_vad_{int(max_duration)}_{int(min_duration)}_{int(frame_ms)}_{vad_threshold:.2f}_{int(min_quiet_ms)}_{int(extra_ms)}"
        for i, segment in enumerate(segments):
            output_path = os.path.join(output_dir, f"{name_tag}_{i + 1}.wav")
            segment.export(output_path, format="wav")
            logging.info(f"save: {output_path}")
            output_path_list.append(os.path.abspath(output_path))
        return output_path_list
    else:
        return segments

def split_audio_vad_f0(input_data, output_dir=None, max_duration=30 * 1000, min_duration=5 * 1000,
                       frame_ms=30, vad_threshold=0.6, min_quiet_ms=300, f0_threshold=80.0, extra_ms=200, save=True):
    
    audio, audio_name = _load_audio(input_data)
    
    if save and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    audio_length = len(audio)
    frame_len = frame_ms
    frames = []
    i = 0
    while i < audio_length:
        end = min(i + frame_len, audio_length)
        frames.append(audio[i:end])
        i = end
    energies = [seg.rms for seg in frames]
    if len(energies) == 0:
        return []
    arr = np.array(energies, dtype=np.float32)
    base = np.percentile(arr, 20)
    eps = 1e-6
    probs = (arr - base) / (arr.max() - base + eps)
    probs = np.clip(probs, 0.0, 1.0)
    speech = probs >= vad_threshold
    hang_on = 3
    hang_off = 6
    smooth = []
    state = False
    on_count = 0
    off_count = 0
    for s in speech:
        if s:
            on_count += 1
            off_count = 0
            if not state and on_count >= hang_on:
                state = True
        else:
            off_count += 1
            on_count = 0
            if state and off_count >= hang_off:
                state = False
        smooth.append(state)
    def estimate_f0(seg, sr):
        x = np.array(seg.get_array_of_samples(), dtype=np.float32)
        if seg.sample_width == 2:
            x = x / 32768.0
        x = x - (x.mean() if x.size else 0.0)
        if x.size < max(1, int(sr / 100)):
            return np.nan
        r = np.correlate(x, x, mode='full')[x.size - 1:]
        if r[0] == 0:
            return np.nan
        r = r / (r[0] + 1e-8)
        min_p = max(1, int(sr / 500))
        max_p = max(min_p + 1, int(sr / 50))
        peak = r[min_p:max_p]
        if peak.size == 0:
            return np.nan
        idx = int(np.argmax(peak))
        val = float(peak[idx])
        if val < 0.3:
            return np.nan
        period = min_p + idx
        if period <= 0:
            return np.nan
        return float(sr) / float(period)
    sr = audio.frame_rate
    f0_vals = [estimate_f0(seg, sr) for seg in frames]
    unvoiced = [np.isnan(f0) or (f0 < f0_threshold) for f0 in f0_vals]
    quiet_span = max(1, min_quiet_ms // frame_len)
    segments = []
    start = 0
    while start < audio_length:
        search_start = start + min_duration
        search_end = min(start + max_duration, audio_length)
        if search_start >= audio_length:
            if segments:
                if len(segments[-1]) + (audio_length - start) <= max_duration:
                    segments[-1] += audio[start:]
                else:
                    segments.append(audio[start:])
            else:
                segments.append(audio[start:])
            break
        cut_point = None
        fs = search_start // frame_len
        fe = max(fs + 1, search_end // frame_len)
        run_len = 0
        run_start = fs
        candidates = []
        for idx in range(fs, fe):
            if not smooth[idx] and unvoiced[idx]:
                if run_len == 0:
                    run_start = idx
                run_len += 1
                if run_len >= quiet_span:
                    cp = (run_start + idx) // 2 * frame_len
                    cut_point = max(search_start, min(cp, search_end))
                    break
            else:
                run_len = 0
        if cut_point:
            seg_end = min(cut_point + extra_ms, audio_length)
            segments.append(audio[start:seg_end])
            start = cut_point
        else:
            seg_end = min(search_end + extra_ms, audio_length)
            segments.append(audio[start:seg_end])
            start = search_end
    total_length_before = sum(len(seg) for seg in segments)
    if total_length_before < audio_length:
        logging.warning(f"VAD+F0 length mismatch: {total_length_before} < {audio_length}")
    segments = merge_audio_segments(segments, max_duration, min_duration)
    
    if save and output_dir:
        output_path_list = []
        name_tag = f"{audio_name}_vadf0_{int(max_duration)}_{int(min_duration)}_{int(frame_ms)}_{vad_threshold:.2f}_{int(min_quiet_ms)}_{int(f0_threshold)}_{int(extra_ms)}"
        for i, segment in enumerate(segments):
            output_path = os.path.join(output_dir, f"{name_tag}_{i + 1}.wav")
            segment.export(output_path, format="wav")
            logging.info(f"save: {output_path}")
            output_path_list.append(os.path.abspath(output_path))
        return output_path_list
    else:
        return segments

def split_audio_whisperx(input_data, output_dir=None, max_duration=30 * 1000, min_duration=5 * 1000,
                         min_quiet_ms=300, extra_ms=200, save=True):
    
    # WhisperX logic heavily relies on file paths for loading and processing
    # If input is AudioSegment, we must save it to temp file first
    # This function keeps file-based logic primarily because whisperx library expects files
    
    temp_input_path = None
    if isinstance(input_data, AudioSegment):
        import tempfile
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        input_data.export(tfile.name, format="wav")
        tfile.close()
        temp_input_path = tfile.name
        input_file = temp_input_path
        audio_name = "audio_segment"
    elif isinstance(input_data, str):
        input_file = input_data
        audio_name = os.path.basename(input_file).split(".")[0]
    else:
        raise ValueError("Input must be path or AudioSegment")
        
    if save and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    try:
        whisperx = importlib.import_module("whisperx")
        import torch
        device = "cuda" if getattr(torch, "cuda", None) and torch.cuda.is_available() else "cpu"
        model = whisperx.load_model("medium", device)
        audio = whisperx.load_audio(input_file)
        result = model.transcribe(audio, batch_size=8)
        align_model, metadata = whisperx.load_align_model(language_code=result.get("language","zh"), device=device)
        aligned = whisperx.align(result["segments"], align_model, metadata, audio, device, return_char_alignments=False)
        words = []
        for seg in aligned.get("segments", []):
            for w in seg.get("words", []):
                s = w.get("start", None)
                e = w.get("end", None)
                if s is not None and e is not None and e > s:
                    words.append((int(s*1000), int(e*1000)))
    except Exception as e:
        print(f"WhisperX failed: {e}, falling back to VAD+F0")
        if temp_input_path: os.unlink(temp_input_path)
        return split_audio_vad_f0(input_data, output_dir, max_duration, min_duration, 30, 0.6, min_quiet_ms, 80.0, extra_ms, save)
    
    if not words:
        if temp_input_path: os.unlink(temp_input_path)
        return split_audio_vad_f0(input_data, output_dir, max_duration, min_duration, 30, 0.6, min_quiet_ms, 80.0, extra_ms, save)
    
    audio = AudioSegment.from_file(input_file)
    audio_length = len(audio)
    gaps = []
    for i in range(len(words)-1):
        g = words[i+1][0] - words[i][1]
        if g >= min_quiet_ms:
            gaps.append(((words[i][1] + words[i+1][0]) // 2))
    segments = []
    start = 0
    while start < audio_length:
        search_start = start + min_duration
        search_end = min(start + max_duration, audio_length)
        if search_start >= audio_length:
            if segments:
                if len(segments[-1]) + (audio_length - start) <= max_duration:
                    segments[-1] += audio[start:]
                else:
                    segments.append(audio[start:])
            else:
                segments.append(audio[start:])
            break
        cands = [cp for cp in gaps if search_start <= cp <= search_end]
        if cands:
            cp = max(cands)
            seg_end = min(cp + extra_ms, audio_length)
            segments.append(audio[start:seg_end])
            start = cp
        else:
            seg_end = min(search_end + extra_ms, audio_length)
            segments.append(audio[start:seg_end])
            start = search_end
    
    if temp_input_path:
        os.unlink(temp_input_path)
            
    if save and output_dir:
        out = []
        elapsed = 0
        for i, seg in enumerate(segments):
            real_start = elapsed
            real_end = elapsed + len(seg) - extra_ms
            real_end = max(real_start + 1, real_end)
            elapsed += len(seg)
            fp = os.path.join(output_dir, f"{audio_name}_wx_{i+1}_{int(real_start)}_{int(real_end)}.wav")
            seg.export(fp, format="wav")
            out.append(os.path.abspath(fp))
        return out
    else:
        return segments
