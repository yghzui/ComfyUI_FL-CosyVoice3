# FL CosyVoice3 节点优化与重构思路

本文档记录了对 `ComfyUI_FL-CosyVoice3` 插件进行的一系列优化与重构的思路和设计细节。主要目的是为了提升节点的用户体验、消除参数歧义，并解决 ComfyUI 中列表（List）与批处理（Batch）逻辑冲突的问题。

## 1. Voice Conversion (VC) 节点优化

### 问题痛点
*   **参数命名歧义**：原有的 `Original Audio` 和 `Reference Audio` 或者 `Source/Target` 容易让用户混淆，不清楚哪个是提供“内容”，哪个是提供“音色”。
*   **功能单一**：仅输出转换后的音频，缺乏对齐原音频时长的功能，也无法直接获取原始音频用于后续对比。

### 解决方案
*   **明确参数命名**：
    *   **Original Audio (Content)**：明确表示这是提供**内容/说话内容**的音频。
    *   **Reference Audio (Timbre)**：明确表示这是提供**音色/说话人特征**的音频。
    *   日志打印同步更新，确保调试信息与界面参数一致。
*   **增加对齐功能**：
    *   新增 `alignment` 布尔参数。
    *   如果开启，会自动将生成的音频进行时间拉伸（Time Stretch），使其时长与 `Original Audio` 严格一致。这对于视频配音（Lip Sync）场景非常重要。
*   **扩展输出端口**：
    *   **Converted Audio**：原始转换结果（时长可能与原音频不同）。
    *   **Aligned Audio**：对齐后的音频（仅当 `alignment=True` 时有值，否则为 None 或未处理）。
    *   **Original Audio**：透传输入的原始音频，方便在工作流中直接连接后续节点（如合并或对比），无需重新引线。

## 2. 音频后处理工具重构 (Post-Process -> Align & Merge)

### 问题痛点
*   **List vs Batch 逻辑冲突**：原有的 `FL_Audio_PostProcess` 节点试图在一个节点内同时处理“列表输入（用于合并）”和“单项输入（用于对齐）”。
*   **ComfyUI 机制限制**：在 ComfyUI 中，如果节点定义了 `INPUT_IS_LIST = True`，那么所有输入都会变成列表。如果此时传入一个 Batch（批次），它会被视为一个列表。但对齐操作通常是“一对一”的（一个生成音频对应一个原音频），而合并操作是“多对一”的（多个片段合并为一个）。将两者混在一起会导致逻辑混乱，且难以利用 ComfyUI 自动的 List 展开（Batch Map）功能。

### 解决方案：拆分为两个独立节点

#### A. FL Audio Align (音频对齐节点)
*   **定位**：**一对一**处理。
*   **逻辑**：
    *   接收 `original_audio` 和 `generated_audio`。
    *   利用 ComfyUI 默认的批处理机制：如果输入是 Batch，节点会自动运行多次（或内部处理 Batch），实现每个片段各自对齐。
    *   **功能**：计算时长比例，调用 `time_stretch` 算法，输出对齐后的音频。
    *   **参数**：支持 `speed_factor` 手动调整速度，或 `alignment` 自动强制对齐。

#### B. FL Audio Merge (音频合并节点)
*   **定位**：**多对一**处理。
*   **逻辑**：
    *   设置 `INPUT_IS_LIST = True`。
    *   接收一个音频列表（通常由 `FL Audio Split` 切分并处理后的结果组成）。
    *   **功能**：将列表中的音频片段按顺序拼接。
    *   **Crossfade (淡入淡出)**：支持设置 `crossfade_ms`，在拼接处应用交叉淡化，使过渡更自然。
        *   *注意*：开启 Crossfade 会导致总时长缩短（因为首尾重叠了），如果需要严格保持总时长，应设为 0。

## 3. 技术实现细节

### 共享工具库 (`utils/audio_utils.py`)
*   为了避免代码重复，将核心算法 `time_stretch`（时间拉伸）封装在工具库中。
*   使用 `ffmpeg` 的 `atempo` 滤镜实现高质量变速不变调。
*   实现了 `tensor` (ComfyUI 格式) 与 `AudioSegment` (pydub 格式) 的互转工具。

### 模块导入处理
*   在节点文件中（如 `nodes/audio_tools.py`），采用健壮的导入策略。
*   优先尝试相对导入，若失败则尝试绝对路径导入，确保在 ComfyUI 复杂的插件加载环境下能正确找到 `utils` 模块。

---
*文档生成时间：2026-03-16*
