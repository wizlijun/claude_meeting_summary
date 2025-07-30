# 会议录音转录工具

基于 WhisperX + Resemblyzer 实现的会议录音转录工具，支持声纹增强识别。

## 功能特点

- **语音转录**: 使用 WhisperX 进行高质量语音转录
- **声纹识别**: 使用 Resemblyzer 进行说话人声纹识别
- **多种输出格式**: 支持 TXT、JSON、SRT 格式输出
- **说话人管理**: 支持预设说话人声纹样本
- **GPU 加速**: 支持 CUDA 和 MPS 加速（默认使用 CPU）

## 环境要求

- Python 3.8+
- 可选：CUDA 或 MPS 设备（用于GPU加速）

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 基本使用

```bash
python meeting_transcription.py input.m4a
```

### 完整参数

```bash
python meeting_transcription.py input.m4a \
    --output output.srt \
    --format srt \
    --model medium \
    --language auto \
    --num-speakers 3
```

### 指定说话人数量

```bash
# 自动检测说话人数量（默认）
python meeting_transcription.py input.m4a

# 指定说话人数量为2
python meeting_transcription.py input.m4a -n 2

# 指定说话人数量为4
python meeting_transcription.py input.m4a --num-speakers 4
```

### 启用 GPU 加速

```bash
python meeting_transcription.py input.m4a --gpu
```

### 参数说明

- `audio_file`: 音频文件路径（必需）
- `-o, --output`: 输出文件路径（可选，默认为输入文件名+扩展名）
- `-f, --format`: 输出格式（txt/json/srt，默认：srt）
- `-m, --model`: WhisperX 模型大小（tiny/base/small/medium/large-v2/large-v3，默认：medium）
- `-l, --language`: 语言代码（默认：auto 自动检测）
- `-n, --num-speakers`: 指定说话人数量（默认：auto 自动检测，或输入具体数字如2、3、4等）
- `--gpu`: 启用 GPU 加速（默认使用 CPU）
- `--no-resemblyzer`: 禁用声纹增强功能（默认启用）

## 说话人管理

### 声纹管理工具

本项目提供了便捷的声纹管理工具 `add_speaker.py`，支持以下功能：

#### 查看当前声纹库
```bash
source whisperx_env/bin/activate && python add_speaker.py list
```

#### 添加新的说话人声纹
```bash
source whisperx_env/bin/activate && python add_speaker.py add <wav文件路径>
```

#### 扫描并重新加载所有声纹
```bash
source whisperx_env/bin/activate && python add_speaker.py scan
```

### 手动添加说话人样本

1. 在 `speaker/` 目录下放置说话人的音频样本文件
2. 文件名格式：`{说话人姓名}.wav`
3. 建议每个样本长度 5-30 秒
4. 运行 `python add_speaker.py scan` 重新加载声纹库

示例目录结构：
```
speaker/
├── Jun.wav
├── Qi.wav
├── Tao.wav
├── Dive.wav
└── Xusw.wav
```

### 声纹管理工具功能特点

- ✅ 自动检测wav文件格式
- ✅ 自动复制文件到speaker目录
- ✅ 实时生成256维声纹特征
- ✅ 显示声纹库统计信息
- ✅ 支持重新加载无需重启程序

### 说话人映射

系统会自动将检测到的 `SPEAKER_00`、`SPEAKER_01` 等映射到对应的说话人姓名。映射关系保存在 `speaker_metadata.json` 文件中。

## 输出格式

### TXT 格式
```
[00:00:05 - 00:00:10] Jun:
大家好，欢迎参加今天的会议。

[00:00:10 - 00:00:15] Qi:
谢谢主持人，我来介绍一下项目进展。
```

### SRT 格式
```
1
00:00:05,000 --> 00:00:10,000
Jun: 大家好，欢迎参加今天的会议。

2
00:00:10,000 --> 00:00:15,000
Qi: 谢谢主持人，我来介绍一下项目进展。
```

### JSON 格式
```json
{
  "audio_file": "input.m4a",
  "language": "zh",
  "segments": [
    {
      "start": 5.0,
      "end": 10.0,
      "text": "大家好，欢迎参加今天的会议。",
      "speaker": "Jun"
    }
  ]
}
```

## 高级功能

### 声纹识别

声纹识别功能默认启用。如需禁用，使用 `--no-resemblyzer` 参数：

```bash
python meeting_transcription.py input.m4a --no-resemblyzer
```

声纹识别会：
- 基于预设的说话人样本进行精确的识别
- 为每个片段提供置信度评分
- 显示识别方法统计信息

## 性能优化

- 使用 GPU 加速：添加 `--gpu` 参数并确保安装 CUDA 或 MPS 支持
- 选择合适的模型大小：`tiny` 速度快但精度低，`large-v3` 精度高但速度慢
- 调整批处理大小：根据显存大小调整（在代码中修改 `batch_size` 参数）

## 故障排除

### 常见问题

1. **GPU 内存不足**
   - 默认使用 CPU 模式，如需 GPU 加速请使用 `--gpu` 参数
   - 选择更小的模型（如 `tiny` 或 `base`）

2. **声纹识别失败**
   - 确保 `speaker/` 目录存在且包含音频样本
   - 检查 Resemblyzer 库是否正确安装

### 依赖问题

如果遇到依赖安装问题，可以尝试：

```bash
# 创建虚拟环境
python -m venv whisperx_env
source whisperx_env/bin/activate  # Linux/Mac
# 或 whisperx_env\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

## 许可证

本项目使用开源许可证。使用的主要依赖库：
- WhisperX: Apache 2.0
- Resemblyzer: MIT
