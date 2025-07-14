#!/usr/bin/env python3
"""
会议录音转录工具 - 基于WhisperX + pyannote-audio实现
将会议录音转换为带说话人标识的SRT字幕文件
支持多说话人识别和声纹增强
"""

import whisperx
import torch
import argparse
import os
from pathlib import Path
import json
from datetime import timedelta
import gc

# 导入说话人管理系统
try:
    from meeting_speaker_manager import MeetingSpeakerManager
    SPEAKER_MANAGER_AVAILABLE = True
except ImportError:
    SPEAKER_MANAGER_AVAILABLE = False
    print("警告: 会议说话人管理系统不可用")

# 导入会议声纹增强功能
try:
    from meeting_voice_enhancement import MeetingVoiceEnhancer
    RESEMBLYZER_AVAILABLE = True
except ImportError:
    RESEMBLYZER_AVAILABLE = False
    print("警告: 会议声纹增强功能不可用")

class MeetingTranscriber:
    def __init__(self, model_size="base", use_gpu=True, language="auto", use_resemblyzer=False):
        """
        初始化会议录音转录系统
        
        Args:
            model_size (str): WhisperX模型大小 ("tiny", "base", "small", "medium", "large-v2", "large-v3")
            use_gpu (bool): 是否使用GPU加速
            language (str): 语言代码，"auto"为自动检测
            use_resemblyzer (bool): 是否使用声纹增强功能
        """
        # 设备选择
        if use_gpu:
            if torch.cuda.is_available():
                self.device = "cuda"
                print("✓ 使用 CUDA 加速")
            elif torch.backends.mps.is_available():
                self.device = "mps"
                print("✓ 使用 MPS 加速")
            else:
                self.device = "cpu"
                print("⚠ GPU不可用，使用CPU")
        else:
            self.device = "cpu"
            print("使用CPU（GPU加速已禁用）")
        
        print(f"最终设备: {self.device}")
        
        # 初始化WhisperX模型
        print(f"正在加载WhisperX模型 ({model_size})...")
        try:
            # 根据设备选择compute_type
            compute_type = "float16" if self.device in ["cuda"] else "int8"
            self.model = whisperx.load_model(
                model_size, 
                self.device, 
                compute_type=compute_type,
                language=language if language != "auto" else None
            )
            print("✓ WhisperX模型加载成功")
        except Exception as e:
            print(f"❌ WhisperX模型加载失败: {e}")
            raise e
        
        # 初始化对齐模型（稍后加载）
        self.align_model = None
        self.metadata = None
        
        # 初始化说话人分离模型（稍后加载）
        self.diarize_model = None
        
        # 初始化说话人管理器
        if SPEAKER_MANAGER_AVAILABLE:
            try:
                self.speaker_manager = MeetingSpeakerManager()
                print("✓ 说话人管理器已加载")
            except Exception as e:
                print(f"⚠ 说话人管理器加载失败: {e}")
                self.speaker_manager = None
        else:
            self.speaker_manager = None
        
        # 初始化Resemblyzer声纹识别增强器
        self.use_resemblyzer = use_resemblyzer
        self.resemblyzer_recognizer = None
        
        if use_resemblyzer and RESEMBLYZER_AVAILABLE:
            try:
                self.resemblyzer_recognizer = MeetingVoiceEnhancer()
                print("✓ 会议声纹增强器已加载")
            except Exception as e:
                print(f"⚠ 会议声纹增强器加载失败: {e}")
                self.resemblyzer_recognizer = None
        elif use_resemblyzer and not RESEMBLYZER_AVAILABLE:
            print("⚠ 声纹增强库不可用，无法启用声纹识别增强")
    
    def load_align_model(self, language_code):
        """加载对齐模型"""
        if self.align_model is None:
            print("正在加载对齐模型...")
            try:
                self.align_model, self.metadata = whisperx.load_align_model(
                    language_code=language_code, 
                    device=self.device
                )
                print("✓ 对齐模型加载成功")
            except Exception as e:
                print(f"⚠ 对齐模型加载失败: {e}")
                self.align_model = None
                self.metadata = None
    
    def load_diarize_model(self, hf_token=None):
        """加载说话人分离模型"""
        if self.diarize_model is None:
            print("正在加载pyannote说话人分离模型...")
            try:
                self.diarize_model = whisperx.DiarizationPipeline(
                    use_auth_token=hf_token, 
                    device=self.device
                )
                print("✓ 说话人分离模型加载成功")
            except Exception as e:
                print(f"⚠ 说话人分离模型加载失败: {e}")
                print("提示: 可能需要Hugging Face token来访问pyannote模型")
                self.diarize_model = None
    
    def transcribe_audio(self, audio_path):
        """
        使用WhisperX转录音频文件
        
        Args:
            audio_path (str): 音频文件路径
            
        Returns:
            dict: 转录结果
        """
        print(f"正在使用WhisperX转录音频文件: {audio_path}")
        try:
            # 加载音频
            audio = whisperx.load_audio(audio_path)
            
            # 转录
            result = self.model.transcribe(audio, batch_size=16)
            
            print(f"✓ 转录完成，检测到语言: {result.get('language', 'unknown')}")
            return result
            
        except Exception as e:
            print(f"❌ WhisperX转录失败: {e}")
            raise e
    
    def align_transcription(self, result, audio_path):
        """
        对转录结果进行时间对齐
        
        Args:
            result (dict): 转录结果
            audio_path (str): 音频文件路径
            
        Returns:
            dict: 对齐后的结果
        """
        language_code = result.get("language", "en")
        
        # 加载对齐模型
        self.load_align_model(language_code)
        
        if self.align_model is None:
            print("⚠ 跳过对齐步骤")
            return result
        
        print("正在进行时间对齐...")
        try:
            # 重新加载音频用于对齐
            audio = whisperx.load_audio(audio_path)
            
            # 对齐
            result = whisperx.align(
                result["segments"], 
                self.align_model, 
                self.metadata, 
                audio, 
                self.device, 
                return_char_alignments=False
            )
            
            print("✓ 时间对齐完成")
            return result
            
        except Exception as e:
            print(f"⚠ 时间对齐失败: {e}")
            return result
    
    def diarize_speakers(self, audio_path, min_speakers=None, max_speakers=None, hf_token=None):
        """
        进行说话人分离
        
        Args:
            audio_path (str): 音频文件路径
            min_speakers (int): 最小说话人数量
            max_speakers (int): 最大说话人数量
            hf_token (str): Hugging Face token
            
        Returns:
            object: 说话人分离结果
        """
        # 加载说话人分离模型
        self.load_diarize_model(hf_token)
        
        if self.diarize_model is None:
            print("⚠ 说话人分离不可用")
            return None
        
        print("正在进行说话人分离...")
        try:
            # 加载音频
            audio = whisperx.load_audio(audio_path)
            
            # 进行说话人分离
            diarize_segments = self.diarize_model(
                audio, 
                min_speakers=min_speakers, 
                max_speakers=max_speakers
            )
            
            print("✓ 说话人分离完成")
            return diarize_segments
            
        except Exception as e:
            print(f"⚠ 说话人分离失败: {e}")
            return None
    
    def assign_speakers_to_segments(self, result, diarize_segments):
        """
        将说话人信息分配给转录片段
        
        Args:
            result (dict): 转录结果
            diarize_segments: 说话人分离结果
            
        Returns:
            dict: 带有说话人信息的转录结果
        """
        if diarize_segments is None:
            print("⚠ 无说话人分离结果，使用默认说话人标签")
            # 为所有片段分配默认说话人
            for segment in result["segments"]:
                segment["speaker"] = "SPEAKER_00"
            return result
        
        print("正在分配说话人标签...")
        try:
            # 使用WhisperX的assign_word_speakers函数
            result = whisperx.assign_word_speakers(diarize_segments, result)
            
            # 如果有说话人管理器，使用它来映射speaker ID到名称
            if self.speaker_manager:
                for segment in result["segments"]:
                    if "speaker" in segment:
                        original_speaker = segment["speaker"]
                        mapped_name = self.speaker_manager.assign_speaker_name(original_speaker)
                        segment["speaker"] = mapped_name
            
            print("✓ 说话人标签分配完成")
            return result
            
        except Exception as e:
            print(f"⚠ 说话人标签分配失败: {e}")
            # 回退：为所有片段分配默认说话人
            for segment in result["segments"]:
                segment["speaker"] = "SPEAKER_00"
            return result
    
    def format_time(self, seconds):
        """
        格式化时间为 HH:MM:SS 格式
        
        Args:
            seconds (float): 秒数
            
        Returns:
            str: 格式化的时间字符串
        """
        return str(timedelta(seconds=int(seconds)))
    
    def process_audio(self, audio_path, min_speakers=None, max_speakers=None, hf_token=None):
        """
        处理音频文件，生成带说话人标识的字幕
        
        Args:
            audio_path (str): 音频文件路径
            min_speakers (int): 最小说话人数量
            max_speakers (int): 最大说话人数量
            hf_token (str): Hugging Face token
            
        Returns:
            dict: 处理结果
        """
        print("=" * 50)
        print("开始处理音频文件")
        print("=" * 50)
        
        # 1. 转录音频
        result = self.transcribe_audio(audio_path)
        
        # 2. 时间对齐
        result = self.align_transcription(result, audio_path)
        
        # 3. 说话人分离
        diarize_segments = self.diarize_speakers(
            audio_path, 
            min_speakers=min_speakers, 
            max_speakers=max_speakers,
            hf_token=hf_token
        )
        
        # 4. 分配说话人标签
        result = self.assign_speakers_to_segments(result, diarize_segments)
        
        # 5. 使用Resemblyzer增强说话人识别（如果启用）
        if self.resemblyzer_recognizer:
            print("正在使用Resemblyzer增强说话人识别...")
            try:
                enhanced_segments = self.resemblyzer_recognizer.enhance_speaker_identification(
                    result.get('segments', []), 
                    audio_path
                )
                result['segments'] = enhanced_segments
                print("✓ Resemblyzer声纹识别增强完成")
            except Exception as e:
                print(f"⚠ Resemblyzer声纹识别增强失败: {e}")
        
        # 6. 生成最终结果
        final_result = {
            'audio_file': audio_path,
            'language': result.get('language', 'unknown'),
            'segments': result.get('segments', []),
            'full_text': ' '.join([seg.get('text', '') for seg in result.get('segments', [])])
        }
        
        # 清理GPU内存
        if self.device in ["cuda", "mps"]:
            gc.collect()
            if self.device == "cuda":
                torch.cuda.empty_cache()
        
        # 保存说话人映射
        if self.speaker_manager:
            try:
                self.speaker_manager.save_metadata()
                print("✓ 说话人映射已保存")
            except Exception as e:
                print(f"⚠ 保存说话人映射失败: {e}")
        
        print("=" * 50)
        print("音频处理完成")
        print("=" * 50)
        
        return final_result
    
    def save_results(self, result, output_path, output_format="txt"):
        """
        保存结果到文件
        
        Args:
            result (dict): 处理结果
            output_path (str): 输出文件路径
            output_format (str): 输出格式
        """
        if output_format == "json":
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        
        elif output_format == "txt":
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(f"音频文件: {result['audio_file']}\n")
                f.write(f"语言: {result['language']}\n")
                f.write("=" * 50 + "\n\n")
                
                for segment in result['segments']:
                    start_time = self.format_time(segment.get('start', 0))
                    end_time = self.format_time(segment.get('end', 0))
                    speaker = segment.get('speaker', 'Unknown')
                    text = segment.get('text', '').strip()
                    
                    f.write(f"[{start_time} - {end_time}] {speaker}:\n")
                    f.write(f"{text}\n\n")
        
        elif output_format == "srt":
            with open(output_path, 'w', encoding='utf-8') as f:
                for i, segment in enumerate(result['segments'], 1):
                    start_time = self.format_srt_time(segment.get('start', 0))
                    end_time = self.format_srt_time(segment.get('end', 0))
                    speaker = segment.get('speaker', 'Unknown')
                    text = segment.get('text', '').strip()
                    
                    f.write(f"{i}\n")
                    f.write(f"{start_time} --> {end_time}\n")
                    f.write(f"{speaker}: {text}\n\n")
    
    def format_srt_time(self, seconds):
        """
        格式化时间为SRT格式 (HH:MM:SS,mmm)
        
        Args:
            seconds (float): 秒数
            
        Returns:
            str: SRT格式的时间字符串
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds % 1) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"

def main():
    parser = argparse.ArgumentParser(description="会议录音转录工具 - 将会议录音转换为带说话人标识的SRT字幕")
    parser.add_argument("audio_file", help="音频文件路径")
    parser.add_argument("-o", "--output", help="输出文件路径 (可选)")
    parser.add_argument("-f", "--format", choices=["txt", "json", "srt"], 
                       default="srt", help="输出格式")
    parser.add_argument("-m", "--model", choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"], 
                       default="base", help="WhisperX模型大小")
    parser.add_argument("-l", "--language", default="auto", help="语言代码 (auto为自动检测)")
    parser.add_argument("--no-gpu", action="store_true", help="不使用GPU")
    parser.add_argument("--min-speakers", type=int, help="最小说话人数量")
    parser.add_argument("--max-speakers", type=int, help="最大说话人数量")
    parser.add_argument("--hf-token", help="Hugging Face token (用于pyannote模型)")
    parser.add_argument("--use-resemblyzer", action="store_true", help="启用声纹增强识别功能")
    
    args = parser.parse_args()
    
    # 检查音频文件是否存在
    if not os.path.exists(args.audio_file):
        print(f"错误: 音频文件不存在: {args.audio_file}")
        return
    
    # 生成输出文件路径
    if args.output:
        output_path = args.output
    else:
        audio_path = Path(args.audio_file)
        output_path = audio_path.with_suffix(f".{args.format}")
    
    # 创建处理器实例
    processor = MeetingTranscriber(
        model_size=args.model,
        use_gpu=not args.no_gpu,
        language=args.language,
        use_resemblyzer=args.use_resemblyzer
    )
    
    try:
        # 处理音频
        print("开始处理音频...")
        result = processor.process_audio(
            args.audio_file,
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers,
            hf_token=args.hf_token
        )
        
        # 保存结果
        processor.save_results(result, output_path, args.format)
        
        print(f"处理完成! 结果已保存到: {output_path}")
        
        # 显示简要统计信息
        print(f"\n统计信息:")
        print(f"- 语言: {result['language']}")
        print(f"- 片段数量: {len(result['segments'])}")
        
        # 显示说话人列表
        speakers = set(segment.get('speaker', 'Unknown') for segment in result['segments'])
        print(f"- 识别到的说话人: {', '.join(sorted(speakers))}")
        
        # 显示每个说话人的发言时长统计
        speaker_durations = {}
        for segment in result['segments']:
            speaker = segment.get('speaker', 'Unknown')
            duration = segment.get('end', 0) - segment.get('start', 0)
            if speaker not in speaker_durations:
                speaker_durations[speaker] = 0
            speaker_durations[speaker] += duration
        
        print(f"\n说话人发言时长统计:")
        for speaker, duration in sorted(speaker_durations.items()):
            minutes = int(duration // 60)
            seconds = int(duration % 60)
            print(f"- {speaker}: {minutes}分{seconds}秒")
        
        # 显示说话人管理器状态
        if processor.speaker_manager:
            processor.speaker_manager.print_summary()
        
        # 显示Resemblyzer增强结果
        if processor.resemblyzer_recognizer:
            processor.resemblyzer_recognizer.print_enhancement_summary(result['segments'])
        
    except Exception as e:
        print(f"处理过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()