#!/usr/bin/env python3
"""
会议声纹增强模块
用于提升会议中说话人识别的精度
"""

import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

try:
    from resemblyzer import VoiceEncoder, preprocess_wav
    import librosa
    RESEMBLYZER_AVAILABLE = True
except ImportError:
    RESEMBLYZER_AVAILABLE = False
    print("警告: Resemblyzer库不可用，请运行 'pip install resemblyzer'")

logger = logging.getLogger(__name__)


class MeetingVoiceEnhancer:
    """会议声纹增强器"""
    
    def __init__(self, speaker_dir: str = "speaker", threshold: float = 0.75):
        """
        初始化会议声纹增强系统
        
        Args:
            speaker_dir: 说话人样本音频目录
            threshold: 声纹匹配相似度阈值
        """
        if not RESEMBLYZER_AVAILABLE:
            raise ImportError("Resemblyzer库不可用，请运行 'pip install resemblyzer'")
        
        self.speaker_dir = Path(speaker_dir)
        self.threshold = threshold
        self.voice_encoder = VoiceEncoder()
        
        # 存储说话人嵌入向量
        self.speaker_embeddings = {}
        self.speaker_names = []
        
        # 加载已知说话人
        self._load_speaker_profiles()
    
    def _load_speaker_profiles(self):
        """加载已知说话人的声纹档案"""
        if not self.speaker_dir.exists():
            logger.warning(f"说话人目录不存在: {self.speaker_dir}")
            return
        
        wav_files = list(self.speaker_dir.glob("*.wav"))
        logger.info(f"发现 {len(wav_files)} 个说话人样本文件")
        
        for wav_file in wav_files:
            speaker_name = wav_file.stem
            try:
                # 加载和预处理音频
                wav, sr = librosa.load(str(wav_file), sr=16000)
                wav = preprocess_wav(wav)
                
                # 提取声纹嵌入向量
                embedding = self.voice_encoder.embed_utterance(wav)
                
                self.speaker_embeddings[speaker_name] = embedding
                self.speaker_names.append(speaker_name)
                
                logger.info(f"已加载说话人声纹: {speaker_name}")
                
            except Exception as e:
                logger.error(f"加载说话人 {speaker_name} 失败: {e}")
        
        print(f"✓ 成功加载 {len(self.speaker_embeddings)} 个说话人声纹档案")
    
    def _extract_segment_embedding(self, audio_data: np.ndarray, start_time: float, end_time: float, sample_rate: int = 16000) -> Optional[np.ndarray]:
        """
        提取音频片段的声纹嵌入向量
        
        Args:
            audio_data: 完整音频数据
            start_time: 开始时间（秒）
            end_time: 结束时间（秒）
            sample_rate: 采样率
            
        Returns:
            声纹嵌入向量或None
        """
        try:
            # 计算样本索引
            start_sample = int(start_time * sample_rate)
            end_sample = int(end_time * sample_rate)
            
            # 提取音频片段
            segment = audio_data[start_sample:end_sample]
            
            # 确保片段长度足够
            if len(segment) < sample_rate * 0.5:  # 至少0.5秒
                return None
            
            # 预处理音频片段
            segment = preprocess_wav(segment)
            
            # 提取嵌入向量
            embedding = self.voice_encoder.embed_utterance(segment)
            
            return embedding
            
        except Exception as e:
            logger.error(f"提取片段嵌入向量失败: {e}")
            return None
    
    def _identify_speaker(self, segment_embedding: np.ndarray) -> Tuple[Optional[str], float]:
        """
        根据嵌入向量识别说话人
        
        Args:
            segment_embedding: 音频片段的嵌入向量
            
        Returns:
            (说话人名称, 相似度分数)
        """
        if not self.speaker_embeddings:
            return None, 0.0
        
        best_match = None
        best_similarity = 0.0
        
        for speaker_name, known_embedding in self.speaker_embeddings.items():
            # 计算余弦相似度
            similarity = np.dot(segment_embedding, known_embedding)
            
            if similarity > best_similarity and similarity > self.threshold:
                best_similarity = similarity
                best_match = speaker_name
        
        return best_match, best_similarity
    
    def enhance_speaker_identification(self, segments: List[Dict], audio_file: str) -> List[Dict]:
        """
        使用Resemblyzer增强说话人识别
        
        Args:
            segments: 带有pyannote说话人标签的片段列表
            audio_file: 原始音频文件路径
            
        Returns:
            增强后的片段列表
        """
        if not self.speaker_embeddings:
            logger.warning("没有加载说话人声纹档案，跳过Resemblyzer增强")
            return segments
        
        try:
            # 加载完整音频文件
            audio_data, sr = librosa.load(audio_file, sr=16000)
            
            enhanced_segments = []
            resemblyzer_matches = 0
            
            for segment in segments:
                start_time = segment.get('start', 0)
                end_time = segment.get('end', 0)
                original_speaker = segment.get('speaker', 'Unknown')
                
                # 提取音频片段的嵌入向量
                segment_embedding = self._extract_segment_embedding(audio_data, start_time, end_time, sr)
                
                # 创建增强的片段
                enhanced_segment = segment.copy()
                
                if segment_embedding is not None:
                    # 使用Resemblyzer识别说话人
                    identified_speaker, similarity = self._identify_speaker(segment_embedding)
                    
                    if identified_speaker:
                        # Resemblyzer成功识别
                        enhanced_segment['speaker'] = identified_speaker
                        enhanced_segment['speaker_confidence'] = 'high'
                        enhanced_segment['recognition_method'] = 'resemblyzer'
                        enhanced_segment['similarity_score'] = float(similarity)
                        resemblyzer_matches += 1
                    else:
                        # Resemblyzer未识别，保持原有标签
                        enhanced_segment['speaker_confidence'] = 'medium'
                        enhanced_segment['recognition_method'] = 'pyannote'
                        enhanced_segment['similarity_score'] = 0.0
                else:
                    # 无法提取嵌入向量，保持原有标签
                    enhanced_segment['speaker_confidence'] = 'low'
                    enhanced_segment['recognition_method'] = 'pyannote'
                    enhanced_segment['similarity_score'] = 0.0
                
                enhanced_segments.append(enhanced_segment)
            
            success_rate = (resemblyzer_matches / len(segments)) * 100 if segments else 0
            logger.info(f"Resemblyzer识别成功率: {success_rate:.1f}% ({resemblyzer_matches}/{len(segments)})")
            
            return enhanced_segments
            
        except Exception as e:
            logger.error(f"Resemblyzer增强说话人识别失败: {e}")
            return segments  # 返回原始片段
    
    def get_speaker_statistics(self, segments: List[Dict]) -> Dict:
        """
        获取说话人统计信息
        
        Args:
            segments: 片段列表
            
        Returns:
            统计信息字典
        """
        stats = {
            'total_segments': len(segments),
            'speakers': {},
            'recognition_methods': {'resemblyzer': 0, 'pyannote': 0},
            'confidence_levels': {'high': 0, 'medium': 0, 'low': 0},
            'average_similarity': 0.0,
            'resemblyzer_success_rate': 0.0
        }
        
        total_similarity = 0.0
        resemblyzer_count = 0
        
        for segment in segments:
            speaker = segment.get('speaker', 'Unknown')
            method = segment.get('recognition_method', 'unknown')
            confidence = segment.get('speaker_confidence', 'low')
            similarity = segment.get('similarity_score', 0.0)
            
            # 统计说话人
            if speaker not in stats['speakers']:
                stats['speakers'][speaker] = {
                    'segments': 0,
                    'total_duration': 0.0,
                    'average_similarity': 0.0,
                    'total_similarity': 0.0
                }
            
            stats['speakers'][speaker]['segments'] += 1
            duration = segment.get('end', 0) - segment.get('start', 0)
            stats['speakers'][speaker]['total_duration'] += duration
            stats['speakers'][speaker]['total_similarity'] += similarity
            
            # 统计识别方法
            if method in stats['recognition_methods']:
                stats['recognition_methods'][method] += 1
                if method == 'resemblyzer':
                    resemblyzer_count += 1
            
            # 统计置信度
            if confidence in stats['confidence_levels']:
                stats['confidence_levels'][confidence] += 1
            
            total_similarity += similarity
        
        # 计算平均相似度
        if len(segments) > 0:
            stats['average_similarity'] = total_similarity / len(segments)
            stats['resemblyzer_success_rate'] = (resemblyzer_count / len(segments)) * 100
        
        # 计算每个说话人的平均相似度
        for speaker_stats in stats['speakers'].values():
            if speaker_stats['segments'] > 0:
                speaker_stats['average_similarity'] = speaker_stats['total_similarity'] / speaker_stats['segments']
        
        return stats
    
    def print_enhancement_summary(self, segments: List[Dict]):
        """打印增强结果摘要"""
        stats = self.get_speaker_statistics(segments)
        
        print("\n" + "="*60)
        print("Resemblyzer声纹识别增强结果")
        print("="*60)
        
        print(f"总片段数: {stats['total_segments']}")
        print(f"识别到的说话人: {len(stats['speakers'])}")
        print(f"Resemblyzer成功率: {stats['resemblyzer_success_rate']:.1f}%")
        print(f"平均相似度: {stats['average_similarity']:.3f}")
        
        print(f"\n识别方法统计:")
        for method, count in stats['recognition_methods'].items():
            percentage = (count / stats['total_segments']) * 100 if stats['total_segments'] > 0 else 0
            print(f"  - {method}: {count} ({percentage:.1f}%)")
        
        print(f"\n置信度统计:")
        for confidence, count in stats['confidence_levels'].items():
            percentage = (count / stats['total_segments']) * 100 if stats['total_segments'] > 0 else 0
            print(f"  - {confidence}: {count} ({percentage:.1f}%)")
        
        print(f"\n说话人发言统计:")
        for speaker, info in stats['speakers'].items():
            duration = info['total_duration']
            minutes = int(duration // 60)
            seconds = int(duration % 60)
            avg_sim = info['average_similarity']
            print(f"  - {speaker}: {info['segments']} 片段, {minutes}分{seconds}秒, 平均相似度: {avg_sim:.3f}")
        
        print("="*60)
    
    def get_known_speakers(self) -> List[str]:
        """获取已知说话人列表"""
        return self.speaker_names.copy()
    
    def set_threshold(self, threshold: float):
        """设置相似度阈值"""
        self.threshold = max(0.0, min(1.0, threshold))
        logger.info(f"相似度阈值已设置为: {self.threshold}")
    
    def reload_speaker_profiles(self):
        """重新加载说话人声纹档案"""
        print("🔄 正在重新加载声纹库...")
        
        # 清空现有声纹
        self.speaker_embeddings.clear()
        self.speaker_names.clear()
        
        # 重新加载
        self._load_speaker_profiles()
        
        print(f"✅ 声纹库重新加载完成，包含 {len(self.speaker_embeddings)} 个说话人")
        return len(self.speaker_embeddings)
    
    def add_speaker_from_file(self, wav_file_path: str) -> bool:
        """
        从文件添加单个说话人声纹
        
        Args:
            wav_file_path: wav文件路径
            
        Returns:
            是否添加成功
        """
        wav_path = Path(wav_file_path)
        
        if not wav_path.exists():
            logger.error(f"文件不存在: {wav_file_path}")
            return False
        
        if wav_path.suffix.lower() != '.wav':
            logger.error(f"只支持wav文件: {wav_file_path}")
            return False
        
        speaker_name = wav_path.stem
        
        try:
            # 加载和预处理音频
            wav, sr = librosa.load(str(wav_path), sr=16000)
            wav = preprocess_wav(wav)
            
            # 提取声纹嵌入向量
            embedding = self.voice_encoder.embed_utterance(wav)
            
            # 添加到声纹库
            self.speaker_embeddings[speaker_name] = embedding
            if speaker_name not in self.speaker_names:
                self.speaker_names.append(speaker_name)
            
            logger.info(f"成功添加说话人声纹: {speaker_name}")
            return True
            
        except Exception as e:
            logger.error(f"添加说话人 {speaker_name} 失败: {e}")
            return False


def demo():
    """演示功能"""
    try:
        # 初始化Resemblyzer声纹识别
        recognizer = ResemblyzerSpeakerRecognition()
        
        print(f"已知说话人: {recognizer.get_known_speakers()}")
        
        # 模拟一些片段数据
        test_segments = [
            {
                'start': 0.0,
                'end': 5.0,
                'text': '大家好，欢迎参加今天的会议。',
                'speaker': 'SPEAKER_00'
            },
            {
                'start': 5.0,
                'end': 10.0,
                'text': '谢谢主持人，我来介绍一下项目进展。',
                'speaker': 'SPEAKER_01'
            }
        ]
        
        print("\n原始片段:")
        for segment in test_segments:
            print(f"  {segment['speaker']}: {segment['text']}")
        
        # 注意：这里需要实际的音频文件来测试
        # enhanced_segments = recognizer.enhance_speaker_identification(test_segments, "test.m4a")
        # recognizer.print_enhancement_summary(enhanced_segments)
        
    except Exception as e:
        print(f"演示失败: {e}")


if __name__ == "__main__":
    demo()