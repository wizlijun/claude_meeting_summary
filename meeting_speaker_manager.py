#!/usr/bin/env python3
"""
会议说话人管理系统
用于管理和识别会议中的不同说话人
"""

import os
import json
from datetime import datetime
from pathlib import Path


class MeetingSpeakerManager:
    """会议说话人管理器"""
    
    def __init__(self, speaker_dir="speaker", metadata_path="speaker_metadata.json"):
        """
        初始化会议说话人管理器
        
        Args:
            speaker_dir (str): 说话人音频样本目录路径
            metadata_path (str): 元数据文件路径
        """
        self.speaker_dir = speaker_dir
        self.metadata_path = metadata_path
        self.speaker_names = {}  # speaker_id -> name映射
        
        # 扫描speaker目录
        self.scan_speaker_directory()
        
        # 加载或创建元数据
        self.load_metadata()
    
    def scan_speaker_directory(self):
        """扫描speaker目录，获取所有人名"""
        self.known_speakers = []
        
        if not os.path.exists(self.speaker_dir):
            print(f"警告: speaker目录不存在: {self.speaker_dir}")
            return
        
        # 获取所有wav文件的文件名（去掉扩展名）
        wav_files = list(Path(self.speaker_dir).glob("*.wav"))
        self.known_speakers = [wav_file.stem for wav_file in wav_files]
        
        if self.known_speakers:
            print(f"发现 {len(self.known_speakers)} 个已知说话人: {', '.join(self.known_speakers)}")
        else:
            print("未发现已知说话人文件")
    
    def load_metadata(self):
        """加载元数据"""
        if os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.speaker_names = data.get('speaker_names', {})
                print(f"已加载说话人映射: {len(self.speaker_names)} 个")
            except Exception as e:
                print(f"加载元数据失败: {e}")
                self.speaker_names = {}
        else:
            self.speaker_names = {}
    
    def save_metadata(self):
        """保存元数据"""
        try:
            data = {
                'speaker_names': self.speaker_names,
                'known_speakers': self.known_speakers,
                'last_updated': datetime.now().isoformat()
            }
            
            with open(self.metadata_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            print(f"元数据已保存到: {self.metadata_path}")
        except Exception as e:
            print(f"保存元数据失败: {e}")
    
    def assign_speaker_name(self, speaker_id, segments_info=None):
        """
        为speaker ID分配名称
        
        Args:
            speaker_id (str): pyannote识别的speaker ID (如 "SPEAKER_00")
            segments_info (dict): 可选的片段信息用于智能分配
            
        Returns:
            str: 分配的说话人名称
        """
        # 如果已经有映射，直接返回
        if speaker_id in self.speaker_names:
            return self.speaker_names[speaker_id]
        
        # 如果有已知说话人，按顺序分配
        if self.known_speakers:
            # 提取speaker编号
            try:
                speaker_num = int(speaker_id.split('_')[-1])
                if speaker_num < len(self.known_speakers):
                    assigned_name = self.known_speakers[speaker_num]
                    self.speaker_names[speaker_id] = assigned_name
                    return assigned_name
            except (ValueError, IndexError):
                pass
            
            # 如果编号超出范围，分配第一个未使用的名称
            used_names = set(self.speaker_names.values())
            for name in self.known_speakers:
                if name not in used_names:
                    self.speaker_names[speaker_id] = name
                    return name
        
        # 默认返回原始speaker ID
        self.speaker_names[speaker_id] = speaker_id
        return speaker_id
    
    def get_speaker_summary(self):
        """获取说话人总结"""
        return {
            'known_speakers': self.known_speakers,
            'speaker_mapping': self.speaker_names,
            'total_mapped': len(self.speaker_names)
        }
    
    def print_summary(self):
        """打印说话人总结"""
        print("\n" + "="*50)
        print("说话人管理系统状态")
        print("="*50)
        print(f"已知说话人: {len(self.known_speakers)} 人")
        if self.known_speakers:
            print(f"  - {', '.join(self.known_speakers)}")
        
        print(f"当前映射: {len(self.speaker_names)} 个")
        for speaker_id, name in self.speaker_names.items():
            print(f"  - {speaker_id} -> {name}")
        print("="*50)


def demo():
    """演示功能"""
    manager = SpeakerManager()
    manager.print_summary()
    
    # 模拟分配一些speaker
    test_speakers = ["SPEAKER_00", "SPEAKER_01", "SPEAKER_02"]
    
    print("\n测试speaker分配:")
    for speaker_id in test_speakers:
        name = manager.assign_speaker_name(speaker_id)
        print(f"{speaker_id} -> {name}")
    
    manager.save_metadata()
    manager.print_summary()


if __name__ == "__main__":
    demo()