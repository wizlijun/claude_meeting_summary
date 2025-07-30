#!/usr/bin/env python3
"""
声纹添加工具
用于将新的wav文件添加到声纹库中
"""

import os
import sys
from pathlib import Path
from meeting_voice_enhancement import MeetingVoiceEnhancer
import argparse

def add_single_speaker(wav_file_path: str, speaker_dir: str = "speaker"):
    """
    添加单个说话人声纹到库中
    
    Args:
        wav_file_path: wav文件路径
        speaker_dir: 声纹库目录
    """
    wav_path = Path(wav_file_path)
    speaker_dir_path = Path(speaker_dir)
    
    # 检查文件是否存在
    if not wav_path.exists():
        print(f"❌ 错误: 文件不存在 {wav_file_path}")
        return False
    
    # 检查是否是wav文件
    if wav_path.suffix.lower() != '.wav':
        print(f"❌ 错误: 只支持wav文件，当前文件: {wav_path.suffix}")
        return False
    
    # 确保speaker目录存在
    speaker_dir_path.mkdir(exist_ok=True)
    
    # 目标文件路径
    speaker_name = wav_path.stem
    target_path = speaker_dir_path / f"{speaker_name}.wav"
    
    # 如果文件不在speaker目录中，复制过去
    if wav_path.absolute() != target_path.absolute():
        import shutil
        shutil.copy2(wav_path, target_path)
        print(f"✓ 已复制文件到声纹库: {target_path}")
    
    # 测试声纹生成
    try:
        print("🔄 正在生成声纹...")
        enhancer = MeetingVoiceEnhancer(speaker_dir=speaker_dir)
        
        if speaker_name in enhancer.get_known_speakers():
            print(f"✅ 成功添加说话人声纹: {speaker_name}")
            print(f"📊 当前声纹库包含 {len(enhancer.get_known_speakers())} 个说话人")
            
            # 显示所有说话人
            speakers = enhancer.get_known_speakers()
            print("📋 当前声纹库中的说话人:")
            for i, speaker in enumerate(speakers, 1):
                print(f"   {i}. {speaker}")
            
            return True
        else:
            print(f"❌ 声纹生成失败: {speaker_name}")
            return False
            
    except Exception as e:
        print(f"❌ 声纹生成出现错误: {e}")
        return False

def list_speakers(speaker_dir: str = "speaker"):
    """列出当前声纹库中的所有说话人"""
    try:
        enhancer = MeetingVoiceEnhancer(speaker_dir=speaker_dir)
        speakers = enhancer.get_known_speakers()
        
        print(f"📋 当前声纹库包含 {len(speakers)} 个说话人:")
        for i, speaker in enumerate(speakers, 1):
            embedding_shape = enhancer.speaker_embeddings[speaker].shape
            print(f"   {i}. {speaker} (声纹维度: {embedding_shape})")
            
    except Exception as e:
        print(f"❌ 读取声纹库失败: {e}")

def scan_and_add_all(speaker_dir: str = "speaker"):
    """扫描speaker目录并添加所有wav文件"""
    speaker_dir_path = Path(speaker_dir)
    
    if not speaker_dir_path.exists():
        print(f"❌ 声纹库目录不存在: {speaker_dir}")
        return
    
    wav_files = list(speaker_dir_path.glob("*.wav"))
    
    if not wav_files:
        print(f"📁 {speaker_dir} 目录中没有找到wav文件")
        return
    
    print(f"🔍 发现 {len(wav_files)} 个wav文件，正在生成声纹...")
    
    try:
        enhancer = MeetingVoiceEnhancer(speaker_dir=speaker_dir)
        speakers = enhancer.get_known_speakers()
        
        print(f"✅ 成功加载声纹库，包含 {len(speakers)} 个说话人:")
        for i, speaker in enumerate(speakers, 1):
            print(f"   {i}. {speaker}")
            
    except Exception as e:
        print(f"❌ 加载声纹库失败: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="声纹管理工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python add_speaker.py add new_speaker.wav     # 添加单个声纹
  python add_speaker.py list                     # 列出所有声纹
  python add_speaker.py scan                     # 扫描并重新加载所有声纹
        """
    )
    
    parser.add_argument(
        'command', 
        choices=['add', 'list', 'scan'],
        help='操作命令: add=添加声纹, list=列出声纹, scan=扫描重新加载'
    )
    
    parser.add_argument(
        'wav_file', 
        nargs='?',
        help='要添加的wav文件路径 (仅用于add命令)'
    )
    
    parser.add_argument(
        '--speaker-dir', 
        default='speaker',
        help='声纹库目录 (默认: speaker)'
    )
    
    args = parser.parse_args()
    
    print("🎙️  声纹管理工具")
    print("=" * 50)
    
    if args.command == 'add':
        if not args.wav_file:
            print("❌ 错误: add命令需要指定wav文件路径")
            print("使用方法: python add_speaker.py add <wav文件路径>")
            sys.exit(1)
        
        success = add_single_speaker(args.wav_file, args.speaker_dir)
        sys.exit(0 if success else 1)
        
    elif args.command == 'list':
        list_speakers(args.speaker_dir)
        
    elif args.command == 'scan':
        scan_and_add_all(args.speaker_dir)

if __name__ == "__main__":
    main()