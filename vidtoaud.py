"""
Video to Audio Converter
========================
Helper script to convert video files to audio using various methods.
"""

import os
import sys
from pathlib import Path

def convert_video_to_audio(video_path, output_path=None):
    """Convert video file to audio using available methods"""
    
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return None
    
    if output_path is None:
        # Create output filename
        video_name = Path(video_path).stem
        output_path = f"{video_name}_audio.wav"
    
    print(f"Converting: {video_path}")
    print(f"Output: {output_path}")
    
    # Method 1: Try using VLC if installed
    try:
        import subprocess
        vlc_paths = [
            r"C:\Program Files\VideoLAN\VLC\vlc.exe",
            r"C:\Program Files (x86)\VideoLAN\VLC\vlc.exe",
            "/usr/bin/vlc",
            "/Applications/VLC.app/Contents/MacOS/VLC"
        ]
        
        vlc_exe = None
        for path in vlc_paths:
            if os.path.exists(path):
                vlc_exe = path
                break
        
        if vlc_exe:
            print("Found VLC, attempting conversion...")
            cmd = [
                vlc_exe,
                "-I", "dummy",
                video_path,
                "--sout=#transcode{acodec=s16l,channels=1,samplerate=16000}:std{access=file,mux=wav,dst=" + output_path + "}",
                "vlc://quit"
            ]
            subprocess.run(cmd, capture_output=True)
            
            if os.path.exists(output_path):
                print("Success! Audio extracted using VLC")
                return output_path
    except Exception as e:
        print(f"VLC method failed: {e}")
    
    # Method 2: Using online converter suggestion
    print("\nVideo to audio conversion requires external tools.")
    print("Here are some options:\n")
    
    print("Option 1: Use an online converter")
    print("   - Visit: https://cloudconvert.com/mp4-to-wav")
    print("   - Or: https://www.online-convert.com/")
    print("   - Upload your video and download as WAV or MP3\n")
    
    print("Option 2: Install imageio-ffmpeg (lightweight)")
    print("   pip install imageio-ffmpeg")
    print("   Then run this script again\n")
    
    print("Option 3: Use VLC Media Player")
    print("   - Open VLC")
    print("   - Media -> Convert/Save")
    print("   - Add your video file")
    print("   - Convert to audio format\n")
    
    # Method 3: Try imageio-ffmpeg if available
    try:
        import imageio_ffmpeg as ffmpeg
        print("Found imageio-ffmpeg, attempting conversion...")
        
        ffmpeg_exe = ffmpeg.get_ffmpeg_exe()
        
        import subprocess
        cmd = [
            ffmpeg_exe,
            '-i', video_path,
            '-vn',
            '-acodec', 'pcm_s16le',
            '-ar', '16000',
            '-ac', '1',
            '-y',
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0 and os.path.exists(output_path):
            print("Success! Audio extracted using imageio-ffmpeg")
            return output_path
        else:
            print(f"imageio-ffmpeg failed: {result.stderr}")
            
    except ImportError:
        pass
    
    return None


def batch_convert_directory(directory_path, output_dir=None):
    """Convert all video files in a directory to audio"""
    
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v']
    
    if not os.path.exists(directory_path):
        print(f"Error: Directory not found: {directory_path}")
        return
    
    if output_dir is None:
        output_dir = os.path.join(directory_path, "audio_files")
    
    os.makedirs(output_dir, exist_ok=True)
    
    video_files = []
    for ext in video_extensions:
        video_files.extend(Path(directory_path).glob(f"*{ext}"))
        video_files.extend(Path(directory_path).glob(f"*{ext.upper()}"))
    
    if not video_files:
        print(f"No video files found in {directory_path}")
        return
    
    print(f"Found {len(video_files)} video files")
    converted = 0
    
    for video_file in video_files:
        output_path = os.path.join(output_dir, f"{video_file.stem}_audio.wav")
        
        if os.path.exists(output_path):
            print(f"Skipping {video_file.name} - audio already exists")
            continue
        
        result = convert_video_to_audio(str(video_file), output_path)
        if result:
            converted += 1
    
    print(f"\nConverted {converted}/{len(video_files)} files")
    if converted > 0:
        print(f"Audio files saved to: {output_dir}")


def main():
    """Main function"""
    
    print("="*60)
    print("VIDEO TO AUDIO CONVERTER")
    print("="*60)
    
    if len(sys.argv) > 1:
        # Command line usage
        input_path = sys.argv[1]
        
        if os.path.isdir(input_path):
            # Convert directory
            output_dir = sys.argv[2] if len(sys.argv) > 2 else None
            batch_convert_directory(input_path, output_dir)
        else:
            # Convert single file
            output_path = sys.argv[2] if len(sys.argv) > 2 else None
            convert_video_to_audio(input_path, output_path)
    else:
        # Interactive mode
        print("Options:")
        print("1. Convert a single video file")
        print("2. Convert all videos in a directory")
        print("3. Exit")
        
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == '1':
            video_path = input("Enter video file path: ").strip()
            if video_path:
                convert_video_to_audio(video_path)
        
        elif choice == '2':
            dir_path = input("Enter directory path: ").strip()
            if dir_path:
                batch_convert_directory(dir_path)
        
        elif choice == '3':
            print("Exiting...")
        else:
            print("Invalid choice")
    
    input("\nPress Enter to exit...")


if __name__ == "__main__":
    main()