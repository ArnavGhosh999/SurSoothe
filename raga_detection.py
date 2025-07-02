"""
Raga Detection Pipeline
======================
Complete pipeline for detecting Ragas from audio files and URLs
using only core Python audio libraries.

Required installations:
pip install torch torchaudio
pip install librosa soundfile
pip install BeatNet
pip install praat-parselmouth
pip install transformers
pip install pandas numpy scikit-learn
pip install matplotlib seaborn scipy
pip install yt-dlp  # Optional for YouTube/web downloads
"""

import os
import torch
import torchaudio
import librosa
import numpy as np
import pandas as pd
import parselmouth
from parselmouth.praat import call
from typing import Tuple, Dict, List, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# Deep learning imports
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from transformers import ASTModel, ASTConfig

# Optional imports
try:
    import yt_dlp
    YTDLP_AVAILABLE = True
except ImportError:
    print("yt-dlp not available. Install with: pip install yt-dlp for YouTube support")
    YTDLP_AVAILABLE = False

# Beat tracking
try:
    from BeatNet.BeatNet import BeatNet
    BEATNET_AVAILABLE = True
except ImportError:
    print("BeatNet not available. Using librosa for beat tracking.")
    BEATNET_AVAILABLE = False

# Utilities
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, top_k_accuracy_score
import matplotlib.pyplot as plt
from scipy import signal
import tempfile
import re
from pathlib import Path
import soundfile as sf


# =====================================
# Audio Processing Module
# =====================================

class UniversalAudioProcessor:
    """Handles audio loading from various sources using only librosa"""
    
    def __init__(self, target_sr: int = 16000, cache_dir: str = "temp_audio"):
        self.target_sr = target_sr
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
    def process_input(self, input_path: str) -> Tuple[np.ndarray, int, str]:
        """
        Process any input and return audio array
        Returns: (audio_array, sample_rate, input_type)
        """
        # Check if it's a URL
        if self._is_url(input_path):
            audio, sr = self._process_url(input_path)
            return audio, sr, "url"
        
        # Check if file exists
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"File not found: {input_path}")
        
        # Get file extension
        ext = Path(input_path).suffix.lower()
        
        # For video files, we need a different approach
        if ext in ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v']:
            return self._process_video_file(input_path)
        
        # Try to load with librosa (for audio files)
        try:
            audio, sr = librosa.load(input_path, sr=self.target_sr, mono=True)
            return audio, sr, "audio"
            
        except Exception as e:
            print(f"Error loading with librosa: {e}")
            # Try alternative loading methods
            return self._fallback_load(input_path)
    
    def _process_video_file(self, video_path: str) -> Tuple[np.ndarray, int, str]:
        """Handle video files using opencv or imageio"""
        print("Attempting to extract audio from video file...")
        
        # Method 1: Try using opencv-python (lightweight)
        try:
            import cv2
            print("Trying OpenCV method...")
            
            # OpenCV can read video but not extract audio directly
            # So we'll check if the file is readable
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                cap.release()
                print("Video file detected but OpenCV cannot extract audio.")
            
        except ImportError:
            pass
        
        # Method 2: Try using imageio-ffmpeg (minimal ffmpeg)
        try:
            import imageio_ffmpeg as ffmpeg
            print("Trying imageio-ffmpeg method...")
            
            # Extract audio using imageio-ffmpeg
            output_path = os.path.join(self.cache_dir, "extracted_audio.wav")
            
            # Get ffmpeg executable from imageio
            ffmpeg_exe = ffmpeg.get_ffmpeg_exe()
            
            # Run extraction
            import subprocess
            cmd = [
                ffmpeg_exe,
                '-i', video_path,
                '-vn',  # No video
                '-acodec', 'pcm_s16le',
                '-ar', str(self.target_sr),
                '-ac', '1',  # Mono
                '-y',  # Overwrite
                output_path
            ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            
            # Load extracted audio
            audio, sr = librosa.load(output_path, sr=self.target_sr, mono=True)
            
            # Clean up
            if os.path.exists(output_path):
                os.remove(output_path)
            
            return audio, sr, "video"
            
        except Exception as e:
            print(f"imageio-ffmpeg method failed: {e}")
        
        # Method 3: Try using VLC python bindings (if available)
        try:
            import vlc
            print("Trying VLC method...")
            
            # This would require VLC to be installed
            # Implementation would be complex, so skipping for now
            
        except ImportError:
            pass
        
        # Method 4: Direct extraction attempt (last resort)
        # Some video containers might have easily extractable audio
        try:
            print("Attempting direct audio stream extraction...")
            
            # Try to find audio stream in the file
            with open(video_path, 'rb') as f:
                # Read first few KB to check format
                header = f.read(1024)
                
                # Check for common audio codec signatures
                if b'mp4a' in header or b'aac' in header:
                    print("Found AAC audio stream marker")
                elif b'mp3' in header:
                    print("Found MP3 audio stream marker")
                
                # This is very basic and won't work for most files
                # But included to show the concept
        
        except Exception:
            pass
        
        # If all methods fail, provide helpful error message
        error_msg = (
            f"Cannot extract audio from video file: {video_path}\n"
            f"Video files require additional tools. Please either:\n"
            f"1. Install imageio-ffmpeg: pip install imageio-ffmpeg\n"
            f"2. Convert the video to audio using an online converter\n"
            f"3. Use VLC or another media player to extract audio\n"
            f"4. Install ffmpeg separately and add to PATH"
        )
        
        raise RuntimeError(error_msg)
    
    def _is_url(self, path: str) -> bool:
        """Check if input is a URL"""
        url_patterns = ['http://', 'https://', 'www.', 'youtube.com', 'youtu.be']
        return any(pattern in path.lower() for pattern in url_patterns)
    
    def _process_url(self, url: str) -> Tuple[np.ndarray, int]:
        """Download audio from URL"""
        if not YTDLP_AVAILABLE:
            # Try direct loading with librosa (works for some URLs)
            try:
                audio, sr = librosa.load(url, sr=self.target_sr, mono=True)
                return audio, sr
            except:
                raise ImportError("yt-dlp is required for most URL processing. Install with: pip install yt-dlp")
        
        # Use yt-dlp to download audio
        output_path = os.path.join(self.cache_dir, "downloaded_audio")
        
        # First try to get audio-only format
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': output_path,
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
            'prefer_ffmpeg': False,  # Don't require ffmpeg
            'extractaudio': True,
        }
        
        print(f"Downloading audio from URL: {url}")
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                
                # Get the actual filename
                filename = ydl.prepare_filename(info)
                
                # Check various possible extensions
                possible_files = [
                    filename,
                    output_path,
                    output_path + ".m4a",
                    output_path + ".webm", 
                    output_path + ".opus",
                    output_path + ".mp3",
                    output_path + ".wav",
                    filename.rsplit('.', 1)[0] + ".m4a",
                    filename.rsplit('.', 1)[0] + ".webm",
                ]
                
                for file_path in possible_files:
                    if os.path.exists(file_path):
                        try:
                            # Try to load with librosa
                            audio, sr = librosa.load(file_path, sr=self.target_sr, mono=True)
                            # Clean up
                            os.remove(file_path)
                            return audio, sr
                        except Exception as e:
                            print(f"Could not load {file_path}: {e}")
                            # Try to clean up anyway
                            try:
                                os.remove(file_path)
                            except:
                                pass
                
                raise FileNotFoundError("Downloaded file not found or could not be loaded")
            
        except Exception as e:
            print(f"Error downloading from URL: {e}")
            raise
    
    def _fallback_load(self, file_path: str) -> Tuple[np.ndarray, int, str]:
        """Fallback loading using soundfile or scipy"""
        ext = Path(file_path).suffix.lower()
        
        # Try soundfile
        try:
            audio, sr = sf.read(file_path)
            
            # Convert to mono if needed
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            
            # Resample if needed
            if sr != self.target_sr:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)
                sr = self.target_sr
            
            return audio, sr, "audio"
            
        except Exception:
            pass
        
        # Try scipy for WAV files
        if ext == '.wav':
            try:
                from scipy.io import wavfile
                sr, audio = wavfile.read(file_path)
                
                # Convert to float32
                if audio.dtype == np.int16:
                    audio = audio.astype(np.float32) / 32768.0
                elif audio.dtype == np.int32:
                    audio = audio.astype(np.float32) / 2147483648.0
                
                # Convert to mono if needed
                if len(audio.shape) > 1:
                    audio = np.mean(audio, axis=1)
                
                # Resample if needed
                if sr != self.target_sr:
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)
                    sr = self.target_sr
                
                return audio, sr, "audio"
                
            except Exception as e:
                pass
        
        # Last resort: try to load as raw audio data
        if ext in ['.pcm', '.raw']:
            try:
                # Assume 16-bit PCM at 44100 Hz
                audio = np.fromfile(file_path, dtype=np.int16)
                audio = audio.astype(np.float32) / 32768.0
                sr = 44100
                
                if sr != self.target_sr:
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)
                    sr = self.target_sr
                
                return audio, sr, "audio"
            except:
                pass
        
        raise RuntimeError(f"Could not load audio from {file_path}. Supported audio formats: WAV, MP3, FLAC, OGG, M4A")
    
    def cleanup(self):
        """Clean up temporary files"""
        import shutil
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)


# =====================================
# Audio Feature Extraction
# =====================================

class AudioFeatureExtractor:
    """Extract various audio features using specified tools"""
    
    def __init__(self, sr: int = 16000):
        self.sr = sr
        if BEATNET_AVAILABLE:
            try:
                self.beatnet = BeatNet(1, mode='offline', inference_model='DBN', plot=False, thread=False)
            except:
                self.beatnet = None
    
    def extract_librosa_features(self, audio: np.ndarray, sr: int) -> Dict:
        """Extract CQT and Tonnetz using Librosa"""
        features = {}
        
        # Ensure audio is not empty
        if len(audio) == 0:
            raise ValueError("Empty audio signal")
        
        # Pad audio if too short
        min_length = 2048
        if len(audio) < min_length:
            audio = np.pad(audio, (0, min_length - len(audio)), mode='constant')
        
        # Constant-Q Transform (CQT)
        cqt = librosa.cqt(
            y=audio, 
            sr=sr, 
            hop_length=512,
            n_bins=84,  # 7 octaves * 12 bins
            bins_per_octave=12,
            fmin=librosa.note_to_hz('C1')
        )
        features['cqt'] = np.abs(cqt)
        features['cqt_phase'] = np.angle(cqt)
        
        # Chroma features
        chroma = librosa.feature.chroma_cqt(y=audio, sr=sr, hop_length=512)
        features['chroma'] = chroma
        
        # Tonnetz (Tonal Centroid Features)
        tonnetz = librosa.feature.tonnetz(y=audio, sr=sr, chroma=chroma)
        features['tonnetz'] = tonnetz
        
        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
        spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
        
        features['spectral_centroid'] = spectral_centroid
        features['spectral_bandwidth'] = spectral_bandwidth
        features['spectral_rolloff'] = spectral_rolloff
        features['spectral_contrast'] = spectral_contrast
        
        # MFCC
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        features['mfcc'] = mfcc
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        features['zcr'] = zcr
        
        # RMS Energy
        rms = librosa.feature.rms(y=audio)[0]
        features['rms'] = rms
        
        return features
    
    def extract_parselmouth_features(self, audio: np.ndarray, sr: int) -> Dict:
        """Extract features using Parselmouth"""
        features = {}
        
        try:
            # Create Parselmouth Sound object
            sound = parselmouth.Sound(audio, sampling_frequency=sr)
            
            # Intensity for tempogram
            intensity = sound.to_intensity(time_step=0.01)
            intensity_values = intensity.values[0]
            
            # Compute tempogram using autocorrelation
            if len(intensity_values) > 1:
                # Normalize intensity
                intensity_norm = intensity_values - np.mean(intensity_values)
                
                # Autocorrelation
                autocorr = np.correlate(intensity_norm, intensity_norm, mode='full')
                autocorr = autocorr[len(autocorr)//2:]
                
                # Normalize
                if autocorr[0] > 0:
                    autocorr = autocorr / autocorr[0]
                
                # Convert to tempo range
                tempo_range = np.arange(40, 200, 2)
                tempogram = []
                for tempo in tempo_range:
                    lag = int(60.0 / tempo / 0.01)  # Convert BPM to lag in samples
                    if lag < len(autocorr):
                        tempogram.append(autocorr[lag])
                    else:
                        tempogram.append(0)
                features['tempogram'] = np.array(tempogram)
            else:
                features['tempogram'] = np.zeros(80)
            
            # Pitch tracking
            pitch = sound.to_pitch_ac(
                time_step=0.01,
                pitch_floor=75.0,
                pitch_ceiling=600.0
            )
            pitch_values = pitch.selected_array['frequency']
            pitch_values[pitch_values == 0] = np.nan
            features['pitch'] = pitch_values
            features['pitch_strength'] = pitch.selected_array['strength']
            
            # Formants (if voiced)
            try:
                formant = sound.to_formant_burg(time_step=0.01)
                formant_tracks = []
                for i in range(1, 6):  # F1-F5
                    track = []
                    for t in formant.ts():
                        try:
                            f = formant.get_value_at_time(i, t)
                            track.append(f if f else np.nan)
                        except:
                            track.append(np.nan)
                    formant_tracks.append(track)
                features['formants'] = np.array(formant_tracks)
            except:
                features['formants'] = np.full((5, 100), np.nan)
            
            # Spectral moments
            spectrum = sound.to_spectrum()
            features['spectral_cog'] = call(spectrum, "Get centre of gravity", 2)
            features['spectral_std'] = call(spectrum, "Get standard deviation", 2)
            features['spectral_skewness'] = call(spectrum, "Get skewness", 2)
            features['spectral_kurtosis'] = call(spectrum, "Get kurtosis", 2)
            
            # Voice quality measures (if applicable)
            try:
                point_process = call(sound, "To PointProcess (periodic, cc)", 75, 600)
                
                # Jitter
                jitter = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
                features['jitter_local'] = jitter if not np.isnan(jitter) else 0
                
                # Shimmer  
                shimmer = call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
                features['shimmer_local'] = shimmer if not np.isnan(shimmer) else 0
                
                # HNR
                harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
                hnr_values = []
                for t in harmonicity.ts():
                    hnr = call(harmonicity, "Get value at time", t, "Cubic")
                    hnr_values.append(hnr if hnr > 0 else 0)
                features['hnr'] = np.array(hnr_values)
            except:
                features['jitter_local'] = 0
                features['shimmer_local'] = 0
                features['hnr'] = np.zeros(100)
                
        except Exception as e:
            print(f"Parselmouth extraction error: {e}")
            # Return default values
            features['tempogram'] = np.zeros(80)
            features['pitch'] = np.full(100, np.nan)
            features['pitch_strength'] = np.zeros(100)
            features['formants'] = np.full((5, 100), np.nan)
            features['spectral_cog'] = 0
            features['spectral_std'] = 0
            features['spectral_skewness'] = 0
            features['spectral_kurtosis'] = 0
            features['jitter_local'] = 0
            features['shimmer_local'] = 0
            features['hnr'] = np.zeros(100)
        
        return features
    
    def extract_beat_features(self, audio: np.ndarray, sr: int) -> Dict:
        """Extract beat tracking features using librosa (BeatNet optional)"""
        features = {}
        
        # Use librosa beat tracking
        tempo, beats = librosa.beat.beat_track(y=audio, sr=sr)
        beat_times = librosa.frames_to_time(beats, sr=sr)
        
        # Onset detection for more detailed rhythm analysis
        onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
        onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)
        
        # Estimate time signature (simplified)
        if len(beat_times) > 4:
            # Look for patterns in beat intervals
            intervals = np.diff(beat_times)
            median_interval = np.median(intervals)
            
            # Group beats into measures (assuming 4/4 as default)
            measure_length = median_interval * 4
            downbeat_times = beat_times[::4]
        else:
            downbeat_times = beat_times
        
        features['tempo'] = float(tempo)
        features['beats'] = beat_times
        features['downbeats'] = downbeat_times
        features['onsets'] = onset_times
        features['onset_strength'] = onset_env
        features['time_signature'] = 4  # Default assumption
        
        # Try BeatNet if available
        if BEATNET_AVAILABLE and hasattr(self, 'beatnet') and self.beatnet is not None:
            try:
                # Save temp file for BeatNet
                temp_path = os.path.join(tempfile.gettempdir(), "temp_beat.wav")
                sf.write(temp_path, audio, sr)
                
                # Process with BeatNet
                output = self.beatnet.process(temp_path)
                
                if output and len(output) > 0:
                    beats_bn = []
                    downbeats_bn = []
                    
                    for time, beat_type in output:
                        if beat_type == 1:  # Downbeat
                            downbeats_bn.append(time)
                        beats_bn.append(time)
                    
                    if len(beats_bn) > 0:
                        features['beats'] = np.array(beats_bn)
                        features['downbeats'] = np.array(downbeats_bn)
                        
                        # Recalculate tempo
                        if len(beats_bn) > 1:
                            intervals = np.diff(beats_bn)
                            features['tempo'] = 60.0 / np.median(intervals)
                
                # Clean up
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    
            except Exception as e:
                print(f"BeatNet failed, using librosa results: {e}")
        
        return features


# =====================================
# Feature Preprocessing
# =====================================

class FeaturePreprocessor:
    """Preprocess features into model-ready formats"""
    
    def __init__(self, feature_type: str = 'log_mel'):
        self.feature_type = feature_type
        
    def to_log_mel_spectrogram(self, audio: np.ndarray, sr: int) -> torch.Tensor:
        """Convert audio to log-mel spectrogram"""
        # Ensure minimum length
        if len(audio) < 2048:
            audio = np.pad(audio, (0, 2048 - len(audio)), mode='constant')
        
        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio, 
            sr=sr, 
            n_fft=2048, 
            hop_length=512, 
            n_mels=128,
            fmin=0,
            fmax=sr//2
        )
        
        # Convert to log scale
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize
        mean = log_mel.mean()
        std = log_mel.std()
        if std > 0:
            log_mel = (log_mel - mean) / std
        else:
            log_mel = log_mel - mean
        
        # Convert to tensor and ensure correct shape
        # Shape should be (1, freq_bins, time_frames) for single channel
        log_mel_tensor = torch.FloatTensor(log_mel)
        
        # Add channel dimension if needed
        if log_mel_tensor.dim() == 2:
            log_mel_tensor = log_mel_tensor.unsqueeze(0)
        
        # Ensure we don't have extra dimensions
        if log_mel_tensor.dim() > 3:
            log_mel_tensor = log_mel_tensor.squeeze()
            # Make sure we still have 3 dimensions
            while log_mel_tensor.dim() < 3:
                log_mel_tensor = log_mel_tensor.unsqueeze(0)
        
        return log_mel_tensor
    
    def to_cqt_spectrogram(self, cqt: np.ndarray) -> torch.Tensor:
        """Convert CQT to model input format"""
        # Convert to magnitude if complex
        if np.iscomplexobj(cqt):
            cqt = np.abs(cqt)
        
        # Log scale
        cqt_db = librosa.amplitude_to_db(cqt, ref=np.max)
        
        # Normalize
        mean = cqt_db.mean()
        std = cqt_db.std()
        if std > 0:
            cqt_db = (cqt_db - mean) / std
        else:
            cqt_db = cqt_db - mean
        
        # Convert to tensor
        return torch.FloatTensor(cqt_db).unsqueeze(0)


# =====================================
# Data Augmentation
# =====================================

class SpecAugment:
    """SpecAugment for training robustness"""
    
    def __init__(self, time_mask_param: int = 20, freq_mask_param: int = 10, n_masks: int = 2):
        self.time_mask_param = time_mask_param
        self.freq_mask_param = freq_mask_param
        self.n_masks = n_masks
        self.training = True
    
    def train(self):
        self.training = True
    
    def eval(self):
        self.training = False
    
    def __call__(self, spec: torch.Tensor) -> torch.Tensor:
        """Apply augmentation"""
        if not self.training:
            return spec
        
        _, freq_dim, time_dim = spec.shape
        
        # Time masking
        for _ in range(self.n_masks):
            if time_dim > self.time_mask_param:
                mask_len = torch.randint(1, self.time_mask_param, (1,)).item()
                mask_start = torch.randint(0, time_dim - mask_len, (1,)).item()
                spec[:, :, mask_start:mask_start + mask_len] = 0
        
        # Frequency masking
        for _ in range(self.n_masks):
            if freq_dim > self.freq_mask_param:
                mask_len = torch.randint(1, self.freq_mask_param, (1,)).item()
                mask_start = torch.randint(0, freq_dim - mask_len, (1,)).item()
                spec[:, mask_start:mask_start + mask_len, :] = 0
        
        return spec


# =====================================
# Model Definitions
# =====================================

class RagaAST(nn.Module):
    """Audio Spectrogram Transformer for Raga Classification"""
    
    def __init__(self, num_ragas: int):
        super().__init__()
        
        # For AST, we'll use a custom architecture since the pretrained one expects specific input
        # Input: (batch, 1, 128, time_frames)
        
        # Convolutional frontend to process spectrograms
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8))  # Fixed size output
        )
        
        # Transformer encoder
        self.embedding_dim = 256 * 8 * 8
        self.positional_encoding = nn.Parameter(torch.randn(1, 100, 512))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=512,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)
        
        # Project conv features to transformer dimension
        self.projection = nn.Linear(self.embedding_dim, 512)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_ragas)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expected input: (batch, 1, 128, time_frames)
        if x.dim() == 5:
            # Remove extra dimension if present
            x = x.squeeze(1)
        
        if x.dim() == 3:
            # Add channel dimension if missing
            x = x.unsqueeze(1)
        
        # Convolutional layers
        conv_out = self.conv_layers(x)  # (batch, 256, 8, 8)
        
        # Flatten and project
        batch_size = conv_out.shape[0]
        conv_out = conv_out.view(batch_size, -1)  # (batch, 256*8*8)
        
        # Project to transformer dimension
        x = self.projection(conv_out)  # (batch, 512)
        x = x.unsqueeze(1)  # (batch, 1, 512)
        
        # Add positional encoding
        x = x + self.positional_encoding[:, :x.shape[1], :]
        
        # Transformer
        x = self.transformer(x)
        
        # Classification
        x = x.mean(dim=1)  # Global average pooling
        logits = self.classifier(x)
        
        return logits


class RagaPaSST(nn.Module):
    """PaSST-style model for Raga Classification"""
    
    def __init__(self, num_ragas: int, input_fdim: int = 128, input_tdim: int = 998):
        super().__init__()
        
        # Architecture parameters
        self.patch_size = 16
        self.stride = 10
        self.embed_dim = 768
        self.depth = 12
        self.num_heads = 12
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(
            1, self.embed_dim,
            kernel_size=(self.patch_size, self.patch_size),
            stride=(self.stride, self.stride)
        )
        
        # Positional embedding
        num_patches = ((input_fdim - self.patch_size) // self.stride + 1) * \
                     ((input_tdim - self.patch_size) // self.stride + 1)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=self.num_heads,
            dim_feedforward=3072,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.depth)
        
        # Classification head
        self.norm = nn.LayerNorm(self.embed_dim)
        self.head = nn.Linear(self.embed_dim, num_ragas)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, embed_dim, H, W)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional embedding
        x = x + self.pos_embed[:, :x.shape[1], :]
        
        # Transformer
        x = self.transformer(x)
        
        # Classification
        x = self.norm(x[:, 0])  # CLS token
        x = self.head(x)
        
        return x


# =====================================
# Dataset
# =====================================

class RagaDataset(Dataset):
    """Dataset for Raga classification"""
    
    def __init__(self, csv_path: str, transform=None, feature_type='log_mel'):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.feature_type = feature_type
        
        # Find filename column
        self.filename_col = None
        for col in ['filename', 'path', 'file', 'audio_path', 'filepath']:
            if col in self.df.columns:
                self.filename_col = col
                break
        
        if self.filename_col is None:
            raise ValueError("No filename column found in dataset")
        
        # Clean paths
        self.df[self.filename_col] = self.df[self.filename_col].apply(self._clean_path)
        
        # Create label mapping
        self.raga_to_idx = {raga: idx for idx, raga in enumerate(self.df['raga'].unique())}
        self.idx_to_raga = {idx: raga for raga, idx in self.raga_to_idx.items()}
        
        # Initialize components
        self.audio_processor = UniversalAudioProcessor()
        self.preprocessor = FeaturePreprocessor(feature_type)
        self.augment = SpecAugment() if transform else None
        
        # Validate files
        self._validate_files()
    
    def _clean_path(self, path):
        """Clean file path"""
        if pd.isna(path):
            return path
        
        path = str(path).strip()
        path = path.replace('\\', '/')
        path = re.sub(r'/+', '/', path)
        path = re.sub(r'\.mp4/', '/', path)
        path = re.sub(r'-\d+$', '', path)
        
        return path
    
    def _validate_files(self):
        """Check which files exist"""
        valid_mask = []
        
        for idx in range(len(self.df)):
            filepath = self.df.iloc[idx][self.filename_col]
            
            # Check various path combinations
            exists = False
            for base in ['', 'data/', 'data/audio/', 'data/Youtube_files/']:
                test_path = os.path.join(base, filepath) if base else filepath
                if os.path.exists(test_path):
                    self.df.at[idx, self.filename_col] = test_path
                    exists = True
                    break
            
            valid_mask.append(exists)
        
        # Filter to valid files
        self.df = self.df[valid_mask].reset_index(drop=True)
        print(f"Dataset: {len(self.df)} valid files found")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filepath = row[self.filename_col]
        
        try:
            # Load audio
            audio, sr, _ = self.audio_processor.process_input(filepath)
            
            # Convert to spectrogram
            if self.feature_type == 'log_mel':
                spec = self.preprocessor.to_log_mel_spectrogram(audio, sr)
            elif self.feature_type == 'cqt':
                cqt = librosa.cqt(y=audio, sr=sr, hop_length=512, n_bins=84)
                spec = self.preprocessor.to_cqt_spectrogram(cqt)
            else:
                raise ValueError(f"Unknown feature type: {self.feature_type}")
            
            # Apply augmentation
            if self.augment and self.transform:
                spec = self.augment(spec)
            
            # Get label
            label = self.raga_to_idx[row['raga']]
            
            return spec, label
            
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            # Return zeros
            dummy_spec = torch.zeros(1, 128, 100)
            return dummy_spec, 0


# =====================================
# Training
# =====================================

class RagaTrainer:
    """Training pipeline"""
    
    def __init__(self, model: nn.Module, device: str = None):
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model = model.to(self.device)
        
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              epochs: int = 50, lr: float = 1e-4, warmup_epochs: int = 5):
        """Train the model"""
        
        # Setup
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # History
        history = {
            'train_loss': [], 'val_loss': [], 
            'val_acc': [], 'val_top3_acc': [], 'lr': []
        }
        
        # Training loop
        best_val_acc = 0
        patience_counter = 0
        
        for epoch in range(epochs):
            # Train
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            # Warmup
            if epoch < warmup_epochs:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr * (epoch + 1) / warmup_epochs
            
            current_lr = optimizer.param_groups[0]['lr']
            history['lr'].append(current_lr)
            
            # Training batches
            for batch_idx, (specs, labels) in enumerate(train_loader):
                specs, labels = specs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(specs)
                loss = criterion(outputs, labels)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
                
                if batch_idx % 10 == 0:
                    print(f'Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, '
                          f'Loss: {loss.item():.4f}, Acc: {100.*train_correct/train_total:.2f}%')
            
            # Validation
            val_loss, val_acc, val_top3_acc = self.evaluate(val_loader, criterion)
            
            # Update history
            history['train_loss'].append(train_loss / len(train_loader))
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['val_top3_acc'].append(val_top3_acc)
            
            print(f'\nEpoch {epoch}: Train Loss: {train_loss/len(train_loader):.4f}, '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, '
                  f'Val Top-3 Acc: {val_top3_acc:.4f}, LR: {current_lr:.6f}\n')
            
            # Scheduler
            if epoch >= warmup_epochs:
                scheduler.step()
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_top3_acc': val_top3_acc
                }, 'best_raga_model.pth')
                print(f"Saved best model with accuracy: {val_acc:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= 15:
                    print(f'Early stopping at epoch {epoch}')
                    break
        
        return history
    
    def evaluate(self, data_loader: DataLoader, criterion: nn.Module):
        """Evaluate the model"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for specs, labels in data_loader:
                specs, labels = specs.to(self.device), labels.to(self.device)
                
                outputs = self.model(specs)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                
                probs = F.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        avg_loss = total_loss / len(data_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        top3_accuracy = top_k_accuracy_score(all_labels, all_probs, k=3)
        
        return avg_loss, accuracy, top3_accuracy


# =====================================
# Inference
# =====================================

class RagaDetector:
    """Main class for Raga detection"""
    
    def __init__(self, model_path: str = None, num_ragas: int = 10, model_type: str = 'ast'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        if model_type == 'ast':
            self.model = RagaAST(num_ragas=num_ragas)
        else:
            self.model = RagaPaSST(num_ragas=num_ragas)
        
        # Load weights if available
        if model_path and os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model from {model_path}")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize components
        self.audio_processor = UniversalAudioProcessor()
        self.feature_extractor = AudioFeatureExtractor()
        self.preprocessor = FeaturePreprocessor()
        
        # Load raga names
        self._load_raga_names()
    
    def _load_raga_names(self):
        """Load raga names from dataset if available"""
        self.idx_to_raga = {i: f"Raga_{i}" for i in range(10)}  # Default
        
        if os.path.exists('data/Dataset.csv'):
            try:
                df = pd.read_csv('data/Dataset.csv')
                unique_ragas = sorted(df['raga'].unique())
                self.idx_to_raga = {i: raga for i, raga in enumerate(unique_ragas)}
            except:
                pass
    
    def detect(self, input_source: str, return_features: bool = True) -> Dict:
        """Detect raga from any input source"""
        
        print(f"\nProcessing: {input_source}")
        
        try:
            # Load audio
            audio, sr, input_type = self.audio_processor.process_input(input_source)
            print(f"Loaded {input_type} ({len(audio)/sr:.2f} seconds)")
            
            # Extract features if requested
            features = {}
            if return_features:
                print("Extracting features...")
                features['librosa'] = self.feature_extractor.extract_librosa_features(audio, sr)
                features['parselmouth'] = self.feature_extractor.extract_parselmouth_features(audio, sr)
                features['rhythm'] = self.feature_extractor.extract_beat_features(audio, sr)
            
            # Convert to model input
            spec = self.preprocessor.to_log_mel_spectrogram(audio, sr)
            
            # Ensure correct shape for model input
            # Should be (batch_size, channels, freq_bins, time_frames)
            if spec.dim() == 3:
                # Add batch dimension
                spec = spec.unsqueeze(0)
            
            # Debug shape
            print(f"Spectrogram shape: {spec.shape}")
            
            # Move to device
            spec = spec.to(self.device)
            
            # Inference
            with torch.no_grad():
                outputs = self.model(spec)
                probs = F.softmax(outputs, dim=1)
            
            # Get results
            top3_probs, top3_indices = torch.topk(probs, k=3, dim=1)
            
            results = {
                'detected_raga': self.idx_to_raga.get(top3_indices[0, 0].item(), "Unknown"),
                'confidence': top3_probs[0, 0].item(),
                'input_type': input_type,
                'duration': len(audio) / sr,
                'top3_predictions': [
                    {
                        'raga': self.idx_to_raga.get(top3_indices[0, i].item(), "Unknown"),
                        'confidence': top3_probs[0, i].item()
                    } for i in range(3)
                ]
            }
            
            # Add features
            if return_features:
                results['features'] = features
                results['audio_metrics'] = {
                    'tempo': features['rhythm'].get('tempo', 0),
                    'beats_count': len(features['rhythm'].get('beats', [])),
                    'pitch_mean': np.nanmean(features['parselmouth'].get('pitch', [])),
                    'spectral_centroid': np.mean(features['librosa'].get('spectral_centroid', []))
                }
            
            # Cleanup
            self.audio_processor.cleanup()
            
            return results
            
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            self.audio_processor.cleanup()
            return {'error': str(e), 'detected_raga': None}


# =====================================
# Visualization
# =====================================

def plot_features(features: Dict, save_path: str = None):
    """Plot audio features"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # CQT
    if 'librosa' in features and 'cqt' in features['librosa']:
        cqt = features['librosa']['cqt']
        axes[0, 0].imshow(librosa.amplitude_to_db(cqt, ref=np.max), 
                         aspect='auto', origin='lower', cmap='viridis')
        axes[0, 0].set_title('Constant-Q Transform')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Frequency Bins')
    
    # Tonnetz
    if 'librosa' in features and 'tonnetz' in features['librosa']:
        tonnetz = features['librosa']['tonnetz']
        axes[0, 1].imshow(tonnetz, aspect='auto', origin='lower', cmap='coolwarm')
        axes[0, 1].set_title('Tonnetz')
        axes[0, 1].set_xlabel('Time')
        axes[0, 1].set_ylabel('Tonnetz Coefficients')
    
    # Tempogram
    if 'parselmouth' in features and 'tempogram' in features['parselmouth']:
        tempogram = features['parselmouth']['tempogram']
        tempo_range = np.arange(40, 200, 2)
        axes[1, 0].plot(tempo_range, tempogram)
        axes[1, 0].set_title('Tempogram')
        axes[1, 0].set_xlabel('Tempo (BPM)')
        axes[1, 0].set_ylabel('Strength')
    
    # Pitch
    if 'parselmouth' in features and 'pitch' in features['parselmouth']:
        pitch = features['parselmouth']['pitch']
        time = np.arange(len(pitch)) * 0.01
        valid = ~np.isnan(pitch)
        if np.any(valid):
            axes[1, 1].plot(time[valid], pitch[valid])
            axes[1, 1].set_title('Pitch Contour')
            axes[1, 1].set_xlabel('Time (s)')
            axes[1, 1].set_ylabel('Frequency (Hz)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()


# =====================================
# Main
# =====================================

def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Raga Detection System')
    parser.add_argument('--mode', choices=['train', 'detect'], default='detect')
    parser.add_argument('--input', type=str, help='Input file/URL for detection')
    parser.add_argument('--dataset', type=str, default='data/Dataset.csv')
    parser.add_argument('--model', type=str, default='best_raga_model.pth')
    parser.add_argument('--model-type', choices=['ast', 'passt'], default='ast')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=16)
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        # Training
        print("="*60)
        print("RAGA DETECTION TRAINING")
        print("="*60)
        
        # Load dataset
        dataset = RagaDataset(args.dataset)
        
        if len(dataset) == 0:
            print("No valid samples found!")
            return
        
        # Split data
        indices = list(range(len(dataset)))
        labels = [dataset[i][1] for i in indices]
        
        train_idx, val_idx = train_test_split(
            indices, test_size=0.2, stratify=labels, random_state=42
        )
        
        # Create loaders
        train_loader = DataLoader(
            dataset, batch_size=args.batch_size,
            sampler=torch.utils.data.SubsetRandomSampler(train_idx),
            num_workers=0
        )
        
        val_loader = DataLoader(
            dataset, batch_size=args.batch_size,
            sampler=torch.utils.data.SubsetRandomSampler(val_idx),
            num_workers=0
        )
        
        # Model
        num_ragas = len(dataset.raga_to_idx)
        print(f"Number of Ragas: {num_ragas}")
        
        if args.model_type == 'ast':
            model = RagaAST(num_ragas)
        else:
            model = RagaPaSST(num_ragas)
        
        # Train
        trainer = RagaTrainer(model)
        history = trainer.train(train_loader, val_loader, epochs=args.epochs)
        
        print("Training completed!")
        
    else:
        # Detection
        print("="*60)
        print("RAGA DETECTION")
        print("="*60)
        
        detector = RagaDetector(args.model, model_type=args.model_type)
        
        if args.input:
            # Detect from file
            results = detector.detect(args.input)
            
            if 'error' not in results:
                print(f"\nDetected Raga: {results['detected_raga']}")
                print(f"Confidence: {results['confidence']:.2%}")
                print(f"Duration: {results['duration']:.2f}s")
                
                print("\nTop 3:")
                for pred in results['top3_predictions']:
                    print(f"  {pred['raga']}: {pred['confidence']:.2%}")
                
                if 'features' in results:
                    plot_features(results['features'], 'features.png')
            else:
                print(f"Error: {results['error']}")
        
        else:
            # Interactive mode
            print("Enter file path or URL (or 'quit' to exit):")
            
            while True:
                input_source = input("\n> ").strip()
                
                if input_source.lower() in ['quit', 'exit', 'q']:
                    break
                
                if input_source:
                    results = detector.detect(input_source)
                    
                    if 'error' not in results:
                        print(f"\nRaga: {results['detected_raga']} ({results['confidence']:.1%})")
                    else:
                        print(f"Error: {results['error']}")


if __name__ == "__main__":
    main()