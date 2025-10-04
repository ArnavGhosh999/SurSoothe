# Cell 1: Imports and Setup
# Install required packages first:
# pip install librosa numpy scipy pandas scikit-learn hmmlearn pydub

import librosa
import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import mode
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

# For HMM
from hmmlearn import hmm

# For audio file handling
from pydub import AudioSegment
import os
import tempfile

print("✓ All libraries imported successfully")
print(f"Librosa version: {librosa.__version__}")

# Configuration Constants
CONFIG = {
    'SR': 22050,  # Sample rate
    'HOP_LENGTH': 512,
    'N_FFT': 2048,
    'N_MFCC': 13,
    'N_CHROMA': 12,
    'CENT_TOLERANCE': 50,  # cents tolerance for swara mapping
    'MIN_SEGMENT_LENGTH': 2.0,  # seconds
    'SEGMENT_WINDOWS': [5, 10],  # segment sizes for analysis
    'NGRAM_ORDER': 2,
    'ALPHA_SMOOTHING': 0.1,
    'HMM_N_STATES': 5,
}

# Swara mapping (12 semitones)
SWARAS = ['S', 'r', 'R', 'g', 'G', 'm', 'M', 'P', 'd', 'D', 'n', 'N']

print("\n✓ Configuration loaded")
print(f"Sample Rate: {CONFIG['SR']} Hz")
print(f"N-gram order: {CONFIG['NGRAM_ORDER']}")

# Cell 2: Audio Loading and Preprocessing

def load_audio_file(filepath):
    """
    Load audio file (supports wav, mp3, mp4, flac, ogg, etc.)
    Returns: audio time series, sample rate
    """
    file_ext = os.path.splitext(filepath)[1].lower()
    
    try:
        # Try direct loading with librosa (works for wav, flac, ogg)
        y, sr = librosa.load(filepath, sr=CONFIG['SR'], mono=True)
        print(f"✓ Loaded audio directly: {filepath}")
        return y, sr
    
    except Exception as e:
        print(f"Direct loading failed, trying pydub conversion...")
        
        # Use pydub for mp3, mp4, m4a, etc.
        try:
            audio = AudioSegment.from_file(filepath)
            
            # Export to temporary wav file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                tmp_path = tmp_file.name
                audio.export(tmp_path, format='wav')
            
            # Load with librosa
            y, sr = librosa.load(tmp_path, sr=CONFIG['SR'], mono=True)
            
            # Clean up temp file
            os.unlink(tmp_path)
            
            print(f"✓ Loaded and converted: {filepath}")
            return y, sr
        
        except Exception as e2:
            raise Exception(f"Failed to load audio file: {str(e2)}")


def preprocess_audio(y, sr):
    """
    Basic preprocessing: normalization and noise reduction
    """
    # Normalize
    y = librosa.util.normalize(y)
    
    # Apply a high-pass filter to remove DC offset and low-frequency noise
    sos = signal.butter(5, 80, 'hp', fs=sr, output='sos')
    y = signal.sosfilt(sos, y)
    
    return y


# Test function
def test_audio_loading():
    """
    Test with user-provided file path
    """
    print("\n" + "="*60)
    print("AUDIO LOADING TEST")
    print("="*60)
    
    filepath = input("\nEnter the path to your audio file: ").strip()
    
    if not os.path.exists(filepath):
        print(f"✗ File not found: {filepath}")
        return None, None
    
    print(f"\nLoading: {filepath}")
    
    try:
        y, sr = load_audio_file(filepath)
        y = preprocess_audio(y, sr)
        
        duration = len(y) / sr
        print(f"\n✓ Audio loaded successfully!")
        print(f"  Duration: {duration:.2f} seconds")
        print(f"  Sample rate: {sr} Hz")
        print(f"  Shape: {y.shape}")
        
        return y, sr
    
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        return None, None


# Uncomment to test:
audio, sr = test_audio_loading()


# Cell 3: Tonic Estimation (Sa Detection)

def estimate_tonic(y, sr):
    """
    Estimate tonic (Sa) using pitch histogram method
    Returns: tonic frequency in Hz
    """
    # Extract pitch using pYIN (probabilistic YIN)
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y,
        fmin=librosa.note_to_hz('C2'),  # ~65 Hz
        fmax=librosa.note_to_hz('C7'),  # ~2093 Hz
        sr=sr,
        hop_length=CONFIG['HOP_LENGTH']
    )
    
    # Filter out unvoiced frames
    valid_f0 = f0[voiced_flag]
    
    if len(valid_f0) == 0:
        print("Warning: No valid pitch detected, using default C4")
        return 261.63  # C4
    
    # Create pitch histogram (fold into one octave)
    pitch_cents = 1200 * np.log2(valid_f0 / valid_f0.min())
    pitch_classes = pitch_cents % 1200  # Fold to one octave
    
    # Find the most stable (sustained) pitch class
    hist, bin_edges = np.histogram(pitch_classes, bins=120, range=(0, 1200))
    
    # Find peaks in histogram (potential tonic candidates)
    peaks, properties = signal.find_peaks(hist, prominence=np.max(hist)*0.1)
    
    if len(peaks) == 0:
        # Fallback: use median of sustained pitches
        tonic_hz = np.median(valid_f0)
    else:
        # Use the most prominent peak
        peak_idx = peaks[np.argmax(properties['prominences'])]
        tonic_cents = bin_edges[peak_idx]
        
        # Convert back to Hz (use the base frequency)
        base_freq = valid_f0.min()
        tonic_hz = base_freq * (2 ** (tonic_cents / 1200))
        
        # Snap to nearest note for stability
        tonic_note = librosa.hz_to_note(tonic_hz)
        tonic_hz = librosa.note_to_hz(tonic_note)
    
    print(f"✓ Estimated tonic (Sa): {tonic_hz:.2f} Hz ({librosa.hz_to_note(tonic_hz)})")
    
    return tonic_hz


def estimate_tonic_robust(y, sr, segment_duration=10):
    """
    Robust tonic estimation using multiple segments
    """
    duration = len(y) / sr
    n_segments = max(1, int(duration / segment_duration))
    
    tonics = []
    segment_length = len(y) // n_segments
    
    for i in range(n_segments):
        start = i * segment_length
        end = min((i + 1) * segment_length, len(y))
        segment = y[start:end]
        
        if len(segment) / sr >= CONFIG['MIN_SEGMENT_LENGTH']:
            try:
                tonic = estimate_tonic(segment, sr)
                tonics.append(tonic)
            except:
                continue
    
    if len(tonics) == 0:
        return estimate_tonic(y, sr)
    
    # Use median tonic across segments
    final_tonic = np.median(tonics)
    print(f"\n✓ Robust tonic (median of {len(tonics)} segments): {final_tonic:.2f} Hz")
    
    return final_tonic


# Test
def test_tonic_estimation(y, sr):
    """Test tonic estimation on loaded audio"""
    if y is None:
        print("Please load audio first (run Cell 2)")
        return None
    
    print("\n" + "="*60)
    print("TONIC ESTIMATION TEST")
    print("="*60)
    
    tonic = estimate_tonic_robust(y, sr)
    return tonic

# Uncomment to test (after loading audio in Cell 2):
tonic = test_tonic_estimation(audio, sr)

# Cell 4: Pitch Tracking and Swara Extraction

def extract_pitch_track(y, sr):
    """
    Extract continuous pitch track using pYIN
    Returns: pitch in Hz, voiced flags, timestamps
    """
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y,
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7'),
        sr=sr,
        hop_length=CONFIG['HOP_LENGTH']
    )
    
    # Smooth pitch track with median filter
    f0_smooth = signal.medfilt(f0, kernel_size=5)
    
    # Time stamps
    times = librosa.frames_to_time(
        np.arange(len(f0)),
        sr=sr,
        hop_length=CONFIG['HOP_LENGTH']
    )
    
    return f0_smooth, voiced_flag, times


def hz_to_cents(freq_hz, tonic_hz):
    """
    Convert frequency to cents relative to tonic
    """
    if freq_hz <= 0 or tonic_hz <= 0:
        return None
    return 1200 * np.log2(freq_hz / tonic_hz)


def cents_to_swara(cents):
    """
    Map cents to swara (semitone)
    Cents are relative to tonic (Sa = 0)
    """
    if cents is None or np.isnan(cents):
        return None
    
    # Fold to one octave
    cents_normalized = cents % 1200
    
    # Map to nearest semitone (12-tone equal temperament)
    semitone = round(cents_normalized / 100)
    
    # Map to swara
    swara_idx = semitone % 12
    return SWARAS[swara_idx]


def extract_swara_sequence(y, sr, tonic_hz):
    """
    Extract sequence of swaras from audio
    Returns: list of (swara, time, cent_deviation)
    """
    # Extract pitch track
    f0, voiced_flag, times = extract_pitch_track(y, sr)
    
    swara_sequence = []
    
    for i, (pitch, is_voiced, time) in enumerate(zip(f0, voiced_flag, times)):
        if is_voiced and not np.isnan(pitch) and pitch > 0:
            # Convert to cents
            cents = hz_to_cents(pitch, tonic_hz)
            
            # Get swara
            swara = cents_to_swara(cents)
            
            if swara is not None:
                # Calculate deviation from perfect semitone
                cents_normalized = cents % 1200
                perfect_cents = (SWARAS.index(swara) % 12) * 100
                deviation = cents_normalized - perfect_cents
                
                swara_sequence.append({
                    'swara': swara,
                    'time': time,
                    'cents': cents,
                    'deviation': deviation,
                    'freq': pitch
                })
    
    return swara_sequence


def get_swara_tokens(swara_sequence):
    """
    Extract just the swara tokens as a list
    """
    return [s['swara'] for s in swara_sequence]


# Test
def test_swara_extraction(y, sr, tonic):
    """Test swara extraction"""
    if y is None or tonic is None:
        print("Please load audio and estimate tonic first")
        return None
    
    print("\n" + "="*60)
    print("SWARA EXTRACTION TEST")
    print("="*60)
    
    swara_seq = extract_swara_sequence(y, sr, tonic)
    tokens = get_swara_tokens(swara_seq)
    
    print(f"\n✓ Extracted {len(swara_seq)} swara frames")
    print(f"  Duration: {swara_seq[-1]['time']:.2f}s")
    print(f"  Unique swaras: {set(tokens)}")
    print(f"\n  First 20 swaras: {' '.join(tokens[:20])}")
    
    # Swara distribution
    swara_counts = Counter(tokens)
    print(f"\n  Swara distribution:")
    for swara in SWARAS:
        if swara in swara_counts:
            pct = 100 * swara_counts[swara] / len(tokens)
            print(f"    {swara}: {swara_counts[swara]:4d} ({pct:5.1f}%)")
    
    return swara_seq, tokens

# Uncomment to test:
swara_seq, tokens = test_swara_extraction(audio, sr, tonic)


# Cell 5: Feature Extraction (MFCC, Chroma, Spectral)

def extract_audio_features(y, sr):
    """
    Extract comprehensive audio features
    Returns: dictionary of features
    """
    features = {}
    
    # 1. MFCCs (Mel-frequency cepstral coefficients)
    mfccs = librosa.feature.mfcc(
        y=y, 
        sr=sr, 
        n_mfcc=CONFIG['N_MFCC'],
        hop_length=CONFIG['HOP_LENGTH']
    )
    
    # Statistics for each MFCC
    for i in range(CONFIG['N_MFCC']):
        features[f'mfcc{i+1}_mean'] = np.mean(mfccs[i])
        features[f'mfcc{i+1}_std'] = np.std(mfccs[i])
    
    # 2. Chroma features (pitch class profile)
    chroma = librosa.feature.chroma_stft(
        y=y,
        sr=sr,
        hop_length=CONFIG['HOP_LENGTH']
    )
    
    for i in range(CONFIG['N_CHROMA']):
        features[f'chroma{i+1}_mean'] = np.mean(chroma[i])
        features[f'chroma{i+1}_std'] = np.std(chroma[i])
    
    # 3. Spectral features
    spectral_centroids = librosa.feature.spectral_centroid(
        y=y, sr=sr, hop_length=CONFIG['HOP_LENGTH']
    )[0]
    features['spectral_centroid_mean'] = np.mean(spectral_centroids)
    features['spectral_centroid_std'] = np.std(spectral_centroids)
    
    spectral_bandwidth = librosa.feature.spectral_bandwidth(
        y=y, sr=sr, hop_length=CONFIG['HOP_LENGTH']
    )[0]
    features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
    features['spectral_bandwidth_std'] = np.std(spectral_bandwidth)
    
    # 4. Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(
        y, hop_length=CONFIG['HOP_LENGTH']
    )[0]
    features['zero_crossing_rate_mean'] = np.mean(zcr)
    features['zero_crossing_rate_std'] = np.std(zcr)
    
    # 5. Tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    features['tempo'] = tempo
    
    return features


def compute_pitch_histogram(swara_sequence, n_bins=12):
    """
    Compute pitch class histogram (normalized)
    Returns: histogram array of length n_bins
    """
    if len(swara_sequence) == 0:
        return np.zeros(n_bins)
    
    # Get cents values
    cents = np.array([s['cents'] % 1200 for s in swara_sequence])
    
    # Create histogram
    hist, _ = np.histogram(cents, bins=n_bins, range=(0, 1200))
    
    # Normalize
    hist = hist.astype(float)
    if hist.sum() > 0:
        hist = hist / hist.sum()
    
    return hist


def detect_ornamentation(swara_sequence, window_size=10):
    """
    Detect ornamentation by analyzing pitch variance
    Returns: ornamentation score (0-1)
    """
    if len(swara_sequence) < window_size:
        return 0.0
    
    # Calculate local pitch variance
    cents = [s['cents'] for s in swara_sequence]
    variances = []
    
    for i in range(len(cents) - window_size):
        window = cents[i:i+window_size]
        variances.append(np.var(window))
    
    # High variance indicates ornamentation
    avg_variance = np.mean(variances)
    ornamentation_score = min(1.0, avg_variance / 10000)  # Normalize
    
    return ornamentation_score


# Test
def test_feature_extraction(y, sr, swara_seq):
    """Test feature extraction"""
    if y is None or swara_seq is None:
        print("Please load audio and extract swaras first")
        return None, None
    
    print("\n" + "="*60)
    print("FEATURE EXTRACTION TEST")
    print("="*60)
    
    # Extract audio features
    features = extract_audio_features(y, sr)
    print(f"\n✓ Extracted {len(features)} audio features")
    print(f"\n  Sample features:")
    for key in list(features.keys())[:10]:
        print(f"    {key}: {features[key]:.4f}")
    
    # Pitch histogram
    pitch_hist = compute_pitch_histogram(swara_seq)
    print(f"\n✓ Pitch histogram (12 bins):")
    for i, val in enumerate(pitch_hist):
        print(f"    Bin {i:2d}: {'█' * int(val * 50)} {val:.3f}")
    
    # Ornamentation
    ornament_score = detect_ornamentation(swara_seq)
    print(f"\n✓ Ornamentation score: {ornament_score:.3f}")
    
    return features, pitch_hist

# Uncomment to test:
features, pitch_hist = test_feature_extraction(audio, sr, swara_seq)

# Cell 6: Load Training Data and Build Raga Profiles

def load_training_data():
    """
    Load training datasets
    Returns: DataFrames for each dataset
    """
    print("Loading training data...")
    
    # Load Final_dataset_s.csv (main dataset with features)
    df_final = pd.read_csv('Final_dataset_s.csv')
    print(f"✓ Loaded Final_dataset_s.csv: {len(df_final)} rows")
    
    # Load Dataset.csv (alternate dataset)
    df_dataset = pd.read_csv('Dataset.csv')
    print(f"✓ Loaded Dataset.csv: {len(df_dataset)} rows")
    
    # Load raga metadata
    df_metadata = pd.read_csv('raga_metadata.csv')
    print(f"✓ Loaded raga_metadata.csv: {len(df_metadata)} ragas")
    
    # Get unique ragas
    ragas_final = sorted(df_final['Raga'].unique())
    ragas_dataset = sorted(df_dataset['raga'].unique())
    
    print(f"\nRagas in Final_dataset_s: {len(ragas_final)}")
    print(f"  {ragas_final}")
    print(f"\nRagas in Dataset: {len(ragas_dataset)}")
    print(f"  {ragas_dataset}")
    
    return df_final, df_dataset, df_metadata


def build_raga_profiles_from_data(df_final):
    """
    Build pitch-class profiles for each raga from training data
    Returns: dictionary {raga: profile_vector}
    """
    profiles = {}
    
    # Group by raga
    for raga in df_final['Raga'].unique():
        raga_data = df_final[df_final['Raga'] == raga]
        
        # Extract chroma features (12 pitch classes)
        chroma_cols = [f'chroma{i}' for i in range(1, 13)]
        chroma_values = raga_data[chroma_cols].values
        
        # Average chroma across all samples for this raga
        profile = np.mean(chroma_values, axis=0)
        
        # Normalize
        profile = profile / (profile.sum() + 1e-10)
        
        profiles[raga] = profile
    
    print(f"\n✓ Built profiles for {len(profiles)} ragas")
    return profiles


def build_tempo_profiles(df_final):
    """
    Build tempo profiles for each raga
    Returns: dictionary {raga: (mean_tempo, std_tempo)}
    """
    tempo_profiles = {}
    
    for raga in df_final['Raga'].unique():
        raga_data = df_final[df_final['Raga'] == raga]
        tempos = raga_data['tempo'].values
        
        tempo_profiles[raga] = {
            'mean': np.mean(tempos),
            'std': np.std(tempos)
        }
    
    return tempo_profiles


def get_raga_priors(df_final):
    """
    Compute prior probabilities for each raga based on frequency
    Returns: dictionary {raga: prior_probability}
    """
    raga_counts = df_final['Raga'].value_counts()
    total = len(df_final)
    
    priors = {}
    for raga, count in raga_counts.items():
        priors[raga] = count / total
    
    return priors


# Test
def test_load_and_build_profiles():
    """Test loading data and building profiles"""
    print("\n" + "="*60)
    print("LOADING TRAINING DATA AND BUILDING PROFILES")
    print("="*60 + "\n")
    
    # Load data
    df_final, df_dataset, df_metadata = load_training_data()
    
    # Build profiles
    print("\nBuilding raga profiles...")
    profiles = build_raga_profiles_from_data(df_final)
    
    # Show sample profile
    sample_raga = list(profiles.keys())[0]
    print(f"\nSample profile ({sample_raga}):")
    for i, val in enumerate(profiles[sample_raga]):
        print(f"  Pitch class {i:2d}: {'█' * int(val * 50)} {val:.4f}")
    
    # Tempo profiles
    tempo_profiles = build_tempo_profiles(df_final)
    print(f"\n✓ Built tempo profiles for {len(tempo_profiles)} ragas")
    
    # Priors
    priors = get_raga_priors(df_final)
    print(f"\n✓ Computed priors for {len(priors)} ragas")
    print(f"\nTop 5 most common ragas:")
    for raga, prob in sorted(priors.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {raga:15s}: {prob:.4f}")
    
    return df_final, df_dataset, df_metadata, profiles, tempo_profiles, priors

# Run this cell to load data
df_final, df_dataset, df_metadata, raga_profiles, tempo_profiles, raga_priors = test_load_and_build_profiles()

# Cell 7: N-gram Model (Markov Chain)

class NgramModel:
    """
    N-gram language model for swara sequences
    """
    def __init__(self, n=2, alpha=0.1):
        self.n = n
        self.alpha = alpha  # Laplace smoothing
        self.ngram_counts = defaultdict(lambda: defaultdict(int))
        self.context_counts = defaultdict(int)
        self.vocab = set()
    
    def train(self, sequences):
        """
        Train on list of swara sequences
        sequences: list of lists of swaras
        """
        for seq in sequences:
            # Add padding
            padded = ['<START>'] * (self.n - 1) + seq + ['<END>']
            
            # Update vocabulary
            self.vocab.update(seq)
            
            # Extract n-grams
            for i in range(len(padded) - self.n + 1):
                context = tuple(padded[i:i+self.n-1])
                next_token = padded[i+self.n-1]
                
                self.ngram_counts[context][next_token] += 1
                self.context_counts[context] += 1
    
    def score_sequence(self, sequence):
        """
        Compute log-likelihood of a sequence
        """
        if len(sequence) == 0:
            return -1e10
        
        padded = ['<START>'] * (self.n - 1) + sequence + ['<END>']
        log_prob = 0.0
        vocab_size = len(self.vocab) + 2  # +2 for START and END
        
        for i in range(len(padded) - self.n + 1):
            context = tuple(padded[i:i+self.n-1])
            next_token = padded[i+self.n-1]
            
            # Laplace smoothing
            count = self.ngram_counts[context][next_token]
            context_total = self.context_counts[context]
            
            prob = (count + self.alpha) / (context_total + self.alpha * vocab_size)
            log_prob += np.log(prob + 1e-10)
        
        return log_prob


def build_ngram_models_per_raga(df_final, n=2):
    """
    Build n-gram model for each raga from training data
    Note: We don't have actual swara sequences in the dataset,
    so we'll simulate them based on chroma features
    """
    models = {}
    
    print(f"\nBuilding {n}-gram models for each raga...")
    
    for raga in df_final['Raga'].unique():
        raga_data = df_final[df_final['Raga'] == raga]
        
        # Simulate swara sequences from chroma features
        sequences = []
        for idx, row in raga_data.iterrows():
            # Get dominant pitch classes from chroma
            chroma_vals = [row[f'chroma{i}'] for i in range(1, 13)]
            
            # Create a sequence based on chroma strengths
            # (simplified approximation)
            sequence = []
            for _ in range(10):  # Short sequence per sample
                # Pick swaras probabilistically based on chroma
                swara_idx = np.random.choice(12, p=np.array(chroma_vals)/sum(chroma_vals))
                sequence.append(SWARAS[swara_idx])
            
            sequences.append(sequence)
        
        # Train model
        model = NgramModel(n=n, alpha=CONFIG['ALPHA_SMOOTHING'])
        model.train(sequences)
        models[raga] = model
    
    print(f"✓ Built {len(models)} n-gram models")
    return models


def score_with_ngram_models(tokens, models):
    """
    Score a swara sequence against all raga models
    Returns: dictionary {raga: log_likelihood}
    """
    scores = {}
    
    for raga, model in models.items():
        scores[raga] = model.score_sequence(tokens)
    
    return scores


# Test
def test_ngram_models(df_final, tokens=None):
    """Test n-gram model building and scoring"""
    print("\n" + "="*60)
    print("N-GRAM MODEL TEST")
    print("="*60)
    
    # Build models
    ngram_models = build_ngram_models_per_raga(df_final, n=CONFIG['NGRAM_ORDER'])
    
    if tokens is not None and len(tokens) > 0:
        print(f"\nScoring test sequence ({len(tokens)} tokens)...")
        scores = score_with_ngram_models(tokens[:100], ngram_models)  # Use first 100 tokens
        
        # Show top 5
        top_ragas = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"\nTop 5 ragas by n-gram score:")
        for raga, score in top_ragas:
            print(f"  {raga:15s}: {score:10.2f}")
    
    return ngram_models

# Uncomment to test (after extracting swaras):
ngram_models = test_ngram_models(df_final, tokens)


# Cell 8: Hidden Markov Models (HMM)

def build_hmm_models_per_raga(df_final, n_states=5):
    """
    Build Gaussian HMM for each raga
    States represent different melodic regions
    Emissions are MFCC features
    """
    hmm_models = {}
    
    print(f"\nBuilding HMMs with {n_states} states for each raga...")
    
    for raga in df_final['Raga'].unique():
        raga_data = df_final[df_final['Raga'] == raga]
        
        # Extract MFCC features
        mfcc_cols = [f'mfcc{i}' for i in range(1, 14)]
        X = raga_data[mfcc_cols].values
        
        if len(X) < n_states:
            print(f"  Warning: {raga} has only {len(X)} samples, skipping")
            continue
        
        # Create and train HMM
        model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type='diag',
            n_iter=100,
            random_state=42
        )
        
        try:
            # Reshape for HMM (needs sequence lengths)
            lengths = [len(X)]  # Treat all as one sequence
            model.fit(X, lengths)
            hmm_models[raga] = model
        except Exception as e:
            print(f"  Warning: Failed to train HMM for {raga}: {str(e)}")
            continue
    
    print(f"✓ Built {len(hmm_models)} HMM models")
    return hmm_models


def extract_hmm_features(y, sr):
    """
    Extract MFCC features for HMM scoring
    Returns: feature matrix (time x features)
    """
    # Extract MFCCs with more temporal resolution
    mfccs = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=13,
        hop_length=512
    )
    
    # Transpose to (time x features)
    return mfccs.T


def score_with_hmm_models(features, models):
    """
    Score features against all HMM models
    Returns: dictionary {raga: log_likelihood}
    """
    scores = {}
    
    for raga, model in models.items():
        try:
            score = model.score(features)
            scores[raga] = score
        except Exception as e:
            scores[raga] = -1e10  # Very low score if error
    
    return scores


# Alternative: Build simpler pitch-based HMMs
def build_pitch_hmm_models(df_final, n_states=4):
    """
    Build HMM based on pitch/chroma features
    More relevant for raga which is melody-based
    """
    hmm_models = {}
    
    print(f"\nBuilding pitch-based HMMs with {n_states} states...")
    
    for raga in df_final['Raga'].unique():
        raga_data = df_final[df_final['Raga'] == raga]
        
        # Use chroma + spectral features
        feature_cols = [f'chroma{i}' for i in range(1, 13)]
        feature_cols += ['spectral_centroid', 'tempo']
        
        X = raga_data[feature_cols].values
        
        if len(X) < n_states:
            continue
        
        # Normalize features
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-10)
        
        model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type='diag',
            n_iter=100,
            random_state=42
        )
        
        try:
            lengths = [len(X)]
            model.fit(X, lengths)
            hmm_models[raga] = model
        except:
            continue
    
    print(f"✓ Built {len(hmm_models)} pitch-based HMM models")
    return hmm_models


# Test
def test_hmm_models(df_final, y=None, sr=None):
    """Test HMM model building and scoring"""
    print("\n" + "="*60)
    print("HMM MODEL TEST")
    print("="*60)
    
    # Build MFCC-based HMMs
    hmm_models = build_hmm_models_per_raga(df_final, n_states=CONFIG['HMM_N_STATES'])
    
    # Build pitch-based HMMs
    pitch_hmm_models = build_pitch_hmm_models(df_final, n_states=CONFIG['HMM_N_STATES'])
    
    if y is not None and sr is not None:
        print(f"\nScoring test audio with HMMs...")
        
        # Extract features
        hmm_features = extract_hmm_features(y, sr)
        
        # Score with MFCC HMMs
        scores = score_with_hmm_models(hmm_features, hmm_models)
        
        # Show top 5
        top_ragas = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"\nTop 5 ragas by HMM score:")
        for raga, score in top_ragas:
            print(f"  {raga:15s}: {score:10.2f}")
    
    return hmm_models, pitch_hmm_models

# Uncomment to test (after loading audio):
hmm_models, pitch_hmm_models = test_hmm_models(df_final, audio, sr)

# Cell 9: Krumhansl-style Scale Matching (Pitch Histogram)

def cosine_similarity(a, b):
    """
    Compute cosine similarity between two vectors
    """
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return dot_product / (norm_a * norm_b)


def kl_divergence(p, q):
    """
    Compute KL divergence: KL(p || q)
    """
    p = np.array(p) + 1e-10
    q = np.array(q) + 1e-10
    
    # Normalize
    p = p / p.sum()
    q = q / q.sum()
    
    return np.sum(p * np.log(p / q))


def score_with_scale_matching(pitch_hist, profiles, method='cosine'):
    """
    Score pitch histogram against raga profiles
    Returns: dictionary {raga: score}
    """
    scores = {}
    
    for raga, profile in profiles.items():
        if method == 'cosine':
            # Cosine similarity (higher is better)
            score = cosine_similarity(pitch_hist, profile)
        
        elif method == 'kl':
            # Negative KL divergence (higher is better)
            score = -kl_divergence(pitch_hist, profile)
        
        elif method == 'log_likelihood':
            # Treat profile as multinomial distribution
            profile_norm = profile / (profile.sum() + 1e-10)
            score = np.sum(pitch_hist * np.log(profile_norm + 1e-10))
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        scores[raga] = score
    
    return scores


def score_with_chroma_profile(chroma_features, profiles):
    """
    Score chroma features against raga profiles
    chroma_features: average chroma vector (12-dimensional)
    """
    scores = {}
    
    # Normalize input chroma
    chroma_norm = chroma_features / (chroma_features.sum() + 1e-10)
    
    for raga, profile in profiles.items():
        # Cosine similarity
        score = cosine_similarity(chroma_norm, profile)
        scores[raga] = score
    
    return scores


def compute_chroma_from_audio(y, sr):
    """
    Compute average chroma vector from audio
    """
    chroma = librosa.feature.chroma_stft(
        y=y,
        sr=sr,
        hop_length=CONFIG['HOP_LENGTH']
    )
    
    # Average across time
    chroma_mean = np.mean(chroma, axis=1)
    
    return chroma_mean


# Test
def test_scale_matching(pitch_hist, profiles, y=None, sr=None):
    """Test scale matching"""
    print("\n" + "="*60)
    print("SCALE MATCHING TEST")
    print("="*60)
    
    if pitch_hist is not None:
        print("\nScoring with pitch histogram...")
        
        # Try different methods
        for method in ['cosine', 'log_likelihood']:
            print(f"\n  Method: {method}")
            scores = score_with_scale_matching(pitch_hist, profiles, method=method)
            
            # Top 5
            top_ragas = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
            print(f"  Top 5 ragas:")
            for raga, score in top_ragas:
                print(f"    {raga:15s}: {score:8.4f}")
    
    if y is not None and sr is not None:
        print("\n\nScoring with chroma features...")
        chroma = compute_chroma_from_audio(y, sr)
        scores = score_with_chroma_profile(chroma, profiles)
        
        # Top 5
        top_ragas = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"  Top 5 ragas:")
        for raga, score in top_ragas:
            print(f"    {raga:15s}: {score:8.4f}")
        
        return scores
    
    return None

# Uncomment to test:
scale_scores = test_scale_matching(pitch_hist, raga_profiles, audio, sr)

# Cell 10: Ensemble Fusion and Final Prediction

def combine_scores_weighted(score_dicts, weights, priors=None):
    """
    Combine scores from multiple models using weighted log-linear combination
    
    score_dicts: list of dictionaries {raga: score}
    weights: list of weights for each score dictionary
    priors: optional dictionary {raga: prior_probability}
    
    Returns: dictionary {raga: posterior_score}
    """
    # Get all ragas
    all_ragas = set()
    for scores in score_dicts:
        all_ragas.update(scores.keys())
    
    combined = {}
    
    for raga in all_ragas:
        log_score = 0.0
        
        # Weighted sum of scores
        for scores, weight in zip(score_dicts, weights):
            if raga in scores:
                log_score += weight * scores[raga]
            else:
                log_score += weight * (-1e10)  # Very low score if missing
        
        # Add prior
        if priors and raga in priors:
            log_score += np.log(priors[raga] + 1e-10)
        
        combined[raga] = log_score
    
    return combined


def softmax_scores(scores):
    """
    Convert scores to probabilities using softmax
    Returns: dictionary {raga: probability}
    """
    ragas = list(scores.keys())
    score_values = np.array([scores[r] for r in ragas])
    
    # Subtract max for numerical stability
    score_values = score_values - np.max(score_values)
    
    # Softmax
    exp_scores = np.exp(score_values)
    probs = exp_scores / exp_scores.sum()
    
    return {raga: prob for raga, prob in zip(ragas, probs)}


def predict_raga(y, sr, raga_profiles, ngram_models, hmm_models, raga_priors, 
                 weights={'scale': 1.0, 'markov': 1.5, 'hmm': 2.0}):
    """
    Main prediction function - combines all three modules
    
    Returns: (predicted_raga, top_k_ragas, all_scores)
    """
    print("\n" + "="*60)
    print("RAGA PREDICTION PIPELINE")
    print("="*60)
    
    # Step 1: Tonic estimation
    print("\n[1/6] Estimating tonic...")
    tonic = estimate_tonic_robust(y, sr)
    
    # Step 2: Extract swara sequence
    print("\n[2/6] Extracting swara sequence...")
    swara_seq = extract_swara_sequence(y, sr, tonic)
    tokens = get_swara_tokens(swara_seq)
    print(f"  Extracted {len(tokens)} swara tokens")
    
    # Step 3: Compute pitch histogram
    print("\n[3/6] Computing pitch histogram...")
    pitch_hist = compute_pitch_histogram(swara_seq)
    
    # Step 4: Score with scale matching (Krumhansl)
    print("\n[4/6] Scoring with scale matching...")
    scale_scores = score_with_scale_matching(
        pitch_hist, raga_profiles, method='log_likelihood'
    )
    
    # Step 5: Score with n-gram model
    print("\n[5/6] Scoring with n-gram model...")
    if len(tokens) > 0:
        markov_scores = score_with_ngram_models(tokens[:200], ngram_models)
    else:
        markov_scores = {raga: -1e10 for raga in raga_profiles.keys()}
    
    # Step 6: Score with HMM
    print("\n[6/6] Scoring with HMM...")
    hmm_features = extract_hmm_features(y, sr)
    hmm_scores = score_with_hmm_models(hmm_features, hmm_models)
    
    # Combine scores
    print("\n" + "-"*60)
    print("COMBINING SCORES")
    print("-"*60)
    
    combined_scores = combine_scores_weighted(
        [scale_scores, markov_scores, hmm_scores],
        [weights['scale'], weights['markov'], weights['hmm']],
        priors=raga_priors
    )
    
    # Convert to probabilities
    posteriors = softmax_scores(combined_scores)
    
    # Get top predictions
    top_ragas = sorted(posteriors.items(), key=lambda x: x[1], reverse=True)
    
    predicted_raga = top_ragas[0][0]
    
    # Display results
    print(f"\n{'='*60}")
    print("PREDICTION RESULTS")
    print(f"{'='*60}")
    print(f"\n🎵 PREDICTED RAGA: {predicted_raga}")
    print(f"   Confidence: {top_ragas[0][1]*100:.2f}%")
    
    print(f"\n📊 Top 5 Predictions:")
    for i, (raga, prob) in enumerate(top_ragas[:5], 1):
        bar = '█' * int(prob * 50)
        print(f"  {i}. {raga:15s}  {bar} {prob*100:5.2f}%")
    
    print(f"\n📈 Component Scores (for top prediction):")
    print(f"  Scale matching: {scale_scores.get(predicted_raga, 0):10.2f}")
    print(f"  N-gram (Markov): {markov_scores.get(predicted_raga, 0):10.2f}")
    print(f"  HMM: {hmm_scores.get(predicted_raga, 0):10.2f}")
    
    return predicted_raga, top_ragas[:5], {
        'combined': combined_scores,
        'posteriors': posteriors,
        'scale': scale_scores,
        'markov': markov_scores,
        'hmm': hmm_scores
    }


# Convenience function for quick prediction
def quick_predict(audio_path, raga_profiles, ngram_models, hmm_models, raga_priors):
    """
    Load audio and predict raga in one call
    """
    print(f"\n{'='*60}")
    print(f"ANALYZING: {os.path.basename(audio_path)}")
    print(f"{'='*60}")
    
    # Load audio
    y, sr = load_audio_file(audio_path)
    y = preprocess_audio(y, sr)
    
    # Predict
    return predict_raga(y, sr, raga_profiles, ngram_models, hmm_models, raga_priors)


# Example usage template:
"""
# After building all models (run cells 6-8), use this:

audio_path = input("Enter path to audio file: ").strip()
predicted_raga, top_5, all_scores = quick_predict(
    audio_path,
    raga_profiles,
    ngram_models,
    hmm_models,
    raga_priors
)
"""

# Cell 11: Complete Pipeline Runner

class RagaPredictor:
    """
    Complete raga prediction system
    """
    def __init__(self):
        self.raga_profiles = None
        self.ngram_models = None
        self.hmm_models = None
        self.raga_priors = None
        self.tempo_profiles = None
        self.is_trained = False
        
        self.weights = {
            'scale': 1.0,
            'markov': 1.5,
            'hmm': 2.0
        }
    
    def train(self, df_final):
        """
        Train all models from dataset
        """
        print("\n" + "="*60)
        print("TRAINING RAGA PREDICTION MODELS")
        print("="*60)
        
        # Build scale profiles
        print("\n[1/5] Building pitch-class profiles...")
        self.raga_profiles = build_raga_profiles_from_data(df_final)
        
        # Build tempo profiles
        print("\n[2/5] Building tempo profiles...")
        self.tempo_profiles = build_tempo_profiles(df_final)
        
        # Build n-gram models
        print("\n[3/5] Building n-gram models...")
        self.ngram_models = build_ngram_models_per_raga(df_final, n=CONFIG['NGRAM_ORDER'])
        
        # Build HMM models
        print("\n[4/5] Building HMM models...")
        self.hmm_models = build_hmm_models_per_raga(df_final, n_states=CONFIG['HMM_N_STATES'])
        
        # Compute priors
        print("\n[5/5] Computing raga priors...")
        self.raga_priors = get_raga_priors(df_final)
        
        self.is_trained = True
        
        print("\n" + "="*60)
        print("✓ TRAINING COMPLETE")
        print("="*60)
        print(f"\nModels trained for {len(self.raga_profiles)} ragas:")
        print(f"  {', '.join(sorted(self.raga_profiles.keys()))}")
    
    def predict(self, audio_path, verbose=True):
        """
        Predict raga from audio file
        """
        if not self.is_trained:
            raise Exception("Models not trained! Call train() first.")
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"ANALYZING: {os.path.basename(audio_path)}")
            print(f"{'='*60}")
        
        # Load and preprocess
        y, sr = load_audio_file(audio_path)
        y = preprocess_audio(y, sr)
        
        # Predict
        predicted, top_k, scores = predict_raga(
            y, sr,
            self.raga_profiles,
            self.ngram_models,
            self.hmm_models,
            self.raga_priors,
            weights=self.weights
        )
        
        return predicted, top_k, scores
    
    def tune_weights(self, validation_data, verbose=False):
        """
        Tune ensemble weights using validation data
        validation_data: list of (audio_path, true_raga) tuples
        """
        print("\n" + "="*60)
        print("TUNING ENSEMBLE WEIGHTS")
        print("="*60)
        
        best_accuracy = 0
        best_weights = self.weights.copy()
        
        # Grid search
        for w_scale in [0.5, 1.0, 1.5, 2.0]:
            for w_markov in [0.5, 1.0, 1.5, 2.0]:
                for w_hmm in [1.0, 1.5, 2.0, 2.5]:
                    self.weights = {
                        'scale': w_scale,
                        'markov': w_markov,
                        'hmm': w_hmm
                    }
                    
                    correct = 0
                    for audio_path, true_raga in validation_data:
                        try:
                            predicted, _, _ = self.predict(audio_path, verbose=False)
                            if predicted == true_raga:
                                correct += 1
                        except:
                            continue
                    
                    accuracy = correct / len(validation_data)
                    
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_weights = self.weights.copy()
                        print(f"\n✓ New best: {best_weights}")
                        print(f"  Accuracy: {accuracy*100:.2f}%")
        
        self.weights = best_weights
        print(f"\n{'='*60}")
        print(f"✓ TUNING COMPLETE")
        print(f"{'='*60}")
        print(f"\nBest weights: {self.weights}")
        print(f"Best accuracy: {best_accuracy*100:.2f}%")
    
    def save_models(self, filepath='raga_predictor.pkl'):
        """
        Save trained models to file
        """
        import pickle
        
        model_data = {
            'raga_profiles': self.raga_profiles,
            'ngram_models': self.ngram_models,
            'hmm_models': self.hmm_models,
            'raga_priors': self.raga_priors,
            'tempo_profiles': self.tempo_profiles,
            'weights': self.weights
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"\n✓ Models saved to {filepath}")
    
    def load_models(self, filepath='raga_predictor.pkl'):
        """
        Load trained models from file
        """
        import pickle
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.raga_profiles = model_data['raga_profiles']
        self.ngram_models = model_data['ngram_models']
        self.hmm_models = model_data['hmm_models']
        self.raga_priors = model_data['raga_priors']
        self.tempo_profiles = model_data['tempo_profiles']
        self.weights = model_data['weights']
        self.is_trained = True
        
        print(f"\n✓ Models loaded from {filepath}")
        print(f"  Ragas available: {len(self.raga_profiles)}")


# Create and train the predictor
def initialize_predictor():
    """
    Initialize and train the complete predictor
    """
    # Load training data
    df_final, _, _ = load_training_data()
    
    # Create predictor
    predictor = RagaPredictor()
    
    # Train
    predictor.train(df_final)
    
    return predictor


# Main execution
print("\n" + "="*60)
print("RAGA PREDICTION SYSTEM - READY")
print("="*60)
print("\nTo use the system:")
print("1. Run: predictor = initialize_predictor()")
print("2. Run: predicted, top_5, scores = predictor.predict('your_audio.mp3')")
print("\nOr for interactive mode:")
print("3. Run the interactive prediction below")


# Cell 12: Interactive Prediction Interface

def interactive_prediction():
    """
    Interactive interface for raga prediction
    """
    print("\n" + "="*60)
    print("INTERACTIVE RAGA PREDICTION")
    print("="*60)
    
    # Initialize predictor
    print("\n[Step 1] Initializing predictor...")
    predictor = initialize_predictor()
    
    print("\n[Step 2] Ready for prediction!")
    print("\nEnter audio file paths to predict ragas.")
    print("Type 'quit' to exit, 'save' to save models, 'load' to load models")
    print("="*60 + "\n")
    
    while True:
        audio_path = input("\n🎵 Enter audio file path (or command): ").strip()
        
        if audio_path.lower() == 'quit':
            print("\n👋 Goodbye!")
            break
        
        elif audio_path.lower() == 'save':
            filepath = input("Enter save path (default: raga_predictor.pkl): ").strip()
            if not filepath:
                filepath = 'raga_predictor.pkl'
            predictor.save_models(filepath)
            continue
        
        elif audio_path.lower() == 'load':
            filepath = input("Enter model path: ").strip()
            if os.path.exists(filepath):
                predictor.load_models(filepath)
            else:
                print(f"✗ File not found: {filepath}")
            continue
        
        elif not os.path.exists(audio_path):
            print(f"✗ File not found: {audio_path}")
            continue
        
        # Predict
        try:
            predicted, top_5, scores = predictor.predict(audio_path)
            
            # Additional info
            print(f"\n📚 Raga Information:")
            # Look up in metadata if available
            try:
                df_metadata = pd.read_csv('raga_metadata.csv')
                raga_info = df_metadata[df_metadata['raga'] == predicted]
                if len(raga_info) > 0:
                    print(f"   {raga_info.iloc[0]['description']}")
                else:
                    print(f"   (No metadata available for {predicted})")
            except:
                print(f"   (Metadata file not available)")
            
            # Ask if user wants to continue
            print("\n" + "-"*60)
        
        except Exception as e:
            print(f"\n✗ Error during prediction: {str(e)}")
            import traceback
            traceback.print_exc()


def batch_prediction(audio_files):
    """
    Predict ragas for multiple audio files
    
    audio_files: list of file paths
    Returns: list of (filepath, predicted_raga, confidence) tuples
    """
    print("\n" + "="*60)
    print("BATCH RAGA PREDICTION")
    print("="*60)
    
    # Initialize predictor
    predictor = initialize_predictor()
    
    results = []
    
    for i, audio_path in enumerate(audio_files, 1):
        print(f"\n[{i}/{len(audio_files)}] Processing: {os.path.basename(audio_path)}")
        
        if not os.path.exists(audio_path):
            print(f"  ✗ File not found, skipping")
            results.append((audio_path, None, 0.0))
            continue
        
        try:
            predicted, top_5, scores = predictor.predict(audio_path, verbose=False)
            confidence = top_5[0][1] * 100
            
            print(f"  ✓ Predicted: {predicted} ({confidence:.1f}% confidence)")
            results.append((audio_path, predicted, confidence))
        
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
            results.append((audio_path, None, 0.0))
    
    # Summary
    print("\n" + "="*60)
    print("BATCH PREDICTION SUMMARY")
    print("="*60)
    
    for audio_path, predicted, confidence in results:
        status = "✓" if predicted else "✗"
        pred_str = f"{predicted} ({confidence:.1f}%)" if predicted else "FAILED"
        print(f"{status} {os.path.basename(audio_path):40s} -> {pred_str}")
    
    return results


def evaluate_on_dataset(test_data):
    """
    Evaluate predictor on labeled test data
    
    test_data: list of (audio_path, true_raga) tuples
    """
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    # Initialize predictor
    predictor = initialize_predictor()
    
    correct = 0
    total = 0
    confusion = defaultdict(lambda: defaultdict(int))
    
    for audio_path, true_raga in test_data:
        print(f"\n[{total+1}/{len(test_data)}] {os.path.basename(audio_path)}")
        print(f"  True raga: {true_raga}")
        
        try:
            predicted, top_5, scores = predictor.predict(audio_path, verbose=False)
            print(f"  Predicted: {predicted}")
            
            if predicted == true_raga:
                print("  ✓ CORRECT")
                correct += 1
            else:
                print("  ✗ WRONG")
            
            confusion[true_raga][predicted] += 1
            total += 1
        
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
            continue
    
    # Results
    accuracy = (correct / total * 100) if total > 0 else 0
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"\nAccuracy: {correct}/{total} = {accuracy:.2f}%")
    
    print(f"\nConfusion Matrix:")
    print(f"{'True \\ Pred':<15s}", end="")
    
    all_ragas = sorted(set(list(confusion.keys())))
    for raga in all_ragas[:10]:  # Show first 10
        print(f"{raga[:8]:>8s}", end="")
    print()
    
    for true_raga in all_ragas[:10]:
        print(f"{true_raga[:15]:<15s}", end="")
        for pred_raga in all_ragas[:10]:
            count = confusion[true_raga][pred_raga]
            print(f"{count:>8d}", end="")
        print()
    
    return accuracy, confusion


# Quick start function
def quick_start():
    """
    Quick start - train and predict in one go
    """
    print("\n" + "="*60)
    print("RAGA PREDICTION SYSTEM - QUICK START")
    print("="*60)
    
    # Load data and train
    print("\n📊 Loading training data...")
    df_final, _, _ = load_training_data()
    
    print("\n🔧 Training models...")
    predictor = RagaPredictor()
    predictor.train(df_final)
    
    print("\n✓ System ready!")
    
    # Get audio file
    audio_path = input("\n🎵 Enter path to audio file to analyze: ").strip()
    
    if os.path.exists(audio_path):
        predicted, top_5, scores = predictor.predict(audio_path)
        return predictor
    else:
        print(f"\n✗ File not found: {audio_path}")
        return predictor


# Example usage patterns:
"""
# Pattern 1: Interactive mode
interactive_prediction()

# Pattern 2: Single prediction
predictor = initialize_predictor()
predicted, top_5, scores = predictor.predict('my_audio.mp3')

# Pattern 3: Batch prediction
audio_files = ['song1.mp3', 'song2.wav', 'song3.mp4']
results = batch_prediction(audio_files)

# Pattern 4: Quick start
predictor = quick_start()

# Pattern 5: Save/Load models
predictor.save_models('my_models.pkl')
# Later...
new_predictor = RagaPredictor()
new_predictor.load_models('my_models.pkl')
predicted, _, _ = new_predictor.predict('audio.mp3')
"""

print("\n" + "="*60)
print("✓ ALL FUNCTIONS LOADED")
print("="*60)
print("\n🚀 Quick commands:")
print("  • interactive_prediction()  - Start interactive mode")
print("  • quick_start()            - Train and predict quickly")
print("  • batch_prediction([...])  - Predict multiple files")
print("\n💡 Or build your own pipeline using the RagaPredictor class")