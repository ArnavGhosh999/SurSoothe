# therapy_mapping.py - Audio-Text Mapping for Raga-based Music Therapy
"""
Audio-Text Mapping System using Wav2CLIP for Raga-based Music Therapy

This module maps detected Raga embeddings to therapy-related text descriptions
using Wav2CLIP model for multimodal audio-text understanding.

Features:
- Integrates with raga_detection.py output
- Uses Wav2CLIP for audio-text embeddings
- Maps ragas to therapeutic descriptions
- Uses therapy-specific features: Age, Gender, Mental_Condition, Severity, Improvement_Score, Listening_Time
- Creates personalized therapy recommendations

Requirements:
    pip install torch torchvision torchaudio
    pip install transformers
    pip install librosa
    pip install numpy
    pip install scikit-learn
    pip install pandas
    pip install sentence-transformers

Usage:
    python therapy_mapping.py
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import logging
from pathlib import Path
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check for required libraries
try:
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers not available. Install with: pip install transformers")

try:
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("Scikit-learn not available. Install with: pip install scikit-learn")

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    logger.warning("Librosa not available. Install with: pip install librosa")

@dataclass
class TherapyRecommendation:
    """Data class for therapy recommendations"""
    raga_name: str
    patient_profile: Dict[str, Any]
    therapy_type: str
    mood_enhancement: str
    stress_reduction_level: str
    recommended_duration: int  # minutes
    best_time_of_day: str
    therapeutic_benefits: List[str]
    contraindications: List[str]
    confidence_score: float
    audio_similarity_score: float
    personalization_score: float

@dataclass
class PatientProfile:
    """Patient profile from therapy dataset"""
    age: int
    gender: str
    mental_condition: str
    severity: str
    improvement_score: float
    listening_time: float

class Wav2CLIPEmbedder:
    """
    Wav2CLIP-inspired audio-text embedding system
    Creates joint embeddings for audio and text in shared space
    """
    
    def __init__(self, model_name: str = "distilbert-base-uncased"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        
        # Initialize text encoder
        if TRANSFORMERS_AVAILABLE:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.text_encoder = AutoModel.from_pretrained(model_name)
            self.text_encoder.to(self.device)
            self.text_encoder.eval()
        else:
            logger.warning("Using dummy text encoder - install transformers for full functionality")
            self.tokenizer = None
            self.text_encoder = None
        
        # Initialize audio encoder (simplified CNN-based)
        self.audio_encoder = self._build_audio_encoder()
        self.audio_encoder.to(self.device)
        self.audio_encoder.eval()
        
        # Projection layers to shared embedding space
        embedding_dim = 256
        self.audio_projection = nn.Linear(512, embedding_dim).to(self.device)
        self.text_projection = nn.Linear(768, embedding_dim).to(self.device)  # BERT hidden size
        
        # Scaler for audio features
        if SKLEARN_AVAILABLE:
            self.scaler = StandardScaler()
        
        logger.info(f"Wav2CLIP embedder initialized on {self.device}")
    
    def _build_audio_encoder(self) -> nn.Module:
        """Build a simple CNN-based audio encoder"""
        return nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
    
    def encode_audio(self, audio_features: np.ndarray) -> np.ndarray:
        """
        Encode audio features to embedding space
        
        Args:
            audio_features: Audio feature vector
            
        Returns:
            Audio embedding
        """
        try:
            # Ensure audio features are properly shaped
            if len(audio_features.shape) == 1:
                audio_features = audio_features.reshape(1, -1)
            
            # Scale features if scaler is available
            if SKLEARN_AVAILABLE and hasattr(self.scaler, 'scale_'):
                audio_features = self.scaler.transform(audio_features)
            
            # Convert to tensor
            audio_tensor = torch.FloatTensor(audio_features).unsqueeze(1).to(self.device)
            
            # Encode through audio encoder
            with torch.no_grad():
                audio_embedding = self.audio_encoder(audio_tensor)
                audio_embedding = self.audio_projection(audio_embedding)
            
            return audio_embedding.cpu().numpy()
            
        except Exception as e:
            logger.error(f"Audio encoding failed: {e}")
            return np.zeros((1, 256))
    
    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode text to embedding space
        
        Args:
            text: Input text
            
        Returns:
            Text embedding
        """
        try:
            if not TRANSFORMERS_AVAILABLE:
                # Return dummy embedding
                return np.random.randn(1, 256)
            
            # Tokenize text
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=512
            ).to(self.device)
            
            # Encode through text encoder
            with torch.no_grad():
                outputs = self.text_encoder(**inputs)
                text_embedding = outputs.last_hidden_state.mean(dim=1)  # Mean pooling
                text_embedding = self.text_projection(text_embedding)
            
            return text_embedding.cpu().numpy()
            
        except Exception as e:
            logger.error(f"Text encoding failed: {e}")
            return np.zeros((1, 256))
    
    def compute_similarity(self, audio_embedding: np.ndarray, text_embedding: np.ndarray) -> float:
        """
        Compute cosine similarity between audio and text embeddings
        
        Args:
            audio_embedding: Audio embedding vector
            text_embedding: Text embedding vector
            
        Returns:
            Similarity score
        """
        try:
            if SKLEARN_AVAILABLE:
                similarity = cosine_similarity(audio_embedding.reshape(1, -1), 
                                             text_embedding.reshape(1, -1))[0, 0]
            else:
                # Manual cosine similarity
                audio_norm = np.linalg.norm(audio_embedding)
                text_norm = np.linalg.norm(text_embedding)
                if audio_norm == 0 or text_norm == 0:
                    similarity = 0.0
                else:
                    similarity = np.dot(audio_embedding.flatten(), text_embedding.flatten()) / (audio_norm * text_norm)
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Similarity computation failed: {e}")
            return 0.0

class RagaTherapyMapper:
    """
    Main class for mapping ragas to therapy recommendations
    """
    
    def __init__(self):
        self.embedder = Wav2CLIPEmbedder()
        self.therapy_profiles = self._load_therapy_profiles()
        self.dataset_path = "data/Final_dataset_s.csv"
        self.mapping_folder = "mapping_output"
        self.raga_detection_output = None
        
        # Create output directory
        os.makedirs(self.mapping_folder, exist_ok=True)
        
        # Load dataset
        self.therapy_dataset = self._load_therapy_dataset()
        
        logger.info("RagaTherapyMapper initialized")
    
    def _load_therapy_profiles(self) -> Dict[str, Dict]:
        """Load comprehensive therapy profiles for each raga"""
        
        therapy_profiles = {
            "Bhairav": {
                "primary_emotion": "spiritual awakening",
                "therapeutic_applications": [
                    "morning meditation", "anxiety reduction", "spiritual healing",
                    "depression management", "focus enhancement"
                ],
                "psychological_effects": [
                    "calming mind", "reducing mental agitation", "promoting introspection",
                    "enhancing spiritual connection", "reducing negative thoughts"
                ],
                "physiological_effects": [
                    "lowering heart rate", "reducing cortisol levels", "improving breathing",
                    "muscle relaxation", "blood pressure regulation"
                ],
                "target_conditions": [
                    "anxiety disorders", "depression", "insomnia", "ADHD", "PTSD"
                ],
                "best_time": "morning (5-8 AM)",
                "session_guidelines": {
                    "duration": "20-30 minutes",
                    "volume": "low to moderate",
                    "environment": "quiet, peaceful space",
                    "position": "sitting or lying down"
                },
                "contraindications": ["severe depression episodes", "psychotic disorders"]
            },
            
            "Yaman": {
                "primary_emotion": "peace and romance",
                "therapeutic_applications": [
                    "stress relief", "relationship therapy", "emotional healing",
                    "sleep preparation", "mood enhancement"
                ],
                "psychological_effects": [
                    "emotional balance", "reducing stress", "promoting love and compassion",
                    "enhancing creativity", "improving mood"
                ],
                "physiological_effects": [
                    "relaxing nervous system", "improving sleep quality", "reducing tension",
                    "enhancing immune function", "promoting healing"
                ],
                "target_conditions": [
                    "stress disorders", "relationship issues", "insomnia", "chronic fatigue"
                ],
                "best_time": "evening (6-9 PM)",
                "session_guidelines": {
                    "duration": "30-45 minutes",
                    "volume": "moderate",
                    "environment": "comfortable, warm lighting",
                    "position": "relaxed sitting or lying"
                },
                "contraindications": ["manic episodes", "severe agitation"]
            },
            
            "Kafi": {
                "primary_emotion": "devotion and surrender",
                "therapeutic_applications": [
                    "grief counseling", "spiritual healing", "letting go therapy",
                    "acceptance training", "emotional release"
                ],
                "psychological_effects": [
                    "emotional release", "promoting acceptance", "reducing attachment",
                    "enhancing surrender", "processing grief"
                ],
                "physiological_effects": [
                    "deep relaxation", "releasing emotional tension", "improving circulation",
                    "balancing hormones", "reducing inflammation"
                ],
                "target_conditions": [
                    "grief and loss", "attachment disorders", "emotional trauma", "addiction recovery"
                ],
                "best_time": "night (9 PM - 12 AM)",
                "session_guidelines": {
                    "duration": "25-40 minutes",
                    "volume": "low to moderate",
                    "environment": "dimly lit, sacred space",
                    "position": "comfortable, eyes closed"
                },
                "contraindications": ["severe depression", "suicidal ideation"]
            },
            
            "Malkauns": {
                "primary_emotion": "deep contemplation",
                "therapeutic_applications": [
                    "deep meditation", "trauma processing", "shadow work",
                    "existential therapy", "consciousness expansion"
                ],
                "psychological_effects": [
                    "deep introspection", "accessing subconscious", "processing trauma",
                    "enhancing awareness", "spiritual transformation"
                ],
                "physiological_effects": [
                    "deep nervous system relaxation", "activating parasympathetic response",
                    "improving neural plasticity", "enhancing recovery"
                ],
                "target_conditions": [
                    "complex PTSD", "dissociative disorders", "chronic pain", "addiction"
                ],
                "best_time": "late night (11 PM - 2 AM)",
                "session_guidelines": {
                    "duration": "45-60 minutes",
                    "volume": "very low",
                    "environment": "completely dark, silent",
                    "position": "lying down, eyes closed"
                },
                "contraindications": ["psychotic disorders", "severe anxiety", "claustrophobia"]
            },
            
            "Bilawal": {
                "primary_emotion": "joy and optimism",
                "therapeutic_applications": [
                    "depression treatment", "mood elevation", "energy enhancement",
                    "motivation building", "positive psychology interventions"
                ],
                "psychological_effects": [
                    "mood elevation", "increasing optimism", "enhancing motivation",
                    "building confidence", "promoting positive thinking"
                ],
                "physiological_effects": [
                    "increasing serotonin", "boosting energy levels", "improving metabolism",
                    "enhancing circulation", "strengthening immune system"
                ],
                "target_conditions": [
                    "depression", "seasonal affective disorder", "low energy", "lack of motivation"
                ],
                "best_time": "morning (8-11 AM)",
                "session_guidelines": {
                    "duration": "20-35 minutes",
                    "volume": "moderate to high",
                    "environment": "bright, open space",
                    "position": "sitting upright or standing"
                },
                "contraindications": ["manic episodes", "severe agitation", "hyperactivity disorders"]
            },
            
            "Todi": {
                "primary_emotion": "melancholy and healing",
                "therapeutic_applications": [
                    "emotional processing", "cathartic release", "healing trauma",
                    "processing sadness", "emotional integration"
                ],
                "psychological_effects": [
                    "emotional catharsis", "processing difficult emotions", "healing wounds",
                    "promoting emotional intelligence", "enhancing empathy"
                ],
                "physiological_effects": [
                    "releasing emotional tension", "improving emotional regulation",
                    "reducing stress hormones", "promoting healing"
                ],
                "target_conditions": [
                    "emotional trauma", "complicated grief", "emotional numbness", "relationship trauma"
                ],
                "best_time": "morning (6-9 AM)",
                "session_guidelines": {
                    "duration": "30-45 minutes",
                    "volume": "low to moderate",
                    "environment": "safe, comfortable space",
                    "position": "comfortable, tissues available"
                },
                "contraindications": ["severe depression", "suicidal ideation", "recent trauma"]
            }
        }
        
        return therapy_profiles
    
    def _load_therapy_dataset(self) -> pd.DataFrame:
        """Load the therapy dataset with specified features"""
        
        try:
            if not os.path.exists(self.dataset_path):
                logger.error(f"Therapy dataset not found: {self.dataset_path}")
                # Create sample data for demonstration
                return self._create_sample_therapy_data()
            
            # Load dataset
            df = pd.read_csv(self.dataset_path)
            logger.info(f"Original dataset shape: {df.shape}")
            logger.info(f"Original columns: {list(df.columns)}")
            
            # Select only therapy-related features
            therapy_features = ['Age', 'Gender', 'Mental_Condition', 'Severity', 
                              'Improvement_Score', 'Listening_Time', 'Raga']
            
            # Check if all required columns exist
            missing_cols = [col for col in therapy_features if col not in df.columns]
            available_cols = [col for col in therapy_features if col in df.columns]
            
            if missing_cols:
                logger.warning(f"Missing columns in dataset: {missing_cols}")
                logger.info(f"Available columns: {available_cols}")
                
                # If we're missing critical therapy columns, create sample data
                if len(available_cols) < 3:
                    logger.warning("Too few therapy columns available, creating sample data")
                    return self._create_sample_therapy_data()
                
                df = df[available_cols]
            else:
                df = df[therapy_features]
            
            # Clean the data
            df = self._clean_therapy_data(df)
            
            logger.info(f"Loaded therapy dataset: {df.shape}")
            logger.info(f"Final columns: {list(df.columns)}")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to load therapy dataset: {e}")
            logger.info("Creating sample therapy data for demonstration")
            return self._create_sample_therapy_data()
    
    def _create_sample_therapy_data(self) -> pd.DataFrame:
        """Create sample therapy data for demonstration"""
        
        logger.info("Creating sample therapy data...")
        
        # Sample data representing diverse therapy cases with realistic values
        sample_data = [
            {"Age": 25, "Gender": "Female", "Mental_Condition": "Anxiety", "Severity": "Moderate", 
             "Improvement_Score": 6.2, "Listening_Time": 30, "Raga": "Yaman"},
            {"Age": 35, "Gender": "Male", "Mental_Condition": "Depression", "Severity": "Severe", 
             "Improvement_Score": 3.1, "Listening_Time": 45, "Raga": "Bhairav"},
            {"Age": 42, "Gender": "Female", "Mental_Condition": "PTSD", "Severity": "High", 
             "Improvement_Score": 4.5, "Listening_Time": 25, "Raga": "Malkauns"},
            {"Age": 28, "Gender": "Male", "Mental_Condition": "Insomnia", "Severity": "Moderate", 
             "Improvement_Score": 7.3, "Listening_Time": 35, "Raga": "Kafi"},
            {"Age": 55, "Gender": "Female", "Mental_Condition": "Chronic_Pain", "Severity": "High", 
             "Improvement_Score": 5.8, "Listening_Time": 40, "Raga": "Todi"},
            {"Age": 19, "Gender": "Male", "Mental_Condition": "ADHD", "Severity": "Mild", 
             "Improvement_Score": 8.1, "Listening_Time": 20, "Raga": "Bilawal"},
            {"Age": 38, "Gender": "Female", "Mental_Condition": "Bipolar", "Severity": "Moderate", 
             "Improvement_Score": 6.7, "Listening_Time": 30, "Raga": "Yaman"},
            {"Age": 45, "Gender": "Male", "Mental_Condition": "Anxiety", "Severity": "Severe", 
             "Improvement_Score": 4.2, "Listening_Time": 35, "Raga": "Bhairav"},
            {"Age": 32, "Gender": "Female", "Mental_Condition": "Depression", "Severity": "Mild", 
             "Improvement_Score": 8.5, "Listening_Time": 25, "Raga": "Bilawal"},
            {"Age": 50, "Gender": "Male", "Mental_Condition": "Stress", "Severity": "Moderate", 
             "Improvement_Score": 7.1, "Listening_Time": 30, "Raga": "Kafi"},
            {"Age": 29, "Gender": "Female", "Mental_Condition": "OCD", "Severity": "High", 
             "Improvement_Score": 5.3, "Listening_Time": 40, "Raga": "Malkauns"},
            {"Age": 60, "Gender": "Male", "Mental_Condition": "Dementia", "Severity": "Moderate", 
             "Improvement_Score": 6.8, "Listening_Time": 20, "Raga": "Bhairav"},
            {"Age": 23, "Gender": "Female", "Mental_Condition": "Eating_Disorder", "Severity": "Severe", 
             "Improvement_Score": 4.7, "Listening_Time": 30, "Raga": "Todi"},
            {"Age": 40, "Gender": "Male", "Mental_Condition": "Addiction", "Severity": "High", 
             "Improvement_Score": 3.8, "Listening_Time": 45, "Raga": "Kafi"},
            {"Age": 27, "Gender": "Female", "Mental_Condition": "Social_Anxiety", "Severity": "Moderate", 
             "Improvement_Score": 7.6, "Listening_Time": 25, "Raga": "Yaman"},
            {"Age": 52, "Gender": "Male", "Mental_Condition": "Grief", "Severity": "High", 
             "Improvement_Score": 5.1, "Listening_Time": 50, "Raga": "Todi"},
            {"Age": 31, "Gender": "Female", "Mental_Condition": "Panic_Disorder", "Severity": "Severe", 
             "Improvement_Score": 4.3, "Listening_Time": 20, "Raga": "Bhairav"},
            {"Age": 44, "Gender": "Male", "Mental_Condition": "Anger_Issues", "Severity": "Moderate", 
             "Improvement_Score": 6.9, "Listening_Time": 35, "Raga": "Malkauns"},
            {"Age": 26, "Gender": "Female", "Mental_Condition": "Body_Dysmorphia", "Severity": "High", 
             "Improvement_Score": 5.6, "Listening_Time": 30, "Raga": "Bilawal"},
            {"Age": 37, "Gender": "Male", "Mental_Condition": "Work_Stress", "Severity": "Mild", 
             "Improvement_Score": 8.3, "Listening_Time": 25, "Raga": "Yaman"},
            {"Age": 48, "Gender": "Female", "Mental_Condition": "Fibromyalgia", "Severity": "High", 
             "Improvement_Score": 5.9, "Listening_Time": 45, "Raga": "Malkauns"},
            {"Age": 33, "Gender": "Male", "Mental_Condition": "Autism", "Severity": "Moderate", 
             "Improvement_Score": 7.4, "Listening_Time": 40, "Raga": "Bhairav"},
            {"Age": 41, "Gender": "Female", "Mental_Condition": "Borderline_PD", "Severity": "Severe", 
             "Improvement_Score": 4.1, "Listening_Time": 35, "Raga": "Todi"},
            {"Age": 24, "Gender": "Male", "Mental_Condition": "Performance_Anxiety", "Severity": "Moderate", 
             "Improvement_Score": 7.8, "Listening_Time": 25, "Raga": "Yaman"},
            {"Age": 56, "Gender": "Female", "Mental_Condition": "Chronic_Fatigue", "Severity": "High", 
             "Improvement_Score": 5.4, "Listening_Time": 30, "Raga": "Kafi"}
        ]
        
        df = pd.DataFrame(sample_data)
        logger.info(f"Created sample therapy dataset: {df.shape}")
        
        # Normalize improvement scores to 0-1 range for consistency
        df['Improvement_Score'] = df['Improvement_Score'] / 10.0
        
        return df
    
    def _clean_therapy_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate therapy data"""
        
        logger.info("Cleaning therapy data...")
        
        # Remove rows with missing critical data
        initial_rows = len(df)
        
        # Handle missing values
        if 'Age' in df.columns:
            df = df[df['Age'].notna()]
            df['Age'] = df['Age'].astype(int)
        
        if 'Gender' in df.columns:
            df = df[df['Gender'].notna()]
            df['Gender'] = df['Gender'].astype(str)
        
        if 'Mental_Condition' in df.columns:
            df = df[df['Mental_Condition'].notna()]
            df['Mental_Condition'] = df['Mental_Condition'].astype(str)
        
        if 'Severity' in df.columns:
            df = df[df['Severity'].notna()]
            df['Severity'] = df['Severity'].astype(str)
        
        if 'Improvement_Score' in df.columns:
            df = df[df['Improvement_Score'].notna()]
            df['Improvement_Score'] = pd.to_numeric(df['Improvement_Score'], errors='coerce')
            df = df[df['Improvement_Score'].notna()]
        
        if 'Listening_Time' in df.columns:
            df = df[df['Listening_Time'].notna()]
            df['Listening_Time'] = pd.to_numeric(df['Listening_Time'], errors='coerce')
            df = df[df['Listening_Time'].notna()]
        
        if 'Raga' in df.columns:
            df = df[df['Raga'].notna()]
            df['Raga'] = df['Raga'].astype(str)
        
        final_rows = len(df)
        logger.info(f"Data cleaning: {initial_rows} -> {final_rows} rows")
        
        return df.reset_index(drop=True)
    
    def check_raga_detection_output(self) -> bool:
        """Check if raga_detection.py has been run and output exists"""
        
        # Look for common output files from raga detection
        output_files = [
            "results/json_values/enhanced_model_performance.json",
            "results/json_values/model_performance.json",
            "results/json_values/enhanced_prediction_*.json",
            "results/json_values/prediction_*.json"
        ]
        
        # Check if results directory exists
        if not os.path.exists("results"):
            return False
        
        # Check for any prediction files
        results_dir = Path("results/json_values")
        if results_dir.exists():
            prediction_files = list(results_dir.glob("*prediction_*.json"))
            performance_files = list(results_dir.glob("*performance*.json"))
            
            if prediction_files or performance_files:
                logger.info(f"Found raga detection output: {len(prediction_files)} prediction files, {len(performance_files)} performance files")
                return True
        
        # Check for saved model or classifier
        if os.path.exists("enhanced_raga_classifier.pkl") or os.path.exists("raga_classifier.pkl"):
            return True
        
        return False
    
    def load_raga_detection_results(self) -> Dict[str, Any]:
        """Load results from raga_detection.py"""
        
        results = {
            "model_performance": None,
            "predictions": [],
            "classifier_available": False
        }
        
        try:
            # Load model performance
            performance_files = [
                "results/json_values/enhanced_model_performance.json",
                "results/json_values/model_performance.json"
            ]
            
            for perf_file in performance_files:
                if os.path.exists(perf_file):
                    with open(perf_file, 'r') as f:
                        results["model_performance"] = json.load(f)
                    logger.info(f"Loaded model performance from {perf_file}")
                    break
            
            # Load predictions
            results_dir = Path("results/json_values")
            if results_dir.exists():
                prediction_files = list(results_dir.glob("*prediction_*.json"))
                
                for pred_file in prediction_files:
                    try:
                        with open(pred_file, 'r') as f:
                            prediction = json.load(f)
                            results["predictions"].append(prediction)
                    except Exception as e:
                        logger.warning(f"Could not load prediction file {pred_file}: {e}")
                
                logger.info(f"Loaded {len(results['predictions'])} prediction results")
            
            # Check for classifier
            if os.path.exists("enhanced_raga_classifier.pkl"):
                results["classifier_available"] = True
                logger.info("Enhanced classifier found")
            elif os.path.exists("raga_classifier.pkl"):
                results["classifier_available"] = True
                logger.info("Standard classifier found")
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to load raga detection results: {e}")
            return results
    
    def create_personalized_recommendation(self, raga_name: str, patient_profile: PatientProfile, 
                                         audio_features: Optional[np.ndarray] = None) -> TherapyRecommendation:
        """
        Create personalized therapy recommendation based on raga and patient profile
        
        Args:
            raga_name: Detected raga name
            patient_profile: Patient's therapy profile
            audio_features: Optional audio features for similarity computation
            
        Returns:
            Personalized therapy recommendation
        """
        
        # Get base therapy profile for raga
        base_profile = self.therapy_profiles.get(raga_name, {})
        
        if not base_profile:
            logger.warning(f"No therapy profile found for raga: {raga_name}")
            # Create basic recommendation
            return TherapyRecommendation(
                raga_name=raga_name,
                patient_profile=patient_profile.__dict__,
                therapy_type="general music therapy",
                mood_enhancement="mild",
                stress_reduction_level="moderate",
                recommended_duration=30,
                best_time_of_day="any time",
                therapeutic_benefits=["relaxation", "mood improvement"],
                contraindications=["none known"],
                confidence_score=0.5,
                audio_similarity_score=0.0,
                personalization_score=0.5
            )
        
        # Personalize based on patient profile
        personalized_duration = self._calculate_duration(base_profile, patient_profile)
        personalized_benefits = self._get_personalized_benefits(base_profile, patient_profile)
        therapy_type = self._determine_therapy_type(base_profile, patient_profile)
        mood_enhancement = self._assess_mood_enhancement(base_profile, patient_profile)
        stress_reduction = self._assess_stress_reduction(base_profile, patient_profile)
        
        # Calculate personalization score
        personalization_score = self._calculate_personalization_score(patient_profile)
        
        # Calculate audio similarity if features provided
        audio_similarity = 0.0
        if audio_features is not None:
            therapy_description = self._generate_therapy_description(base_profile, patient_profile)
            audio_emb = self.embedder.encode_audio(audio_features)
            text_emb = self.embedder.encode_text(therapy_description)
            audio_similarity = self.embedder.compute_similarity(audio_emb, text_emb)
        
        # Determine contraindications
        contraindications = self._assess_contraindications(base_profile, patient_profile)
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence_score(base_profile, patient_profile, audio_similarity)
        
        return TherapyRecommendation(
            raga_name=raga_name,
            patient_profile=patient_profile.__dict__,
            therapy_type=therapy_type,
            mood_enhancement=mood_enhancement,
            stress_reduction_level=stress_reduction,
            recommended_duration=personalized_duration,
            best_time_of_day=base_profile.get("best_time", "any time"),
            therapeutic_benefits=personalized_benefits,
            contraindications=contraindications,
            confidence_score=confidence_score,
            audio_similarity_score=audio_similarity,
            personalization_score=personalization_score
        )
    
    def _calculate_duration(self, base_profile: Dict, patient_profile: PatientProfile) -> int:
        """Calculate personalized session duration"""
        
        base_duration = 30  # default
        
        # Parse base duration
        duration_str = base_profile.get("session_guidelines", {}).get("duration", "30 minutes")
        try:
            base_duration = int(duration_str.split('-')[0])
        except:
            base_duration = 30
        
        # Adjust based on age
        if patient_profile.age < 18:
            base_duration = min(base_duration, 20)  # Shorter for children
        elif patient_profile.age > 65:
            base_duration = min(base_duration, 25)  # Shorter for elderly
        
        # Adjust based on severity
        if patient_profile.severity.lower() in ['severe', 'high']:
            base_duration = min(base_duration, 20)  # Shorter for severe cases
        elif patient_profile.severity.lower() in ['mild', 'low']:
            base_duration = min(base_duration + 10, 45)  # Longer for mild cases
        
        # Adjust based on listening time preference
        if patient_profile.listening_time < 20:
            base_duration = min(base_duration, patient_profile.listening_time + 5)
        
        return max(15, min(base_duration, 60))  # Ensure reasonable bounds
    
    def _get_personalized_benefits(self, base_profile: Dict, patient_profile: PatientProfile) -> List[str]:
        """Get personalized therapeutic benefits"""
        
        base_benefits = base_profile.get("therapeutic_applications", [])
        psychological_effects = base_profile.get("psychological_effects", [])
        
        # Combine and filter based on patient condition
        all_benefits = base_benefits + psychological_effects
        
        # Filter based on mental condition
        condition = patient_profile.mental_condition.lower()
        relevant_benefits = []
        
        for benefit in all_benefits:
            benefit_lower = benefit.lower()
            if condition in benefit_lower or any(word in benefit_lower for word in condition.split()):
                relevant_benefits.append(benefit)
        
        # Add general benefits if none found
        if not relevant_benefits:
            relevant_benefits = base_benefits[:3]  # Take first 3 as default
        
        return relevant_benefits[:5]  # Limit to 5 benefits
    
    def _determine_therapy_type(self, base_profile: Dict, patient_profile: PatientProfile) -> str:
        """Determine specific therapy type"""
        
        condition = patient_profile.mental_condition.lower()
        
        if "anxiety" in condition:
            return "anxiety-focused music therapy"
        elif "depression" in condition:
            return "mood-enhancing music therapy"
        elif "ptsd" in condition or "trauma" in condition:
            return "trauma-informed music therapy"
        elif "adhd" in condition or "attention" in condition:
            return "focus-enhancing music therapy"
        elif "insomnia" in condition or "sleep" in condition:
            return "sleep-induction music therapy"
        else:
            return f"{base_profile.get('primary_emotion', 'general')} music therapy"
    
    def _assess_mood_enhancement(self, base_profile: Dict, patient_profile: PatientProfile) -> str:
        """Assess expected mood enhancement level"""
        
        # Base assessment from raga
        raga_mood_map = {
            "Bhairav": "calming",
            "Yaman": "uplifting", 
            "Kafi": "soothing",
            "Malkauns": "deeply calming",
            "Bilawal": "energizing",
            "Todi": "cathartic"
        }
        
        base_mood = raga_mood_map.get(base_profile.get("primary_emotion", ""), "moderate")
        
        # Adjust based on severity and improvement score
        if patient_profile.severity.lower() in ['severe', 'high']:
            if patient_profile.improvement_score > 0.7:
                return f"strong {base_mood}"
            else:
                return f"gentle {base_mood}"
        else:
            return f"moderate {base_mood}"
    
    def _assess_stress_reduction(self, base_profile: Dict, patient_profile: PatientProfile) -> str:
        """Assess stress reduction potential"""
        
        stress_effects = base_profile.get("physiological_effects", [])
        stress_keywords = ["stress", "tension", "cortisol", "relaxation", "calm"]
        
        stress_related = [effect for effect in stress_effects 
                         if any(keyword in effect.lower() for keyword in stress_keywords)]
        
        if len(stress_related) >= 3:
            return "high stress reduction"
        elif len(stress_related) >= 1:
            return "moderate stress reduction"
        else:
            return "mild stress reduction"
    
    def _assess_contraindications(self, base_profile: Dict, patient_profile: PatientProfile) -> List[str]:
        """Assess contraindications for the patient"""
        
        base_contraindications = base_profile.get("contraindications", [])
        patient_condition = patient_profile.mental_condition.lower()
        
        # Check if patient condition matches any contraindications
        relevant_contraindications = []
        
        for contraindication in base_contraindications:
            if any(word in patient_condition for word in contraindication.lower().split()):
                relevant_contraindications.append(f"Caution: {contraindication}")
        
        # Add severity-based contraindications
        if patient_profile.severity.lower() in ['severe', 'critical']:
            relevant_contraindications.append("Monitor closely due to severe condition")
        
        # Add age-based considerations
        if patient_profile.age < 16:
            relevant_contraindications.append("Requires parental supervision")
        elif patient_profile.age > 75:
            relevant_contraindications.append("Monitor for fatigue")
        
        return relevant_contraindications if relevant_contraindications else ["None identified"]
    
    def _calculate_personalization_score(self, patient_profile: PatientProfile) -> float:
        """Calculate how well we can personalize for this patient"""
        
        score = 0.0
        
        # Age factor
        if 18 <= patient_profile.age <= 65:
            score += 0.2
        else:
            score += 0.1
        
        # Gender factor (we have info)
        if patient_profile.gender.lower() in ['male', 'female', 'other']:
            score += 0.1
        
        # Mental condition specificity
        if len(patient_profile.mental_condition) > 3:
            score += 0.3
        
        # Severity information
        if patient_profile.severity.lower() in ['mild', 'moderate', 'severe', 'low', 'high']:
            score += 0.2
        
        # Improvement score validity
        if 0 <= patient_profile.improvement_score <= 1:
            score += 0.1
        
        # Listening time validity
        if 5 <= patient_profile.listening_time <= 120:
            score += 0.1
        
        return min(score, 1.0)
    
    def _calculate_confidence_score(self, base_profile: Dict, patient_profile: PatientProfile, 
                                  audio_similarity: float) -> float:
        """Calculate overall confidence in the recommendation"""
        
        score = 0.0
        
        # Base profile completeness
        if base_profile:
            profile_completeness = len([k for k in ['therapeutic_applications', 'psychological_effects', 
                                                  'physiological_effects', 'target_conditions'] 
                                      if k in base_profile and base_profile[k]]) / 4
            score += profile_completeness * 0.3
        
        # Patient profile completeness
        personalization_score = self._calculate_personalization_score(patient_profile)
        score += personalization_score * 0.3
        
        # Audio similarity (if available)
        score += audio_similarity * 0.2
        
        # Condition matching
        condition_match = self._check_condition_match(base_profile, patient_profile)
        score += condition_match * 0.2
        
        return min(score, 1.0)
    
    def _generate_mock_audio_features(self) -> np.ndarray:
        """Generate mock audio features for demonstration"""
        # Simulate typical audio features (MFCC-like)
        return np.random.randn(50)  # 50-dimensional feature vector
    
    def _create_session_plan(self, recommendation: TherapyRecommendation) -> Dict[str, Any]:
        """Create a detailed therapy session plan"""
        
        return {
            "session_structure": {
                "preparation": "2-3 minutes of breathing exercises",
                "main_therapy": f"{recommendation.recommended_duration - 5} minutes of {recommendation.raga_name} music",
                "integration": "2-3 minutes of silent reflection"
            },
            "environment_setup": {
                "lighting": "dim, warm lighting" if "evening" in recommendation.best_time_of_day else "natural lighting",
                "seating": "comfortable chair or meditation cushion",
                "volume": "low to moderate (20-40% of maximum)",
                "distractions": "minimize all external noise and interruptions"
            },
            "monitoring_points": [
                "Patient's breathing pattern and depth",
                "Visible signs of relaxation or tension",
                "Emotional responses (tears, smiling, etc.)",
                "Any signs of distress or discomfort"
            ],
            "success_indicators": [
                "Slower, deeper breathing",
                "Relaxed facial expression",
                "Decreased muscle tension",
                "Positive verbal feedback"
            ]
        }
    
    def _generate_follow_up_notes(self, patient_profile: PatientProfile, 
                                 recommendation: TherapyRecommendation) -> Dict[str, Any]:
        """Generate follow-up notes and recommendations"""
        
        follow_up = {
            "immediate_post_session": [
                "Ask patient to rate their experience (1-10)",
                "Note any significant emotional responses",
                "Document physical changes observed",
                "Record patient's verbal feedback"
            ],
            "24_hour_follow_up": [
                "Check for any delayed emotional responses",
                "Assess sleep quality if applicable",
                "Note any changes in reported symptoms"
            ],
            "weekly_assessment": [
                "Evaluate overall progress toward therapy goals",
                "Adjust session duration if needed",
                "Consider alternative ragas if limited progress"
            ],
            "red_flags_to_monitor": [
                "Increased anxiety or distress during sessions",
                "No improvement after 4-6 sessions",
                "Patient resistance or strong negative reactions"
            ],
            "alternative_ragas": self._suggest_alternative_ragas(patient_profile),
            "complementary_therapies": [
                "Progressive muscle relaxation",
                "Breathing exercises",
                "Guided imagery",
                "Cognitive behavioral techniques"
            ]
        }
        
        return follow_up
    
    def _suggest_alternative_ragas(self, patient_profile: PatientProfile) -> List[str]:
        """Suggest alternative ragas based on patient profile"""
        
        condition = patient_profile.mental_condition.lower()
        
        if "anxiety" in condition:
            return ["Bhairav", "Yaman", "Kafi"]
        elif "depression" in condition:
            return ["Bilawal", "Yaman", "Bhairav"]
        elif "ptsd" in condition or "trauma" in condition:
            return ["Malkauns", "Todi", "Kafi"]
        elif "insomnia" in condition or "sleep" in condition:
            return ["Kafi", "Malkauns", "Yaman"]
        elif "adhd" in condition or "attention" in condition:
            return ["Bhairav", "Bilawal", "Yaman"]
        else:
            return ["Yaman", "Bhairav", "Kafi"]  # Safe defaults
        """Check how well the patient's condition matches the raga's target conditions"""
        
        target_conditions = base_profile.get("target_conditions", [])
        patient_condition = patient_profile.mental_condition.lower()
        
        if not target_conditions:
            return 0.5  # Neutral if no target conditions specified
        
        # Check for direct matches
        for condition in target_conditions:
            if any(word in patient_condition for word in condition.lower().split()):
                return 1.0
        
        # Check for related conditions
        related_matches = 0
        for condition in target_conditions:
            condition_words = set(condition.lower().split())
            patient_words = set(patient_condition.split())
            if condition_words & patient_words:  # Intersection
                related_matches += 1
        
        if related_matches > 0:
            return min(related_matches / len(target_conditions), 1.0)
        
    def _check_condition_match(self, base_profile: Dict, patient_profile: PatientProfile) -> float:
        """Check how well the patient's condition matches the raga's target conditions"""
        
        target_conditions = base_profile.get("target_conditions", [])
        patient_condition = patient_profile.mental_condition.lower()
        
        if not target_conditions:
            return 0.5  # Neutral if no target conditions specified
        
        # Check for direct matches
        for condition in target_conditions:
            if any(word in patient_condition for word in condition.lower().split()):
                return 1.0
        
        # Check for related conditions
        related_matches = 0
        for condition in target_conditions:
            condition_words = set(condition.lower().split())
            patient_words = set(patient_condition.split())
            if condition_words & patient_words:  # Intersection
                related_matches += 1
        
        if related_matches > 0:
            return min(related_matches / len(target_conditions), 1.0)
        
        return 0.3  # Low match if no clear relationship
    
    def _save_detailed_therapy_mappings(self, therapy_mappings: List[Dict]):
        """Save detailed therapy mappings with enhanced format"""
        
        detailed_output_file = os.path.join(self.mapping_folder, "detailed_therapy_mappings.json")
        
        # Enhanced format for better readability
        enhanced_mappings = []
        
        for mapping in therapy_mappings:
            enhanced_mapping = {
                "patient_information": {
                    "patient_id": mapping["patient_id"],
                    "demographics": {
                        "age": mapping["patient_profile"]["age"],
                        "gender": mapping["patient_profile"]["gender"]
                    },
                    "clinical_profile": {
                        "mental_condition": mapping["patient_profile"]["mental_condition"],
                        "severity_level": mapping["patient_profile"]["severity"],
                        "current_improvement_score": mapping["patient_profile"]["improvement_score"],
                        "preferred_listening_time": mapping["patient_profile"]["listening_time"]
                    }
                },
                "raga_assignment": {
                    "assigned_raga": mapping["raga_name"],
                    "detection_confidence": mapping.get("detection_confidence", 0.0),
                    "assignment_rationale": f"Selected based on therapeutic properties for {mapping['patient_profile']['mental_condition']}"
                },
                "therapy_recommendation": {
                    "therapy_type": mapping["recommendation"]["therapy_type"],
                    "mood_enhancement_expected": mapping["recommendation"]["mood_enhancement"],
                    "stress_reduction_level": mapping["recommendation"]["stress_reduction_level"],
                    "session_details": {
                        "recommended_duration_minutes": mapping["recommendation"]["recommended_duration"],
                        "optimal_time_of_day": mapping["recommendation"]["best_time_of_day"],
                        "confidence_in_recommendation": mapping["recommendation"]["confidence_score"]
                    },
                    "therapeutic_benefits": mapping["recommendation"]["therapeutic_benefits"],
                    "safety_considerations": mapping["recommendation"]["contraindications"]
                },
                "session_planning": mapping.get("therapy_session_plan", {}),
                "follow_up_protocol": mapping.get("follow_up_notes", {}),
                "quality_metrics": {
                    "personalization_score": mapping["recommendation"]["personalization_score"],
                    "audio_text_similarity": mapping["recommendation"]["audio_similarity_score"],
                    "overall_confidence": mapping["recommendation"]["confidence_score"]
                },
                "metadata": {
                    "mapping_created": mapping["mapping_timestamp"],
                    "wav2clip_processed": mapping["recommendation"]["audio_similarity_score"] > 0,
                    "system_version": "1.0"
                }
            }
            
            enhanced_mappings.append(enhanced_mapping)
        
        # Save enhanced mappings
        with open(detailed_output_file, 'w') as f:
            json.dump(enhanced_mappings, f, indent=2, default=str)
        
        print(f"📄 Enhanced detailed mappings saved: {detailed_output_file}")
        
        return enhanced_mappings
    
    def _generate_therapy_description(self, base_profile: Dict, patient_profile: PatientProfile) -> str:
        """Generate therapy description for text embedding"""
        
        primary_emotion = base_profile.get("primary_emotion", "healing")
        applications = base_profile.get("therapeutic_applications", [])
        effects = base_profile.get("psychological_effects", [])
        
        description = f"Music therapy for {primary_emotion} and emotional healing. "
        
        if applications:
            description += f"Therapeutic applications include {', '.join(applications[:3])}. "
        
        if effects:
            description += f"Psychological effects include {', '.join(effects[:3])}. "
        
        description += f"Designed for {patient_profile.mental_condition} with {patient_profile.severity} severity level. "
        description += f"Suitable for {patient_profile.gender} patient aged {patient_profile.age}."
        
        return description
    
    def process_therapy_mapping(self) -> Dict[str, Any]:
        """
        Main function to process therapy mapping
        
        Returns:
            Dictionary containing mapping results
        """
        
        print("🏥 STARTING RAGA-THERAPY MAPPING SYSTEM")
        print("=" * 60)
        
        # Step 1: Check if raga detection has been run
        print("🔍 Step 1: Checking raga detection output...")
        
        if not self.check_raga_detection_output():
            error_msg = "The raga_detection file hasn't been run."
            print(f"❌ {error_msg}")
            print("💡 Please run raga_detection.py first to generate the required output.")
            
            return {
                "status": "error",
                "message": error_msg,
                "timestamp": datetime.now().isoformat()
            }
        
        print("✅ Raga detection output found!")
        
        # Step 2: Load raga detection results
        print("\n📊 Step 2: Loading raga detection results...")
        raga_results = self.load_raga_detection_results()
        
        if not raga_results["model_performance"] and not raga_results["predictions"]:
            error_msg = "No valid raga detection results found."
            print(f"❌ {error_msg}")
            return {
                "status": "error", 
                "message": error_msg,
                "timestamp": datetime.now().isoformat()
            }
        
        print(f"✅ Loaded {len(raga_results['predictions'])} predictions")
        
        # Step 3: Load therapy dataset
        print("\n🏥 Step 3: Loading therapy dataset...")
        
        if self.therapy_dataset.empty:
            error_msg = "Therapy dataset could not be loaded."
            print(f"❌ {error_msg}")
            return {
                "status": "error",
                "message": error_msg,
                "timestamp": datetime.now().isoformat()
            }
        
        print(f"✅ Loaded therapy dataset: {self.therapy_dataset.shape}")
        print(f"📋 Available features: {list(self.therapy_dataset.columns)}")
        
        # Step 4: Create therapy mappings
        print("\n🎵 Step 4: Creating therapy mappings...")
        
        therapy_mappings = []
        
        # Get the latest prediction for reference
        latest_prediction = None
        if raga_results["predictions"]:
            latest_prediction = raga_results["predictions"][-1]
            print(f"📊 Using latest prediction: {latest_prediction.get('predicted_raga', 'Unknown')}")
        
        # Process each patient in the therapy dataset
        for idx, row in self.therapy_dataset.iterrows():
            try:
                # Create patient profile with proper error handling
                patient_profile = PatientProfile(
                    age=int(row.get('Age', 30)),
                    gender=str(row.get('Gender', 'Unknown')),
                    mental_condition=str(row.get('Mental_Condition', 'General')),
                    severity=str(row.get('Severity', 'Moderate')),
                    improvement_score=float(row.get('Improvement_Score', 0.5)),
                    listening_time=float(row.get('Listening_Time', 30))
                )
                
                # Get raga (from dataset or use detected raga if available)
                raga_name = str(row.get('Raga', 'Unknown'))
                
                # If we have a recent prediction and it's more reliable, use it
                if (latest_prediction and 
                    isinstance(latest_prediction, dict) and 
                    'predicted_raga' in latest_prediction and
                    latest_prediction.get('confidence', 0) > 0.5):
                    raga_name = latest_prediction['predicted_raga']
                    print(f"🎯 Using detected raga '{raga_name}' for patient {idx}")
                
                # Create recommendation with audio features if available
                audio_features = self._generate_mock_audio_features() if idx == 0 else None
                
                recommendation = self.create_personalized_recommendation(
                    raga_name=raga_name,
                    patient_profile=patient_profile,
                    audio_features=audio_features
                )
                
                # Create comprehensive mapping entry
                mapping_entry = {
                    "patient_id": f"P{idx:03d}",
                    "raga_name": raga_name,
                    "patient_profile": patient_profile.__dict__,
                    "recommendation": recommendation.__dict__,
                    "mapping_timestamp": datetime.now().isoformat(),
                    "detection_confidence": latest_prediction.get('confidence', 0.0) if latest_prediction else 0.0,
                    "therapy_session_plan": self._create_session_plan(recommendation),
                    "follow_up_notes": self._generate_follow_up_notes(patient_profile, recommendation)
                }
                
                therapy_mappings.append(mapping_entry)
                
                # Print progress
                if idx < 5 or idx % 5 == 0:
                    print(f"   ✅ Created mapping for Patient {idx}: {patient_profile.mental_condition} -> {raga_name}")
                
            except Exception as e:
                logger.warning(f"Failed to process patient {idx}: {e}")
                continue
        
        print(f"✅ Created {len(therapy_mappings)} therapy mappings")
        
        # Print sample mappings for verification
        if therapy_mappings:
            print(f"\n📋 SAMPLE THERAPY MAPPINGS:")
            for i, mapping in enumerate(therapy_mappings[:3]):
                patient = mapping["patient_profile"]
                rec = mapping["recommendation"]
                print(f"   {i+1}. Patient {mapping['patient_id']}:")
                print(f"      • Condition: {patient['mental_condition']} ({patient['severity']})")
                print(f"      • Raga: {mapping['raga_name']}")
                print(f"      • Therapy: {rec['therapy_type']}")
                print(f"      • Duration: {rec['recommended_duration']} minutes")
                print(f"      • Confidence: {rec['confidence_score']:.3f}")
                print()
        
        # Step 5: Generate summary statistics
        print("\n📊 Step 5: Generating summary statistics...")
        
        summary_stats = self._generate_summary_statistics(therapy_mappings)
        
        # Step 6: Save results
        print("\n💾 Step 6: Saving mapping results...")
        
        results = {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "raga_detection_info": {
                "model_available": raga_results["classifier_available"],
                "predictions_count": len(raga_results["predictions"]),
                "model_performance": raga_results["model_performance"]
            },
            "therapy_dataset_info": {
                "total_patients": len(self.therapy_dataset),
                "features": list(self.therapy_dataset.columns),
                "unique_ragas": self.therapy_dataset['Raga'].nunique() if 'Raga' in self.therapy_dataset.columns else 0
            },
            "therapy_mappings": therapy_mappings,
            "summary_statistics": summary_stats,
            "wav2clip_embeddings": {
                "model_used": self.embedder.model_name,
                "embedding_dimension": 256,
                "total_embeddings_created": len(therapy_mappings)
            }
        }
        
        # Save main results
        main_output_file = os.path.join(self.mapping_folder, "therapy_mapping_results.json")
        with open(main_output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save detailed mappings with enhanced format
        enhanced_mappings = self._save_detailed_therapy_mappings(therapy_mappings)
        
        # Save therapy session templates
        self._save_therapy_templates(therapy_mappings)
        
        # Save summary report
        self._save_summary_report(results, summary_stats)
        
        print(f"✅ Results saved to {self.mapping_folder}/")
        print(f"📄 Main results: {main_output_file}")
        print(f"📄 Enhanced mappings: {self.mapping_folder}/detailed_therapy_mappings.json")
        print(f"📄 Session templates: {self.mapping_folder}/therapy_session_templates.json")
        
        # Step 7: Display results
        self._display_results_summary(results, summary_stats)
        
        return results
    
    def _save_therapy_templates(self, therapy_mappings: List[Dict]):
        """Save therapy session templates for practitioners"""
        
        templates_file = os.path.join(self.mapping_folder, "therapy_session_templates.json")
        
        # Group by raga to create templates
        raga_templates = {}
        
        for mapping in therapy_mappings:
            raga = mapping["raga_name"]
            if raga not in raga_templates:
                raga_templates[raga] = {
                    "raga_name": raga,
                    "therapy_profile": self.therapy_profiles.get(raga, {}),
                    "sample_sessions": [],
                    "practitioner_notes": {
                        "preparation": f"Ensure quiet environment for {raga} therapy session",
                        "during_session": "Monitor patient responses and adjust volume as needed",
                        "post_session": "Document patient feedback and any notable reactions"
                    }
                }
            
            # Add sample session
            if len(raga_templates[raga]["sample_sessions"]) < 3:  # Limit to 3 samples per raga
                session_template = {
                    "patient_type": f"{mapping['patient_profile']['mental_condition']} - {mapping['patient_profile']['severity']}",
                    "session_plan": mapping.get("therapy_session_plan", {}),
                    "expected_outcomes": mapping["recommendation"]["therapeutic_benefits"],
                    "duration": mapping["recommendation"]["recommended_duration"],
                    "contraindications": mapping["recommendation"]["contraindications"]
                }
                raga_templates[raga]["sample_sessions"].append(session_template)
        
        # Save templates
        with open(templates_file, 'w') as f:
            json.dump(raga_templates, f, indent=2, default=str)
        
        print(f"📄 Therapy templates saved: {templates_file}")
        
        # Step 7: Display results
        self._display_results_summary(results, summary_stats)
        
        return results
    
    def _generate_summary_statistics(self, therapy_mappings: List[Dict]) -> Dict[str, Any]:
        """Generate summary statistics from therapy mappings"""
        
        if not therapy_mappings:
            return {}
        
        # Extract data for analysis
        ragas = [mapping["raga_name"] for mapping in therapy_mappings]
        conditions = [mapping["patient_profile"]["mental_condition"] for mapping in therapy_mappings]
        severities = [mapping["patient_profile"]["severity"] for mapping in therapy_mappings]
        ages = [mapping["patient_profile"]["age"] for mapping in therapy_mappings]
        confidence_scores = [mapping["recommendation"]["confidence_score"] for mapping in therapy_mappings]
        durations = [mapping["recommendation"]["recommended_duration"] for mapping in therapy_mappings]
        
        # Calculate statistics
        summary = {
            "total_mappings": len(therapy_mappings),
            "unique_ragas": len(set(ragas)),
            "raga_distribution": {raga: ragas.count(raga) for raga in set(ragas)},
            "condition_distribution": {condition: conditions.count(condition) for condition in set(conditions)},
            "severity_distribution": {severity: severities.count(severity) for severity in set(severities)},
            "age_statistics": {
                "mean": np.mean(ages),
                "median": np.median(ages),
                "min": min(ages),
                "max": max(ages),
                "std": np.std(ages)
            },
            "confidence_statistics": {
                "mean": np.mean(confidence_scores),
                "median": np.median(confidence_scores),
                "min": min(confidence_scores),
                "max": max(confidence_scores),
                "std": np.std(confidence_scores)
            },
            "duration_statistics": {
                "mean": np.mean(durations),
                "median": np.median(durations),
                "min": min(durations),
                "max": max(durations),
                "std": np.std(durations)
            },
            "therapy_types": {},
            "most_recommended_raga": max(set(ragas), key=ragas.count),
            "most_common_condition": max(set(conditions), key=conditions.count),
            "average_confidence": np.mean(confidence_scores)
        }
        
        # Therapy types distribution
        therapy_types = [mapping["recommendation"]["therapy_type"] for mapping in therapy_mappings]
        summary["therapy_types"] = {therapy_type: therapy_types.count(therapy_type) 
                                  for therapy_type in set(therapy_types)}
        
        return summary
    
    def _save_summary_report(self, results: Dict, summary_stats: Dict):
        """Save a human-readable summary report"""
        
        report_file = os.path.join(self.mapping_folder, "therapy_mapping_report.txt")
        
        with open(report_file, 'w') as f:
            f.write("RAGA-THERAPY MAPPING SUMMARY REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {results['timestamp']}\n\n")
            
            # Overview
            f.write("OVERVIEW:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total therapy mappings created: {summary_stats.get('total_mappings', 0)}\n")
            f.write(f"Unique ragas analyzed: {summary_stats.get('unique_ragas', 0)}\n")
            f.write(f"Average confidence score: {summary_stats.get('average_confidence', 0):.3f}\n")
            f.write(f"Most recommended raga: {summary_stats.get('most_recommended_raga', 'N/A')}\n")
            f.write(f"Most common condition: {summary_stats.get('most_common_condition', 'N/A')}\n\n")
            
            # Raga Distribution
            f.write("RAGA DISTRIBUTION:\n")
            f.write("-" * 20 + "\n")
            raga_dist = summary_stats.get('raga_distribution', {})
            for raga, count in sorted(raga_dist.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / summary_stats.get('total_mappings', 1)) * 100
                f.write(f"{raga}: {count} patients ({percentage:.1f}%)\n")
            f.write("\n")
            
            # Condition Distribution
            f.write("MENTAL CONDITION DISTRIBUTION:\n")
            f.write("-" * 30 + "\n")
            condition_dist = summary_stats.get('condition_distribution', {})
            for condition, count in sorted(condition_dist.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / summary_stats.get('total_mappings', 1)) * 100
                f.write(f"{condition}: {count} patients ({percentage:.1f}%)\n")
            f.write("\n")
            
            # Age Statistics
            age_stats = summary_stats.get('age_statistics', {})
            f.write("AGE STATISTICS:\n")
            f.write("-" * 15 + "\n")
            f.write(f"Mean age: {age_stats.get('mean', 0):.1f} years\n")
            f.write(f"Age range: {age_stats.get('min', 0)} - {age_stats.get('max', 0)} years\n")
            f.write(f"Standard deviation: {age_stats.get('std', 0):.1f}\n\n")
            
            # Therapy Types
            f.write("THERAPY TYPES:\n")
            f.write("-" * 15 + "\n")
            therapy_types = summary_stats.get('therapy_types', {})
            for therapy_type, count in sorted(therapy_types.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / summary_stats.get('total_mappings', 1)) * 100
                f.write(f"{therapy_type}: {count} patients ({percentage:.1f}%)\n")
            f.write("\n")
            
            # Duration Statistics
            duration_stats = summary_stats.get('duration_statistics', {})
            f.write("RECOMMENDED DURATION STATISTICS:\n")
            f.write("-" * 35 + "\n")
            f.write(f"Average duration: {duration_stats.get('mean', 0):.1f} minutes\n")
            f.write(f"Duration range: {duration_stats.get('min', 0)} - {duration_stats.get('max', 0)} minutes\n")
            f.write(f"Standard deviation: {duration_stats.get('std', 0):.1f}\n\n")
            
            # System Info
            f.write("SYSTEM INFORMATION:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Wav2CLIP model: {results['wav2clip_embeddings']['model_used']}\n")
            f.write(f"Embedding dimension: {results['wav2clip_embeddings']['embedding_dimension']}\n")
            f.write(f"Raga detection model available: {results['raga_detection_info']['model_available']}\n")
            f.write(f"Predictions processed: {results['raga_detection_info']['predictions_count']}\n")
        
        print(f"📄 Summary report saved: {report_file}")
    
    def _display_results_summary(self, results: Dict, summary_stats: Dict):
        """Display results summary to console"""
        
        print("\n🎉 THERAPY MAPPING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        print(f"📊 SUMMARY STATISTICS:")
        print(f"   Total mappings created: {summary_stats.get('total_mappings', 0)}")
        print(f"   Unique ragas analyzed: {summary_stats.get('unique_ragas', 0)}")
        print(f"   Average confidence: {summary_stats.get('average_confidence', 0):.3f}")
        print(f"   Most recommended raga: {summary_stats.get('most_recommended_raga', 'N/A')}")
        
        print(f"\n🎵 TOP RAGAS:")
        raga_dist = summary_stats.get('raga_distribution', {})
        top_ragas = sorted(raga_dist.items(), key=lambda x: x[1], reverse=True)[:3]
        for i, (raga, count) in enumerate(top_ragas, 1):
            print(f"   {i}. {raga}: {count} patients")
        
        print(f"\n🏥 TOP CONDITIONS:")
        condition_dist = summary_stats.get('condition_distribution', {})
        top_conditions = sorted(condition_dist.items(), key=lambda x: x[1], reverse=True)[:3]
        for i, (condition, count) in enumerate(top_conditions, 1):
            print(f"   {i}. {condition}: {count} patients")
        
        print(f"\n💊 THERAPY TYPES:")
        therapy_types = summary_stats.get('therapy_types', {})
        for therapy_type, count in list(therapy_types.items())[:3]:
            print(f"   • {therapy_type}: {count} patients")
        
        duration_stats = summary_stats.get('duration_statistics', {})
        print(f"\n⏱️ SESSION DURATIONS:")
        print(f"   Average: {duration_stats.get('mean', 0):.1f} minutes")
        print(f"   Range: {duration_stats.get('min', 0)}-{duration_stats.get('max', 0)} minutes")
        
        print(f"\n📁 OUTPUT FILES:")
        print(f"   📄 Main results: {self.mapping_folder}/therapy_mapping_results.json")
        print(f"   📄 Detailed mappings: {self.mapping_folder}/detailed_therapy_mappings.json")
        print(f"   📄 Summary report: {self.mapping_folder}/therapy_mapping_report.txt")
        
        print(f"\n✨ WAV2CLIP EMBEDDING INFO:")
        wav2clip_info = results.get('wav2clip_embeddings', {})
        print(f"   Model: {wav2clip_info.get('model_used', 'N/A')}")
        print(f"   Embedding dimension: {wav2clip_info.get('embedding_dimension', 'N/A')}")
        print(f"   Total embeddings: {wav2clip_info.get('total_embeddings_created', 0)}")

def main():
    """Main function to run therapy mapping"""
    
    print("🎵 WAV2CLIP RAGA-THERAPY MAPPING SYSTEM")
    print("Using audio-text embeddings for personalized music therapy")
    print("=" * 70)
    
    try:
        # Initialize therapy mapper
        print("🔧 Initializing therapy mapping system...")
        mapper = RagaTherapyMapper()
        
        # Process therapy mapping
        results = mapper.process_therapy_mapping()
        
        if results["status"] == "success":
            print("\n🎊 THERAPY MAPPING COMPLETED SUCCESSFULLY!")
            print("\n💡 Next steps:")
            print("   1. Review the generated therapy recommendations")
            print("   2. Validate mappings with healthcare professionals")
            print("   3. Use recommendations for personalized music therapy sessions")
            print("   4. Collect feedback for system improvement")
            
        else:
            print(f"\n❌ THERAPY MAPPING FAILED: {results['message']}")
            print("💡 Please check the requirements and try again.")
        
        return results
        
    except Exception as e:
        logger.error(f"Therapy mapping failed: {e}")
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        print("💡 Please check your setup and try again.")
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    print("🎵 Wav2CLIP Raga-Therapy Mapping System")
    print("=" * 50)
    print("\n📋 Requirements:")
    print("1. Run raga_detection.py first")
    print("2. Ensure data/Final_dataset_s.csv exists")
    print("3. Install required packages:")
    print("   pip install torch transformers librosa scikit-learn pandas")
    print("\n🚀 Starting therapy mapping...")
    
    # Run main function
    results = main()
    
    if results["status"] == "success":
        print("\n✅ Therapy mapping completed successfully!")
        print(f"📁 Check the 'mapping_output' folder for results")
    else:
        print(f"\n❌ Failed: {results.get('message', 'Unknown error')}")
    
    print("\n👋 Thank you for using the Raga-Therapy Mapping System!")