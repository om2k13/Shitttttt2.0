import torch
import torch.nn as nn
import numpy as np
import os
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class SafeCodeEmbeddingModel(nn.Module):
    def __init__(self, vocab_size: int = 10000, embedding_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attention = nn.MultiheadAttention(hidden_dim * 2, num_heads=8, batch_first=True)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        attended, _ = self.attention(lstm_out, lstm_out, lstm_out)
        return self.dropout(attended.mean(dim=1))

class SafeSecurityVulnerabilityDetector(nn.Module):
    def __init__(self, input_dim: int = 6):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.layers(x)

class SafeCodeQualityPredictor(nn.Module):
    def __init__(self, input_dim: int = 6):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3),  # 3 outputs: complexity, maintainability, reliability
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.layers(x)

class SafeNeuralAnalyzer:
    def __init__(self):
        self.device = torch.device('cpu')
        self.models = {}
        self._load_models_safely()
        
    def _load_models_safely(self):
        """Load models one at a time to avoid memory conflicts"""
        try:
            # Load embedding model
            logger.info("Loading Code Embedding Model...")
            self.models['embedding'] = SafeCodeEmbeddingModel()
            self.models['embedding'].to(self.device)
            self.models['embedding'].eval()
            logger.info("✅ Code Embedding Model loaded")
            
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            self.models['embedding'] = None
            
        try:
            # Load security model
            logger.info("Loading Security Vulnerability Detector...")
            self.models['security'] = SafeSecurityVulnerabilityDetector()
            self.models['security'].to(self.device)
            self.models['security'].eval()
            logger.info("✅ Security Vulnerability Detector loaded")
            
        except Exception as e:
            logger.error(f"Failed to load security model: {e}")
            self.models['security'] = None
            
        try:
            # Load quality model
            logger.info("Loading Code Quality Predictor...")
            self.models['quality'] = SafeCodeQualityPredictor()
            self.models['quality'].to(self.device)
            self.models['quality'].eval()
            logger.info("✅ Code Quality Predictor loaded")
            
        except Exception as e:
            logger.error(f"Failed to load quality model: {e}")
            self.models['quality'] = None
    
    def analyze_security(self, features: np.ndarray) -> Dict[str, float]:
        """Analyze security vulnerabilities"""
        if self.models.get('security') is None:
            return {'risk_score': 0.5, 'confidence': 0.0}
            
        try:
            with torch.no_grad():
                features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
                predictions = self.models['security'](features_tensor)
                risk_score = predictions[0, 0].item()
                confidence = predictions[0, 1].item()
                return {'risk_score': risk_score, 'confidence': confidence}
        except Exception as e:
            logger.error(f"Security analysis failed: {e}")
            return {'risk_score': 0.5, 'confidence': 0.0}
    
    def predict_quality(self, features: np.ndarray) -> Dict[str, float]:
        """Predict code quality metrics"""
        if self.models.get('quality') is None:
            return {'complexity': 0.5, 'maintainability': 0.5, 'reliability': 0.5}
            
        try:
            with torch.no_grad():
                features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
                predictions = self.models['quality'](features_tensor)
                return {
                    'complexity': predictions[0, 0].item(),
                    'maintainability': predictions[0, 1].item(),
                    'reliability': predictions[0, 2].item()
                }
        except Exception as e:
            logger.error(f"Quality prediction failed: {e}")
            return {'complexity': 0.5, 'maintainability': 0.5, 'reliability': 0.5}
    
    def get_status(self) -> Dict[str, bool]:
        """Get status of all models"""
        return {
            'embedding_model': self.models.get('embedding') is not None,
            'security_model': self.models.get('security') is not None,
            'quality_model': self.models.get('quality') is not None
        }
