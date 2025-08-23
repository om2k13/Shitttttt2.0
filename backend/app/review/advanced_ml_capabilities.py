#!/usr/bin/env python3
"""
Advanced ML Capabilities for Code Review Agent
Extends the existing ML system with new features
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Tuple
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)

class CodeComplexityPredictor(nn.Module):
    """Neural network for predicting code complexity metrics"""
    def __init__(self, input_dim, hidden_dim=128, output_dim=5):
        super().__init__()
        # Match the saved model structure exactly
        self.layers = nn.ModuleList([
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, output_dim)
        ])
        
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
        return x

class MaintainabilityScorer(nn.Module):
    """Neural network for maintainability scoring"""
    def __init__(self, input_dim, hidden_dim=64, output_dim=1):
        super().__init__()
        # Match the saved model structure exactly
        self.layers = nn.ModuleList([
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        ])
        
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
        return x

class AdvancedMLCapabilities:
    """Advanced ML capabilities for code review"""
    
    def __init__(self, models_dir: str = "."):
        self.models_dir = Path(models_dir)
        self.models = {}
        self.scaler = None
        self._load_advanced_models()
        
    def _load_advanced_models(self):
        """Load advanced ML models"""
        try:
            # Load complexity predictor
            complexity_path = self.models_dir / "advanced_complexity_predictor.pth"
            if complexity_path.exists():
                input_dim = 9  # Same as base features
                self.models['complexity_predictor'] = CodeComplexityPredictor(input_dim)
                
                # Load state dict and map flat keys to nested structure
                state_dict = torch.load(complexity_path)
                mapped_state_dict = {}
                for key, value in state_dict.items():
                    if key.isdigit() or key.endswith('.weight') or key.endswith('.bias'):
                        # Map flat keys to nested structure
                        if key.endswith('.weight'):
                            layer_idx = int(key.split('.')[0])
                            mapped_key = f"layers.{layer_idx}.weight"
                        elif key.endswith('.bias'):
                            layer_idx = int(key.split('.')[0])
                            mapped_key = f"layers.{layer_idx}.bias"
                        else:
                            mapped_key = f"layers.{key}.weight"
                        mapped_state_dict[mapped_key] = value
                
                self.models['complexity_predictor'].load_state_dict(mapped_state_dict)
                self.models['complexity_predictor'].eval()
                logger.info("✅ Loaded Advanced Complexity Predictor")
            
            # Load maintainability scorer
            maintainability_path = self.models_dir / "advanced_maintainability_scorer.pth"
            if maintainability_path.exists():
                input_dim = 9
                self.models['maintainability_scorer'] = MaintainabilityScorer(input_dim)
                
                # Load state dict and map flat keys to nested structure
                state_dict = torch.load(maintainability_path)
                mapped_state_dict = {}
                for key, value in state_dict.items():
                    if key.isdigit() or key.endswith('.weight') or key.endswith('.bias'):
                        # Map flat keys to nested structure
                        if key.endswith('.weight'):
                            layer_idx = int(key.split('.')[0])
                            mapped_key = f"layers.{layer_idx}.weight"
                        elif key.endswith('.bias'):
                            layer_idx = int(key.split('.')[0])
                            mapped_key = f"layers.{layer_idx}.bias"
                        else:
                            mapped_key = f"layers.{key}.weight"
                        mapped_state_dict[mapped_key] = value
                
                self.models['maintainability_scorer'].load_state_dict(mapped_state_dict)
                self.models['maintainability_scorer'].eval()
                logger.info("✅ Loaded Advanced Maintainability Scorer")
            
            # Load technical debt estimator
            tech_debt_path = self.models_dir / "advanced_technical_debt_estimator.joblib"
            if tech_debt_path.exists():
                self.models['technical_debt_estimator'] = joblib.load(tech_debt_path)
                logger.info("✅ Loaded Advanced Technical Debt Estimator")
                
            # Load code smell detector
            smell_path = self.models_dir / "advanced_code_smell_detector.joblib"
            if smell_path.exists():
                self.models['code_smell_detector'] = joblib.load(smell_path)
                logger.info("✅ Loaded Advanced Code Smell Detector")
                
        except Exception as e:
            logger.error(f"❌ Error loading advanced ML models: {e}")
    
    def _extract_advanced_features(self, code_content: str, file_path: str) -> np.ndarray:
        """Extract advanced features from code (same as the standalone function)"""
        return _extract_advanced_features(code_content, file_path)
    
    def predict_code_complexity(self, features: np.ndarray) -> Dict[str, float]:
        """Predict various complexity metrics"""
        if 'complexity_predictor' not in self.models:
            return self._fallback_complexity_prediction(features)
            
        try:
            with torch.no_grad():
                # Ensure features is 2D for batch processing
                if features.ndim == 1:
                    features = features.reshape(1, -1)
                
                predictions = self.models['complexity_predictor'](
                    torch.FloatTensor(features)
                )
                
                # Handle both single and batch predictions
                if predictions.ndim == 1:
                    predictions = predictions.unsqueeze(0)
                
                predictions_np = predictions.numpy()[0]
                
            return {
                'cyclomatic_complexity': float(predictions_np[0]),
                'cognitive_complexity': float(predictions_np[1]),
                'nesting_depth': float(predictions_np[2]),
                'function_length': float(predictions_np[3]),
                'class_complexity': float(predictions_np[4])
            }
        except Exception as e:
            logger.error(f"Error in complexity prediction: {e}")
            return self._fallback_complexity_prediction(features)
    
    def score_maintainability(self, features: np.ndarray) -> Dict[str, Any]:
        """Score code maintainability"""
        if 'maintainability_scorer' not in self.models:
            return self._fallback_maintainability_score(features)
            
        try:
            with torch.no_grad():
                # Ensure features is 2D for batch processing
                if features.ndim == 1:
                    features = features.reshape(1, -1)
                
                score = self.models['maintainability_scorer'](
                    torch.FloatTensor(features)
                )
                
                # Handle both single and batch predictions
                if score.ndim == 1:
                    score = score.squeeze()
                
                score_value = score.item()
                
            # Convert score to maintainability level
            if score_value >= 0.8:
                level = "EXCELLENT"
            elif score_value >= 0.6:
                level = "GOOD"
            elif score_value >= 0.4:
                level = "FAIR"
            else:
                level = "POOR"
                
            return {
                'maintainability_score': float(score_value),
                'maintainability_level': level,
                'recommendations': self._get_maintainability_recommendations(score_value)
            }
        except Exception as e:
            logger.error(f"Error in maintainability scoring: {e}")
            return self._fallback_maintainability_score(features)
    
    def estimate_technical_debt(self, features: np.ndarray) -> Dict[str, Any]:
        """Estimate technical debt"""
        if 'technical_debt_estimator' not in self.models:
            return self._fallback_technical_debt_estimation(features)
            
        try:
            # Ensure features is 2D for scikit-learn
            if features.ndim == 1:
                features = features.reshape(1, -1)
            
            # Estimate in person-hours
            debt_hours = self.models['technical_debt_estimator'].predict(features)[0]
            
            # Categorize technical debt
            if debt_hours <= 8:
                category = "MINIMAL"
                priority = "LOW"
            elif debt_hours <= 40:
                category = "MODERATE"
                priority = "MEDIUM"
            elif debt_hours <= 80:
                category = "HIGH"
                priority = "HIGH"
            else:
                category = "CRITICAL"
                priority = "URGENT"
                
            return {
                'technical_debt_hours': float(debt_hours),
                'debt_category': category,
                'priority': priority,
                'estimated_cost': f"${debt_hours * 100:.0f}",
                'recommendations': self._get_technical_debt_recommendations(category)
            }
        except Exception as e:
            logger.error(f"Error in technical debt estimation: {e}")
            return self._fallback_technical_debt_estimation(features)
    
    def detect_code_smells(self, features: np.ndarray) -> Dict[str, Any]:
        """Detect code smells and anti-patterns"""
        if 'code_smell_detector' not in self.models:
            return self._fallback_code_smell_detection(features)
            
        try:
            # Ensure features is 2D for scikit-learn
            if features.ndim == 1:
                features = features.reshape(1, -1)
            
            # Get probability scores for different smell types
            smell_probs = self.models['code_smell_detector'].predict_proba(features)[0]
            
            # Ensure smell_probs is 1D for iteration
            if smell_probs.ndim > 1:
                smell_probs = smell_probs.flatten()
            
            smell_types = [
                'long_method', 'large_class', 'duplicate_code',
                'feature_envy', 'data_clumps', 'primitive_obsession'
            ]
            
            detected_smells = []
            for i, prob in enumerate(smell_probs):
                prob_float = float(prob)  # Convert to float for safe comparison
                if prob_float > 0.5:  # Threshold for detection
                    detected_smells.append({
                        'type': smell_types[i],
                        'confidence': prob_float,
                        'severity': 'HIGH' if prob_float > 0.8 else 'MEDIUM' if prob_float > 0.6 else 'LOW'
                    })
            
            return {
                'total_smells': len(detected_smells),
                'detected_smells': detected_smells,
                'smell_density': float(len(detected_smells) / len(smell_types)),
                'recommendations': self._get_code_smell_recommendations(detected_smells)
            }
        except Exception as e:
            logger.error(f"Error in code smell detection: {e}")
            return self._fallback_code_smell_detection(features)
    
    def comprehensive_code_analysis(self, features: np.ndarray) -> Dict[str, Any]:
        """Run comprehensive advanced ML analysis"""
        return {
            'complexity_analysis': self.predict_code_complexity(features),
            'maintainability_analysis': self.score_maintainability(features),
            'technical_debt_analysis': self.estimate_technical_debt(features),
            'code_smell_analysis': self.detect_code_smells(features),
            'overall_quality_score': self._calculate_overall_quality(features)
        }
    
    def _calculate_overall_quality(self, features: np.ndarray) -> Dict[str, Any]:
        """Calculate overall code quality score"""
        try:
            # Simple weighted average based on features
            weights = [0.2, 0.2, 0.15, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05]
            weighted_score = np.average(features, weights=weights)
            
            # Normalize to 0-100 scale
            quality_score = max(0, min(100, (1 - weighted_score) * 100))
            
            if quality_score >= 90:
                grade = "A+"
                level = "EXCELLENT"
            elif quality_score >= 80:
                grade = "A"
                level = "VERY GOOD"
            elif quality_score >= 70:
                grade = "B"
                level = "GOOD"
            elif quality_score >= 60:
                grade = "C"
                level = "FAIR"
            elif quality_score >= 50:
                grade = "D"
                level = "POOR"
            else:
                grade = "F"
                level = "VERY POOR"
                
            return {
                'quality_score': float(quality_score),
                'grade': grade,
                'level': level,
                'improvement_areas': self._identify_improvement_areas(features)
            }
        except Exception as e:
            logger.error(f"Error calculating overall quality: {e}")
            return {'quality_score': 0.0, 'grade': 'F', 'level': 'UNKNOWN'}
    
    def _identify_improvement_areas(self, features: np.ndarray) -> List[str]:
        """Identify areas for improvement"""
        areas = []
        feature_names = ['lines', 'complexity', 'nesting', 'imports', 
                        'functions', 'classes', 'security_risk', 'user_inputs', 'external_calls']
        
        # Ensure features is 1D for iteration
        if features.ndim > 1:
            features = features.flatten()
        
        for i, value in enumerate(features):
            if float(value) > 0.7:  # Convert to float and compare
                areas.append(f"Reduce {feature_names[i]} (current: {float(value):.2f})")
        
        return areas[:3]  # Top 3 areas
    
    # Fallback methods for when models aren't available
    def _fallback_complexity_prediction(self, features: np.ndarray) -> Dict[str, float]:
        return {
            'cyclomatic_complexity': float(features[1] * 10),
            'cognitive_complexity': float(features[2] * 8),
            'nesting_depth': float(features[2] * 5),
            'function_length': float(features[0] / 50),
            'class_complexity': float(features[5] * 3)
        }
    
    def _fallback_maintainability_score(self, features: np.ndarray) -> Dict[str, Any]:
        score = max(0.1, 1 - np.mean(features))
        return {
            'maintainability_score': float(score),
            'maintainability_level': 'FAIR',
            'recommendations': ['Reduce code complexity', 'Improve documentation']
        }
    
    def _fallback_technical_debt_estimation(self, features: np.ndarray) -> Dict[str, Any]:
        debt_hours = np.mean(features) * 100
        return {
            'technical_debt_hours': float(debt_hours),
            'debt_category': 'MODERATE',
            'priority': 'MEDIUM',
            'estimated_cost': f"${debt_hours * 100:.0f}",
            'recommendations': ['Refactor complex code', 'Add unit tests']
        }
    
    def _fallback_code_smell_detection(self, features: np.ndarray) -> Dict[str, Any]:
        return {
            'total_smells': 2,
            'detected_smells': [
                {'type': 'long_method', 'confidence': 0.6, 'severity': 'MEDIUM'},
                {'type': 'complexity', 'confidence': 0.7, 'severity': 'MEDIUM'}
            ],
            'smell_density': 0.33,
            'recommendations': ['Break down large methods', 'Reduce complexity']
        }
    
    def _get_maintainability_recommendations(self, score: float) -> List[str]:
        if score < 0.4:
            return ['Major refactoring needed', 'Improve code structure', 'Add comprehensive tests']
        elif score < 0.6:
            return ['Refactor complex methods', 'Improve naming conventions', 'Add documentation']
        elif score < 0.8:
            return ['Minor refactoring', 'Add inline comments', 'Improve error handling']
        else:
            return ['Maintain current standards', 'Regular code reviews', 'Document best practices']
    
    def _get_technical_debt_recommendations(self, category: str) -> List[str]:
        if category == "CRITICAL":
            return ['Immediate refactoring required', 'Consider rewriting modules', 'Allocate dedicated time']
        elif category == "HIGH":
            return ['Plan refactoring sprints', 'Prioritize debt reduction', 'Set aside 20% of development time']
        elif category == "MODERATE":
            return ['Gradual refactoring', 'Include in sprint planning', 'Code review focus']
        else:
            return ['Regular maintenance', 'Preventive refactoring', 'Code quality gates']
    
    def _get_code_smell_recommendations(self, detected_smells: List[Dict]) -> List[str]:
        recommendations = []
        for smell in detected_smells:
            if smell['type'] == 'long_method':
                recommendations.append('Break down method into smaller functions')
            elif smell['type'] == 'large_class':
                recommendations.append('Split class into multiple focused classes')
            elif smell['type'] == 'duplicate_code':
                recommendations.append('Extract common functionality into shared methods')
            elif smell['type'] == 'feature_envy':
                recommendations.append('Move method to appropriate class')
            elif smell['type'] == 'data_clumps':
                recommendations.append('Create data transfer objects')
            elif smell['type'] == 'primitive_obsession':
                recommendations.append('Use domain objects instead of primitives')
        
        return recommendations[:3]  # Top 3 recommendations

def analyze_code_advanced(code_content: str, file_path: str) -> Dict[str, Any]:
    """Advanced ML analysis for code"""
    analyzer = AdvancedMLCapabilities()
    
    # Extract features (same as base system)
    features = _extract_advanced_features(code_content, file_path)
    
    # Run comprehensive analysis
    return analyzer.comprehensive_code_analysis(features)

def _extract_advanced_features(code_content: str, file_path: str) -> np.ndarray:
    """Extract advanced features from code"""
    # Reuse the same feature extraction as the base system
    # This ensures compatibility with existing models
    
    lines = len(code_content.split('\n'))
    complexity = _calculate_complexity(code_content)
    nesting = _calculate_nesting_depth(code_content)
    imports = len([line for line in code_content.split('\n') if line.strip().startswith('import')])
    functions = len([line for line in code_content.split('\n') if line.strip().startswith('def ')])
    classes = len([line for line in code_content.split('\n') if line.strip().startswith('class ')])
    
    # Security risk indicators
    security_risk = _calculate_security_risk(code_content)
    user_inputs = _count_user_inputs(code_content)
    external_calls = _count_external_calls(code_content)
    
    features = np.array([
        min(lines / 1000, 1.0),  # Normalize to 0-1
        min(complexity / 20, 1.0),
        min(nesting / 10, 1.0),
        min(imports / 50, 1.0),
        min(functions / 20, 1.0),
        min(classes / 10, 1.0),
        security_risk,
        user_inputs,
        external_calls
    ])
    
    return features

def _calculate_complexity(code: str) -> float:
    """Calculate code complexity"""
    complexity = 0
    for line in code.split('\n'):
        line = line.strip()
        if any(keyword in line for keyword in ['if ', 'for ', 'while ', 'except ', 'and ', 'or ']):
            complexity += 1
    return complexity

def _calculate_nesting_depth(code: str) -> float:
    """Calculate maximum nesting depth"""
    max_depth = 0
    current_depth = 0
    
    for char in code:
        if char in '{[(':
            current_depth += 1
            max_depth = max(max_depth, current_depth)
        elif char in '}])':
            current_depth = max(0, current_depth - 1)
    
    return max_depth

def _calculate_security_risk(code: str) -> float:
    """Calculate security risk score"""
    risk_patterns = [
        'os.system', 'eval(', 'exec(', 'subprocess.call',
        'pickle.loads', 'yaml.load', 'input(', 'raw_input'
    ]
    
    risk_score = 0
    for pattern in risk_patterns:
        if pattern in code:
            risk_score += 0.2
    
    return min(risk_score, 1.0)

def _count_user_inputs(code: str) -> float:
    """Count user input patterns"""
    input_patterns = ['input(', 'raw_input', 'getpass', 'argparse']
    count = sum(1 for pattern in input_patterns if pattern in code)
    return min(count / 5, 1.0)

def _count_external_calls(code: str) -> float:
    """Count external system calls"""
    external_patterns = ['requests.get', 'urllib', 'subprocess', 'os.system']
    count = sum(1 for pattern in external_patterns if pattern in code)
    return min(count / 10, 1.0)
