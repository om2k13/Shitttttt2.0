#!/usr/bin/env python3
"""
Production ML and Neural Network Analyzer
Integrates ALL trained models into the code review pipeline
"""

import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import joblib
from typing import Dict, List, Any, Optional, Tuple
import ast
import re
import logging
from sklearn.preprocessing import StandardScaler

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SecurityVulnerabilityDetector(nn.Module):
    """PyTorch model for security vulnerability detection"""
    def __init__(self, input_dim=9, hidden_dim=256):
        super(SecurityVulnerabilityDetector, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.fc4 = nn.Linear(hidden_dim // 4, 1)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc4(x))
        return x

class CodeQualityPredictor(nn.Module):
    """PyTorch model for code quality prediction"""
    def __init__(self, input_dim=9, hidden_dim=128, num_classes=5):
        super(CodeQualityPredictor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, num_classes)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class ProductionMLAnalyzer:
    """Production ML Analyzer with all trained models"""
    
    def __init__(self, models_dir: str = "."):
        self.models_dir = Path(models_dir)
        self.traditional_models = {}
        self.neural_models = {}
        self.scaler = None
        self.ensemble_data = None
        self.metadata = None
        self.feature_names = [
            'lines', 'complexity', 'nesting', 'imports', 
            'functions', 'classes', 'security_risk', 'user_inputs', 'external_calls'
        ]
        
        # Load all models
        self._load_all_models()
    
    def _load_all_models(self):
        """Load all trained models"""
        logger.info("🔄 Loading Production ML Models...")
        
        # Load traditional ML models
        traditional_model_files = {
            'RandomForest': 'production_randomforest.joblib',
            'GradientBoosting': 'production_gradientboosting.joblib',
            'LogisticRegression': 'production_logisticregression.joblib',
            'SVM': 'production_svm.joblib',
            'MLP': 'production_mlp.joblib',
            'IsolationForest': 'production_isolationforest.joblib',
            'XGBoost': 'production_xgboost.joblib',
            'LightGBM': 'production_lightgbm.joblib'
        }
        
        for name, filename in traditional_model_files.items():
            file_path = self.models_dir / filename
            if file_path.exists():
                try:
                    model = joblib.load(file_path)
                    self.traditional_models[name] = model
                    logger.info(f"  ✅ Loaded {name}")
                except Exception as e:
                    logger.error(f"  ❌ Error loading {name}: {e}")
        
        # Load neural networks
        neural_model_files = {
            'SecurityDetector': 'production_securitydetector_neural.pth',
            'QualityPredictor': 'production_qualitypredictor_neural.pth'
        }
        
        for name, filename in neural_model_files.items():
            file_path = self.models_dir / filename
            if file_path.exists():
                try:
                    if name == 'SecurityDetector':
                        model = SecurityVulnerabilityDetector()
                        model.load_state_dict(torch.load(file_path, map_location='cpu'))
                        model.eval()
                        self.neural_models[name] = model
                        logger.info(f"  ✅ Loaded Neural {name}")
                    elif name == 'QualityPredictor':
                        model = CodeQualityPredictor()
                        model.load_state_dict(torch.load(file_path, map_location='cpu'))
                        model.eval()
                        self.neural_models[name] = model
                        logger.info(f"  ✅ Loaded Neural {name}")
                except Exception as e:
                    logger.error(f"  ❌ Error loading Neural {name}: {e}")
        
        # Load scaler
        scaler_path = self.models_dir / 'production_scaler.joblib'
        if scaler_path.exists():
            try:
                self.scaler = joblib.load(scaler_path)
                logger.info("  ✅ Loaded feature scaler")
            except Exception as e:
                logger.error(f"  ❌ Error loading scaler: {e}")
        
        # Load ensemble data
        ensemble_path = self.models_dir / 'production_super_ensemble.json'
        if ensemble_path.exists():
            try:
                with open(ensemble_path, 'r') as f:
                    self.ensemble_data = json.load(f)
                logger.info("  ✅ Loaded ensemble configuration")
            except Exception as e:
                logger.error(f"  ❌ Error loading ensemble: {e}")
        
        # Load metadata
        metadata_path = self.models_dir / 'production_training_metadata.json'
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                logger.info("  ✅ Loaded training metadata")
            except Exception as e:
                logger.error(f"  ❌ Error loading metadata: {e}")
        
        logger.info(f"📊 Loaded {len(self.traditional_models)} traditional ML models")
        logger.info(f"📊 Loaded {len(self.neural_models)} neural network models")
    
    def extract_code_features(self, code: str, file_path: str = "") -> np.ndarray:
        """Extract features from code for ML analysis"""
        try:
            # Initialize features
            features = {
                'lines': 0,
                'complexity': 0,
                'nesting': 0,
                'imports': 0,
                'functions': 0,
                'classes': 0,
                'security_risk': 0,
                'user_inputs': 0,
                'external_calls': 0
            }
            
            lines = code.strip().split('\n')
            features['lines'] = len(lines)
            
            # Detect language
            if file_path.endswith(('.py', '.pyw')):
                features.update(self._extract_python_features(code))
            elif file_path.endswith(('.js', '.jsx', '.ts', '.tsx')):
                features.update(self._extract_javascript_features(code))
            elif file_path.endswith(('.java', '.class')):
                features.update(self._extract_java_features(code))
            else:
                # Generic analysis
                features.update(self._extract_generic_features(code))
            
            # Convert to numpy array
            feature_array = np.array([features[name] for name in self.feature_names])
            return feature_array.reshape(1, -1)
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            # Return default features
            return np.array([[10, 1, 1, 0, 1, 0, 0, 0, 0]])
    
    def _extract_python_features(self, code: str) -> Dict[str, int]:
        """Extract Python-specific features"""
        features = {}
        
        try:
            tree = ast.parse(code)
            
            # Count functions and classes
            features['functions'] = len([node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)])
            features['classes'] = len([node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)])
            
            # Count imports
            imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
            features['imports'] = len(imports)
            
            # Calculate complexity (approximate)
            complexity_nodes = [node for node in ast.walk(tree) if isinstance(node, (ast.If, ast.While, ast.For, ast.Try))]
            features['complexity'] = len(complexity_nodes) + 1
            
            # Calculate nesting depth
            features['nesting'] = self._calculate_nesting_depth(tree)
            
            # Security risk indicators
            security_patterns = ['eval', 'exec', 'input', 'raw_input', 'subprocess', 'os.system']
            security_risk = sum(1 for pattern in security_patterns if pattern in code)
            features['security_risk'] = min(security_risk, 1)  # Binary flag
            
            # User input patterns
            input_patterns = ['input(', 'raw_input(', 'request.', 'flask.request']
            features['user_inputs'] = sum(1 for pattern in input_patterns if pattern in code)
            
            # External calls
            external_patterns = ['requests.', 'urllib', 'http', 'subprocess.', 'os.system']
            features['external_calls'] = sum(1 for pattern in external_patterns if pattern in code)
            
        except Exception as e:
            logger.error(f"Error parsing Python code: {e}")
            features = self._extract_generic_features(code)
        
        return features
    
    def _extract_javascript_features(self, code: str) -> Dict[str, int]:
        """Extract JavaScript-specific features"""
        features = {}
        
        # Count functions
        function_patterns = [r'function\s+\w+', r'\w+\s*=\s*function', r'\w+\s*=>\s*']
        features['functions'] = sum(len(re.findall(pattern, code)) for pattern in function_patterns)
        
        # Count classes
        features['classes'] = len(re.findall(r'class\s+\w+', code))
        
        # Count imports
        import_patterns = [r'import\s+', r'require\s*\(', r'from\s+\w+\s+import']
        features['imports'] = sum(len(re.findall(pattern, code)) for pattern in import_patterns)
        
        # Complexity (control structures)
        complexity_patterns = [r'\bif\s*\(', r'\bwhile\s*\(', r'\bfor\s*\(', r'\btry\s*{']
        features['complexity'] = sum(len(re.findall(pattern, code)) for pattern in complexity_patterns) + 1
        
        # Nesting (approximate by counting braces)
        features['nesting'] = min(code.count('{'), 20)
        
        # Security risks
        security_patterns = ['eval(', 'innerHTML', 'document.write', 'setTimeout', 'setInterval']
        features['security_risk'] = min(sum(1 for pattern in security_patterns if pattern in code), 1)
        
        # User inputs
        input_patterns = ['prompt(', 'confirm(', 'alert(', 'input']
        features['user_inputs'] = sum(1 for pattern in input_patterns if pattern in code)
        
        # External calls
        external_patterns = ['fetch(', 'axios.', '$.ajax', 'XMLHttpRequest']
        features['external_calls'] = sum(1 for pattern in external_patterns if pattern in code)
        
        return features
    
    def _extract_java_features(self, code: str) -> Dict[str, int]:
        """Extract Java-specific features"""
        features = {}
        
        # Count methods and classes
        features['functions'] = len(re.findall(r'(public|private|protected|static).*?\w+\s*\([^)]*\)\s*{', code))
        features['classes'] = len(re.findall(r'class\s+\w+', code))
        
        # Count imports
        features['imports'] = len(re.findall(r'import\s+', code))
        
        # Complexity
        complexity_patterns = [r'\bif\s*\(', r'\bwhile\s*\(', r'\bfor\s*\(', r'\btry\s*{']
        features['complexity'] = sum(len(re.findall(pattern, code)) for pattern in complexity_patterns) + 1
        
        # Nesting
        features['nesting'] = min(code.count('{'), 20)
        
        # Security risks
        security_patterns = ['Runtime.exec', 'System.exit', 'ProcessBuilder', 'reflection']
        features['security_risk'] = min(sum(1 for pattern in security_patterns if pattern in code), 1)
        
        # User inputs
        input_patterns = ['Scanner', 'BufferedReader', 'System.in']
        features['user_inputs'] = sum(1 for pattern in input_patterns if pattern in code)
        
        # External calls
        external_patterns = ['HttpURLConnection', 'Socket', 'URL(', 'URI(']
        features['external_calls'] = sum(1 for pattern in external_patterns if pattern in code)
        
        return features
    
    def _extract_generic_features(self, code: str) -> Dict[str, int]:
        """Extract generic features for any language"""
        features = {}
        
        # Basic counts
        features['functions'] = len(re.findall(r'function|def |func |sub |method', code, re.IGNORECASE))
        features['classes'] = len(re.findall(r'class |struct |interface', code, re.IGNORECASE))
        features['imports'] = len(re.findall(r'import|include|require|using', code, re.IGNORECASE))
        
        # Complexity indicators
        complexity_indicators = ['if', 'while', 'for', 'switch', 'case', 'try', 'catch']
        features['complexity'] = sum(len(re.findall(f'\\b{keyword}\\b', code, re.IGNORECASE)) for keyword in complexity_indicators) + 1
        
        # Nesting (count common bracket types)
        features['nesting'] = min(max(code.count('{'), code.count('(')), 20)
        
        # Security patterns
        security_keywords = ['system', 'exec', 'eval', 'shell', 'command']
        features['security_risk'] = min(sum(1 for keyword in security_keywords if keyword in code.lower()), 1)
        
        # Input patterns
        input_keywords = ['input', 'read', 'scan', 'get']
        features['user_inputs'] = sum(1 for keyword in input_keywords if keyword in code.lower())
        
        # External calls
        external_keywords = ['http', 'url', 'request', 'socket', 'connect']
        features['external_calls'] = sum(1 for keyword in external_keywords if keyword in code.lower())
        
        return features
    
    def _calculate_nesting_depth(self, tree) -> int:
        """Calculate maximum nesting depth in AST"""
        def depth(node):
            if not hasattr(node, 'body'):
                return 1
            if not node.body:
                return 1
            return 1 + max(depth(child) for child in ast.walk(node) if child != node)
        
        try:
            return min(depth(tree), 20)  # Cap at 20
        except:
            return 1
    
    def analyze_code_ml(self, code: str, file_path: str = "") -> Dict[str, Any]:
        """Perform comprehensive ML analysis on code"""
        logger.info(f"🤖 Analyzing code with ML models: {file_path}")
        
        try:
            # Extract features
            features = self.extract_code_features(code, file_path)
            
            # Scale features if scaler is available
            if self.scaler is not None:
                features_scaled = self.scaler.transform(features)
            else:
                features_scaled = features
            
            results = {
                'features': {name: float(val) for name, val in zip(self.feature_names, features[0])},
                'predictions': {},
                'confidence_scores': {},
                'ensemble_prediction': None,
                'risk_assessment': {},
                'recommendations': []
            }
            
            # Traditional ML predictions
            for name, model in self.traditional_models.items():
                try:
                    if name == 'IsolationForest':
                        # Anomaly detection
                        anomaly_score = model.decision_function(features_scaled)[0]
                        is_anomaly = model.predict(features_scaled)[0] == -1
                        results['predictions'][name] = {
                            'anomaly_score': float(anomaly_score),
                            'is_anomaly': bool(is_anomaly)
                        }
                    else:
                        # Classification
                        prediction = model.predict(features_scaled)[0]
                        if hasattr(model, 'predict_proba'):
                            proba = model.predict_proba(features_scaled)[0]
                            confidence = float(max(proba))
                        else:
                            confidence = 0.8  # Default confidence
                        
                        results['predictions'][name] = {
                            'prediction': int(prediction),
                            'confidence': confidence
                        }
                except Exception as e:
                    logger.error(f"Error with {name}: {e}")
            
            # Neural network predictions
            features_tensor = torch.FloatTensor(features_scaled)
            
            for name, model in self.neural_models.items():
                try:
                    with torch.no_grad():
                        if name == 'SecurityDetector':
                            output = model(features_tensor)
                            vulnerability_score = float(output.squeeze())
                            results['predictions'][f'Neural_{name}'] = {
                                'vulnerability_score': vulnerability_score,
                                'is_vulnerable': vulnerability_score > 0.5
                            }
                        elif name == 'QualityPredictor':
                            output = model(features_tensor)
                            quality_scores = torch.softmax(output, dim=1).squeeze()
                            quality_class = int(torch.argmax(quality_scores))
                            results['predictions'][f'Neural_{name}'] = {
                                'quality_class': quality_class,
                                'quality_scores': quality_scores.tolist()
                            }
                except Exception as e:
                    logger.error(f"Error with Neural {name}: {e}")
            
            # Ensemble prediction
            if self.ensemble_data and len(results['predictions']) >= 2:
                ensemble_score = self._calculate_ensemble_prediction(results['predictions'])
                results['ensemble_prediction'] = {
                    'vulnerability_score': ensemble_score,
                    'is_vulnerable': ensemble_score > 0.5,
                    'confidence': abs(ensemble_score - 0.5) * 2  # Convert to 0-1 scale
                }
            
            # Risk assessment
            results['risk_assessment'] = self._assess_risk(results, features[0])
            
            # Generate recommendations
            results['recommendations'] = self._generate_recommendations(results, code, file_path)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in ML analysis: {e}")
            return self._get_fallback_analysis()
    
    def _calculate_ensemble_prediction(self, predictions: Dict) -> float:
        """Calculate ensemble prediction from all models"""
        weights = {
            'RandomForest': 0.2,
            'GradientBoosting': 0.2,
            'XGBoost': 0.2,
            'LightGBM': 0.15,
            'LogisticRegression': 0.1,
            'SVM': 0.05,
            'MLP': 0.05,
            'Neural_SecurityDetector': 0.15
        }
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for model_name, weight in weights.items():
            if model_name in predictions:
                if 'prediction' in predictions[model_name]:
                    weighted_sum += weight * predictions[model_name]['prediction']
                    total_weight += weight
                elif 'vulnerability_score' in predictions[model_name]:
                    weighted_sum += weight * predictions[model_name]['vulnerability_score']
                    total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.5
    
    def _assess_risk(self, results: Dict, features: np.ndarray) -> Dict[str, Any]:
        """Assess overall risk based on ML predictions and features"""
        risk_factors = []
        risk_score = 0.0
        
        # Check ensemble prediction
        if results.get('ensemble_prediction'):
            if results['ensemble_prediction']['is_vulnerable']:
                risk_score += 0.4
                risk_factors.append("High vulnerability probability detected")
        
        # Check individual model agreements
        vuln_predictions = [
            pred.get('prediction', 0) for pred in results['predictions'].values()
            if 'prediction' in pred
        ]
        if vuln_predictions and sum(vuln_predictions) / len(vuln_predictions) > 0.6:
            risk_score += 0.2
            risk_factors.append("Multiple models predict vulnerability")
        
        # Check features
        if features[6] > 0:  # security_risk feature
            risk_score += 0.2
            risk_factors.append("Security-sensitive patterns detected")
        
        if features[1] > 20:  # high complexity
            risk_score += 0.1
            risk_factors.append("High code complexity")
        
        if features[2] > 10:  # deep nesting
            risk_score += 0.1
            risk_factors.append("Deep nesting levels")
        
        # Determine risk level
        if risk_score >= 0.7:
            risk_level = "HIGH"
        elif risk_score >= 0.4:
            risk_level = "MEDIUM"
        elif risk_score >= 0.2:
            risk_level = "LOW"
        else:
            risk_level = "MINIMAL"
        
        return {
            'risk_score': min(risk_score, 1.0),
            'risk_level': risk_level,
            'risk_factors': risk_factors
        }
    
    def _generate_recommendations(self, results: Dict, code: str, file_path: str) -> List[str]:
        """Generate actionable recommendations based on analysis"""
        recommendations = []
        
        # Based on risk assessment
        risk_level = results.get('risk_assessment', {}).get('risk_level', 'MINIMAL')
        
        if risk_level in ['HIGH', 'MEDIUM']:
            recommendations.append("🔒 Conduct security review for potential vulnerabilities")
            recommendations.append("🧪 Add comprehensive unit tests")
            recommendations.append("🔍 Consider static analysis tools")
        
        # Based on features
        features = results.get('features', {})
        
        if features.get('complexity', 0) > 20:
            recommendations.append("📦 Consider breaking down complex functions")
            recommendations.append("♻️ Refactor to reduce cyclomatic complexity")
        
        if features.get('nesting', 0) > 5:
            recommendations.append("🎯 Reduce nesting depth for better readability")
            recommendations.append("🔄 Extract nested logic into separate functions")
        
        if features.get('security_risk', 0) > 0:
            recommendations.append("🛡️ Review security-sensitive operations")
            recommendations.append("✅ Validate all user inputs")
            recommendations.append("🔐 Use parameterized queries for database operations")
        
        if features.get('lines', 0) > 500:
            recommendations.append("📝 Consider splitting large files into smaller modules")
        
        # Neural network specific recommendations
        if 'Neural_QualityPredictor' in results['predictions']:
            quality_class = results['predictions']['Neural_QualityPredictor'].get('quality_class', 2)
            if quality_class < 2:
                recommendations.append("📈 Code quality could be improved")
                recommendations.append("📚 Follow coding best practices and style guides")
        
        return recommendations[:5]  # Limit to top 5 recommendations
    
    def _get_fallback_analysis(self) -> Dict[str, Any]:
        """Provide fallback analysis when ML fails"""
        return {
            'features': {name: 0 for name in self.feature_names},
            'predictions': {},
            'confidence_scores': {},
            'ensemble_prediction': None,
            'risk_assessment': {
                'risk_score': 0.0,
                'risk_level': 'UNKNOWN',
                'risk_factors': ['ML analysis unavailable']
            },
            'recommendations': ['🔧 Manual code review recommended'],
            'error': 'ML analysis failed - using fallback'
        }
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get status of all loaded models"""
        return {
            'traditional_models_loaded': len(self.traditional_models),
            'neural_models_loaded': len(self.neural_models),
            'scaler_loaded': self.scaler is not None,
            'ensemble_loaded': self.ensemble_data is not None,
            'metadata_loaded': self.metadata is not None,
            'traditional_models': list(self.traditional_models.keys()),
            'neural_models': list(self.neural_models.keys()),
            'total_models': len(self.traditional_models) + len(self.neural_models)
        }

# Global instance
_ml_analyzer = None

def get_ml_analyzer() -> ProductionMLAnalyzer:
    """Get or create global ML analyzer instance"""
    global _ml_analyzer
    if _ml_analyzer is None:
        _ml_analyzer = ProductionMLAnalyzer()
    return _ml_analyzer

def analyze_code_with_ml(code: str, file_path: str = "") -> Dict[str, Any]:
    """Convenience function for ML code analysis"""
    analyzer = get_ml_analyzer()
    return analyzer.analyze_code_ml(code, file_path)
