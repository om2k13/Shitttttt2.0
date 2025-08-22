import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime, timedelta
import pickle
import os
import warnings
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import joblib
import xgboost as xgb
import lightgbm as lgb
from .neural_analyzer import NeuralAnalyzer

warnings.filterwarnings('ignore')

class EnhancedMLAnalyzer:
    """Enhanced ML analyzer with neural networks and traditional ML for comprehensive code analysis"""
    
    def __init__(self, model_dir: Path = None):
        self.model_dir = model_dir or Path("./ml_models")
        self.model_dir.mkdir(exist_ok=True)
        
        # Initialize neural analyzer
        self.neural_analyzer = NeuralAnalyzer(self.model_dir)
        
        # Traditional ML models
        self.severity_classifier = None
        self.false_positive_detector = None
        self.code_suggestion_model = None
        self.risk_predictor = None
        self.anomaly_detector = None
        self.priority_classifier = None
        
        # Feature extractors and preprocessors
        self.text_vectorizer = TfidfVectorizer(max_features=2000, stop_words='english', ngram_range=(1, 3))
        self.code_vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 4))
        self.scaler = RobustScaler()
        self.label_encoder = LabelEncoder()
        
        # Advanced ML models
        self.xgb_classifier = None
        self.lgb_classifier = None
        self.svm_classifier = None
        self.mlp_classifier = None
        
        # Load or initialize models
        self._load_or_initialize_models()
        
    def _load_or_initialize_models(self):
        """Load existing models or initialize new ones with advanced configurations"""
        try:
            # Load traditional ML models
            self._load_traditional_models()
            
            # Load advanced ML models
            self._load_advanced_models()
            
        except Exception as e:
            print(f"Warning: Could not load ML models: {e}")
            self._initialize_and_train_all_models()
    
    def _load_traditional_models(self):
        """Load traditional ML models"""
        try:
            # Severity classifier
            severity_path = self.model_dir / "severity_classifier.pkl"
            if severity_path.exists():
                self.severity_classifier = joblib.load(severity_path)
            else:
                self.severity_classifier = RandomForestClassifier(
                    n_estimators=200, 
                    max_depth=15, 
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1
                )
            
            # False positive detector
            fp_path = self.model_dir / "false_positive_detector.pkl"
            if fp_path.exists():
                self.false_positive_detector = joblib.load(fp_path)
            else:
                self.false_positive_detector = GradientBoostingClassifier(
                    n_estimators=150,
                    learning_rate=0.1,
                    max_depth=8,
                    random_state=42
                )
            
            # Risk predictor
            risk_path = self.model_dir / "risk_predictor.pkl"
            if risk_path.exists():
                self.risk_predictor = joblib.load(risk_path)
            else:
                self.risk_predictor = RandomForestClassifier(
                    n_estimators=300,
                    max_depth=20,
                    min_samples_split=3,
                    random_state=42,
                    n_jobs=-1
                )
                
            # Anomaly detector
            anomaly_path = self.model_dir / "anomaly_detector.pkl"
            if anomaly_path.exists():
                self.anomaly_detector = joblib.load(anomaly_path)
            else:
                self.anomaly_detector = IsolationForest(
                    contamination=0.1,
                    random_state=42,
                    n_estimators=200
                )
                
            # Priority classifier
            priority_path = self.model_dir / "priority_classifier.pkl"
            if priority_path.exists():
                self.priority_classifier = joblib.load(priority_path)
            else:
                self.priority_classifier = LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                    solver='liblinear'
                )
                
        except Exception as e:
            print(f"Error loading traditional models: {e}")
            self._initialize_traditional_models()
    
    def _load_advanced_models(self):
        """Load advanced ML models"""
        try:
            # XGBoost classifier
            xgb_path = self.model_dir / "xgb_classifier.pkl"
            if xgb_path.exists():
                self.xgb_classifier = joblib.load(xgb_path)
            else:
                self.xgb_classifier = xgb.XGBClassifier(
                    n_estimators=200,
                    max_depth=8,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=-1
                )
            
            # LightGBM classifier
            lgb_path = self.model_dir / "lgb_classifier.pkl"
            if lgb_path.exists():
                self.lgb_classifier = joblib.load(lgb_path)
            else:
                self.lgb_classifier = lgb.LGBMClassifier(
                    n_estimators=200,
                    max_depth=8,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=-1
                )
            
            # SVM classifier
            svm_path = self.model_dir / "svm_classifier.pkl"
            if svm_path.exists():
                self.svm_classifier = joblib.load(svm_path)
            else:
                self.svm_classifier = SVC(
                    kernel='rbf',
                    C=1.0,
                    gamma='scale',
                    probability=True,
                    random_state=42
                )
            
            # MLP classifier
            mlp_path = self.model_dir / "mlp_classifier.pkl"
            if mlp_path.exists():
                self.mlp_classifier = joblib.load(mlp_path)
            else:
                self.mlp_classifier = MLPClassifier(
                    hidden_layer_sizes=(256, 128, 64),
                    activation='relu',
                    solver='adam',
                    alpha=0.001,
                    learning_rate='adaptive',
                    max_iter=500,
                    random_state=42
                )
                
        except Exception as e:
            print(f"Error loading advanced models: {e}")
            self._initialize_advanced_models()
    
    def _initialize_and_train_all_models(self):
        """Initialize and train all models"""
        print("🤖 Initializing and training all ML models...")
        
        self._initialize_traditional_models()
        self._initialize_advanced_models()
        
        # Train all models
        self._train_all_models()
    
    def _initialize_traditional_models(self):
        """Initialize traditional ML models"""
        self.severity_classifier = RandomForestClassifier(
            n_estimators=200, 
            max_depth=15, 
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        self.false_positive_detector = GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=8,
            random_state=42
        )
        
        self.risk_predictor = RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            min_samples_split=3,
            random_state=42,
            n_jobs=-1
        )
        
        self.anomaly_detector = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_estimators=200
        )
        
        self.priority_classifier = LogisticRegression(
            max_iter=1000,
            random_state=42,
            solver='liblinear'
        )
    
    def _initialize_advanced_models(self):
        """Initialize advanced ML models"""
        self.xgb_classifier = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        
        self.lgb_classifier = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        
        self.svm_classifier = SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            probability=True,
            random_state=42
        )
        
        self.mlp_classifier = MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation='relu',
            solver='adam',
            alpha=0.001,
            learning_rate='adaptive',
            max_iter=500,
            random_state=42
        )
    
    def _train_all_models(self):
        """Train all ML models with comprehensive data"""
        print("🔄 Training all ML models...")
        
        # Generate comprehensive training data
        training_data = self._generate_comprehensive_training_data()
        
        if len(training_data) > 0:
            # Extract features and labels
            X, y_severity, y_false_positive, y_risk, y_priority = zip(*training_data)
            X = np.array(X)
            
            # Split data
            X_train, X_test, y_severity_train, y_severity_test = train_test_split(
                X, y_severity, test_size=0.2, random_state=42
            )
            
            print(f"Training with {len(X_train)} samples, testing with {len(X_test)} samples")
            
            # Train severity classifier
            print("Training severity classifier...")
            self.severity_classifier.fit(X_train, y_severity_train)
            
            # Train false positive detector
            print("Training false positive detector...")
            self.false_positive_detector.fit(X_train, y_false_positive)
            
            # Train risk predictor
            print("Training risk predictor...")
            self.risk_predictor.fit(X_train, y_risk)
            
            # Train priority classifier
            print("Training priority classifier...")
            self.priority_classifier.fit(X_train, y_priority)
            
            # Train anomaly detector
            print("Training anomaly detector...")
            self.anomaly_detector.fit(X_train)
            
            # Train advanced models
            self._train_advanced_models(X_train, y_severity_train)
            
            # Save all models
            self._save_all_models()
            
            # Evaluate models
            self._evaluate_models(X_test, y_severity_test)
            
            print("✅ All ML models trained successfully!")
        else:
            print("❌ No training data generated!")
            # Generate fallback training data
            self._generate_fallback_training_data()
    
    def _train_advanced_models(self, X_train, y_train):
        """Train advanced ML models"""
        print("Training advanced ML models...")
        
        # Train XGBoost
        self.xgb_classifier.fit(X_train, y_train)
        
        # Train LightGBM
        self.lgb_classifier.fit(X_train, y_train)
        
        # Train SVM (with smaller sample for efficiency)
        if len(X_train) > 1000:
            sample_indices = np.random.choice(len(X_train), 1000, replace=False)
            X_sample = X_train[sample_indices]
            y_sample = y_train[sample_indices]
        else:
            X_sample, y_sample = X_train, y_train
        
        self.svm_classifier.fit(X_sample, y_sample)
        
        # Train MLP
        self.mlp_classifier.fit(X_train, y_train)
    
    def _generate_comprehensive_training_data(self) -> List[Tuple[List[float], str, bool, str, str]]:
        """Generate comprehensive training data for all models"""
        training_data = []
        
        # Real-world code patterns and their characteristics
        patterns = [
            # High severity, high risk, high priority
            ([0.9, 0.8, 0.7, 0.9, 0.8, 0.9, 0.8, 0.9], "high", False, "high", "critical"),
            
            # Medium severity, medium risk, medium priority
            ([0.5, 0.6, 0.5, 0.4, 0.5, 0.6, 0.5, 0.4], "medium", False, "medium", "medium"),
            
            # Low severity, low risk, low priority
            ([0.2, 0.3, 0.2, 0.1, 0.2, 0.3, 0.2, 0.1], "low", True, "low", "low"),
            
            # False positive patterns
            ([0.1, 0.2, 0.1, 0.1, 0.1, 0.2, 0.1, 0.1], "low", True, "low", "low"),
        ]
        
        training_data.extend(patterns)
        
        # Generate synthetic data with realistic distributions
        for i in range(500):
            # Generate features with realistic correlations
            base_risk = np.random.beta(2, 3)
            features = [
                base_risk + np.random.normal(0, 0.1),  # Severity
                base_risk + np.random.normal(0, 0.1),  # Risk
                np.random.beta(1, 2),  # Complexity
                np.random.beta(1, 3),  # Maintainability
                np.random.beta(2, 2),  # Reliability
                base_risk + np.random.normal(0, 0.1),  # Security
                np.random.beta(1, 2),  # Performance
                np.random.beta(1, 3),  # Readability
            ]
            
            # Normalize features
            features = [min(1.0, max(0.0, f)) for f in features]
            
            # Determine labels based on features
            avg_severity = (features[0] + features[1] + features[5]) / 3
            severity = "high" if avg_severity > 0.7 else "medium" if avg_severity > 0.4 else "low"
            
            false_positive = avg_severity < 0.3 and np.random.random() < 0.8
            
            risk = "high" if avg_severity > 0.7 else "medium" if avg_severity > 0.4 else "low"
            
            priority = "critical" if avg_severity > 0.8 else "high" if avg_severity > 0.6 else "medium" if avg_severity > 0.4 else "low"
            
            training_data.append((features, severity, false_positive, risk, priority))
        
        return training_data
    
    def _generate_fallback_training_data(self):
        """Generate fallback training data if main method fails"""
        print("🔄 Generating fallback training data...")
        
        # Simple but effective training data
        X = []
        y_severity = []
        y_false_positive = []
        y_risk = []
        y_priority = []
        
        # Generate 100 simple training samples
        for i in range(100):
            # Simple features: [severity_score, complexity, security_risk, quality_score]
            features = [
                np.random.random(),  # severity
                np.random.random(),  # complexity
                np.random.random(),  # security risk
                np.random.random()   # quality
            ]
            X.append(features)
            
            # Labels based on features
            avg_severity = (features[0] + features[2]) / 2
            severity = "high" if avg_severity > 0.7 else "medium" if avg_severity > 0.4 else "low"
            y_severity.append(severity)
            
            y_false_positive.append(avg_severity < 0.3)
            y_risk.append("high" if avg_severity > 0.7 else "medium" if avg_severity > 0.4 else "low")
            y_priority.append("critical" if avg_severity > 0.8 else "high" if avg_severity > 0.6 else "medium" if avg_severity > 0.4 else "low")
        
        X = np.array(X)
        
        # Train models with fallback data
        print("Training models with fallback data...")
        
        # Train severity classifier
        self.severity_classifier.fit(X, y_severity)
        
        # Train false positive detector
        self.false_positive_detector.fit(X, y_false_positive)
        
        # Train risk predictor
        self.risk_predictor.fit(X, y_risk)
        
        # Train priority classifier
        self.priority_classifier.fit(X, y_priority)
        
        # Train anomaly detector
        self.anomaly_detector.fit(X)
        
        print("✅ Fallback models trained successfully!")
    
    def _save_all_models(self):
        """Save all trained models"""
        try:
            # Save traditional models
            joblib.dump(self.severity_classifier, self.model_dir / "severity_classifier.pkl")
            joblib.dump(self.false_positive_detector, self.model_dir / "false_positive_detector.pkl")
            joblib.dump(self.risk_predictor, self.model_dir / "risk_predictor.pkl")
            joblib.dump(self.anomaly_detector, self.model_dir / "anomaly_detector.pkl")
            joblib.dump(self.priority_classifier, self.model_dir / "priority_classifier.pkl")
            
            # Save advanced models
            joblib.dump(self.xgb_classifier, self.model_dir / "xgb_classifier.pkl")
            joblib.dump(self.lgb_classifier, self.model_dir / "lgb_classifier.pkl")
            joblib.dump(self.svm_classifier, self.model_dir / "svm_classifier.pkl")
            joblib.dump(self.mlp_classifier, self.model_dir / "mlp_classifier.pkl")
            
            print("✅ All models saved successfully")
            
        except Exception as e:
            print(f"Warning: Could not save some models: {e}")
    
    def _evaluate_models(self, X_test, y_test):
        """Evaluate model performance"""
        try:
            # Evaluate severity classifier
            y_pred = self.severity_classifier.predict(X_test)
            accuracy = np.mean(y_pred == y_test)
            print(f"Severity Classifier Accuracy: {accuracy:.4f}")
            
            # Cross-validation score
            cv_score = cross_val_score(self.severity_classifier, X_test, y_test, cv=5)
            print(f"Cross-validation score: {cv_score.mean():.4f} (+/- {cv_score.std() * 2:.4f})")
            
        except Exception as e:
            print(f"Model evaluation failed: {e}")
    
    def extract_enhanced_features(self, finding: Dict) -> np.ndarray:
        """Extract enhanced features from a finding for ML analysis"""
        features = []
        
        # Text-based features
        text_features = self._extract_text_features(finding)
        features.extend(text_features)
        
        # Code-based features
        code_features = self._extract_code_features(finding)
        features.extend(code_features)
        
        # Contextual features
        context_features = self._extract_context_features(finding)
        features.extend(context_features)
        
        # Tool-specific features
        tool_features = self._extract_tool_features(finding)
        features.extend(tool_features)
        
        # Neural network features
        neural_features = self._extract_neural_features(finding)
        features.extend(neural_features)
        
        return np.array(features)
    
    def _extract_text_features(self, finding: Dict) -> List[float]:
        """Extract enhanced text-based features"""
        text = f"{finding.get('message', '')} {finding.get('suggestion', '')}"
        
        # Basic text statistics
        features = [
            len(text) / 1000.0,  # Normalized length
            text.count('error') / max(1, len(text.split())),
            text.count('warning') / max(1, len(text.split())),
            text.count('critical') / max(1, len(text.split())),
            text.count('security') / max(1, len(text.split())),
            text.count('performance') / max(1, len(text.split())),
            text.count('bug') / max(1, len(text.split())),
            text.count('fix') / max(1, len(text.split())),
        ]
        
        return features
    
    def _extract_code_features(self, finding: Dict) -> List[float]:
        """Extract enhanced code-based features"""
        code_snippet = finding.get('code_snippet', '')
        
        features = [
            len(code_snippet) / 1000.0,  # Normalized length
            code_snippet.count(';') / max(1, len(code_snippet.split('\n'))),
            code_snippet.count('if') / max(1, len(code_snippet.split('\n'))),
            code_snippet.count('for') / max(1, len(code_snippet.split('\n'))),
            code_snippet.count('while') / max(1, len(code_snippet.split('\n'))),
            code_snippet.count('try') / max(1, len(code_snippet.split('\n'))),
            code_snippet.count('except') / max(1, len(code_snippet.split('\n'))),
            code_snippet.count('def') / max(1, len(code_snippet.split('\n'))),
            code_snippet.count('class') / max(1, len(code_snippet.split('\n'))),
        ]
        
        return features
    
    def _extract_context_features(self, finding: Dict) -> List[float]:
        """Extract contextual features"""
        features = [
            finding.get('confidence', 0.5),
            finding.get('severity_score', 0.5),
            finding.get('impact_score', 0.5),
            finding.get('effort_score', 0.5),
        ]
        
        return features
    
    def _extract_tool_features(self, finding: Dict) -> List[float]:
        """Extract tool-specific features"""
        tool = finding.get('tool', 'unknown')
        
        # Tool confidence scores
        tool_scores = {
            'ruff': 0.9,
            'bandit': 0.85,
            'radon': 0.8,
            'eslint': 0.9,
            'semgrep': 0.88,
            'unknown': 0.5
        }
        
        features = [
            tool_scores.get(tool, 0.5),
            finding.get('line_number', 0) / 1000.0,  # Normalized line number
            finding.get('file_size', 0) / 10000.0,   # Normalized file size
        ]
        
        return features
    
    def _extract_neural_features(self, finding: Dict) -> List[float]:
        """Extract features using neural networks"""
        try:
            # Use neural analyzer for additional features
            code_snippet = finding.get('code_snippet', '')
            context = {'file': finding.get('file', ''), 'line': finding.get('line', 0)}
            
            # Security analysis
            security_result = self.neural_analyzer.analyze_code_security(code_snippet, context)
            security_features = [
                security_result.get('risk_score', 0.5),
                security_result.get('confidence', 0.5),
            ]
            
            # Quality prediction
            quality_metrics = {
                'lines': len(code_snippet.split('\n')),
                'complexity': finding.get('complexity', 5),
                'nesting': finding.get('nesting', 3),
                'imports': finding.get('imports', 5),
                'functions': finding.get('functions', 3),
                'classes': finding.get('classes', 2),
            }
            
            quality_result = self.neural_analyzer.predict_code_quality(quality_metrics)
            quality_features = [
                quality_result.get('quality_score', 0.5),
                quality_result.get('predicted_maintainability', 0.5),
                quality_result.get('predicted_reliability', 0.5),
            ]
            
            return security_features + quality_features
            
        except Exception as e:
            print(f"Neural feature extraction failed: {e}")
            return [0.5, 0.5, 0.5, 0.5, 0.5]  # Fallback features
    
    def analyze_finding_with_ml(self, finding: Dict) -> Dict[str, Any]:
        """Analyze a finding using all ML models"""
        try:
            # Extract features
            features = self.extract_enhanced_features(finding)
            features_reshaped = features.reshape(1, -1)
            
            # Get predictions from all models
            results = {}
            
            # Traditional ML predictions
            if self.severity_classifier and hasattr(self.severity_classifier, 'estimators_'):
                try:
                    results['predicted_severity'] = self.severity_classifier.predict(features_reshaped)[0]
                    results['severity_confidence'] = np.max(self.severity_classifier.predict_proba(features_reshaped))
                except Exception as e:
                    print(f"Severity classifier failed: {e}")
            
            if self.false_positive_detector and hasattr(self.false_positive_detector, 'estimators_'):
                try:
                    results['is_false_positive'] = self.false_positive_detector.predict(features_reshaped)[0]
                    results['false_positive_confidence'] = np.max(self.false_positive_detector.predict_proba(features_reshaped))
                except Exception as e:
                    print(f"False positive detector failed: {e}")
            
            if self.risk_predictor and hasattr(self.risk_predictor, 'estimators_'):
                try:
                    results['predicted_risk'] = self.risk_predictor.predict(features_reshaped)[0]
                    results['risk_confidence'] = np.max(self.risk_predictor.predict_proba(features_reshaped))
                except Exception as e:
                    print(f"Risk predictor failed: {e}")
            
            if self.priority_classifier and hasattr(self.priority_classifier, 'coef_'):
                try:
                    results['predicted_priority'] = self.priority_classifier.predict(features_reshaped)[0]
                    results['priority_confidence'] = np.max(self.priority_classifier.predict_proba(features_reshaped))
                except Exception as e:
                    print(f"Priority classifier failed: {e}")
            
            if self.anomaly_detector and hasattr(self.anomaly_detector, 'estimators_'):
                try:
                    results['anomaly_score'] = self.anomaly_detector.decision_function(features_reshaped)[0]
                    results['is_anomaly'] = results['anomaly_score'] < -0.1
                except Exception as e:
                    print(f"Anomaly detector failed: {e}")
            
            # Advanced ML predictions
            if self.xgb_classifier and hasattr(self.xgb_classifier, 'estimators_'):
                try:
                    results['xgb_severity'] = self.xgb_classifier.predict(features_reshaped)[0]
                    results['xgb_confidence'] = np.max(self.xgb_classifier.predict_proba(features_reshaped))
                except Exception as e:
                    print(f"XGBoost classifier failed: {e}")
            
            if self.lgb_classifier and hasattr(self.lgb_classifier, 'estimators_'):
                try:
                    results['lgb_severity'] = self.lgb_classifier.predict(features_reshaped)[0]
                    results['lgb_confidence'] = np.max(self.lgb_classifier.predict_proba(features_reshaped))
                except Exception as e:
                    print(f"LightGBM classifier failed: {e}")
            
            # Ensemble prediction
            results['ensemble_severity'] = self._get_ensemble_prediction(features_reshaped)
            results['ensemble_confidence'] = self._get_ensemble_confidence(features_reshaped)
            
            # Enhanced analysis
            results['enhanced_analysis'] = self._generate_enhanced_analysis(results)
            results['ml_recommendations'] = self._generate_ml_recommendations(results)
            
            return results
            
        except Exception as e:
            print(f"ML analysis failed: {e}")
            return {
                'error': str(e),
                'fallback_analysis': True,
                'predicted_severity': 'medium',
                'confidence': 0.5
            }
    
    def _get_ensemble_prediction(self, features: np.ndarray) -> str:
        """Get ensemble prediction from multiple models"""
        predictions = []
        weights = []
        
        if self.severity_classifier:
            pred = self.severity_classifier.predict(features)[0]
            predictions.append(pred)
            weights.append(0.3)
        
        if self.xgb_classifier:
            pred = self.xgb_classifier.predict(features)[0]
            predictions.append(pred)
            weights.append(0.25)
        
        if self.lgb_classifier:
            pred = self.lgb_classifier.predict(features)[0]
            predictions.append(pred)
            weights.append(0.25)
        
        if self.mlp_classifier:
            pred = self.mlp_classifier.predict(features)[0]
            predictions.append(pred)
            weights.append(0.2)
        
        if not predictions:
            return 'medium'
        
        # Weighted voting
        severity_scores = {'low': 0, 'medium': 1, 'high': 2}
        weighted_sum = sum(severity_scores[pred] * weight for pred, weight in zip(predictions, weights))
        total_weight = sum(weights)
        
        if total_weight == 0:
            return 'medium'
        
        avg_score = weighted_sum / total_weight
        
        if avg_score < 0.7:
            return 'low'
        elif avg_score < 1.3:
            return 'medium'
        else:
            return 'high'
    
    def _get_ensemble_confidence(self, features: np.ndarray) -> float:
        """Get ensemble confidence score"""
        confidences = []
        weights = []
        
        if self.severity_classifier:
            conf = np.max(self.severity_classifier.predict_proba(features))
            confidences.append(conf)
            weights.append(0.3)
        
        if self.xgb_classifier:
            conf = np.max(self.xgb_classifier.predict_proba(features))
            confidences.append(conf)
            weights.append(0.25)
        
        if self.lgb_classifier:
            conf = np.max(self.lgb_classifier.predict_proba(features))
            confidences.append(conf)
            weights.append(0.25)
        
        if self.mlp_classifier:
            conf = np.max(self.mlp_classifier.predict_proba(features))
            confidences.append(conf)
            weights.append(0.2)
        
        if not confidences:
            return 0.5
        
        # Weighted average confidence
        weighted_sum = sum(conf * weight for conf, weight in zip(confidences, weights))
        total_weight = sum(weights)
        
        return weighted_sum / total_weight if total_weight > 0 else 0.5
    
    def _generate_enhanced_analysis(self, results: Dict) -> str:
        """Generate enhanced analysis based on ML results"""
        analysis_parts = []
        
        # Severity analysis
        if 'predicted_severity' in results:
            severity = results['predicted_severity']
            confidence = results.get('severity_confidence', 0.5)
            analysis_parts.append(f"ML predicts {severity} severity with {confidence:.2f} confidence")
        
        # False positive analysis
        if 'is_false_positive' in results:
            if results['is_false_positive']:
                analysis_parts.append("High probability of false positive")
            else:
                analysis_parts.append("Low false positive probability")
        
        # Risk analysis
        if 'predicted_risk' in results:
            risk = results['predicted_risk']
            analysis_parts.append(f"Risk level: {risk}")
        
        # Anomaly detection
        if 'is_anomaly' in results and results['is_anomaly']:
            analysis_parts.append("Anomaly detected - unusual pattern")
        
        # Ensemble analysis
        if 'ensemble_severity' in results:
            ensemble = results['ensemble_severity']
            ensemble_conf = results.get('ensemble_confidence', 0.5)
            analysis_parts.append(f"Ensemble prediction: {ensemble} severity ({ensemble_conf:.2f} confidence)")
        
        return ". ".join(analysis_parts) if analysis_parts else "ML analysis completed"
    
    def _generate_ml_recommendations(self, results: Dict) -> List[str]:
        """Generate ML-based recommendations"""
        recommendations = []
        
        # Severity-based recommendations
        severity = results.get('predicted_severity', 'medium')
        if severity == 'high':
            recommendations.append("Immediate attention required - high severity issue")
            recommendations.append("Consider code review and testing before deployment")
        elif severity == 'medium':
            recommendations.append("Address in next development cycle")
            recommendations.append("Monitor for impact on system performance")
        
        # False positive recommendations
        if results.get('is_false_positive', False):
            recommendations.append("Verify this finding manually - high false positive probability")
            recommendations.append("Consider adjusting tool configuration")
        
        # Risk-based recommendations
        risk = results.get('predicted_risk', 'medium')
        if risk == 'high':
            recommendations.append("Implement additional security measures")
            recommendations.append("Add comprehensive testing for this component")
        
        # Anomaly recommendations
        if results.get('is_anomaly', False):
            recommendations.append("Investigate unusual patterns in this code")
            recommendations.append("Consider peer review for anomaly verification")
        
        if not recommendations:
            recommendations.append("No specific ML recommendations at this time")
        
        return recommendations
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get comprehensive status of all ML models"""
        neural_status = self.neural_analyzer.get_model_status()
        
        return {
            "neural_models": neural_status,
            "traditional_ml_models": {
                "severity_classifier": self.severity_classifier is not None,
                "false_positive_detector": self.false_positive_detector is not None,
                "risk_predictor": self.risk_predictor is not None,
                "anomaly_detector": self.anomaly_detector is not None,
                "priority_classifier": self.priority_classifier is not None,
            },
            "advanced_ml_models": {
                "xgb_classifier": self.xgb_classifier is not None,
                "lgb_classifier": self.lgb_classifier is not None,
                "svm_classifier": self.svm_classifier is not None,
                "mlp_classifier": self.mlp_classifier is not None,
            },
            "total_models": 11,  # 3 neural + 5 traditional + 3 advanced
            "models_directory": str(self.model_dir),
            "feature_extractors": {
                "text_vectorizer": self.text_vectorizer is not None,
                "code_vectorizer": self.code_vectorizer is not None,
                "scaler": self.scaler is not None,
            }
        }
