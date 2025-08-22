import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import joblib

class MLAnalyzer:
    """Machine Learning-based code analysis for intelligent findings and false positive reduction"""
    
    def __init__(self, model_dir: Path = None):
        self.model_dir = model_dir or Path("./ml_models")
        self.model_dir.mkdir(exist_ok=True)
        
        # Initialize models
        self.severity_classifier = None
        self.false_positive_detector = None
        self.code_suggestion_model = None
        self.risk_predictor = None
        
        # Feature extractors
        self.text_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.code_vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 3))
        
        # Load or initialize models
        self._load_or_initialize_models()
    
    def _load_or_initialize_models(self):
        """Load existing models or initialize new ones"""
        try:
            # Load severity classifier
            severity_model_path = self.model_dir / "severity_classifier.pkl"
            if severity_model_path.exists():
                self.severity_classifier = joblib.load(severity_model_path)
            else:
                self.severity_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
            
            # Load false positive detector
            fp_model_path = self.model_dir / "false_positive_detector.pkl"
            if fp_model_path.exists():
                self.false_positive_detector = joblib.load(fp_model_path)
            else:
                self.false_positive_detector = RandomForestClassifier(n_estimators=100, random_state=42)
            
            # Load risk predictor
            risk_model_path = self.model_dir / "risk_predictor.pkl"
            if risk_model_path.exists():
                self.risk_predictor = joblib.load(risk_model_path)
            else:
                self.risk_predictor = RandomForestClassifier(n_estimators=100, random_state=42)
                
        except Exception as e:
            print(f"Warning: Could not load ML models: {e}")
            # Initialize with default models
            self.severity_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
            self.false_positive_detector = RandomForestClassifier(n_estimators=100, random_state=42)
            self.risk_predictor = RandomForestClassifier(n_estimators=100, random_state=42)
    
    def _save_models(self):
        """Save trained models to disk"""
        try:
            joblib.dump(self.severity_classifier, self.model_dir / "severity_classifier.pkl")
            joblib.dump(self.false_positive_detector, self.model_dir / "false_positive_detector.pkl")
            joblib.dump(self.risk_predictor, self.model_dir / "risk_predictor.pkl")
        except Exception as e:
            print(f"Warning: Could not save ML models: {e}")
    
    def extract_features(self, finding: Dict) -> np.ndarray:
        """Extract features from a finding for ML analysis"""
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
        
        return np.array(features)
    
    def _extract_text_features(self, finding: Dict) -> List[float]:
        """Extract text-based features from finding message and description"""
        text = f"{finding.get('message', '')} {finding.get('remediation', '')}"
        
        # Basic text statistics
        features = [
            len(text),  # Text length
            text.count('error'),  # Error keyword count
            text.count('warning'),  # Warning keyword count
            text.count('critical'),  # Critical keyword count
            text.count('vulnerability'),  # Vulnerability keyword count
            text.count('security'),  # Security keyword count
            text.count('performance'),  # Performance keyword count
            len(text.split()),  # Word count
            len(set(text.split())),  # Unique word count
        ]
        
        # TF-IDF features (if vectorizer is fitted)
        try:
            if hasattr(self.text_vectorizer, 'vocabulary_'):
                text_vector = self.text_vectorizer.transform([text]).toarray()[0]
                features.extend(text_vector[:50])  # First 50 features
            else:
                features.extend([0] * 50)
        except:
            features.extend([0] * 50)
        
        return features
    
    def _extract_code_features(self, finding: Dict) -> List[float]:
        """Extract code-based features from code snippets"""
        code_snippet = finding.get('code_snippet', '')
        
        if not code_snippet:
            return [0] * 60
        
        # Code complexity features
        features = [
            len(code_snippet),  # Code length
            code_snippet.count('\n'),  # Line count
            code_snippet.count(';'),  # Statement count
            code_snippet.count('('),  # Parenthesis count
            code_snippet.count('{'),  # Brace count
            code_snippet.count('['),  # Bracket count
            code_snippet.count('='),  # Assignment count
            code_snippet.count('+'),  # Addition count
            code_snippet.count('-'),  # Subtraction count
            code_snippet.count('*'),  # Multiplication count
            code_snippet.count('/'),  # Division count
            code_snippet.count('if'),  # Conditional count
            code_snippet.count('for'),  # Loop count
            code_snippet.count('while'),  # While loop count
            code_snippet.count('def'),  # Function definition count
            code_snippet.count('class'),  # Class definition count
            code_snippet.count('import'),  # Import count
            code_snippet.count('from'),  # From import count
            code_snippet.count('try'),  # Try block count
            code_snippet.count('except'),  # Exception count
        ]
        
        # TF-IDF features for code
        try:
            if hasattr(self.code_vectorizer, 'vocabulary_'):
                code_vector = self.code_vectorizer.transform([code_snippet]).toarray()[0]
                features.extend(code_vector[:40])  # First 40 features
            else:
                features.extend([0] * 40)
        except:
            features.extend([0] * 40)
        
        return features
    
    def _extract_context_features(self, finding: Dict) -> List[float]:
        """Extract contextual features from finding"""
        features = []
        
        # File type features
        file_path = finding.get('file', '')
        file_extension = Path(file_path).suffix.lower()
        
        # File type encoding
        file_types = ['.py', '.js', '.ts', '.java', '.cpp', '.c', '.go', '.rs']
        for ext in file_types:
            features.append(1.0 if file_extension == ext else 0.0)
        
        # Line number features
        line_number = finding.get('line', 0)
        features.extend([
            line_number,  # Absolute line number
            line_number / 1000 if line_number > 0 else 0,  # Normalized line number
        ])
        
        # Tool features
        tool = finding.get('tool', '')
        tools = ['ruff', 'mypy', 'bandit', 'semgrep', 'eslint', 'owasp-security']
        for t in tools:
            features.append(1.0 if tool == t else 0.0)
        
        # Rule ID features
        rule_id = finding.get('rule_id', '')
        features.extend([
            len(rule_id),  # Rule ID length
            1.0 if 'security' in rule_id.lower() else 0.0,
            1.0 if 'performance' in rule_id.lower() else 0.0,
            1.0 if 'style' in rule_id.lower() else 0.0,
        ])
        
        return features
    
    def _extract_tool_features(self, finding: Dict) -> List[float]:
        """Extract tool-specific features"""
        features = []
        
        # Autofixable feature
        features.append(1.0 if finding.get('autofixable', False) else 0.0)
        
        # Vulnerability type features
        vuln_type = finding.get('vulnerability_type', '')
        vuln_types = ['sql_injection', 'xss', 'command_injection', 'path_traversal', 
                     'hardcoded_secrets', 'weak_crypto', 'memory_leak', 'n_plus_one']
        
        for vt in vuln_types:
            features.append(1.0 if vt in vuln_type else 0.0)
        
        # PR context features
        pr_context = finding.get('pr_context', {})
        features.extend([
            1.0 if pr_context.get('file_status') == 'A' else 0.0,  # Added file
            1.0 if pr_context.get('file_status') == 'M' else 0.0,  # Modified file
            1.0 if pr_context.get('file_status') == 'D' else 0.0,  # Deleted file
        ])
        
        return features
    
    def predict_severity(self, finding: Dict) -> str:
        """Predict the severity of a finding using ML"""
        if not self.severity_classifier:
            return finding.get('severity', 'medium')
        
        try:
            features = self.extract_features(finding)
            features = features.reshape(1, -1)
            
            # Ensure features array has correct shape
            if features.shape[1] != self._get_expected_feature_count():
                return finding.get('severity', 'medium')
            
            prediction = self.severity_classifier.predict(features)[0]
            return prediction
        except Exception as e:
            print(f"Warning: Could not predict severity: {e}")
            return finding.get('severity', 'medium')
    
    def detect_false_positive(self, finding: Dict) -> float:
        """Detect if a finding is likely a false positive (0.0 = likely true, 1.0 = likely false)"""
        if not self.false_positive_detector:
            return 0.0  # Default to not false positive
        
        try:
            features = self.extract_features(finding)
            features = features.reshape(1, -1)
            
            if features.shape[1] != self._get_expected_feature_count():
                return 0.0
            
            # Get probability of being false positive
            proba = self.false_positive_detector.predict_proba(features)[0]
            # Assuming class 1 is false positive
            return proba[1] if len(proba) > 1 else 0.0
        except Exception as e:
            print(f"Warning: Could not detect false positive: {e}")
            return 0.0
    
    def predict_risk_score(self, finding: Dict) -> int:
        """Predict risk score (0-10) for a finding"""
        if not self.risk_predictor:
            return self._calculate_basic_risk_score(finding)
        
        try:
            features = self.extract_features(finding)
            features = features.reshape(1, -1)
            
            if features.shape[1] != self._get_expected_feature_count():
                return self._calculate_basic_risk_score(finding)
            
            # Predict risk score (0-10)
            prediction = self.risk_predictor.predict(features)[0]
            return max(0, min(10, int(prediction)))
        except Exception as e:
            print(f"Warning: Could not predict risk score: {e}")
            return self._calculate_basic_risk_score(finding)
    
    def _calculate_basic_risk_score(self, finding: Dict) -> int:
        """Calculate basic risk score without ML"""
        severity = finding.get('severity', 'medium')
        tool = finding.get('tool', '')
        vuln_type = finding.get('vulnerability_type', '')
        
        # Base score by severity
        base_scores = {
            'critical': 9,
            'high': 7,
            'medium': 5,
            'low': 3
        }
        
        score = base_scores.get(severity, 5)
        
        # Adjust for security tools
        if 'security' in tool.lower() or 'owasp' in tool.lower():
            score += 1
        
        # Adjust for critical vulnerability types
        critical_vulns = ['sql_injection', 'command_injection', 'hardcoded_secrets']
        if any(vt in vuln_type.lower() for vt in critical_vulns):
            score += 1
        
        return min(10, score)
    
    def _get_expected_feature_count(self) -> int:
        """Get the expected number of features for the models"""
        # This should match the total features from all extractors
        return 50 + 60 + 20 + 20  # text + code + context + tool features
    
    def generate_intelligent_suggestions(self, finding: Dict) -> List[str]:
        """Generate intelligent code suggestions based on ML analysis"""
        suggestions = []
        
        # Get false positive probability
        fp_prob = self.detect_false_positive(finding)
        
        # If likely false positive, suggest verification
        if fp_prob > 0.7:
            suggestions.append("⚠️ This finding has a high probability of being a false positive. Please verify manually.")
        
        # Get predicted severity vs actual
        predicted_severity = self.predict_severity(finding)
        actual_severity = finding.get('severity', 'medium')
        
        if predicted_severity != actual_severity:
            suggestions.append(f"🤖 ML suggests this should be {predicted_severity} severity (currently {actual_severity})")
        
        # Get risk score
        risk_score = self.predict_risk_score(finding)
        if risk_score >= 8:
            suggestions.append("🚨 High-risk finding detected. Consider immediate attention.")
        elif risk_score >= 6:
            suggestions.append("⚠️ Medium-high risk. Review before next release.")
        
        # Tool-specific suggestions
        tool = finding.get('tool', '')
        if 'security' in tool.lower():
            suggestions.append("🔒 Security finding detected. Consider adding to security review checklist.")
        
        if 'performance' in tool.lower():
            suggestions.append("⚡ Performance issue detected. Consider adding to performance review checklist.")
        
        # Context-aware suggestions
        pr_context = finding.get('pr_context', {})
        if pr_context.get('file_status') == 'A':  # Added file
            suggestions.append("🆕 New file with issues. Consider adding tests for this code.")
        
        return suggestions
    
    def train_models(self, training_data: List[Dict]):
        """Train ML models with historical data"""
        if not training_data:
            print("No training data provided")
            return
        
        try:
            # Prepare training data
            X = []
            y_severity = []
            y_false_positive = []
            y_risk_score = []
            
            for item in training_data:
                features = self.extract_features(item['finding'])
                X.append(features)
                
                # Severity labels
                y_severity.append(item['finding'].get('severity', 'medium'))
                
                # False positive labels (assuming 'is_false_positive' field exists)
                y_false_positive.append(1 if item.get('is_false_positive', False) else 0)
                
                # Risk score labels (assuming 'actual_risk_score' field exists)
                y_risk_score.append(item.get('actual_risk_score', 5))
            
            X = np.array(X)
            
            # Split data
            X_train, X_test, y_sev_train, y_sev_test = train_test_split(
                X, y_severity, test_size=0.2, random_state=42
            )
            
            X_train, X_test, y_fp_train, y_fp_test = train_test_split(
                X, y_false_positive, test_size=0.2, random_state=42
            )
            
            X_train, X_test, y_risk_train, y_risk_test = train_test_split(
                X, y_risk_score, test_size=0.2, random_state=42
            )
            
            # Fit text vectorizers
            text_data = [f"{item['finding'].get('message', '')} {item['finding'].get('remediation', '')}" 
                        for item in training_data]
            self.text_vectorizer.fit(text_data)
            
            code_data = [item['finding'].get('code_snippet', '') for item in training_data]
            self.code_vectorizer.fit(code_data)
            
            # Train severity classifier
            print("Training severity classifier...")
            self.severity_classifier.fit(X_train, y_sev_train)
            sev_score = self.severity_classifier.score(X_test, y_sev_test)
            print(f"Severity classifier accuracy: {sev_score:.3f}")
            
            # Train false positive detector
            print("Training false positive detector...")
            self.false_positive_detector.fit(X_train, y_fp_train)
            fp_score = self.false_positive_detector.score(X_test, y_fp_test)
            print(f"False positive detector accuracy: {fp_score:.3f}")
            
            # Train risk predictor
            print("Training risk predictor...")
            self.risk_predictor.fit(X_train, y_risk_train)
            risk_score = self.risk_predictor.score(X_test, y_risk_test)
            print(f"Risk predictor accuracy: {risk_score:.3f}")
            
            # Save models
            self._save_models()
            
            print("ML models trained and saved successfully!")
            
        except Exception as e:
            print(f"Error training models: {e}")
    
    def get_model_performance(self) -> Dict:
        """Get performance metrics for ML models"""
        performance = {
            "severity_classifier": {
                "loaded": self.severity_classifier is not None,
                "trained": hasattr(self.severity_classifier, 'n_estimators') if self.severity_classifier else False
            },
            "false_positive_detector": {
                "loaded": self.false_positive_detector is not None,
                "trained": hasattr(self.false_positive_detector, 'n_estimators') if self.false_positive_detector else False
            },
            "risk_predictor": {
                "loaded": self.risk_predictor is not None,
                "trained": hasattr(self.risk_predictor, 'n_estimators') if self.risk_predictor else False
            },
            "feature_extractors": {
                "text_vectorizer": hasattr(self.text_vectorizer, 'vocabulary_'),
                "code_vectorizer": hasattr(self.code_vectorizer, 'vocabulary_')
            }
        }
        
        return performance

# Global ML analyzer instance
ml_analyzer = MLAnalyzer()

# Convenience functions
async def predict_finding_severity(finding: Dict) -> str:
    """Predict severity for a finding using ML"""
    return ml_analyzer.predict_severity(finding)

async def detect_false_positive(finding: Dict) -> float:
    """Detect false positive probability for a finding"""
    return ml_analyzer.detect_false_positive(finding)

async def predict_risk_score(finding: Dict) -> int:
    """Predict risk score for a finding using ML"""
    return ml_analyzer.predict_risk_score(finding)

async def generate_intelligent_suggestions(finding: Dict) -> List[str]:
    """Generate intelligent suggestions for a finding"""
    return ml_analyzer.generate_intelligent_suggestions(finding)

async def train_ml_models(training_data: List[Dict]):
    """Train ML models with historical data"""
    ml_analyzer.train_models(training_data)

async def get_ml_model_performance() -> Dict:
    """Get performance metrics for ML models"""
    return ml_analyzer.get_model_performance()
