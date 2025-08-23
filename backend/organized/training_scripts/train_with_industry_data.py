"""
Train ML Models with Industry-Standard Training Data

This script trains our ML models using the comprehensive training data that combines:
- Real code analysis data
- Industry-standard security patterns (OWASP, Bandit, SonarQube)
- Industry-standard quality patterns
- Synthetic data based on industry best practices
"""

import numpy as np
import json
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
import xgboost as xgb
import lightgbm as lgb
import joblib

class IndustryDataTrainer:
    """Trains ML models with industry-standard training data"""
    
    def __init__(self, data_dir: str = "comprehensive_training_data"):
        self.data_dir = Path(data_dir)
        self.models = {}
        self.training_results = {}
        
    def load_industry_data(self):
        """Load the comprehensive industry training data"""
        print("📊 Loading Industry-Standard Training Data...")
        
        # Load features and labels
        self.X = np.load(self.data_dir / "comprehensive_features.npy")
        self.y_quality = np.load(self.data_dir / "comprehensive_quality_labels.npy")
        self.y_security = np.load(self.data_dir / "comprehensive_security_labels.npy")
        
        # Load metadata
        with open(self.data_dir / "dataset_metadata.json", "r") as f:
            self.metadata = json.load(f)
        
        print(f"✅ Loaded {len(self.X)} samples with {self.X.shape[1]} features")
        print(f"📊 Quality distribution: {self.metadata['quality_distribution']}")
        print(f"📊 Security distribution: {self.metadata['security_distribution']}")
        print(f"📊 Real samples: {self.metadata['real_samples']}")
        print(f"📊 Industry-based synthetic samples: {self.metadata['synthetic_samples']}")
        
        # Split data for training
        self.X_train_quality, self.X_test_quality, self.y_train_quality, self.y_test_quality = train_test_split(
            self.X, self.y_quality, test_size=0.2, random_state=42, stratify=self.y_quality
        )
        
        self.X_train_security, self.X_test_security, self.y_train_security, self.y_test_security = train_test_split(
            self.X, self.y_security, test_size=0.2, random_state=42, stratify=self.y_security
        )
        
        print(f"🔧 Training set: {len(self.X_train_quality)} samples")
        print(f"🧪 Test set: {len(self.X_test_quality)} samples")
    
    def train_quality_models(self):
        """Train models for code quality prediction"""
        print("\n🎯 Training Code Quality Models with Industry Data...")
        
        # Random Forest
        print("🌲 Training Random Forest...")
        rf_quality = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        rf_quality.fit(self.X_train_quality, self.y_train_quality)
        self.models['quality_random_forest'] = rf_quality
        
        # Gradient Boosting
        print("📈 Training Gradient Boosting...")
        gb_quality = GradientBoostingClassifier(n_estimators=200, random_state=42)
        gb_quality.fit(self.X_train_quality, self.y_train_quality)
        self.models['quality_gradient_boosting'] = gb_quality
        
        # XGBoost
        print("🚀 Training XGBoost...")
        xgb_quality = xgb.XGBClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        xgb_quality.fit(self.X_train_quality, self.y_train_quality)
        self.models['quality_xgboost'] = xgb_quality
        
        # LightGBM
        print("💡 Training LightGBM...")
        lgb_quality = lgb.LGBMClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        lgb_quality.fit(self.X_train_quality, self.y_train_quality)
        self.models['quality_lightgbm'] = lgb_quality
        
        # SVM
        print("🔧 Training SVM...")
        svm_quality = SVC(kernel='rbf', random_state=42, probability=True)
        svm_quality.fit(self.X_train_quality, self.y_train_quality)
        self.models['quality_svm'] = svm_quality
        
        # Neural Network
        print("🧠 Training Neural Network...")
        mlp_quality = MLPClassifier(hidden_layer_sizes=(200, 100, 50), max_iter=1000, random_state=42)
        mlp_quality.fit(self.X_train_quality, self.y_train_quality)
        self.models['quality_mlp'] = mlp_quality
        
        print("✅ Quality models trained successfully!")
    
    def train_security_models(self):
        """Train models for security vulnerability prediction"""
        print("\n🔒 Training Security Models with Industry Data...")
        
        # Random Forest
        print("🌲 Training Random Forest...")
        rf_security = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        rf_security.fit(self.X_train_security, self.y_train_security)
        self.models['security_random_forest'] = rf_security
        
        # Gradient Boosting
        print("📈 Training Gradient Boosting...")
        gb_security = GradientBoostingClassifier(n_estimators=200, random_state=42)
        gb_security.fit(self.X_train_security, self.y_train_security)
        self.models['security_gradient_boosting'] = gb_security
        
        # XGBoost
        print("🚀 Training XGBoost...")
        xgb_security = xgb.XGBClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        xgb_security.fit(self.X_train_security, self.y_train_security)
        self.models['security_xgboost'] = xgb_security
        
        # LightGBM
        print("💡 Training LightGBM...")
        lgb_security = lgb.LGBMClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        lgb_security.fit(self.X_train_security, self.y_train_security)
        self.models['security_lightgbm'] = lgb_security
        
        # Logistic Regression
        print("📊 Training Logistic Regression...")
        lr_security = LogisticRegression(random_state=42, max_iter=1000, n_jobs=-1)
        lr_security.fit(self.X_train_security, self.y_train_security)
        self.models['security_logistic_regression'] = lr_security
        
        print("✅ Security models trained successfully!")
    
    def evaluate_models(self):
        """Evaluate all trained models"""
        print("\n📊 Evaluating Models...")
        
        self.training_results = {}
        
        # Evaluate quality models
        print("\n🎯 Code Quality Models:")
        for name, model in self.models.items():
            if 'quality' in name:
                if hasattr(model, 'predict'):
                    y_pred = model.predict(self.X_test_quality)
                    accuracy = accuracy_score(self.y_test_quality, y_pred)
                    
                    # Calculate precision, recall, F1-score
                    precision, recall, f1, _ = precision_recall_fscore_support(
                        self.y_test_quality, y_pred, average='weighted'
                    )
                    
                    self.training_results[name] = {
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1_score': f1,
                        'test_predictions': y_pred.tolist()
                    }
                    print(f"  {name}: Accuracy={accuracy:.3f}, Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")
        
        # Evaluate security models
        print("\n🔒 Security Models:")
        for name, model in self.models.items():
            if 'security' in name:
                if hasattr(model, 'predict'):
                    y_pred = model.predict(self.X_test_security)
                    accuracy = accuracy_score(self.y_test_security, y_pred)
                    
                    # Calculate precision, recall, F1-score
                    precision, recall, f1, _ = precision_recall_fscore_support(
                        self.y_test_security, y_pred, average='weighted'
                    )
                    
                    self.training_results[name] = {
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1_score': f1,
                        'test_predictions': y_pred.tolist()
                    }
                    print(f"  {name}: Accuracy={accuracy:.3f}, Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")
    
    def cross_validate_models(self):
        """Perform cross-validation on models"""
        print("\n🔄 Performing Cross-Validation...")
        
        cv_results = {}
        
        # Cross-validate quality models
        print("\n🎯 Code Quality Models (5-fold CV):")
        for name, model in self.models.items():
            if 'quality' in name:
                try:
                    cv_scores = cross_val_score(model, self.X, self.y_quality, cv=5, scoring='accuracy')
                    cv_results[name] = {
                        'mean_accuracy': cv_scores.mean(),
                        'std_accuracy': cv_scores.std(),
                        'cv_scores': cv_scores.tolist()
                    }
                    print(f"  {name}: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
                except Exception as e:
                    print(f"  {name}: CV failed - {e}")
        
        # Cross-validate security models
        print("\n🔒 Security Models (5-fold CV):")
        for name, model in self.models.items():
            if 'security' in name:
                try:
                    cv_scores = cross_val_score(model, self.X, self.y_security, cv=5, scoring='accuracy')
                    cv_results[name] = {
                        'mean_accuracy': cv_scores.mean(),
                        'std_accuracy': cv_scores.std(),
                        'cv_scores': cv_scores.tolist()
                    }
                    print(f"  {name}: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
                except Exception as e:
                    print(f"  {name}: CV failed - {e}")
        
        return cv_results
    
    def create_ensemble_model(self):
        """Create an ensemble model that combines predictions"""
        print("\n🤝 Creating Ensemble Model...")
        
        # For quality prediction
        quality_predictions = []
        quality_models = [name for name in self.models.keys() if 'quality' in name]
        
        for name in quality_models:
            model = self.models[name]
            if hasattr(model, 'predict_proba'):
                try:
                    pred = model.predict_proba(self.X_test_quality)
                    quality_predictions.append(pred)
                except:
                    pass
        
        if quality_predictions:
            # Average the probabilities
            ensemble_quality = np.mean(quality_predictions, axis=0)
            ensemble_quality_pred = np.argmax(ensemble_quality, axis=1)
            ensemble_quality_acc = accuracy_score(self.y_test_quality, ensemble_quality_pred)
            print(f"  Ensemble Quality Accuracy: {ensemble_quality_acc:.3f}")
        
        # For security prediction
        security_predictions = []
        security_models = [name for name in self.models.keys() if 'security' in name]
        
        for name in security_models:
            model = self.models[name]
            if hasattr(model, 'predict_proba'):
                try:
                    pred = model.predict_proba(self.X_test_security)
                    security_predictions.append(pred)
                except:
                    pass
        
        if security_predictions:
            # Average the probabilities
            ensemble_security = np.mean(security_predictions, axis=0)
            ensemble_security_pred = np.argmax(ensemble_security, axis=1)
            ensemble_security_acc = accuracy_score(self.y_test_security, ensemble_security_pred)
            print(f"  Ensemble Security Accuracy: {ensemble_security_acc:.3f}")
    
    def save_models(self, output_dir: str = "industry_trained_models"):
        """Save all trained models"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print(f"\n💾 Saving models to {output_path}...")
        
        # Save individual models
        for name, model in self.models.items():
            model_path = output_path / f"{name}.joblib"
            joblib.dump(model, model_path)
            print(f"  ✅ Saved {name}")
        
        # Save training results
        results_path = output_path / "training_results.json"
        with open(results_path, "w") as f:
            json.dump(self.training_results, f, indent=2)
        
        # Save feature names
        feature_names = ['lines', 'complexity', 'nesting', 'imports', 'functions', 'classes', 'security_risk', 'user_inputs', 'external_calls']
        features_path = output_path / "feature_names.json"
        with open(features_path, "w") as f:
            json.dump(feature_names, f, indent=2)
        
        # Save metadata
        metadata_path = output_path / "model_metadata.json"
        model_metadata = {
            "total_models": len(self.models),
            "quality_models": [name for name in self.models.keys() if 'quality' in name],
            "security_models": [name for name in self.models.keys() if 'security' in name],
            "features": self.X.shape[1],
            "training_samples": len(self.X_train_quality),
            "test_samples": len(self.X_test_quality),
            "data_source": "comprehensive_industry_training_data",
            "training_data_metadata": self.metadata
        }
        with open(metadata_path, "w") as f:
            json.dump(model_metadata, f, indent=2)
        
        print(f"✅ All models saved to {output_path}")
        return output_path
    
    def train_all(self):
        """Train all models with industry data"""
        print("🚀 Starting ML Model Training with Industry-Standard Data...")
        
        # Load data
        self.load_industry_data()
        
        # Train models
        self.train_quality_models()
        self.train_security_models()
        
        # Evaluate models
        self.evaluate_models()
        
        # Cross-validate models
        cv_results = self.cross_validate_models()
        
        # Create ensemble
        self.create_ensemble_model()
        
        # Save models
        output_dir = self.save_models()
        
        print(f"\n🎉 Training Complete! Models saved to: {output_dir}")
        print(f"🎯 Your ML models are now trained with INDUSTRY-STANDARD data!")
        
        return output_dir

def main():
    """Main function to train models with industry data"""
    trainer = IndustryDataTrainer()
    output_dir = trainer.train_all()
    print(f"🎯 Industry-trained ML models ready in: {output_dir}")

if __name__ == "__main__":
    main()
