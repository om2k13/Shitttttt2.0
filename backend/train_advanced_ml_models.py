#!/usr/bin/env python3
"""
Train Advanced ML Models for Code Review Agent
Trains models for complexity prediction, maintainability scoring, technical debt estimation, and code smell detection
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, classification_report, accuracy_score
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings
import logging

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedMLTrainer:
    """Trainer for advanced ML capabilities"""
    
    def __init__(self, data_dir: str = "."):
        self.data_dir = Path(data_dir)
        self.models = {}
        self.scaler = None
        
    def generate_advanced_training_data(self, num_samples: int = 10000) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Generate synthetic training data for advanced ML models"""
        logger.info(f"🔄 Generating {num_samples} synthetic training samples...")
        
        np.random.seed(42)
        
        # Generate base features (same as existing system)
        X = np.random.random((num_samples, 9))
        
        # Generate target variables for each model
        
        # 1. Complexity metrics (5 outputs)
        complexity_targets = np.column_stack([
            X[:, 1] * 15 + np.random.normal(0, 2, num_samples),  # cyclomatic
            X[:, 2] * 10 + np.random.normal(0, 1.5, num_samples),  # cognitive
            X[:, 2] * 8 + np.random.normal(0, 1, num_samples),  # nesting
            X[:, 0] * 100 + np.random.normal(0, 20, num_samples),  # function length
            X[:, 5] * 5 + np.random.normal(0, 1, num_samples)  # class complexity
        ])
        
        # 2. Maintainability score (1 output, 0-1)
        maintainability_scores = np.maximum(0, np.minimum(1, 
            1 - (X[:, 1] * 0.4 + X[:, 2] * 0.3 + X[:, 6] * 0.3) + np.random.normal(0, 0.1, num_samples)
        ))
        
        # 3. Technical debt hours (1 output)
        tech_debt_hours = (X[:, 1] * 50 + X[:, 2] * 30 + X[:, 0] * 20) + np.random.normal(0, 10, num_samples)
        tech_debt_hours = np.maximum(0, tech_debt_hours)
        
        # 4. Code smells (6 binary outputs)
        smell_probs = X[:, 1:7] * 0.8 + np.random.normal(0, 0.1, (num_samples, 6))
        code_smells = (smell_probs > 0.5).astype(int)
        
        targets = {
            'complexity': complexity_targets,
            'maintainability': maintainability_scores,
            'technical_debt': tech_debt_hours,
            'code_smells': code_smells
        }
        
        logger.info(f"✅ Generated training data with shapes:")
        logger.info(f"   Features: {X.shape}")
        for name, target in targets.items():
            logger.info(f"   {name}: {target.shape}")
        
        return X, targets
    
    def train_complexity_predictor(self, X: np.ndarray, y: np.ndarray) -> nn.Module:
        """Train neural network for complexity prediction"""
        logger.info("🧠 Training Complexity Predictor...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train)
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.FloatTensor(y_test)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        # Initialize model
        input_dim = X.shape[1]
        model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, y.shape[1])
        )
        
        # Training
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        model.train()
        for epoch in range(100):
            total_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if epoch % 20 == 0:
                logger.info(f"   Epoch {epoch}, Loss: {total_loss/len(train_loader):.4f}")
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            test_predictions = model(X_test_tensor)
            mse = mean_squared_error(y_test, test_predictions.numpy())
            logger.info(f"   Test MSE: {mse:.4f}")
        
        return model
    
    def train_maintainability_scorer(self, X: np.ndarray, y: np.ndarray) -> nn.Module:
        """Train neural network for maintainability scoring"""
        logger.info("🧠 Training Maintainability Scorer...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        # Initialize model
        input_dim = X.shape[1]
        model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Training
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        model.train()
        for epoch in range(100):
            total_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if epoch % 20 == 0:
                logger.info(f"   Epoch {epoch}, Loss: {total_loss/len(train_loader):.4f}")
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            test_predictions = model(X_test_tensor)
            test_predictions_binary = (test_predictions > 0.5).float()
            accuracy = accuracy_score(y_test, test_predictions_binary.numpy())
            logger.info(f"   Test Accuracy: {accuracy:.4f}")
        
        return model
    
    def train_technical_debt_estimator(self, X: np.ndarray, y: np.ndarray) -> RandomForestRegressor:
        """Train Random Forest for technical debt estimation"""
        logger.info("🌲 Training Technical Debt Estimator...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        # Evaluation
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        test_predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, test_predictions)
        
        logger.info(f"   Train R²: {train_score:.4f}")
        logger.info(f"   Test R²: {test_score:.4f}")
        logger.info(f"   Test MSE: {mse:.4f}")
        
        return model
    
    def train_code_smell_detector(self, X: np.ndarray, y: np.ndarray) -> RandomForestClassifier:
        """Train Random Forest for code smell detection"""
        logger.info("🌲 Training Code Smell Detector...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        # Evaluation
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        test_predictions = model.predict(X_test)
        
        logger.info(f"   Train Accuracy: {train_score:.4f}")
        logger.info(f"   Test Accuracy: {test_score:.4f}")
        
        # Detailed classification report
        logger.info("   Classification Report:")
        report = classification_report(y_test, test_predictions, target_names=[
            'long_method', 'large_class', 'duplicate_code',
            'feature_envy', 'data_clumps', 'primitive_obsession'
        ])
        for line in report.split('\n'):
            if line.strip():
                logger.info(f"     {line}")
        
        return model
    
    def save_models(self):
        """Save all trained models"""
        logger.info("💾 Saving trained models...")
        
        # Save PyTorch models
        if 'complexity_predictor' in self.models:
            torch.save(self.models['complexity_predictor'].state_dict(), 
                      self.data_dir / "advanced_complexity_predictor.pth")
            logger.info("   ✅ Saved Complexity Predictor")
        
        if 'maintainability_scorer' in self.models:
            torch.save(self.models['maintainability_scorer'].state_dict(), 
                      self.data_dir / "advanced_maintainability_scorer.pth")
            logger.info("   ✅ Saved Maintainability Scorer")
        
        # Save scikit-learn models
        if 'technical_debt_estimator' in self.models:
            joblib.dump(self.models['technical_debt_estimator'], 
                       self.data_dir / "advanced_technical_debt_estimator.joblib")
            logger.info("   ✅ Saved Technical Debt Estimator")
        
        if 'code_smell_detector' in self.models:
            joblib.dump(self.models['code_smell_detector'], 
                       self.data_dir / "advanced_code_smell_detector.joblib")
            logger.info("   ✅ Saved Code Smell Detector")
        
        # Save metadata
        metadata = {
            'models_trained': list(self.models.keys()),
            'training_date': pd.Timestamp.now().isoformat(),
            'feature_count': 9,
            'model_types': {
                'complexity_predictor': 'PyTorch Neural Network',
                'maintainability_scorer': 'PyTorch Neural Network',
                'technical_debt_estimator': 'Random Forest Regressor',
                'code_smell_detector': 'Random Forest Classifier'
            }
        }
        
        with open(self.data_dir / "advanced_ml_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info("   ✅ Saved training metadata")
    
    def train_all_advanced_models(self):
        """Train all advanced ML models"""
        logger.info("🚀 Starting Advanced ML Model Training...")
        logger.info("=" * 60)
        
        try:
            # Generate training data
            X, targets = self.generate_advanced_training_data(10000)
            
            # Train Complexity Predictor
            self.models['complexity_predictor'] = self.train_complexity_predictor(
                X, targets['complexity']
            )
            
            # Train Maintainability Scorer
            self.models['maintainability_scorer'] = self.train_maintainability_scorer(
                X, targets['maintainability']
            )
            
            # Train Technical Debt Estimator
            self.models['technical_debt_estimator'] = self.train_technical_debt_estimator(
                X, targets['technical_debt']
            )
            
            # Train Code Smell Detector
            self.models['code_smell_detector'] = self.train_code_smell_detector(
                X, targets['code_smells']
            )
            
            # Save all models
            self.save_models()
            
            logger.info("=" * 60)
            logger.info("🎉 Advanced ML Model Training Completed Successfully!")
            logger.info(f"📊 Total models trained: {len(self.models)}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error during training: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Main training function"""
    trainer = AdvancedMLTrainer()
    success = trainer.train_all_advanced_models()
    
    if success:
        print("\n🎉 Advanced ML models trained successfully!")
        print("📁 Models saved in current directory:")
        print("   - advanced_complexity_predictor.pth")
        print("   - advanced_maintainability_scorer.pth")
        print("   - advanced_technical_debt_estimator.joblib")
        print("   - advanced_code_smell_detector.joblib")
        print("   - advanced_ml_metadata.json")
    else:
        print("\n❌ Training failed. Check logs for details.")
        exit(1)

if __name__ == "__main__":
    main()
