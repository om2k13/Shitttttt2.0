import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime, timedelta
import asyncio
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
import joblib
import pickle
from dataclasses import dataclass
from enum import Enum

class TrendDirection(Enum):
    IMPROVING = "improving"
    STABLE = "stable"
    DETERIORATING = "deteriorating"
    FLUCTUATING = "fluctuating"

class PredictionConfidence(Enum):
    HIGH = "high"      # 80-100%
    MEDIUM = "medium"  # 60-79%
    LOW = "low"        # 40-59%
    VERY_LOW = "very_low"  # <40%

@dataclass
class TrendAnalysis:
    """Trend analysis result"""
    metric: str
    direction: TrendDirection
    confidence: float
    slope: float
    r_squared: float
    prediction_next_month: float
    recommendation: str

@dataclass
class PredictionResult:
    """Prediction result"""
    metric: str
    predicted_value: float
    confidence: float
    confidence_level: PredictionConfidence
    factors: List[str]
    uncertainty_range: Tuple[float, float]

class PredictiveAnalyticsEngine:
    """Advanced predictive analytics for code review and quality trends"""
    
    def __init__(self, model_dir: Path = None):
        self.model_dir = model_dir or Path("./predictive_models")
        self.model_dir.mkdir(exist_ok=True)
        
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        
        # Load or initialize models
        self._load_or_initialize_models()
    
    def _load_or_initialize_models(self):
        """Load existing models or initialize new ones"""
        model_files = {
            "security_trend": "security_trend_model.pkl",
            "quality_trend": "quality_trend_model.pkl",
            "performance_trend": "performance_trend_model.pkl",
            "compliance_trend": "compliance_trend_model.pkl",
            "issue_prediction": "issue_prediction_model.pkl",
            "risk_prediction": "risk_prediction_model.pkl"
        }
        
        for model_name, filename in model_files.items():
            model_path = self.model_dir / filename
            if model_path.exists():
                try:
                    with open(model_path, 'rb') as f:
                        self.models[model_name] = pickle.load(f)
                    print(f"Loaded {model_name} model")
                except Exception as e:
                    print(f"Failed to load {model_name} model: {e}")
                    self.models[model_name] = None
            else:
                self.models[model_name] = None
    
    async def analyze_quality_trends(self, historical_data: List[Dict]) -> Dict:
        """Analyze quality trends over time"""
        trend_analysis = {
            "timestamp": datetime.now().isoformat(),
            "trends": {},
            "predictions": {},
            "insights": {},
            "recommendations": []
        }
        
        try:
            # Convert to DataFrame for analysis
            df = pd.DataFrame(historical_data)
            
            # Analyze trends for different metrics
            metrics = ["security_score", "quality_score", "performance_score", "compliance_score"]
            
            for metric in metrics:
                if metric in df.columns:
                    trend = await self._analyze_metric_trend(df, metric)
                    trend_analysis["trends"][metric] = trend
                    
                    # Generate prediction
                    prediction = await self._predict_metric_value(df, metric)
                    trend_analysis["predictions"][metric] = prediction
            
            # Generate insights and recommendations
            trend_analysis["insights"] = await self._generate_trend_insights(trend_analysis["trends"])
            trend_analysis["recommendations"] = await self._generate_trend_recommendations(trend_analysis["trends"])
            
        except Exception as e:
            trend_analysis["error"] = str(e)
        
        return trend_analysis
    
    async def _analyze_metric_trend(self, df: pd.DataFrame, metric: str) -> TrendAnalysis:
        """Analyze trend for a specific metric"""
        try:
            # Prepare data for trend analysis
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df = df.sort_values("timestamp")
                df["days_since_start"] = (df["timestamp"] - df["timestamp"].min()).dt.days
            else:
                # If no timestamp, use index as time proxy
                df["days_since_start"] = range(len(df))
            
            # Remove rows with missing values
            df_clean = df.dropna(subset=[metric])
            
            if len(df_clean) < 3:
                return TrendAnalysis(
                    metric=metric,
                    direction=TrendDirection.STABLE,
                    confidence=0.0,
                    slope=0.0,
                    r_squared=0.0,
                    prediction_next_month=df_clean[metric].mean() if len(df_clean) > 0 else 0.0,
                    recommendation="Insufficient data for trend analysis"
                )
            
            # Fit linear regression
            X = df_clean["days_since_start"].values.reshape(-1, 1)
            y = df_clean[metric].values
            
            model = LinearRegression()
            model.fit(X, y)
            
            # Calculate trend statistics
            slope = model.coef_[0]
            r_squared = model.score(X, y)
            y_pred = model.predict(X)
            
            # Determine trend direction
            if r_squared > 0.3:  # Only consider trend if it explains >30% of variance
                if slope > 0.01:
                    direction = TrendDirection.IMPROVING
                elif slope < -0.01:
                    direction = TrendDirection.DETERIORATING
                else:
                    direction = TrendDirection.STABLE
            else:
                direction = TrendDirection.FLUCTUATING
            
            # Calculate confidence based on R-squared and data points
            confidence = min(1.0, r_squared * (len(df_clean) / 10))
            
            # Predict next month value
            days_in_month = 30
            next_month_days = df_clean["days_since_start"].max() + days_in_month
            prediction_next_month = model.predict([[next_month_days]])[0]
            
            # Generate recommendation
            recommendation = self._generate_trend_recommendation(direction, metric, slope, confidence)
            
            return TrendAnalysis(
                metric=metric,
                direction=direction,
                confidence=confidence,
                slope=slope,
                r_squared=r_squared,
                prediction_next_month=prediction_next_month,
                recommendation=recommendation
            )
            
        except Exception as e:
            return TrendAnalysis(
                metric=metric,
                direction=TrendDirection.STABLE,
                confidence=0.0,
                slope=0.0,
                r_squared=0.0,
                prediction_next_month=0.0,
                recommendation=f"Error in trend analysis: {str(e)}"
            )
    
    def _generate_trend_recommendation(self, direction: TrendDirection, metric: str, slope: float, confidence: float) -> str:
        """Generate recommendation based on trend analysis"""
        metric_names = {
            "security_score": "security posture",
            "quality_score": "code quality",
            "performance_score": "performance characteristics",
            "compliance_score": "compliance status"
        }
        
        metric_name = metric_names.get(metric, metric)
        
        if direction == TrendDirection.IMPROVING:
            if confidence > 0.7:
                return f"Excellent! {metric_name.title()} is improving steadily. Continue current practices."
            else:
                return f"{metric_name.title()} shows improvement trend but with low confidence. Monitor closely."
        
        elif direction == TrendDirection.DETERIORATING:
            if confidence > 0.7:
                return f"Warning: {metric_name.title()} is declining. Immediate intervention required."
            else:
                return f"{metric_name.title()} shows decline trend but with low confidence. Investigate further."
        
        elif direction == TrendDirection.STABLE:
            return f"{metric_name.title()} is stable. Consider optimization opportunities."
        
        else:  # FLUCTUATING
            return f"{metric_name.title()} shows high variability. Standardize processes and reduce noise."
    
    async def _predict_metric_value(self, df: pd.DataFrame, metric: str) -> PredictionResult:
        """Predict future value of a metric"""
        try:
            # Prepare features for prediction
            features = await self._extract_prediction_features(df, metric)
            
            if features is None or len(features) < 10:
                return PredictionResult(
                    metric=metric,
                    predicted_value=df[metric].mean() if metric in df.columns else 0.0,
                    confidence=0.0,
                    confidence_level=PredictionConfidence.VERY_LOW,
                    factors=["Insufficient data for prediction"],
                    uncertainty_range=(0.0, 0.0)
                )
            
            # Train prediction model
            model = await self._train_prediction_model(features, metric)
            
            if model is None:
                return PredictionResult(
                    metric=metric,
                    predicted_value=df[metric].mean() if metric in df.columns else 0.0,
                    confidence=0.0,
                    confidence_level=PredictionConfidence.VERY_LOW,
                    factors=["Failed to train prediction model"],
                    uncertainty_range=(0.0, 0.0)
                )
            
            # Make prediction
            latest_features = features.iloc[-1:].values
            prediction = model.predict(latest_features)[0]
            
            # Calculate confidence and uncertainty
            confidence = self._calculate_prediction_confidence(model, features, metric)
            confidence_level = self._get_confidence_level(confidence)
            uncertainty_range = self._calculate_uncertainty_range(prediction, confidence)
            
            # Identify key factors
            key_factors = self._identify_key_factors(model, features.columns)
            
            return PredictionResult(
                metric=metric,
                predicted_value=prediction,
                confidence=confidence,
                confidence_level=confidence_level,
                factors=key_factors,
                uncertainty_range=uncertainty_range
            )
            
        except Exception as e:
            return PredictionResult(
                metric=metric,
                predicted_value=0.0,
                confidence=0.0,
                confidence_level=PredictionConfidence.VERY_LOW,
                factors=[f"Prediction error: {str(e)}"],
                uncertainty_range=(0.0, 0.0)
            )
    
    async def _extract_prediction_features(self, df: pd.DataFrame, target_metric: str) -> Optional[pd.DataFrame]:
        """Extract features for prediction"""
        try:
            # Create lagged features
            features = pd.DataFrame()
            
            # Time-based features
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                features["month"] = df["timestamp"].dt.month
                features["day_of_week"] = df["timestamp"].dt.dayofweek
                features["days_since_start"] = (df["timestamp"] - df["timestamp"].min()).dt.days
            
            # Lagged target values
            for lag in [1, 2, 3]:
                features[f"{target_metric}_lag_{lag}"] = df[target_metric].shift(lag)
            
            # Rolling statistics
            for window in [3, 5, 7]:
                features[f"{target_metric}_rolling_mean_{window}"] = df[target_metric].rolling(window=window).mean()
                features[f"{target_metric}_rolling_std_{window}"] = df[target_metric].rolling(window=window).std()
            
            # Other metrics as features
            other_metrics = ["security_score", "quality_score", "performance_score", "compliance_score"]
            for metric in other_metrics:
                if metric in df.columns and metric != target_metric:
                    features[f"{metric}_current"] = df[metric]
                    features[f"{metric}_lag_1"] = df[metric].shift(1)
            
            # Remove rows with missing values
            features = features.dropna()
            
            return features if len(features) > 0 else None
            
        except Exception as e:
            print(f"Feature extraction failed: {e}")
            return None
    
    async def _train_prediction_model(self, features: pd.DataFrame, target_metric: str) -> Optional[RandomForestRegressor]:
        """Train prediction model"""
        try:
            # Prepare target variable
            target = features[f"{target_metric}_lag_1"].dropna()
            features_clean = features.loc[target.index].drop(columns=[f"{target_metric}_lag_1"])
            
            if len(features_clean) < 10:
                return None
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                features_clean, target, test_size=0.2, random_state=42
            )
            
            # Train model
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Store feature importance
            self.feature_importance[target_metric] = dict(zip(features_clean.columns, model.feature_importances_))
            
            return model
            
        except Exception as e:
            print(f"Model training failed: {e}")
            return None
    
    def _calculate_prediction_confidence(self, model: RandomForestRegressor, features: pd.DataFrame, target_metric: str) -> float:
        """Calculate prediction confidence"""
        try:
            # Use model's feature importances and data quality as confidence indicators
            if target_metric in self.feature_importance:
                avg_importance = np.mean(list(self.feature_importance[target_metric].values()))
                data_quality = min(1.0, len(features) / 50)  # Normalize by expected data size
                
                confidence = (avg_importance * 0.6) + (data_quality * 0.4)
                return min(1.0, confidence)
            else:
                return 0.5  # Default confidence
            
        except Exception:
            return 0.5
    
    def _get_confidence_level(self, confidence: float) -> PredictionConfidence:
        """Get confidence level enum"""
        if confidence >= 0.8:
            return PredictionConfidence.HIGH
        elif confidence >= 0.6:
            return PredictionConfidence.MEDIUM
        elif confidence >= 0.4:
            return PredictionConfidence.LOW
        else:
            return PredictionConfidence.VERY_LOW
    
    def _calculate_uncertainty_range(self, prediction: float, confidence: float) -> Tuple[float, float]:
        """Calculate uncertainty range for prediction"""
        # Higher confidence means lower uncertainty
        uncertainty_factor = (1 - confidence) * 0.5  # 0.5 = 50% uncertainty at 0% confidence
        
        range_size = prediction * uncertainty_factor
        return (prediction - range_size, prediction + range_size)
    
    def _identify_key_factors(self, model: RandomForestRegressor, feature_names: List[str]) -> List[str]:
        """Identify key factors influencing predictions"""
        try:
            if hasattr(model, 'feature_importances_'):
                # Get top 3 most important features
                feature_importance = list(zip(feature_names, model.feature_importances_))
                feature_importance.sort(key=lambda x: x[1], reverse=True)
                
                top_features = feature_importance[:3]
                return [f"{feature} (importance: {importance:.3f})" for feature, importance in top_features]
            else:
                return ["Feature importance not available"]
                
        except Exception:
            return ["Unable to identify key factors"]
    
    async def _generate_trend_insights(self, trends: Dict) -> Dict:
        """Generate insights from trend analysis"""
        insights = {
            "overall_direction": "stable",
            "improving_metrics": [],
            "declining_metrics": [],
            "stable_metrics": [],
            "fluctuating_metrics": [],
            "key_observations": []
        }
        
        # Categorize metrics by trend direction
        for metric, trend in trends.items():
            if trend.direction == TrendDirection.IMPROVING:
                insights["improving_metrics"].append(metric)
            elif trend.direction == TrendDirection.DETERIORATING:
                insights["declining_metrics"].append(metric)
            elif trend.direction == TrendDirection.STABLE:
                insights["stable_metrics"].append(metric)
            else:
                insights["fluctuating_metrics"].append(metric)
        
        # Determine overall direction
        if len(insights["improving_metrics"]) > len(insights["declining_metrics"]):
            insights["overall_direction"] = "improving"
        elif len(insights["declining_metrics"]) > len(insights["improving_metrics"]):
            insights["overall_direction"] = "declining"
        else:
            insights["overall_direction"] = "stable"
        
        # Generate key observations
        if insights["improving_metrics"]:
            insights["key_observations"].append(
                f"Positive trends observed in: {', '.join(insights['improving_metrics'])}"
            )
        
        if insights["declining_metrics"]:
            insights["key_observations"].append(
                f"Attention needed for: {', '.join(insights['declining_metrics'])}"
            )
        
        if insights["fluctuating_metrics"]:
            insights["key_observations"].append(
                f"High variability in: {', '.join(insights['fluctuating_metrics'])}"
            )
        
        return insights
    
    async def _generate_trend_recommendations(self, trends: Dict) -> List[str]:
        """Generate recommendations based on trends"""
        recommendations = []
        
        # Count trends by direction
        improving_count = sum(1 for t in trends.values() if t.direction == TrendDirection.IMPROVING)
        declining_count = sum(1 for t in trends.values() if t.direction == TrendDirection.DETERIORATING)
        fluctuating_count = sum(1 for t in trends.values() if t.direction == TrendDirection.FLUCTUATING)
        
        # Overall recommendations
        if improving_count > declining_count:
            recommendations.append("Overall quality is improving. Continue current practices and document successful strategies.")
        elif declining_count > improving_count:
            recommendations.append("Overall quality is declining. Implement immediate quality improvement initiatives.")
        else:
            recommendations.append("Quality is stable. Focus on optimization and continuous improvement.")
        
        # Specific recommendations
        for metric, trend in trends.items():
            if trend.direction == TrendDirection.DETERIORATING and trend.confidence > 0.6:
                recommendations.append(f"Address declining {metric} trend with targeted improvement plan.")
            
            elif trend.direction == TrendDirection.FLUCTUATING and trend.confidence < 0.4:
                recommendations.append(f"Standardize processes for {metric} to reduce variability.")
        
        # Add general recommendations
        if fluctuating_count > 0:
            recommendations.append("Implement consistent review processes to reduce quality variability.")
        
        if improving_count > 0:
            recommendations.append("Document and share best practices from improving metrics.")
        
        return recommendations
    
    async def predict_future_issues(self, current_findings: List[Dict], historical_data: List[Dict]) -> Dict:
        """Predict future issues based on current state and historical patterns"""
        prediction_results = {
            "timestamp": datetime.now().isoformat(),
            "issue_predictions": {},
            "risk_forecast": {},
            "prevention_strategies": [],
            "confidence_assessment": {}
        }
        
        try:
            # Predict issue counts by category
            categories = ["security", "performance", "quality", "compliance"]
            
            for category in categories:
                prediction = await self._predict_issue_category(category, current_findings, historical_data)
                prediction_results["issue_predictions"][category] = prediction
            
            # Predict overall risk
            risk_prediction = await self._predict_overall_risk(current_findings, historical_data)
            prediction_results["risk_forecast"] = risk_prediction
            
            # Generate prevention strategies
            prediction_results["prevention_strategies"] = await self._generate_prevention_strategies(
                prediction_results["issue_predictions"]
            )
            
            # Assess prediction confidence
            prediction_results["confidence_assessment"] = await self._assess_prediction_confidence(
                prediction_results["issue_predictions"]
            )
            
        except Exception as e:
            prediction_results["error"] = str(e)
        
        return prediction_results
    
    async def _predict_issue_category(self, category: str, current_findings: List[Dict], historical_data: List[Dict]) -> Dict:
        """Predict issues for a specific category"""
        try:
            # Count current issues by category
            current_count = len([f for f in current_findings if self._categorize_finding(f) == category])
            
            # Simple prediction based on current trend
            # In practice, you'd use more sophisticated time series analysis
            if len(historical_data) > 0:
                # Calculate trend from historical data
                trend_factor = self._calculate_trend_factor(historical_data, category)
                predicted_count = max(0, current_count + trend_factor)
            else:
                predicted_count = current_count
            
            # Calculate confidence based on data availability
            confidence = min(1.0, len(historical_data) / 20)  # 20 data points = 100% confidence
            
            return {
                "current_count": current_count,
                "predicted_count": predicted_count,
                "trend": "increasing" if predicted_count > current_count else "decreasing" if predicted_count < current_count else "stable",
                "confidence": confidence,
                "factors": [f"Current {category} issues: {current_count}", "Historical trend analysis"]
            }
            
        except Exception as e:
            return {
                "current_count": 0,
                "predicted_count": 0,
                "trend": "unknown",
                "confidence": 0.0,
                "factors": [f"Prediction failed: {str(e)}"]
            }
    
    def _categorize_finding(self, finding: Dict) -> str:
        """Categorize a finding"""
        tool = finding.get("tool", "").lower()
        message = finding.get("message", "").lower()
        
        if any(sec in tool for sec in ["bandit", "semgrep", "safety", "npm-audit"]):
            return "security"
        elif any(perf in tool for perf in ["performance", "complexity", "memory"]):
            return "performance"
        elif any(comp in tool for comp in ["compliance", "gdpr", "hipaa", "pci"]):
            return "compliance"
        else:
            return "quality"
    
    def _calculate_trend_factor(self, historical_data: List[Dict], category: str) -> float:
        """Calculate trend factor for prediction"""
        try:
            # Simple trend calculation
            if len(historical_data) < 2:
                return 0.0
            
            # Count issues by category over time
            category_counts = []
            for data_point in historical_data:
                if "findings" in data_point:
                    count = len([f for f in data_point["findings"] if self._categorize_finding(f) == category])
                    category_counts.append(count)
            
            if len(category_counts) < 2:
                return 0.0
            
            # Calculate simple trend
            recent_avg = np.mean(category_counts[-3:]) if len(category_counts) >= 3 else category_counts[-1]
            older_avg = np.mean(category_counts[:-3]) if len(category_counts) >= 6 else category_counts[0]
            
            trend_factor = recent_avg - older_avg
            return trend_factor
            
        except Exception:
            return 0.0
    
    async def _predict_overall_risk(self, current_findings: List[Dict], historical_data: List[Dict]) -> Dict:
        """Predict overall risk level"""
        try:
            # Calculate current risk score
            current_risk = self._calculate_risk_score(current_findings)
            
            # Predict future risk based on trends
            if len(historical_data) > 0:
                risk_trend = self._calculate_risk_trend(historical_data)
                predicted_risk = max(0, min(10, current_risk + risk_trend))
            else:
                predicted_risk = current_risk
            
            # Determine risk level
            if predicted_risk >= 8:
                risk_level = "Critical"
            elif predicted_risk >= 6:
                risk_level = "High"
            elif predicted_risk >= 4:
                risk_level = "Medium"
            else:
                risk_level = "Low"
            
            return {
                "current_risk": current_risk,
                "predicted_risk": predicted_risk,
                "risk_level": risk_level,
                "trend": "increasing" if predicted_risk > current_risk else "decreasing" if predicted_risk < current_risk else "stable",
                "confidence": min(1.0, len(historical_data) / 20)
            }
            
        except Exception as e:
            return {
                "current_risk": 0,
                "predicted_risk": 0,
                "risk_level": "Unknown",
                "trend": "unknown",
                "confidence": 0.0,
                "error": str(e)
            }
    
    def _calculate_risk_score(self, findings: List[Dict]) -> float:
        """Calculate current risk score"""
        if not findings:
            return 0.0
        
        total_score = 0
        for finding in findings:
            severity = finding.get("severity", "medium")
            if severity == "critical":
                total_score += 10
            elif severity == "high":
                total_score += 7
            elif severity == "medium":
                total_score += 4
            else:
                total_score += 1
        
        # Normalize to 0-10 scale
        return min(10.0, total_score / len(findings))
    
    def _calculate_risk_trend(self, historical_data: List[Dict]) -> float:
        """Calculate risk trend over time"""
        try:
            if len(historical_data) < 2:
                return 0.0
            
            # Calculate risk scores over time
            risk_scores = []
            for data_point in historical_data:
                if "findings" in data_point:
                    risk_score = self._calculate_risk_score(data_point["findings"])
                    risk_scores.append(risk_score)
            
            if len(risk_scores) < 2:
                return 0.0
            
            # Calculate trend
            recent_avg = np.mean(risk_scores[-3:]) if len(risk_scores) >= 3 else risk_scores[-1]
            older_avg = np.mean(risk_scores[:-3]) if len(risk_scores) >= 6 else risk_scores[0]
            
            return recent_avg - older_avg
            
        except Exception:
            return 0.0
    
    async def _generate_prevention_strategies(self, issue_predictions: Dict) -> List[str]:
        """Generate prevention strategies based on predictions"""
        strategies = []
        
        for category, prediction in issue_predictions.items():
            if prediction.get("trend") == "increasing":
                if category == "security":
                    strategies.append("Implement additional security training and code review checklists")
                elif category == "performance":
                    strategies.append("Add performance testing to CI/CD pipeline")
                elif category == "quality":
                    strategies.append("Enhance code quality gates and automated testing")
                elif category == "compliance":
                    strategies.append("Strengthen compliance review processes and documentation")
        
        # Add general strategies
        strategies.append("Implement proactive monitoring and early warning systems")
        strategies.append("Establish quality metrics dashboards for real-time visibility")
        strategies.append("Regular team training on best practices and emerging threats")
        
        return strategies
    
    async def _assess_prediction_confidence(self, issue_predictions: Dict) -> Dict:
        """Assess confidence in predictions"""
        confidence_assessment = {
            "overall_confidence": 0.0,
            "category_confidence": {},
            "factors_affecting_confidence": [],
            "recommendations": []
        }
        
        # Calculate confidence by category
        total_confidence = 0
        for category, prediction in issue_predictions.items():
            confidence = prediction.get("confidence", 0.0)
            confidence_assessment["category_confidence"][category] = confidence
            total_confidence += confidence
        
        # Overall confidence
        if issue_predictions:
            confidence_assessment["overall_confidence"] = total_confidence / len(issue_predictions)
        
        # Factors affecting confidence
        if confidence_assessment["overall_confidence"] < 0.5:
            confidence_assessment["factors_affecting_confidence"].append("Limited historical data")
            confidence_assessment["factors_affecting_confidence"].append("High variability in patterns")
            confidence_assessment["recommendations"].append("Collect more historical data for better predictions")
            confidence_assessment["recommendations"].append("Implement consistent data collection processes")
        
        if confidence_assessment["overall_confidence"] < 0.7:
            confidence_assessment["recommendations"].append("Use predictions as guidance, not absolute forecasts")
            confidence_assessment["recommendations"].append("Monitor actual vs. predicted values closely")
        
        return confidence_assessment
    
    async def save_models(self):
        """Save trained models to disk"""
        try:
            for model_name, model in self.models.items():
                if model is not None:
                    model_path = self.model_dir / f"{model_name}.pkl"
                    with open(model_path, 'wb') as f:
                        pickle.dump(model, f)
                    print(f"Saved {model_name} model")
        except Exception as e:
            print(f"Failed to save models: {e}")
    
    async def load_models(self):
        """Load models from disk"""
        self._load_or_initialize_models()

# Global predictive analytics engine instance
predictive_analytics_engine = PredictiveAnalyticsEngine()

# Convenience functions
async def analyze_quality_trends(historical_data: List[Dict]) -> Dict:
    """Analyze quality trends over time"""
    return await predictive_analytics_engine.analyze_quality_trends(historical_data)

async def predict_future_issues(current_findings: List[Dict], historical_data: List[Dict]) -> Dict:
    """Predict future issues based on current state and historical patterns"""
    return await predictive_analytics_engine.predict_future_issues(current_findings, historical_data)

async def save_predictive_models():
    """Save trained predictive models"""
    await predictive_analytics_engine.save_models()

async def load_predictive_models():
    """Load trained predictive models"""
    await predictive_analytics_engine.load_models()
