"""
Multi-model ensemble for issue difficulty classification.
"""
import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModel
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import xgboost as xgb
from typing import Dict, List, Tuple


class EnsembleIssueClassifier:
    """
    Ensemble of multiple ML models for robust issue classification.
    Combines BERT, XGBoost, and Random Forest for maximum accuracy.
    """

    def __init__(self, model_dir: str = "models/"):
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        self.bert_model = self._load_bert_model(model_dir + "bert")
        self.xgb_model = self._load_xgboost_model(model_dir + "xgboost.json")
        self.rf_model = self._load_random_forest(model_dir + "rf.pkl")

        # Model weights (tuned on validation set)
        self.weights = {
            'bert': 0.5,
            'xgboost': 0.3,
            'random_forest': 0.2
        }

    def predict(self, issue: Dict) -> Tuple[str, float, Dict]:
        """
        Predict issue difficulty with confidence scores.

        Args:
            issue: Dict with 'title', 'body', 'labels', 'repo_stats'

        Returns:
            Tuple of (difficulty_level, confidence, detailed_scores)
        """
        # Extract features
        text_features = self._extract_text_features(issue)
        numeric_features = self._extract_numeric_features(issue)

        # Get predictions from each model
        bert_pred, bert_conf = self._bert_predict(text_features)
        xgb_pred, xgb_conf = self._xgb_predict(numeric_features)
        rf_pred, rf_conf = self._rf_predict(numeric_features)

        # Ensemble voting
        final_pred, final_conf = self._ensemble_vote([
            (bert_pred, bert_conf, self.weights['bert']),
            (xgb_pred, xgb_conf, self.weights['xgboost']),
            (rf_pred, rf_conf, self.weights['random_forest'])
        ])

        difficulty_levels = ['beginner', 'intermediate', 'advanced', 'expert']

        return (
            difficulty_levels[final_pred],
            final_conf,
            {
                'bert': {'class': bert_pred, 'confidence': bert_conf},
                'xgboost': {'class': xgb_pred, 'confidence': xgb_conf},
                'random_forest': {'class': rf_pred, 'confidence': rf_conf}
            }
        )

    def _extract_text_features(self, issue: Dict) -> Dict:
        """Extract text features for BERT model."""
        text = f"{issue['title']} {issue['body']}"
        tokens = self.tokenizer(
            text,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='tf'
        )
        return tokens

    def _extract_numeric_features(self, issue: Dict) -> np.ndarray:
        """Extract numeric features for XGBoost and RF."""
        features = [
            len(issue['body']),
            len(issue['title']),
            len(issue.get('labels', [])),
            1 if 'code' in issue['body'] else 0,
            1 if 'error' in issue['body'].lower() else 0,
            issue.get('repo_stats', {}).get('stars', 0),
            issue.get('repo_stats', {}).get('contributors', 0),
            issue.get('repo_stats', {}).get('open_issues', 0),
        ]
        return np.array(features).reshape(1, -1)

    def _bert_predict(self, tokens: Dict) -> Tuple[int, float]:
        """BERT model prediction."""
        outputs = self.bert_model.predict(tokens['input_ids'], verbose=0)
        pred_class = np.argmax(outputs[0])
        confidence = float(outputs[0][pred_class])
        return int(pred_class), confidence

    def _xgb_predict(self, features: np.ndarray) -> Tuple[int, float]:
        """XGBoost prediction."""
        probs = self.xgb_model.predict_proba(features)[0]
        pred_class = np.argmax(probs)
        confidence = float(probs[pred_class])
        return int(pred_class), confidence

    def _rf_predict(self, features: np.ndarray) -> Tuple[int, float]:
        """Random Forest prediction."""
        probs = self.rf_model.predict_proba(features)[0]
        pred_class = np.argmax(probs)
        confidence = float(probs[pred_class])
        return int(pred_class), confidence

    def _ensemble_vote(self, predictions: List[Tuple]) -> Tuple[int, float]:
        """Weighted ensemble voting."""
        weighted_probs = np.zeros(4)  # 4 difficulty classes

        for pred, conf, weight in predictions:
            weighted_probs[pred] += conf * weight

        final_class = int(np.argmax(weighted_probs))
        final_conf = float(weighted_probs[final_class])

        return final_class, final_conf

    def _load_bert_model(self, path: str):
        """Load fine-tuned BERT model."""
        try:
            return tf.keras.models.load_model(path)
        except:
            # Return base model if fine-tuned not available
            return TFAutoModel.from_pretrained("microsoft/codebert-base")

    def _load_xgboost_model(self, path: str):
        """Load XGBoost model."""
        model = xgb.XGBClassifier()
        try:
            model.load_model(path)
        except:
            pass  # Return untrained model
        return model

    def _load_random_forest(self, path: str):
        """Load Random Forest model."""
        import pickle
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except:
            return RandomForestClassifier()  # Return untrained model


class RecommendationEngine:
    """Neural collaborative filtering for issue recommendations."""

    def __init__(self, embedding_dim: int = 64):
        self.embedding_dim = embedding_dim
        self.model = self._build_model()

    def _build_model(self) -> tf.keras.Model:
        """Build neural CF model."""
        # User input
        user_input = tf.keras.Input(shape=(1,), name='user_id')
        user_embedding = tf.keras.layers.Embedding(10000, self.embedding_dim)(user_input)
        user_vec = tf.keras.layers.Flatten()(user_embedding)

        # Issue input
        issue_input = tf.keras.Input(shape=(1,), name='issue_id')
        issue_embedding = tf.keras.layers.Embedding(100000, self.embedding_dim)(issue_input)
        issue_vec = tf.keras.layers.Flatten()(issue_embedding)

        # Concatenate and process
        concat = tf.keras.layers.Concatenate()([user_vec, issue_vec])
        x = tf.keras.layers.Dense(128, activation='relu')(concat)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        output = tf.keras.layers.Dense(1, activation='sigmoid')(x)

        model = tf.keras.Model(inputs=[user_input, issue_input], outputs=output)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        return model

    def recommend(self, user_id: int, candidate_issues: List[int], top_k: int = 10) -> List[Tuple[int, float]]:
        """Get top-k issue recommendations for user."""
        user_ids = np.array([user_id] * len(candidate_issues))
        issue_ids = np.array(candidate_issues)

        scores = self.model.predict([user_ids, issue_ids], verbose=0).flatten()

        # Get top-k
        top_indices = np.argsort(scores)[-top_k:][::-1]
        recommendations = [(candidate_issues[i], float(scores[i])) for i in top_indices]

        return recommendations
