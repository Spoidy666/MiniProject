import numpy as np
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

class EnsembleNIDS:
    def __init__(self, models_and_feats=None):
        if models_and_feats is not None:
            self.models_and_feats = models_and_feats
        else:
            self.models_and_feats = [
                (RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1), 'nrf'),
                (LGBMClassifier(random_state=42, verbosity=-1, n_jobs=-1), 'hgi'),
                (LGBMClassifier(random_state=42, verbosity=-1, n_jobs=-1), 'hga')
            ]
            
    def fit(self, X_nrf, X_hgi, X_hga, y):
        feature_map = {'nrf': X_nrf, 'hgi': X_hgi, 'hga': X_hga}
        for model, feat_type in self.models_and_feats:
            model.fit(feature_map[feat_type], y)
            
    def predict_proba(self, X_nrf, X_hgi, X_hga):
        feature_map = {'nrf': X_nrf, 'hgi': X_hgi, 'hga': X_hga}
        probas = []
        for model, feat_type in self.models_and_feats:
            p = self._get_proba_class_1(model, feature_map[feat_type])
            probas.append(p)
        return np.mean(probas, axis=0)
        
    def predict(self, X_nrf, X_hgi, X_hga, threshold=0.5):
        p_ensemble = self.predict_proba(X_nrf, X_hgi, X_hga)
        return np.where(p_ensemble >= threshold, 1, 0)
        
    def _get_proba_class_1(self, model, X):
        proba = model.predict_proba(X)
        if proba.shape[1] == 2:
            return proba[:, 1]
        else:
            if len(model.classes_) > 0 and (model.classes_[0] == 1 or model.classes_[0] == 'PORT SCAN'):
                return np.ones(len(X))
            else:
                return np.zeros(len(X))
                
    def copy(self):
        """
        Creates a copy of the ensemble (referencing the underlying trained models).
        """
        return EnsembleNIDS(list(self.models_and_feats))
