import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

_LGBM_DEVICE_LOGGED = False
_RF_DEVICE_LOGGED = False


def get_lightgbm_device():
    """
    Checks if a GPU is available for LightGBM.
    Returns 'gpu' if available, otherwise falls back to 'cpu' and prints a message.
    """
    global _LGBM_DEVICE_LOGGED
    device = 'cpu'

    # Check via PyTorch if CUDA GPU is available
    try:
        import torch
        if torch.cuda.is_available():
            device = 'gpu'
    except ImportError:
        pass

    if device == 'cpu' and not _LGBM_DEVICE_LOGGED:
        print("[LightGBM] GPU not detected or unavailable, running on CPU.")
        _LGBM_DEVICE_LOGGED = True
    elif device == 'gpu' and not _LGBM_DEVICE_LOGGED:
        print("[LightGBM] GPU detected. Setting device='gpu'.")
        _LGBM_DEVICE_LOGGED = True

    return device


class SafeLGBMClassifier(LGBMClassifier):
    """
    LightGBM Classifier with automatic GPU enablement and runtime CPU fallback.
    """
    def __init__(self, **kwargs):
        kwargs.setdefault('random_state', 42)
        kwargs.setdefault('verbosity', -1)
        kwargs.setdefault('n_jobs', -1)
        if 'device' not in kwargs and 'device_type' not in kwargs:
            kwargs['device'] = get_lightgbm_device()
        super().__init__(**kwargs)

    def fit(self, X, y, **kwargs):
        current_device = self.get_params().get('device', 'cpu')
        if current_device != 'cpu':
            try:
                res = super().fit(X, y, **kwargs)
                print("[LightGBM] Training successfully executed on GPU.")
                return res
            except Exception as e:
                print(f"[LightGBM] GPU mode failed ({type(e).__name__}), switching to CPU.")
                self.set_params(device='cpu')
                return super().fit(X, y, **kwargs)
        return super().fit(X, y, **kwargs)


class SafeRandomForestClassifier:
    """
    GPU-accelerated Random Forest using cuML if available,
    otherwise falls back to Scikit-Learn on CPU with a notification.
    """
    def __init__(self, n_estimators=100, random_state=42, n_jobs=-1, **kwargs):
        global _RF_DEVICE_LOGGED
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.kwargs = kwargs
        self.is_gpu = False

        try:
            from cuml.ensemble import RandomForestClassifier as cuRF
            self.model = cuRF(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                output_type='numpy',
                **self.kwargs
            )
            self.is_gpu = True
            if not _RF_DEVICE_LOGGED:
                print("[RandomForest] GPU detected (cuML). Running on GPU.")
                _RF_DEVICE_LOGGED = True
        except (ImportError, Exception):
            if not _RF_DEVICE_LOGGED:
                print("[RandomForest] cuML (GPU) not available, running on CPU (scikit-learn).")
                _RF_DEVICE_LOGGED = True
            self.model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                **self.kwargs
            )

    def fit(self, X, y, **fit_kwargs):
        if self.is_gpu:
            try:
                X_arr = np.ascontiguousarray(X, dtype=np.float32)
                y_arr = np.ascontiguousarray(y, dtype=np.int32)
                res = self.model.fit(X_arr, y_arr, **fit_kwargs)
                print("[RandomForest] Training successfully executed on GPU (cuML).")
                return res
            except Exception as e:
                print(f"[RandomForest] cuML GPU fit failed ({type(e).__name__}), switching to CPU.")
                self.is_gpu = False
                self.model = RandomForestClassifier(
                    n_estimators=self.n_estimators,
                    random_state=self.random_state,
                    n_jobs=self.n_jobs,
                    **self.kwargs
                )
                return self.model.fit(X, y, **fit_kwargs)
        return self.model.fit(X, y, **fit_kwargs)

    def predict(self, X):
        if self.is_gpu:
            X_arr = np.ascontiguousarray(X, dtype=np.float32)
            preds = self.model.predict(X_arr)
            return np.asarray(preds)
        return self.model.predict(X)

    def predict_proba(self, X):
        if self.is_gpu:
            X_arr = np.ascontiguousarray(X, dtype=np.float32)
            proba = self.model.predict_proba(X_arr)
            return np.asarray(proba)
        return self.model.predict_proba(X)

    @property
    def classes_(self):
        return getattr(self.model, 'classes_', np.array([0, 1]))


class EnsembleNIDS:
    def __init__(self, models_and_feats=None):
        if models_and_feats is not None:
            self.models_and_feats = models_and_feats
        else:
            self.models_and_feats = [
                (SafeRandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1), 'nrf'),
                (SafeLGBMClassifier(random_state=42, verbosity=-1, n_jobs=-1), 'hgi'),
                (SafeLGBMClassifier(random_state=42, verbosity=-1, n_jobs=-1), 'hga')
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
