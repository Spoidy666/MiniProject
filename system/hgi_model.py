# 00000000000000000000000000000000000000000000000000000000000000000000000000000000
# NEW IMPORTS FOR MODEL PERSISTENCE AND ARTIFACT MANAGEMENT
# 00000000000000000000000000000000000000000000000000000000000000000000000000000000
import os
import pickle
from datetime import datetime
try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False
# 00000000000000000000000000000000000000000000000000000000000000000000000000000000

import lightgbm as lgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

#000000000000000000000000000000000000000000000000000000000000000000000000000000000
#for gpu acceleration
def get_lightgbm_device():
    try:
        import torch
        if torch.cuda.is_available():
            print("[LightGBM] GPU detected. Using device='gpu'.")
            return 'gpu'
    except ImportError:
        pass
    print("[LightGBM] GPU not detected or unavailable, running on CPU.")
    return 'cpu'


def train_hgi_model(X, y, save_path=None, s_values=None, feature_names=None):

    # Internal train/test split ONLY inside the 80% training dataset.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("\nTraining HGI LightGBM model...")

    device = get_lightgbm_device()   #for gpu acceleration
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=100,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        verbosity=-1,   
        device=device
    )

    #000000000000000000000000000000000000000000000000000000000000000000000000000000000
    #try catch for gpu 

    try:
        model.fit(
            X_train,
            y_train
        )
        if device != 'cpu':
            print("[LightGBM] Training successfully executed on GPU.")
    except Exception as e:
        if device != 'cpu':
            print(f"[LightGBM] GPU fit failed ({type(e).__name__}), switching to CPU.")
            model.set_params(device='cpu')
            model.fit(
                X_train,
                y_train
            )
        else:
            raise e

    predictions = model.predict(X_test)

    print("\n===== INITIAL HGI MODEL RESULTS =====")

    print(
        "Accuracy :",
        accuracy_score(y_test, predictions)
    )

    print(
        "Precision:",
        precision_score(
            y_test,
            predictions,
            zero_division=0
        )
    )

    print(
        "Recall   :",
        recall_score(
            y_test,
            predictions,
            zero_division=0
        )
    )

    print(
        "F1 Score :",
        f1_score(
            y_test,
            predictions,
            zero_division=0
        )
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=[
                "BENIGN",
                "PORT SCAN"
            ],
            zero_division=0
        )
    )

    print("\nConfusion Matrix:")

    print(
        confusion_matrix(
            y_test,
            predictions,
            labels=[0, 1]
        )
    )

    # 00000000000000000000000000000000000000000000000000000000000000000000000000000000
    # COLLECT METRICS DICTIONARY AND SAVE MODEL ARTIFACT IF REQUESTED
    # 00000000000000000000000000000000000000000000000000000000000000000000000000000000
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1_score": float(f1_score(y_test, predictions, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=[0, 1]).tolist(),
        "test_samples": int(len(y_test)),
        "train_samples": int(len(y_train))
    }

    if save_path:
        save_hgi_model(
            model=model,
            filepath=save_path,
            s_values=s_values,
            feature_names=feature_names,
            training_metrics=metrics
        )
    # 00000000000000000000000000000000000000000000000000000000000000000000000000000000

    return model


# 00000000000000000000000000000000000000000000000000000000000000000000000000000000
# MODEL SAVING AND LOADING FUNCTIONS (ARTIFACT MANAGEMENT)
# 00000000000000000000000000000000000000000000000000000000000000000000000000000000
def save_hgi_model(model, filepath, s_values=None, feature_names=None, training_metrics=None):
    """
    Saves the trained LightGBM model along with calibrated s-values,
    feature names, and training metrics into a persistent artifact.
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    artifact = {
        "model": model,
        "s_values": s_values,
        "feature_names": feature_names,
        "training_metrics": training_metrics,
        "saved_at": datetime.now().isoformat()
    }
    if HAS_JOBLIB:
        joblib.dump(artifact, filepath)
    else:
        with open(filepath, "wb") as f:
            pickle.dump(artifact, f)
    print(f"\n[Artifact Saved] Successfully saved HGI model and configuration to: {filepath}")


def load_hgi_model(filepath):
    """
    Loads a saved HGI model artifact from disk and returns the artifact dictionary.
    Keys: 'model', 's_values', 'feature_names', 'training_metrics', 'saved_at'
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model artifact not found at: {filepath}")

    if HAS_JOBLIB:
        artifact = joblib.load(filepath)
    else:
        with open(filepath, "rb") as f:
            artifact = pickle.load(f)

    print(f"\n[Artifact Loaded] Successfully loaded HGI model from: {filepath}")
    if artifact.get("s_values"):
        print(f"  Loaded calibrated s-values: {artifact['s_values']}")
    if artifact.get("training_metrics"):
        tm = artifact["training_metrics"]
        print(f"  Internal Training Metrics -> Accuracy: {tm.get('accuracy', 0):.4f}, F1: {tm.get('f1_score', 0):.4f}")
    return artifact
# 00000000000000000000000000000000000000000000000000000000000000000000000000000000



