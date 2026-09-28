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


def train_hgi_model(X, y):

    # Internal train/test split ONLY inside the 80% training dataset.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("\nTraining HGI LightGBM model...")

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=100,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        verbosity=-1
    )

    model.fit(
        X_train,
        y_train
    )

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

    return model



