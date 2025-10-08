from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold


def get_pipelines(enabled_models: dict[str, bool]) -> dict[str, Pipeline]:
    """
    Liefert je nach Einstellungen in config.json die benötigten Parameter für Pipline

    Args:
        enabled_models (dict[str, bool]): Werte aus json, was benötigt wird

    Returns:
        dict[str, Pipeline]: Paramter Dictionary für Piplines
    """
    pipelines = {}

    if enabled_models.get("Logistic Regression", False):
        pipelines["Logistic Regression"] = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression())
        ])

    if enabled_models.get("Random Forest", False):
        pipelines["Random Forest"] = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", RandomForestClassifier())
        ])

    if enabled_models.get("K-Nearest Neighbors", False):
        pipelines["K-Nearest Neighbors"] = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", KNeighborsClassifier())
        ])

    if enabled_models.get("Decision Tree", False):
        pipelines["Decision Tree"] = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", DecisionTreeClassifier())
        ])

    return pipelines


def test_models(iris, cfg: dict) -> dict:
    """
    Führt GridSearchCV für mehrere Modelle/Pipelines mit gegebenen Parametern und Cross-Validation durch.

    Args:
        iris (Any): Objekt mit Attributen X_train, y_train, X_test, y_test sowie Methode save_model.
        pipelines (Dict[str, Pipeline]): Dictionary von Modellnamen zu sklearn Pipeline-Objekten.
        classifier_Parameters (Dict[str, Dict[str, list]]): Dictionary von Modellnamen zu Parameterrastern für GridSearchCV.
        
        Die Parameter müssen den Pipeline-Schritt-Prefix enthalten (z.B. 'classifier__C').
        stratified_kfold (BaseCrossValidator): Cross-Validator-Instanz, z.B. StratifiedKFold.

        cfg: Dictionary aus json Datei

    Returns:
        Dict[str, Dict[str, Any]]: Dictionary mit Modellnamen als Schlüssel und Werten:
            {
                "Best Parameters": dict der besten Hyperparameter,
                "Test Accuracy": float, Genauigkeit auf Testdaten,
                "Best CV Score": float, bestes Cross-Validation-Scoring
            }
    """
    # Hyperparameter festlegen für jeden classifier
    classifier_Parameters = cfg.classifier_Parameters
    pipelines = get_pipelines(cfg.enabled)
    stratified_kfold = StratifiedKFold(n_splits=10, shuffle=True)

    results = {}  # {"model name": "Best Parameters Best Accuracy Best CV Score"}

    for model_name, pipeline in pipelines.items():
        param_grid = classifier_Parameters[model_name]
        grid_search = GridSearchCV(pipeline, param_grid, cv=stratified_kfold, n_jobs=-1)
        grid_search.fit(iris.X_train, iris.y_train)
        best_model = grid_search.best_estimator_
        test_accuracy = best_model.score(iris.X_test, iris.y_test)
        
        results[model_name] = {
            "Best Parameters": grid_search.best_params_,
            "Test Accuracy": test_accuracy,
            "Best CV Score": grid_search.best_score_
        }

        filename = model_name.replace(" ", "_") + "_v1"
        iris.save_model(best_model, filename)

    return results


def best_model(results: dict) -> dict:
    """findet den besten Modelnamen und die Paramenter

    Args:
        results (_dict_): Dictionary der Models mit Parameter

    Returns:
        Dict[str, Dict[str, Any]]: Dictionary mit besten Modell als Schlüssel und Werten:
            {
                "Best Parameters": dict der besten Hyperparameter,
                "Test Accuracy": float, Genauigkeit auf Testdaten,
                "Best CV Score": float, bestes Cross-Validation-Scoring
            }
    """
    best_model_name = None
    best_cv_score = -float('inf')  # Sehr kleiner Startwert

    for model_name, result in results.items():
        if result["Best CV Score"] > best_cv_score:
            best_cv_score = result["Best CV Score"]
            best_model_name = model_name

    return best_model_name