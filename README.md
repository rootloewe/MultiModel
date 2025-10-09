# MultiModel
Dieses Python-Projekt stellt verschiedene Klassen von Logistic Regression, Decision Tree, Random Forest und KNN nebeneinadner dar,
um am Beispiel des iris Datensatzes aus Sklearn herauszufinden, welches Modell am besten ist.
Die Config.json können die Parameter zu eingestellt werden.

## Funktionen
- Die Modelle werden in pickle oder Joblib gespeichert
- Protokollierung der Schritte über Python Logging

## Installation
Stelle sicher, dass Python 3 und alle Requirements installiert ist:
pip install -r Requirements.txt


## Methoden
-------
def load_data(self):
def preprocess(self):
def save_model(self, model: str, path: str):
def load_model(self, filename: str) -> Any:
def get_pipelines(enabled_models: dict[str, bool]) -> dict[str, Pipeline]:
def test_models(iris, cfg: dict) -> dict:
def best_model(results: dict) -> dict:


## Lizenz
GNU GENERAL PUBLIC LICENSE