__author__ = "Armin"
__version__ = "1.4.5"
__doc__ = """
Dieses Programm führt Logstic Regression Random Forest DT und KNN aus und speichert die Modelle ab, 
zur Bewertung, weches am Besten geeignet ist
Sie können Informationen darüber, wie die Auswertung erfolgen soll, über eine JSON-Datei einstellen.
"""
import config as cfg
import logging
from setting import test_models, best_model
from dataset import Dataset


logger = logging.getLogger(__name__)


def main():    
    """
    Hauptfunktion zur Ausführung der einzelnen Models zur Bestimmung des Besten.

    Parameters:
        None

    Returns:
        None
    """    
    iris = Dataset()
    iris.load_data()
    iris.preprocess()

    # results = {}  # {"model name": "Best Parameters Best Accuracy Best CV Score"}
    results = test_models(iris, cfg) 

    # Finde das Modell mit dem höchsten CV-Score aus results
    best_model_name = best_model(results)

    # Ausgabe des besten Modells
    best_result = results[best_model_name]
    logging.info(f"Bestes Modell: {best_model_name}")
    logging.info(f"Beste Kreuzvalidierungs-Score: {best_result['Best CV Score']:.4f}")
    logging.info(f"Beste Hyperparameter: {best_result['Best Parameters']}")
    logging.info(f"Test Accuracy: {best_result['Test Accuracy']:.4f}")

    # Bestes Modell laden
    file = "Logistic_Regression_v1.pkl"
    iris.load_model(file)


if __name__ == "__main__":    
    logger.info(f"Anwendung {cfg.APP_TITLE} der Version {cfg.APP_VERSION} gestartet.")
    main()
    logger.info(f"Anwendung {cfg.APP_TITLE} der Version {cfg.APP_VERSION} beendet.")