import logging
import pickle 
import joblib
import config as cfg
from typing import Any
from pathlib import Path
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


logger = logging.getLogger(__name__)


class Dataset:
    def __init__(self):
        self.iris = load_iris()
        self.X = None
        self.y = None
        self.feature_names = None
        self.target_names = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None


    def load_data(self):
        """
        Läd den iris Datensatz ein

        Returns:
            X, y und namen der Werte
        """
        self.X = self.iris.data
        self.y = self.iris.target
        self.feature_names = self.iris.feature_names
        self.target_names = self.iris.target_names

        return self.X, self.y, self.feature_names, self.target_names


    def preprocess(self):
        """
        Teilt Datensatz jeweils in test und train Daten auf

        Raises:
            ValueError: Datensatz iris vorher laden

        Returns:
            train und test Werte
        """
        if self.X is None or self.y is None:
            logger.error("Daten müssen zuerst geladen werden.")
            raise ValueError("Daten müssen vor der Preprocessing-Methode geladen werden.")

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        
        return self.X_train, self.X_test, self.y_train, self.y_test


    def save_model(self, model: str, path: str):
        """Speichert ein Python-Objekt in einer Pickle oder Job Binary

        Args:
            model (_string_): Modelname
            path (_string_): Dateiname
        """
        if cfg.pkl:
            pkl_path = Path("../models") / f"{path}.pkl"
            try:
                file_pkl = open(pkl_path, mode = "wb") 
                pickle.dump(model, file_pkl, protocol=pickle.HIGHEST_PROTOCOL)
            except (FileNotFoundError, PermissionError) as e:
                logger.error(f"Fehler beim Speichern des Pickle-Modells: {e}")
            except pickle.PicklingError as e:
                logger.error(f"Pickling Fehler: {e}")
            except Exception as e:
                logger.exception(f"Unerwarteter Fehler beim Speichern des Pickle-Modells: {e}")
            else:
                file_pkl.close()
                logger.info(f"Model als Pickle erfolgreich gespeichert in {pkl_path}")


        if cfg.job:
            job_path = Path("../models") / f"{path}.jbl"
            try:
                joblib.dump(model, job_path, compress=5)
            except (FileNotFoundError, PermissionError) as e:
                logger.error(f"Fehler beim Speichern des Modells: {e}")
            except Exception as e:
                logger.exception(f"Unerwarteter Fehler beim Speichern: {e}")
            else:
                logger.info(f"Model als joblib erfolgreich gespeichert in {job_path}")


    def load_model(self, filename: str) -> Any:
        """
        Lädt ein Python-Objekt aus einer Pickle oder Job Datei.

        Args:
            filename (str): Pfad zur Datei.

        Returns:
            Pikel oder Joblib binary
        """
        if filename.find("pkl") != -1:
            pkl_path = Path("../models") / f"{filename}"

            try:
                file_pkl = open(pkl_path, 'rb') 
                obj = pickle.load(file_pkl)
            except FileNotFoundError:
                logging.error(f"Datei nicht gefunden: {pkl_path}")
            except pickle.UnpicklingError:
                logging.error(f"Fehler beim Laden der Pickle-Datei (ungültiges Format): {pkl_path}")
            except EOFError:
                logging.error(f"Datei ist unvollständig oder beschädigt: {pkl_path}")
            except Exception as e:
                logging.exception(f"Unerwarteter Fehler beim Laden der Pickle-Datei: {e}")
            else:
                file_pkl.close()
                logging.info(f"Pickel erfolgreich von {filename} geladen.")
                return obj

        else:
            logging.error(f"Dateiname ist kein Pikle oder Joblib, {filename}")


        if filename.find("jbl") != -1:
            job_path = Path("../models") / f"{filename}"

            try:
                file_job = joblib.load(job_path)
            except FileNotFoundError:
                logging.error(f"Datei nicht gefunden: {job_path}")
            except PermissionError:
                logging.error(f"Keine Berechtigung zum Lesen der Datei: {job_path}")
            except EOFError:
                logging.error(f"Die Datei ist beschädigt oder unvollständig: {job_path}")
            except Exception as e:
                logging.exception(f"Unerwarteter Fehler beim Laden der joblib Datei: {e}")
            else:
                logging.info(f"Joblib erfolgreich von {filename} geladen.")
                return file_job

        else:
            logging.error(f"Dateiname ist kein Pikle oder Joblib, {filename}")
