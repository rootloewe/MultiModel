import logging.config
from pathlib import Path
import json
import os


os.chdir(str(Path(__file__).parent.resolve()))

config_path = Path("../configs") / "config.json"
logging_pfad = Path("../configs") / "logging.ini"

logging.config.fileConfig(logging_pfad)

try:
    file = open(config_path, mode = "r", encoding= "UTF-8")
    config = json.load(file)
except FileNotFoundError as fnf_error:
    logging.error(f"config Datei nicht gefunden: {fnf_error}")
except IOError as e:
    logging.error(f"Ein I/O Fehler ist aufgetreten: {e}")
else:
    logging.info("Config.json erfolgreich geladen.")
    file.close()


pkl: bool = config["saving"]["pkl"]
job: bool = config["saving"]["job"]

enabled = config["models"]

classifier_Parameters = config["classifier_Parameters"]

APP_TITLE = config["app"]["app_title"]
APP_VERSION = config["app"]["app_version"]