import os
import importlib

# Hämta den aktuella mappens namn
PACKAGE_DIR = os.path.dirname(__file__)

# Iterera över alla Python-filer i `ai_Model/`
for filename in os.listdir(PACKAGE_DIR):
    if filename.endswith(".py") and filename != "__init__.py":
        module_name = filename[:-3]  # Ta bort `.py`-suffixet
        try:
            # Dynamiskt importera modulen
            globals()[module_name] = importlib.import_module(f"ai_Model.{module_name}")
        except ImportError as e:
            print(f"Kunde inte importera {module_name}: {e}")