import json
import subprocess
import os

class CodeExecutor:
    def __init__(self, config_path="ai_Model/code_exec_config.json"):
        with open(config_path, "r") as f:
            self.config = json.load(f)

    def run_code(self, code: str, language: str) -> str:
        language = language.lower()
        if language not in self.config:
            return f"Språket '{language}' stöds inte."

        lang_conf = self.config[language]
        file_ext = lang_conf.get("file_extension", "txt")
        filename = f"temp_code.{file_ext}"

        try:
            # Skriv koden till temporär fil
            with open(filename, "w") as f:
                f.write(code)

            # Om kompilering krävs, kör den
            if "compile_command" in lang_conf:
                compile_cmd = [arg.format(filename=filename) for arg in lang_conf["compile_command"]]
                compile_proc = subprocess.run(compile_cmd, capture_output=True, text=True, timeout=10)
                if compile_proc.returncode != 0:
                    return f"Kompilationsfel:\n{compile_proc.stderr}"

            # Kör koden
            run_cmd = [arg.format(filename=filename) for arg in lang_conf["run_command"]]
            run_proc = subprocess.run(run_cmd, capture_output=True, text=True, timeout=10)

            if run_proc.returncode == 0:
                return run_proc.stdout
            else:
                return f"Fel vid körning:\n{run_proc.stderr}"

        except Exception as e:
            return f"Undantag vid körning av koden: {e}"
        finally:
            # Ta bort temporär fil (valfritt)
            if os.path.exists(filename):
                os.remove(filename)
