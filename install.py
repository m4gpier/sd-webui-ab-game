import subprocess
import sys


def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


try:
    import wordfreq
    print("[prompt-evolution] wordfreq already installed.")
except ImportError:
    print("[prompt-evolution] Installing wordfreq...")
    install("wordfreq")
    print("[prompt-evolution] wordfreq installed successfully.")
