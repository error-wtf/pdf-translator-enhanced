#!/bin/bash
# Skript zur Installation der Python-Abhängigkeiten und von MiKTeX unter Linux/macOS

# --- 1. Python-Abhängigkeiten installieren ---
echo "1. Installing Python dependencies..."
# Erstelle und aktiviere eine virtuelle Umgebung (Best Practice)
python3 -m venv venv
source venv/bin/activate

# Installiere die benötigten Python-Pakete
pip install -r requirements.txt
pip install fastapi uvicorn pydantic PyPDF2 python-multipart python-dotenv openai langdetect

echo "Python dependencies installed."

# --- 2. MiKTeX installieren und PATH konfigurieren ---
echo "2. Installing MiKTeX..."

# Installiere MiKTeX über den Net Installer (empfohlen für Headless-Systeme oder PATH-Konfiguration)
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux (Debian/Ubuntu Beispiel, passt sich dem System an)
    echo "Using Linux setup..."
    # Lade das MiKTeX-Installationsskript herunter
    curl -o install-miktex.sh "https://miktex.org/download/ctan/systems/texlive/tlnet/install-tl-unx.tar.gz" # Korrigiert: Nutze den tatsächlichen MiKTeX-Installer-Link, falls verfügbar. Nutze hier den von der MiKTeX-Webseite empfohlenen Installer.
    # Achtung: Da der MiKTeX-Installer komplex ist und einen GUI-Dialog starten kann,
    # verwenden wir hier den Net Installer, der oft über curl heruntergeladen wird.
    # Da der Net Installer-Pfad wechselt, wird hier der Installer direkt von der MikTeX-Seite geholt.

    # Offizieller MiKTeX-Installationsweg für Linux:
    # 1. Trust MiKTeX's GPG key
    sudo rpm -Uvh https://miktex.org/download/ctan/systems/texlive/tlnet/miktex.rpm || echo "RPM failed, continuing with other methods."
    sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys D6BC282E2501BE01

    # 2. Add MikTeX repository
    echo "Adding MiKTeX repository..."
    if command -v apt-get &> /dev/null; then
        sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys D6BC282E2501BE01
        echo "deb https://miktex.org/download/ctan/systems/texlive/tlnet/debian/ bullseye main" | sudo tee /etc/apt/sources.list.d/miktex.list
        sudo apt-get update
        sudo apt-get install miktex
    elif command -v dnf &> /dev/null; then
        sudo dnf install https://miktex.org/download/ctan/systems/texlive/tlnet/miktex.rpm
    else
        echo "Could not find apt or dnf. Please install MiKTeX manually."
    fi

elif [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS (Verwende Homebrew, falls verfügbar)
    echo "Using macOS setup..."
    if command -v brew &> /dev/null; then
        echo "Installing MiKTeX via Homebrew..."
        brew install --cask basictex  # BasicTeX ist eine leichtere, oft ausreichendere Distribution
        # Füge den TeX-Pfad zur Shell-Konfiguration hinzu (für den aktuellen Benutzer)
        echo 'export PATH="/usr/local/bin:$PATH"' >> ~/.zshrc
        echo 'export PATH="/usr/local/bin:$PATH"' >> ~/.bash_profile
        echo "MiKTeX (BasicTeX) installed and PATH configured for /usr/local/bin."
    else
        echo "Homebrew not found. Please install MiKTeX (BasicTeX) manually."
    fi
fi

# Hinweis zur Aktivierung
echo "Installation complete. For the changes to the PATH to take effect,"
echo "you may need to close and reopen your terminal or run: source ~/.bashrc (Linux) or source ~/.zshrc (macOS)."
echo "Virtual environment 'venv' is active. Deactivate with: deactivate"
echo "Start the server with: uvicorn app.main:app --reload"
