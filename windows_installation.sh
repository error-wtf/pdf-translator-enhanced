# PowerShell-Skript zur Installation der Python-Abhängigkeiten und von MiKTeX unter Windows

# --- 1. Python-Abhängigkeiten installieren ---
Write-Host "1. Installing Python dependencies..."

# Überprüfe, ob Python installiert ist
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "Python not found. Please install Python and ensure it is in your PATH."
    exit 1
}

# Erstelle und aktiviere eine virtuelle Umgebung (Best Practice)
python -m venv venv
.\venv\Scripts\activate

# Installiere die benötigten Python-Pakete
pip install -r requirements.txt
pip install fastapi uvicorn pydantic PyPDF2 python-multipart python-dotenv openai langdetect

Write-Host "Python dependencies installed."

# --- 2. MiKTeX installieren und PATH konfigurieren ---
Write-Host "2. Installing MiKTeX..."

# MiKTeX Net Installer herunterladen
$installerUrl = "https://miktex.org/download/ctan/systems/miktex/setup/windows/setup-2.9.exe"
$installerPath = Join-Path $env:TEMP "MiKTeXSetup.exe"

Write-Host "Downloading MiKTeX Net Installer..."
Invoke-WebRequest -Uri $installerUrl -OutFile $installerPath

# MiKTeX Net Installer ausführen
# Wichtig: Der Installer muss mit Administratorrechten ausgeführt werden,
# um eine systemweite Installation (und PATH-Konfiguration) zu gewährleisten.
Write-Host "Starting MiKTeX Installer. Please choose 'Install for all users' and follow the prompts."
Write-Host "You may need to run this PowerShell script as Administrator!"

Start-Process -FilePath $installerPath -Wait

# Pfad nach der Installation testen (MiKTeX installiert standardmäßig nach C:\Program Files\MiKTeX\miktex\bin\x64)
# Da der genaue Pfad variieren kann (32-Bit/64-Bit, Version), verlassen wir uns auf die PATH-Konfiguration durch den Installer.
Write-Host "MiKTeX installation finished. Please confirm 'pdflatex' is available in your PATH."

# Hinweis zur Aktivierung
Write-Host "Installation complete. The virtual environment 'venv' is active."
Write-Host "Deactivate with: deactivate"
Write-Host "Start the server with: .\venv\Scripts\uvicorn app.main:app --reload"
