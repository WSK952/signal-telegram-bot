# Utiliser une image officielle Python 3.10
FROM python:3.10-slim

# Définir le dossier de travail
WORKDIR /app

# Copier tous les fichiers du repo dans le conteneur
COPY . .

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Lancer le bot
CMD ["python", "main.py"]
