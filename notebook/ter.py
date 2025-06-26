#!/usr/bin/env python
# coding: utf-8

# # TER - Indicateurs de mobilité durable dans l'Eurométropole de Strasbourg <a class="jp-toc-ignore"></a>
# Fabian MATHIEU (M1 OTG) - 2025
# 
# Ce notebook python sert à calculer des indicateurs pour déterminer le caractère durable (dans le sens du développement durable) des moyens de transports dans un EPCI. Je préconise de l'ouvrir via Jupyter Lab, car les opérations spatiales les plus complexes demandent beaucoup de ram, et feront planter google colab.
# 
# D'abord concu pour calculer ceux de l'Eurométropole de Strasbourg (EMS), il peut cependant être adapté pour d'autres métropoles. Si c'est le cas, je conseille de créer une copie de ce notebook avant toute modifications : il pourra servir de référence.
# 
# Pour télécharger les données, ce notebook à besoin d'un .csv 'tableau_sources_donnees.csv' contenant notamment leurs liens de téléchargement (voir partie 1.4) : ce fichier est le seul à avoir besoin d'être importé dans le notebook. Si ce code venait à être réutilisé, il faudra modifier ce fichier pour y intégrer des données actuelles, et trouver des équivalements aux données non-disponibles à l'échelle de la France (voir partie 3).

# ## 1. Initialisation
# ---

# ### 1.1. Installation et import des libraires
# --- 
# * Pandas : permet de manipuler des tableaux de données - https://pandas.pydata.org/docs/
# * Geopandas : permet de manipuler des tableaux avec des données spatialisées - https://geopandas.org/en/stable/docs.html
# * py7zr : décompresser des fichiers .zip, .7z, ou autres - https://py7zr.readthedocs.io/en/latest/user_guide.html
# * Contextily : permet d'ajouter des fonds de carte - https://contextily.readthedocs.io/en/latest/intro_guide.html
# * Matplotlib : utilisé conjointement avec contextily pour afficher des cartes - https://matplotlib.org/stable/api/index.html
# * tqdm : ajoute des barres de progression - https://tqdm.github.io/
# * authlib / requests-oauthli : utilisés lors de la connexion aux API - https://oauthlib.readthedocs.io/en/latest/index.html / https://requests-oauthlib.readthedocs.io/en/latest
# * Rasterio : similaire à GDAL, mais est plus intéropérable - https://rasterio.readthedocs.io/en/stable/quickstart.html
# * Pillow : permet de manipuler des images - https://pillow.readthedocs.io/en/stable/handbook/tutorial.html
# * Aria2 : un protocole de transfert très performant, permettant de télécharger plusieurs fichiers simultanéments ou un seul fichier avec plusieurs connexions - https://pypi.org/project/aria2p
# * Joblib : permet de faire plusieurs calculs en parallèle - https://joblib.readthedocs.io/en/stable/
# * Pyarrow : permet d'utiliser le format .parquet (plus rapide pour les traitements python) - https://joblib.readthedocs.io/en/stable/

# In[15]:


pip install pandas geopandas requests py7zr contextily matplotlib shapely tqdm oauthlib requests-oauthlib rasterio pillow aria2 networkx joblib pyarrow osmnx beautifulsoup4 unidecode scipy --quiet


# In[ ]:


import sys
sys.path.append(r'C:\Users\fabian.mathieu\AppData\Roaming\Python\Python311\site-packages')
import pandas as gpd


# In[16]:


# Utilisés pour télécharger des fichiers avec aria2c
import os
import shutil
import subprocess
import sys
# Utilisé pour faire des barres des progression
from tqdm import tqdm
# Utilisé pour traiter des tableaux de données (dataframe / geodataframe)
import pandas as pd
import geopandas as gpd
# Utilisé pour manipuler nos graphes
import networkx as nx
# Utilisé pour ne pas surcharger l'API sirene
import time
# Utilisé pour ajouter des fonds de carte
import contextily as ctx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
# Dézipper des fichiers compressés
import py7zr
import requests, zipfile, os, shutil
from pathlib import Path
# Utilisé pour manipuler des géométries
import shapely
from shapely.geometry import MultiLineString, LineString, Polygon, Point, shape, box, mapping
from shapely.prepared import prep
from shapely.ops import linemerge, unary_union, substring, nearest_points, split
from shapely import force_2d
import fiona
from tqdm import tqdm
# Utilisé pour interroger l'API Copernicus - Sentinel 2
from oauthlib.oauth2 import BackendApplicationClient
from requests_oauthlib import OAuth2Session
# Utilisé pour afficher et manipuler des images
from PIL import Image
# Utilisé pour gérer des opérations d'entrée / sortie des fichiers
import io
# Utilisé pour des calculs numériques
import numpy as np
# Utilisé pour manipuler des données raster géospatiales
import rasterio
from rasterio.mask import mask
from rasterio.merge import merge
from rasterio.enums import Resampling
from rasterio.shutil import copy as rio_copy
from rasterio.enums import Compression
#  Utilisé pour manipuler des formats datetime 
from datetime import datetime, timedelta
# Utilisé pour récupérer les informations de Citiz (Autopartage de voitures)
import json
# utilisé pour afficher la taille des variables en mémoire
from IPython import get_ipython
from IPython.display import display
# Utilisé pour explorer les fichiers ou dossiers suivant un certain pattern
import glob
# Utilisé pour construire le graphe routier
import networkx as nx
# Permet de paralléliser les calculs d'itinéraires
import joblib
from joblib import Parallel, delayed
# Utilisé pour calculer des valeurs aléatoires
import random
# Utilisé pour exporter les graphes dans un format compris par igraph
import pickle
# Utilisé pour utiliser le format paquet
import pyarrow
import pyarrow.parquet as pq
# Test
import numpy as np
# Utilisé pour savoir quelles routes du graphe sont inutilisées
from collections import Counter
# Utilisé pour l'aléatoire
import math
# Utilisé pour manipuler des dictionnaires
from collections import defaultdict
# Utilisé pour modifier les paramètres d'enregistrement des .csv quand nécessaire
import csv
# Utilisé pour convertir les chaînes de type liste/dictionnaire
import ast
# Utilisé pour récupérer des données OSM
import osmnx as ox
# Utilisé pour télécharger les .csv pour les infos sur les parkings de l'EMS
from bs4 import BeautifulSoup
# Utilisé pour les expressions régulières (re : reguliar expressions)
import re
# Utilisé pour convertir des caractères unicode (à, é, ù) en leur version ascii
import unidecode
# Utilisé pour analyser différents formats de dates et d'heures
from dateutil.parser import parse
# Utilisé pour créer des répertoires temporaires
from tempfile import TemporaryDirectory
# Utilisé pour traiter des données sur un calendrier
import calendar
# Utilisé pour créer un index spatial des voisins les plus proches, accélère les calculs
from scipy.spatial import cKDTree

print("Toutes les libraires ont été importées avec succès")


# ### 1.2. Création des sous-dossiers pour stocker les données
# ---
# Créé un sous-dossier (nommé 'data' par défaut) à l'endroit où ce fichier .ipynb est exécuté. Afin de mieux s'y retrouver, deux sous-dossiers y sont créés : les fichiers .zip et .7z décompressés sont mis dans "extraction", "exports" contient les fichiers traités, prêts pour le calcul d'indicateurs.
# 
# Les fichiers sont sont exportés en 2 formats open source, et peuvent être ouvert dans des SIG pour vérification : 
# 1. gpkg : un format universel
# 2. parquet : souvent plus petit pour les mêmes données, mais semble ne pas fonctionner avec QGIS pour linux pour le moment
# 
# Documentation : 
# * https://fr.wikipedia.org/wiki/Geopackage
# * https://fr.wikipedia.org/wiki/Apache_Parquet

# In[17]:


dir = "data"
os.makedirs(dir, exist_ok=True)

# Répertoire où les fichiers compressés (.zip, .7z) seront extraits
extraction_dir = os.path.join(dir, "extraction")
os.makedirs(extraction_dir, exist_ok=True)

# Répertoire "de travail", où les données seront présentes
exports_dir = os.path.join(dir, "exports")
os.makedirs(exports_dir, exist_ok=True)

# Répertoire contenant les données utilisées par le tableau de bord
tableau_bord_dir = os.path.join(dir, "tableau_bord")
os.makedirs(tableau_bord_dir, exist_ok=True)

# Répertoire où sont exportées les images (optionnelles) issues des visualisations
images_dir = os.path.join(dir, "images")
os.makedirs(tableau_bord_dir, exist_ok=True)


# ### 1.3. Définition des fonctions utilisées dans la suite du notebook
# ---

# In[18]:


# Cette fonction utilise aria2c dès que possible à la place du module 'requests' de base utilisé sur python : 
# cette librairie permet de télécharger plusieurs fichiers en même temps, et un même fichier avec plusieurs 
# connexions. Par exemple, le téléchargement du MNT passe d'1h10 à 2-3 minutes
def telecharger_fichier(url, destination, verbose=True):
    def aria2_disponible():
        return shutil.which("aria2c") is not None

    if aria2_disponible():
        cmd = [
            "aria2c",
            "-x", "8",  # nombre de connexions par fichier
            "-s", "8",  # nombre de connexions en parallèle
            "-d", os.path.dirname(destination),
            "-o", os.path.basename(destination),
            url
        ]

        if not verbose:
            cmd.insert(1, "--quiet=true")

        try:
            subprocess.run(cmd, check=True)
            if verbose:
                print(f"Téléchargement terminé : {destination}")
            return
        # Si problème : téléchargement classique via requests
        except subprocess.CalledProcessError:
            if verbose:
                print("Échec avec aria2c, bascule vers requests.")

    if verbose:
        print(f"Téléchargement depuis : {url}")

    with requests.get(url, stream=True) as r:
        total_length = r.headers.get('content-length')
        total_length = int(total_length) if total_length and total_length.isdigit() else 0
        chunk_size = 8192
        downloaded = 0

        with open(destination, 'wb') as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)

                    if verbose and total_length > 0:
                        done_kb = downloaded / 1024
                        total_kb = total_length / 1024
                        percent = downloaded * 100 / total_length
                        print(f"\r Téléchargement : {done_kb:.1f} KB / {total_kb:.1f} KB ({percent:.1f}%)", end='')

    if verbose:
        print("\n Téléchargement terminé.")


# In[19]:


# Idem que la précédente mais pour plusieurs fichiers
def telecharger_fichiers(urls, dossier_destination):
    os.makedirs(dossier_destination, exist_ok=True)

    def aria2_disponible():
        return shutil.which("aria2c") is not None

    if aria2_disponible():
        # Crée un fichier temporaire de configuration pour aria2c
        input_file = os.path.join(dossier_destination, "aria2_input.txt")
        with open(input_file, "w") as f:
            for i, url in enumerate(urls):
                filename = os.path.join(dossier_destination, f"zip_{i:04d}.zip")
                if not os.path.exists(filename):
                    f.write(f"{url}\n\tout={os.path.basename(filename)}\n")

        # Téléchargement via aria2c (8 connexions par fichier, 4 fichiers en parallèle)
        cmd = [
            "aria2c",
            "-x", "8",
            "-j", "4",
            "-d", dossier_destination,
            "-i", input_file
        ]

        print("Téléchargement des fichiers avec aria2c...")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print("Échec de aria2c. Aucun fichier téléchargé.")
        finally:
            os.remove(input_file)
        return

    # Fallback si aria2c indisponible
    print("aria2c non disponible, téléchargement avec requests (lent)...")

    for i, url in enumerate(tqdm(urls, desc="Téléchargement des fichiers")):
        filename = os.path.join(dossier_destination, f"zip_{i:04d}.zip")
        if not os.path.exists(filename):
            try:
                telecharger_fichier(url, filename, verbose=False)
            except Exception as e:
                print(f"Erreur lors du téléchargement : {url}\n{e}")


# In[20]:


# # Extrait un .zip (zip_path) dans un répertoire cible (extract_path)
def extraire_zip(zip_path, extract_path, verbose=True):
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        os.remove(zip_path)
        if verbose:
            print(f"ZIP extrait et supprimé : {zip_path}")
    except zipfile.BadZipFile:
        print(f"[!] Fichier corrompu ou invalide (non ZIP) : {zip_path}")
    except Exception as e:
        print(f"[!] Erreur lors de l'extraction de {zip_path} : {e}")


# In[21]:


# Extrait un .7z (sevenz_path) dans un répertoire cible (extract_path)
def extraire_7z(sevenz_path, extract_path, verbose=True):
    try:
        with py7zr.SevenZipFile(sevenz_path, mode='r') as z:
            z.extractall(path=extract_path)
        os.remove(sevenz_path)
        if verbose:
            print(f"7z extrait et supprimé : {sevenz_path}")
    except py7zr.Bad7zFile:
        print(f"[!] Fichier corrompu ou invalide (non 7z) : {sevenz_path}")
    except Exception as e:
        print(f"[!] Erreur lors de l'extraction de {sevenz_path} : {e}")


# In[22]:


# Extrait un .zip (zip_path) dans un répertoire cible (extract_path),
# puis extrait tous les .7z trouvés à l’intérieur.
def extraire_zip_puis_7z(zip_path, extract_path):
    tmp_dir = os.path.join(extract_path, "_tmp_zip")
    os.makedirs(tmp_dir, exist_ok=True)
    extraire_zip(zip_path, tmp_dir)
    for root, _, files in os.walk(tmp_dir):
        for file in files:
            if file.lower().endswith(".7z"):
                sevenz_path = os.path.join(root, file)
                extraire_7z(sevenz_path, extract_path)
    print(f"Tous les .7z extraits dans : {extract_path}")


# In[23]:


# Cherche récursivement un fichier dans un dossier, renvoie le chemin complet vers le fichier trouvé.
# nom_fichier (str): Nom du fichier à rechercher 
# dossier_base (str): Dossier racine à explorer.
def trouver_fichier(nom_fichier, dossier_base):
    for root, _, files in os.walk(dossier_base):
        for file in files:
            if file.lower() == nom_fichier.lower():
                return os.path.join(root, file)
    raise FileNotFoundError(f"Fichier '{nom_fichier}' introuvable dans '{dossier_base}'")


# In[24]:


# Récupère l'url de téléchargement dans le fichier sources_donnees.csv correspondant au nom donné 
def trouver_source_url(nom_source):
    ligne = source_donnees[source_donnees['nom'] == nom_source].iloc[0]
    return ligne['url_telechargement']


# In[25]:


def definir_chemins(nom_source, ext="geojson"):
    zip_path = os.path.join(dir, f"{nom_source}.zip")
    extract_path = os.path.join(extraction_dir, nom_source)
    export_path = os.path.join(exports_dir, f"{nom_source}.{ext}")
    return zip_path, extract_path, export_path


# In[26]:


def decouper_par_epci(gdf, limites_epci, predicate="intersects"):
    if gdf.crs != limites_epci.crs:
        gdf = gdf.to_crs(limites_epci.crs)
    return gpd.sjoin(gdf, limites_epci, how="inner", predicate=predicate)


# In[27]:


# Exporte un GeoDataFrame en .geosjon
def exporter_geojson(gdf, nom_fichier, dossier=exports_dir):
    path = os.path.join(dossier, f"{nom_fichier}.geojson")
    gdf.to_file(path, driver="GeoJSON")
    print(f"Fichier '{path}' exporté avec succès.")


# In[28]:


# Exporte un GeoDataFrame en .gpkg
def exporter_gpkg(gdf, nom_fichier):
    path = os.path.join(exports_dir, f"{nom_fichier}.gpkg")
    gdf.to_file(path, driver="GPKG", layer=nom_fichier)
    print(f"Fichier '{path}' exporté avec succès.")


# In[29]:


# Exporte un GeoDataFrame en .parquet
def exporter_parquet(gdf, nom_fichier):
    path = os.path.join(exports_dir, f"{nom_fichier}.parquet")
    gdf.to_parquet(path, engine="pyarrow")
    print(f"Fichier '{path}' exporté avec succès.")


# In[30]:


# ANCIENNE VERSION
"""
def charger_fichier_parquet(nom_fichier, crs=None):
    path = os.path.join(exports_dir, nom_fichier + ".parquet")
    gdf = gpd.read_parquet(path)
    if crs is not None:
        gdf = gdf.to_crs(crs)
    return gdf
"""


# In[31]:


# Charge un fichier parquet, avec conversion du crs si besoin
def charger_fichier_parquet(nom_fichier, crs=None):
    path = os.path.join(exports_dir, nom_fichier + ".parquet")

    gdf = gpd.read_parquet(path)

    # Par défaut, il semble que l'enregistrement en fichier .parquet convertisse
    # les listes python en array numpy. En attedant de comprendre et de corriger ça,
    # on fait une conversion array numpy -> liste python lors du chargement
    # pour les colonnes concernées
    for col in gdf.columns:
        # Vérifie si la colonne contient des arrays numpy
        if gdf[col].apply(lambda x: isinstance(x, np.ndarray)).any():
            gdf[col] = gdf[col].apply(
                lambda x: x.tolist() if isinstance(x, np.ndarray) else (
                    list(x) if hasattr(x, '__iter__') and not isinstance(x, (str, bytes)) else x
                )
            )

    # Reprojection si nécessaire
    if crs is not None:
        gdf = gdf.to_crs(crs)

    return gdf


# In[32]:


# Charge un fichier GeoPackage (.gpkg), avec conversion du crs si besoin
def charger_fichier_gpkg(nom_fichier, crs=None):
    path = os.path.join(exports_dir, nom_fichier + ".gpkg")

    gdf = gpd.read_file(path)

    # Conversion array numpy -> liste python (même logique que pour le .parquet)
    for col in gdf.columns:
        if gdf[col].apply(lambda x: isinstance(x, np.ndarray)).any():
            gdf[col] = gdf[col].apply(
                lambda x: x.tolist() if isinstance(x, np.ndarray) else (
                    list(x) if hasattr(x, '__iter__') and not isinstance(x, (str, bytes)) else x
                )
            )

    # Reprojection si nécessaire
    if crs is not None:
        gdf = gdf.to_crs(crs)

    return gdf


# In[33]:


# Chargement d’un fichier GeoJSON, avec conversion du CRS si besoin
def charger_fichier_geojson(nom_fichier, crs=None, dossier=exports_dir):
    # 1. Construction du chemin
    path = os.path.join(dossier, f"{nom_fichier}.geojson")

    # 2. Lecture du fichier
    gdf = gpd.read_file(path)

    # 3. Conversion des colonnes contenant des arrays en listes Python
    for col in gdf.columns:
        if gdf[col].apply(lambda x: isinstance(x, np.ndarray)).any():
            gdf[col] = gdf[col].apply(
                lambda x: x.tolist() if isinstance(x, np.ndarray) else (
                    list(x) if hasattr(x, '__iter__') and not isinstance(x, (str, bytes)) else x
                )
            )

    # 4. Reprojection si nécessaire
    if crs is not None:
        gdf = gdf.to_crs(crs)

    return gdf


# In[34]:


# Charge le graphe et ses informations associées en format pickle
# Par exemple, charger_graphe("velos") retourne : graphe_velos, id_to_coord_velos, coord_to_id_velos
def charger_graphe(nom_graphe = "vl"):
    nom_fichier = f"graphe_routier_{nom_graphe}.pkl"
    chemin = os.path.join(exports_dir, nom_fichier)

    if not os.path.exists(chemin):
        raise FileNotFoundError(f"Fichier introuvable : {chemin}")

    with open(chemin, "rb") as f:
        graphe, id_to_coord = pickle.load(f)

    print(f"Ouverture du graphe {chemin} ")
    return graphe, id_to_coord


# In[35]:


# Convertit les champs non pris en charge par OGR (listes, dicts) en chaînes JSON
def nettoyer_champs_ogr(gdf):
    gdf = gdf.copy()
    for col in gdf.columns:
        if col != gdf.geometry.name:
            if gdf[col].apply(lambda x: isinstance(x, (list, dict))).any():
                gdf[col] = gdf[col].apply(lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, (list, dict)) else x)
    return gdf


# In[36]:


# LARGEUR REELLE DES RUES A RE-VERIFIER
def calculer_largeur_rues(routes_nom = "bd_topo_routes_epci"):
    routes = charger_fichier_parquet(routes_nom, crs=2154)
    # 1. Conversion du type de 'LARGEUR' en entier
    routes["LARGEUR"] = pd.to_numeric(routes.get("LARGEUR", None), errors="coerce")
    routes["largeur_calculee"] = routes["LARGEUR"].where(routes["LARGEUR"] > 0)

    # 2. Nettoyage pour comparaison
    for col in ["NATURE", "NAT_RESTR", "SENS"]:
        if col in routes.columns:
            routes[f"{col}_clean"] = routes[col].astype(str).str.strip().str.lower()
        else:
            routes[f"{col}_clean"] = ""

    # 3. Cas : pistes cyclables
    condition_piste = (
        routes["largeur_calculee"].isna() &
        (routes["NAT_RESTR_clean"] == "piste cyclable")
    )
    routes.loc[condition_piste & (routes["SENS_clean"] == "double sens"), "largeur_calculee"] = 3.0
    routes.loc[condition_piste & (routes["SENS_clean"] != "double sens"), "largeur_calculee"] = 2.0

    # 4. Cas : sentiers
    condition_sentier = (
        routes["largeur_calculee"].isna() &
        (routes["NATURE_clean"] == "sentier")
    )
    routes.loc[condition_sentier, "largeur_calculee"] = 1.5

    # 5. Cas : chemins
    condition_chemin = (
        routes["largeur_calculee"].isna() &
        (routes["NATURE_clean"] == "chemin")
    )
    routes.loc[condition_chemin, "largeur_calculee"] = 2.0

    # 6. Cas : routes empierrées
    condition_empreinte = (
        routes["largeur_calculee"].isna() &
        (routes["NATURE_clean"] == "route empierrée")
    )
    routes.loc[condition_empreinte, "largeur_calculee"] = 2.5

    # 7. Autres cas
    routes.loc[routes["largeur_calculee"].isna(), "largeur_calculee"] = 2.0

    # 8. Nettoyage final
    routes.drop(columns=["NATURE_clean", "NAT_RESTR_clean", "SENS_clean"], inplace=True)

    # 9. Export
    exporter_parquet(routes, routes_nom)
    exporter_gpkg(routes, routes_nom)

    print(f"Largeur calculée attribuée à {routes['largeur_calculee'].notnull().sum()} tronçons.")


# In[37]:


# Retire les colonnes de jointure d'un dataframe ou geodataframe
def nettoyer_colonnes(df, colonnes_supplementaires=None):
    # Colonnes par défaut à supprimer (index de jointure)
    cols_a_supprimer = ['index_right', 'index_left']

    # Ajoute les colonnes supplémentaires si elles sont spécifiées
    if colonnes_supplementaires is not None:
        if isinstance(colonnes_supplementaires, (list, tuple, set)):
            cols_a_supprimer.extend(colonnes_supplementaires)
        else:
            cols_a_supprimer.append(colonnes_supplementaires)

    # Supprime uniquement les colonnes existantes
    colonnes_existantes = [col for col in cols_a_supprimer if col in df.columns]
    if colonnes_existantes:
        df = df.drop(columns=colonnes_existantes)

    return df


# In[38]:


# Fonction utilisée pour retourner des datetime au format string
def safe_parse_time(h):
    if not isinstance(h, str):
        return None
    try:
        if h.startswith("24:"):
            base = datetime.strptime("00" + h[2:], "%H:%M:%S")
            return base + timedelta(days=1)
        return datetime.strptime(h, "%H:%M:%S")
    except ValueError:
        return None


# ### 1.4. Import du .csv contenant les liens de téléchargement des données
# ---
# Import du .csv contenant notamment le nom des données, et son URL de téléchargement associée

# In[39]:


source_donnees = pd.read_csv(os.path.join(dir, "tableau_sources_donnees.csv"), sep=';', encoding='utf-8')
source_donnees


# ## 2. Téléchargement et traitement des données à l'échelle française
# ---

# ### 2.1. Limites de l'EPCI étudiée depuis la BD Topo
# ---
# Les données sont récupérées depuis la BD Topo pour la région étudiée. IL existe une version distribuée à l'échelle française, mais sa taille (22 gb avant décompression) la rend peu pratique à utiliser
# 
# Documentation :
# * https://geoservices.ign.fr/bdtopo
# * https://documentation.geoservices.ign.fr/?id_theme=1&id_classe=31&BDTopo

# In[95]:


def recupere_limites_epci():
    # 1. Trouver l’URL pour 'bd_topo'
    nom_source = "bd_topo"
    url = trouver_source_url(nom_source)

    # 2. Définition des dossiers de téléchargement et d'extraction du .7z
    archive_path, extract_path, _ = definir_chemins(nom_source, ext="7z")

    # 3. Téléchargement et extraction
    telecharger_fichier(url, archive_path)
    extraire_7z(archive_path, extract_path)

    # 4. Lecture du fichier EPCI.shp et filtrage
    epci_path = trouver_fichier("EPCI.shp", extract_path)
    epci = gpd.read_file(epci_path)
    limites_epci = epci[epci['NOM'] == 'Eurométropole de Strasbourg']

    # 5. Export
    exporter_gpkg(limites_epci, "limites_epci")
    exporter_parquet(limites_epci, "limites_epci")

# Exécution
recupere_limites_epci()


# In[80]:


# Les données sont affichées grâce à matplotlib et contextily : ces cartes requièrent une projection en 3857
def afficher_limites_epci(export = False):
    # 1. Chargement des données
    limites_epci = charger_fichier_parquet("limites_epci", crs=3857)

    # 2. Affichage
    fig, ax = plt.subplots(figsize=(10, 10))
    limites_epci.plot(ax=ax, alpha=1, edgecolor='black', facecolor='none')

    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)

    ax.set_axis_off()
    plt.title("Limites de l'EPCI étudié")

    # (Optionnel) Export PNG
    if export :
        images_export_dir = os.path.join(images_dir, f"visualisation_limites_epci.png")
        plt.savefig(images_export_dir, dpi=300, bbox_inches='tight')
        print(f"Carte exportée vers : {images_export_dir}")

    plt.show()
    plt.close()

# Exécution
afficher_limites_epci(export = True)


# ### 2.2. Routes depuis la BD Topo
# 
# Les données sont récupérées sur une zone tampon autour de l'EPCI étudié : sinon, seules les routes strictement à l'intérieur seraient conservées. Ce qui fausserai les résultats : il est parfois plus rapide de prendre une route à l'extérieur de l'EPCI lors d'un trajet.
# 
# Documentation :
# * https://documentation.geoservices.ign.fr/troncon_de_route&BDTopo
# 
# 
# ![schema_zone_tampon](images/schema_zone_tampon.png)

# #### 2.2.1. Création de la zone tampon
# ---
# 
# Par défaut, on conserve les routes dans une zone tampon de 10 km. La fonction utilisation_troncons dans la partie 4.7. calcule les trajets les plus courts pour partir et arriver à X carreaux de la maille, et retourne le nombre de fois où chaque tronçon de route à été utilisé. L'idée est d'avoir un tampon assez large pour que tout les itinéraires probables soient conservés, mais pas trop pour éviter d'allonger les temps de calculs.

# In[630]:


def recuperer_zone_tampon(taille_metres = 10000):
    # 1. Chargement des données
    limites_epci = charger_fichier_parquet("limites_epci", 2154) # projection métrique (obligatoire pour un buffer en mètres)

    # 2. Construction du buffer de X mètres
    limites_epci_tampon = limites_epci.buffer(taille_metres)

    # 3. Conversion en GeoDataFrame
    tampon_gdf = gpd.GeoDataFrame(geometry=limites_epci_tampon, crs=limites_epci.crs)

    # 4. Export du tampon
    exporter_gpkg(tampon_gdf, "limites_epci_tampon")
    exporter_parquet(tampon_gdf, "limites_epci_tampon")

# Exécution
recuperer_zone_tampon(7000) # Pour l'EMS : valeur adaptée de 7 km


# In[81]:


def afficher_zone_tampon(export = False):
    # 1. Chargement des données
    limites_epci = charger_fichier_parquet("limites_epci", crs=3857)
    limites_epci_tampon = charger_fichier_parquet("limites_epci_tampon", crs=3857)

    # 2. Affichage
    fig, ax = plt.subplots(figsize=(10, 10))
    limites_epci.plot(ax=ax, alpha=1, edgecolor='black', facecolor='none')
    limites_epci_tampon.plot(ax=ax, alpha=1, edgecolor='orange', facecolor='none')

    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)

    ax.set_axis_off()
    plt.title("Limites de l'EPCI et sa zone tampon", fontsize=14)

    # (Optionnel) Export PNG
    if export :
        images_export_dir = os.path.join(images_dir, "visualisation_limites_epci_tampon.png")
        plt.savefig(images_export_dir, dpi=300, bbox_inches='tight')
        print(f"Carte exportée vers : {images_export_dir}")

    plt.show()
    plt.close()

# Exécution
afficher_zone_tampon(export = True)


# #### 2.2.2. Récupération des routes (BD Topo) dans le tampon
# ---

# In[633]:


def recuperer_routes_tampon():
    # 1. Chargement des données, cherche le fichier 'TRONCON_DE_ROUTE.shp' dans l'arborescence
    routes_path = trouver_fichier("TRONCON_DE_ROUTE.shp", extraction_dir)
    limites_epci = charger_fichier_parquet("limites_epci", crs=2154)
    limites_epci_tampon = charger_fichier_parquet("limites_epci_tampon", crs=2154)

    # 2. Charger les tronçons de route
    gdf_routes = gpd.read_file(routes_path)

    # 3. S'assurer que limites_epci_tampon est un GeoDataFrame (sinon, erreur lors de l'export)
    if isinstance(limites_epci_tampon, gpd.GeoSeries):
        print("est une geo serie")
        limites_epci_tampon = gpd.GeoDataFrame(geometry=limites_epci_tampon)

    # 4. Intersection
    bd_topo_routes_tampon = gdf_routes[gdf_routes.intersects(limites_epci_tampon.geometry.union_all())].copy()
    bd_topo_routes_epci = gdf_routes[gdf_routes.intersects(limites_epci.geometry.union_all())].copy()

    # 5. Découpe les tronçons pour correspondre aux limites
    bd_topo_routes_tampon['geometry'] = bd_topo_routes_tampon.intersection(limites_epci_tampon.union_all())
    bd_topo_routes_epci['geometry'] = bd_topo_routes_epci.intersection(limites_epci.union_all())

    # 6. Ne conserve que le type de géométrie original (élimine les points et geometry collection)
    bd_topo_routes_tampon = bd_topo_routes_tampon[bd_topo_routes_tampon.geometry.type.isin(['LineString', 'MultiLineString'])].copy()
    bd_topo_routes_epci = bd_topo_routes_epci[bd_topo_routes_epci.geometry.type.isin(['LineString', 'MultiLineString'])].copy()

    # 7. Calcule la largeur pour tout les tronçons de rues et export
    calculer_largeur_rues(routes_nom = "bd_topo_routes_tampon")

    exporter_gpkg(bd_topo_routes_epci, "bd_topo_routes_epci")
    exporter_parquet(bd_topo_routes_epci, "bd_topo_routes_epci")

# Exécution
recuperer_routes_tampon()


# In[83]:


def afficher_routes_tampon(export = False):
    # 1. Chargement des données
    limites_epci = charger_fichier_parquet("limites_epci", crs=3857)
    limites_epci_tampon = charger_fichier_parquet("limites_epci_tampon", crs=3857)
    bd_topo_routes_tampon = charger_fichier_parquet("bd_topo_routes_tampon", crs=3857)

    # 2. Affichage
    fig, ax = plt.subplots(figsize=(10, 10))

    limites_epci.plot(ax=ax, alpha=1, edgecolor='black', facecolor='none')
    limites_epci_tampon.plot(ax=ax, alpha=1, edgecolor='orange', facecolor='none')
    bd_topo_routes_tampon.plot(ax=ax, alpha=1, linewidth=0.3, edgecolor='green', facecolor='none')

    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)

    ax.set_axis_off()
    plt.title("Routes dans la zone tampon de l'EPCI", fontsize=14)

    # (Optionnel) Export PNG
    if export :
        images_export_dir = os.path.join(images_dir, "visualisation_limites_epci_tampon_routes.png")
        plt.savefig(images_export_dir, dpi=300, bbox_inches='tight')
        print(f"Carte exportée vers : {images_export_dir}")

    plt.show()
    plt.close()

# Exécution
afficher_routes_tampon(export = True)


# ### 2.3. Emprise de la France
# ---
# Données issue d'Admin Express pour obtenir les limites de la France. Utilisées pour ne pas fausser les résultats à cause des limites frontalières dans le calcul d'indicateurs.

# In[24]:


def recuperer_emprise_france():
    # 1. Trouver l’URL pour 'admin_express'
    nom_source = "admin_express"
    url = trouver_source_url(nom_source)

    # 2. Définition des dossiers de téléchargement et d'extraction du .7z
    archive_path, extract_path, _ = definir_chemins(nom_source, ext="7z")

    # 3. Téléchargement et extraction
    telecharger_fichier(url, archive_path)
    extraire_7z(archive_path, extract_path)

    # 4. Trouver le fichier GPKG extrait
    try:
        gpkg_files = []
        for root, _, files in os.walk(extract_path):
            for file in files:
                if file.lower().endswith('.gpkg'):
                    gpkg_files.append(os.path.join(root, file))

        if not gpkg_files:
            raise FileNotFoundError("Aucun fichier .gpkg trouvé dans l'extraction")
        elif len(gpkg_files) > 1:
            print("Attention : plusieurs fichiers GPKG trouvés, utilisation du premier")

        gpkg_path = gpkg_files[0]
        print(f"Fichier GPKG trouvé : {gpkg_path}")

    except Exception as e:
        print(f"Erreur lors de la recherche du GPKG : {e}")
        raise

    # 5. Chargement des régions avec vérification de la couche
    try:
        # Vérifier les couches disponibles
        layers = fiona.listlayers(gpkg_path)
        #print("Couches disponibles dans le GPKG:", layers)

        # Charger la couche 'region' (avec fallback sur d'autres noms possibles)
        layer_name = next((l for l in layers if 'region' in l.lower()), None)
        if not layer_name:
            raise ValueError("Aucune couche 'region' trouvée dans le GPKG")

        regions = gpd.read_file(gpkg_path, layer=layer_name)

    except Exception as e:
        print(f"Erreur lors du chargement des régions : {e}")
        raise

    # 6. Charger les régions françaises
    regions = gpd.read_file(gpkg_path, layer='region')

    # 7. Dissoudre les régions pour obtenir le contour France entière
    limites_france = regions.dissolve().reset_index()  # Fusionne toutes les régions

    # (Optionnel) Simplification
    #limites_france.geometry = limites_france.geometry.simplify(10)  # Tolérance 10m

    # 8. Export
    exporter_parquet(limites_france, "limites_france")
    exporter_gpkg(limites_france, "limites_france")

# Exécution
recuperer_emprise_france()


# In[85]:


def afficher_emprise_france(export = False):
    # 1. Chargement des données
    limites_epci = charger_fichier_parquet("limites_epci", crs=3857)
    limites_france = charger_fichier_parquet("limites_france", crs=3857)

    # 2. Affichage
    fig, ax = plt.subplots(figsize=(10, 10))

    limites_epci.plot(ax=ax, alpha=1, edgecolor='red', facecolor='none')
    limites_france.plot(ax=ax, alpha=1, edgecolor='black', facecolor='none')

    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)

    ax.set_axis_off()
    plt.title("Limites de la france métropolitaine", fontsize=14)

    # (Optionnel) Export PNG
    if export :
        images_export_dir = os.path.join(images_dir, "visualisation_limites_france.png")
        plt.savefig(images_export_dir, dpi=300, bbox_inches='tight')
        print(f"Carte exportée vers : {images_export_dir}")

    plt.show()
    plt.close()

# Exécution
afficher_emprise_france(export = True)


# ### 2.4. Bâtiments depuis la zone d'étude
# ---
# Les bâtiments dans la zone tampon de l'EPCI sont utilisés pour les indicateurs de surface bâtie à 400 mètres des stations de tram et de bus. Si ces stations se trouvent en bordure de l'EPCI étudié, il faut conserver les bâtiments présents dans sa zone tampon.

# In[26]:


def recuperer_bati_zone_etude():
    # 1. Chercher le fichier 'BATIMENT.shp' dans l'arborescence
    limites_epci = charger_fichier_parquet("limites_epci", crs=2154)
    limites_epci_tampon = charger_fichier_parquet("limites_epci_tampon", crs=2154)
    batiments_path = trouver_fichier("BATIMENT.shp", extraction_dir)

    # 2. Charger les bâtiments
    gdf_batiments = gpd.read_file(batiments_path)

    # 3. Filtrer les bâtiments à l'intérieur de l'EPCI et de son tampon
    bd_topo_batiments_epci = gpd.overlay(gdf_batiments, limites_epci, how="intersection")
    bd_topo_batiments_tampon = gpd.overlay(gdf_batiments, limites_epci_tampon, how="intersection")

    # 4. Exports
    exporter_gpkg(bd_topo_batiments_epci, "bd_topo_batiments_epci")
    exporter_parquet(bd_topo_batiments_epci, "bd_topo_batiments_epci")
    exporter_gpkg(bd_topo_batiments_tampon, "bd_topo_batiments_tampon")
    exporter_parquet(bd_topo_batiments_tampon, "bd_topo_batiments_tampon")

# Exécution
recuperer_bati_zone_etude()


# In[86]:


def afficher_bati_zone_etude(export = False):
    # 1. Chargement des données
    limites_epci = charger_fichier_parquet("limites_epci", crs=3857)
    bd_topo_batiments_epci = charger_fichier_parquet("bd_topo_batiments_epci", crs=3857)

    # 2. Affichage
    fig, ax = plt.subplots(figsize=(10, 10))
    limites_epci.plot(ax=ax, alpha=1, edgecolor='black', facecolor='none')
    bd_topo_batiments_epci.plot(ax=ax, alpha=1, linewidth=0.5, edgecolor='orange', facecolor='none')

    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)

    ax.set_axis_off()
    plt.title("Bâtiments dans la zone étudiée (BD Topo)", fontsize=14)

    # (Optionnel) Export PNG
    if export :
        images_export_dir = os.path.join(images_dir, "visualisation_bati_epci.png")
        plt.savefig(images_export_dir, dpi=300, bbox_inches='tight')
        print(f"Carte exportée vers : {images_export_dir}")

    plt.show()
    plt.close()

# Exécution
afficher_bati_zone_etude(export = True)


# ### 2.5. Parkings
# ---
# Données tirées de la BD Topo

# In[ ]:


def recuperer_parkings():
    # 1. Chargement des données
    limites_epci = charger_fichier_parquet("limites_epci", crs=2154)

    equip_transport_path = trouver_fichier("EQUIPEMENT_DE_TRANSPORT.dbf", extraction_dir)
    equipements = gpd.read_file(equip_transport_path)
    equipements = equipements.to_crs(2154)

    # 2. Filtrer les parkings sauf les parkings relais
    parkings = equipements[
        (equipements["NATURE"] == "Parking") &
        (equipements["NAT_DETAIL"].fillna("") != "Parking relais") &
        (equipements["FICTIF"].fillna("") == "Non") &
        (equipements["ETAT"].fillna("") == "En service")
    ].copy()

    # 3. Filtrage spatial : parkings dans l’EPCI
    parkings_epci = gpd.overlay(parkings, limites_epci, how="intersection")

    # 4. Exports
    exporter_gpkg(parkings_epci, "parkings_epci")
    exporter_parquet(parkings_epci, "parkings_epci")

    print(f"{len(parkings_epci)} parkings trouvées dans l'EPCI")

# Exécution
recuperer_parkings()


# ### 2.6. Parkings relais
# ---
# Données tiréees de la BD Topo

# In[84]:


def recuperer_parkings_relais():
    # 1. Chargement des données
    limites_epci = charger_fichier_parquet("limites_epci", crs=2154)

    equip_transport_path = trouver_fichier("EQUIPEMENT_DE_TRANSPORT.dbf", extraction_dir)
    equipements = gpd.read_file(equip_transport_path)
    equipements = equipements.to_crs(2154)

    # 2. Filtrer les parkings sauf les parkings relais
    parkings = equipements[
        (equipements["NAT_DETAIL"].fillna("") == "Parking relais") &
        (equipements["FICTIF"].fillna("") == "Non") &
        (equipements["ETAT"].fillna("") == "En service")
    ].copy()

    # 3. Filtrage spatial : parkings relais dans l’EPCI
    parkings_relais_epci = gpd.overlay(parkings, limites_epci, how="intersection")

    # 4. Exports
    exporter_gpkg(parkings_relais_epci, "parkings_relais_epci")
    exporter_parquet(parkings_relais_epci, "parkings_relais_epci")

    print(f"{len(parkings_relais_epci)} parkings trouvées dans l'EPCI")

# Exécution
recuperer_parkings_relais()


# In[85]:


def afficher_parkings_relais(export = False):
    # 1. Chargement des données
    limites_epci = charger_fichier_parquet("limites_epci", crs=3857)
    parkings_relais = charger_fichier_parquet("parkings_relais_epci", crs=3857)

    # 2. Affichage
    fig, ax = plt.subplots(figsize=(10, 10))
    limites_epci.plot(ax=ax, alpha=1, edgecolor='black', facecolor='none')
    parkings_relais.plot(ax=ax, color='red', alpha=0.7, edgecolor='black')
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)

    ax.set_axis_off()
    plt.title("Localisation des parkings relais", fontsize=14)

    # (Optionnel) Export PNG
    if export :
        images_export_dir = os.path.join(images_dir, "visualisation_parkins_relais.png")
        plt.savefig(images_export_dir, dpi=300, bbox_inches='tight')
        print(f"Carte exportée vers : {images_export_dir}")

    plt.show()

# Exécution
afficher_parkings_relais(export = True)


# ### 2.7. Maillage INSEE (couverture complète, sans données)
# --- 
# Téléchargement d'un maillage sans données de 200m par 200m sur la zone étudiée 
# 
# Documentation :
# * https://www.insee.fr/fr/statistiques/6214726?sommaire=6215217

# In[29]:


def recuperer_maille_insee_200m():
    # 1. Trouver l'URL pour 'insee_carreaux_sans_donnees'
    nom_source = "insee_carreaux_sans_donnees"
    url = trouver_source_url(nom_source)

    # 2. Charger les limites de l'EPCI
    limites_epci = charger_fichier_parquet("limites_epci", crs=2154)

    # 3. Définir les chemins et télécharger le fichier
    zip_path, extract_path, _ = definir_chemins(nom_source, ext="zip")
    telecharger_fichier(url, zip_path)

    # 4. Extraire le .zip puis le .7z dans le bon dossier
    os.makedirs(extract_path, exist_ok=True)
    extraire_zip_puis_7z(zip_path, extract_path)

    # 5. Trouver le fichier GPKG
    carreaux_met_path = trouver_fichier("grille200m_metropole.gpkg", extract_path)

    # 6. Créer une emprise sur l'EPCI étudié
    epci_union = limites_epci.geometry.union_all()

    # 7. Traitement par lots
    def process_large_gpkg(carreaux_met_path, limites_geom, batch_size=50000):
        total_features = len(gpd.read_file(carreaux_met_path))
        results = []

        for offset in tqdm(range(0, total_features, batch_size)):
            batch = gpd.read_file(carreaux_met_path, rows=slice(offset, offset + batch_size))
            filtered = batch[batch.geometry.within(limites_geom)]
            results.append(filtered)

        return gpd.GeoDataFrame(pd.concat(results), crs="EPSG:2154") if results else gpd.GeoDataFrame()

    carreaux_sans_donnees_epci = process_large_gpkg(carreaux_met_path, epci_union)

    # 8. Export
    if not carreaux_sans_donnees_epci.empty:
        exporter_gpkg(carreaux_sans_donnees_epci, "maille_200m_epci")
        exporter_parquet(carreaux_sans_donnees_epci, "maille_200m_epci")
    else:
        print("Aucun carreau trouvé dans les limites de l'EPCI")

# Exécution
recuperer_maille_insee_200m()


# NOTE : carreau_id fait doublon avec idINSPIRE, à retirer du reste du code

# In[195]:


"""
# Calcule le point centre de chaque carreau de la maille
# 1. Charger les carreaux existants
carreaux = charger_fichier_parquet("maille_200m_epci", crs=3857)

# 2. Calculer les centroïdes
centres = carreaux.copy()
centres["geometry"] = centres.geometry.centroid

# 3. Rajoute un attribut 'carreau_id' pour identifier chaque carreau
if "carreau_id" not in carreaux.columns:
    carreaux["carreau_id"] = carreaux.index

# 3. Exporter le résultat
exporter_gpkg(carreaux, "maille_200m_epci")
exporter_parquet(carreaux, "maille_200m_epci")
"""


# In[88]:


def afficher_maille_insee_200m(export = False):
    # 1. Chargement des données
    limites_epci = charger_fichier_parquet("limites_epci", crs=3857)
    maille_epci = charger_fichier_parquet("maille_200m_epci", crs=3857)

    # 2. Affichage
    fig, ax = plt.subplots(figsize=(10, 10))
    limites_epci.plot(ax=ax, alpha=1, edgecolor='black', facecolor='none')
    maille_epci.plot(ax=ax, alpha=0.5, edgecolor='orange', facecolor='none')
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)

    ax.set_axis_off()
    plt.title("Affichage des limites de l'EPCI étudié et des carreaux INSEE ne possédant pas de données")

    # (Optionnel) Export PNG
    if export :
        images_export_dir = os.path.join(images_dir, "visualisation_maille_sans_donnees.png")
        plt.savefig(images_export_dir, dpi=300, bbox_inches='tight')
        print(f"Carte exportée vers : {images_export_dir}")

    plt.show()
    plt.close()

# Exécution
afficher_maille_insee_200m(export = True)


# ### 2.8. Données Filosofi (Revenus localisés sociaux et fiscaux de l'INSEE)
# ---
# Théoriquement, ce fichier est mis à jour annuellement, mais "la production du millésime 2022 qui aurait dû être diffusé début 2025 ne pourra avoir lieu en raison d’une qualité statistique insuffisante des sources".
# De plus, la donnée de 2019 semble être la dernière distribuée sous la forme d'un carroyage de 200m (avec 2017 et 2015). Le code commenté récupère la version de 2019, et aucun traitement supplémentaire n'est nécessaire : c'est la version à utiliser de préférence.
# 
# La dernière version de la donnée (2021) est distribuée à l'échelle de l'IRIS. Si l'INSEE décide de ne plus distribuer de données carroyées, il faut l'utiliser et appliquer les traitements suivants : 
# 1. Télécharger la géométrie de l'IRIS pour l'année étudiée
# 2. Télécharger les données voulues à l'échelle de l'IRIS (population, revenus, nombre de salariés, etc.) et les joindre sur la géométrie de l'IRIS
# 3. Désagrérer les données IRIS pour passer les données sur une maille de 200m par 200m
# Etant donné que la structure des champs utilisé par l'INSEE semble varier selon les années, je ne peux pas garantir que les scripts suivants fonctionneront dans X années. Ils auront sans doute besoin d'être modifiés / adaptés.
# 
# La donnée 2019 contient (en autres) les colonnes suivantes :
# * ind : Nombre d’individus
# * ind_snv : Somme des niveaux de vie winsorisés des individus
# 
# La donnée 2021 ne contient pas les mêmes colonnes que 2019, notamment la population : 
# * DISP_MED21 : Médiane du revenu disponible par unité de consommation (en euros)
# 
# Documentation :
# * https://fr.wikipedia.org/wiki/Winsorisation
# * https://www.insee.fr/fr/statistiques/7756729?sommaire=7756859
# * https://www.insee.fr/fr/statistiques/7655475?sommaire=7655515
# * https://www.insee.fr/fr/metadonnees/source/serie/s1172

# #### 2.8.1. Version pour les données déjà carroyées
# ---

# In[32]:


def recuperer_filosofi_carroyage_200m():
    # 1. Trouver l’URL pour 'insee_filosofi_2019_200m'
    nom_source = "insee_filosofi_2019_200m"
    url = trouver_source_url(nom_source)

    # 2. Charger les limites de l'EPCI
    limites_epci = charger_fichier_parquet("limites_epci", crs=2154)

    # 3. Définir les chemins et télécharger le fichier
    zip_path, extract_path, _ = definir_chemins(nom_source, ext="zip")
    telecharger_fichier(url, zip_path)

    # 4. Extraire le .zip puis le .7z dans le bon dossier
    os.makedirs(extract_path, exist_ok=True)
    extraire_zip_puis_7z(zip_path, extract_path)

    # 5. Trouver le fichier GPKG
    carreaux_path = trouver_fichier("carreaux_200m_met.gpkg", extract_path)

    # 6. Charger les données Filosofi et ne conserver que ceux entièrement contenus dans la zone étudiée
    carreaux = gpd.read_file(carreaux_path)
    carreaux_epci = carreaux[carreaux.within(limites_epci.geometry.union_all())]

    # 7. Exporter le résultat
    exporter_gpkg(carreaux_epci, "filosofi_maille_200m_epci")
    exporter_parquet(carreaux_epci, "filosofi_maille_200m_epci")

# Exécution
recuperer_filosofi_carroyage_200m()


# In[89]:


def afficher_filofi_carroyage_200m(export = False):
    # 1. Chargement des données
    limites_epci = charger_fichier_parquet("limites_epci", crs=3857)
    carreaux_epci = charger_fichier_parquet("filosofi_maille_200m_epci", crs=3857)

    # 2. Affichage
    fig, ax = plt.subplots(figsize=(10, 10))
    limites_epci.plot(ax=ax, alpha=1, edgecolor='black', facecolor='none')
    carreaux_epci.plot(ax=ax, alpha=0.5, edgecolor='orange', facecolor='none')

    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)

    ax.set_axis_off()
    plt.title("Affichage des limites de l'EPCI étudié et des carreaux Filosofi de l'INSEE")

    # (Optionnel) Export PNG
    if export :
        images_export_dir = os.path.join(images_dir, "visualisation_filosofi.png")
        plt.savefig(images_export_dir, dpi=300, bbox_inches='tight')
        print(f"Carte exportée vers : {images_export_dir}")

    plt.show()
    plt.close()

# Exécution
afficher_filofi_carroyage_200m(export = True)


# #### 2.8.2. Version pour les données IRIS
# ---
# L'INSEE propose ses données sur le recensement de la population avec différentes échelles : 
# * Le découpage par IRIS, un niveau infra-communal : https://fr.wikipedia.org/wiki/%C3%8Elots_regroup%C3%A9s_pour_l%27information_statistique
# * Par carroyage d'1 km
# 
# Pour obtenir des données utilisables avec notre maillage de 200 m, il faut donc subdiviser les données. La population présente dans IRIS sera répartie dans les carreaux de 200m en fonction de leur quantité de surface bâtie (issue de la BD Topo). C'est pour cela que les carreaux présents en partie dans la zone étudiée sont conservés. 
# 
# Ici, 5 étapes sont réalisées : 
# 1. Téléchargement de la géométrie des IRIS pour les données les plus récentes (2021) - https://geoservices.ign.fr/contoursiris
# 2. Téléchargement des données contenant la population par IRIS et jointure avec la géométrie - https://www.insee.fr/fr/statistiques/8268806
# 3. Téléchargement des donnéees contenant les revenus déclarés et disponibles par IRIS et jointure avec la géométrie - https://www.insee.fr/fr/statistiques/8229323
# 4. Téléchargement des données filosofi au niveau communal pour les communes non-divisées au niveau IRIS et jointure avec la géométrie - https://www.insee.fr/fr/statistiques/7756855?sommaire=7756859
# 5. Téléchargement des données sur l'activité professionnelle de la population et leur catégorie d'emploi (ouvriers, employés, apprentis/stagiaires, etc.) et jointure avec la géométrie - https://www.insee.fr/fr/statistiques/8268843
# 
# Documentation : 
# * https://www.insee.fr/fr/metadonnees/definition/c1458
# * https://www.insee.fr/fr/metadonnees/definition/c1710

# ##### 2.8.2.1. Téléchargement de la géométrie des IRIS pour les données les plus récentes (2021)
# ---

# In[34]:


# Télécharge les contours géographiques IRIS sans données
def recuperer_contours_geo_iris():
    # 1. Trouver l'URL pour 'insee_iris'
    nom_source = "insee_iris"
    url = trouver_source_url(nom_source)

    # 2. Charger les données
    limites_epci = charger_fichier_parquet("limites_epci", crs=2154)
    emprise_epci = limites_epci.geometry.union_all()

    # 3. Définir les chemins et télécharger le fichier
    archive_path, extract_path, _ = definir_chemins(nom_source, ext="7z")
    #telecharger_fichier(url, archive_path)

    # 4. Extraire le .7z
    os.makedirs(extract_path, exist_ok=True)
    #extraire_7z(archive_path, extract_path)

    # 5. Trouver le fichier .shp
    iris_path = trouver_fichier("CONTOURS-IRIS.shp", extract_path)
    iris = gpd.read_file(iris_path).to_crs(2154)

    # 6. Jointure spatiale 
    iris_epci = iris[iris.intersects(emprise_epci)]

    # 7. Exporter le résultat
    exporter_gpkg(iris_epci, "iris_epci")
    exporter_parquet(iris_epci, "iris_epci")

# Exécution
recuperer_contours_geo_iris()


# ##### 2.8.2.2. Téléchargement des données contenant la population par IRIS et jointure avec la géométrie
# ---

# In[35]:


# Télécharge et applique la population de 2021 aux contours IRIS
def recuperer_population_iris():
    # 1. Trouver l'URL pour 'insee_pop_2021_iris'
    nom_source = "insee_pop_2021_iris"
    url = trouver_source_url(nom_source)

    # 2. Charger les données
    iris_epci = charger_fichier_parquet("iris_epci", crs=2154)

    # 3. Définir les chemins et télécharger le fichier
    zip_path, extract_path, _ = definir_chemins(nom_source, ext="zip")
    #telecharger_fichier(url, zip_path)

    # 4. Extraire le .zip
    os.makedirs(extract_path, exist_ok=True)
    #extraire_zip(zip_path, extract_path)

    # 5. Trouver et charger le fichier .CSV contenant les données sur la population
    pop_iris_path = trouver_fichier("base-ic-evol-struct-pop-2021.CSV", extract_path)

    # 6. Définition explicite des types pour les colonnes problématiques
    dtype_spec = {'IRIS': str, 'COM': str,'LIBCOM': str}

    # 7. Chargement des donnéees 
    pop_iris = pd.read_csv(pop_iris_path, sep=";", encoding='utf-8', dtype=dtype_spec, low_memory=False)

    # 8. Nettoyage des données de population
    # 8.1. Sélectionner les colonnes pertinentes (à modifier selon les besoin)
    colonnes_pop = [
        'IRIS', 'P21_POP', 'P21_POP0014', 'P21_POP1529', 
        'P21_POP3044', 'P21_POP4559', 'P21_POP6074', 'P21_POP75P',
        'P21_POPH', 'P21_POPF'
    ]
    pop_iris = pop_iris[colonnes_pop]

    # 8.2. Conversion des colonnes numériques (remplacer les virgules par des points si nécessaire)
    for col in colonnes_pop[1:]:  # Toutes sauf 'IRIS'
        if pop_iris[col].dtype == object:
            pop_iris[col] = pop_iris[col].str.replace(',', '.').astype(float).fillna(0).astype(int)

    # 9. Préparation des contours IRIS
    # 9.1. Supprimer la colonne 'IRIS' si elle existe
    if 'IRIS' in iris_epci.columns:
        iris_epci = iris_epci.drop(columns='IRIS')

    # 9.2. Renommer 'CODE_IRIS' en 'IRIS'
    if 'CODE_IRIS' in iris_epci.columns:
        iris_epci = iris_epci.rename(columns={'CODE_IRIS': 'IRIS'})

    # 9.3. Nettoyage des index résiduels
    iris_epci = iris_epci.drop(columns=['index_right', 'index_left'], errors='ignore')

    # 10. Jointure des données avec vérification
    print(f"Nombre de lignes avant jointure: {len(iris_epci)}")
    print(f"Nombre de codes IRIS uniques dans pop_iris: {pop_iris['IRIS'].nunique()}")

    iris_complet = iris_epci.merge(
        pop_iris,
        on='IRIS',
        how='left',  # Conserve tous les IRIS même sans correspondance
        indicator=True  # Pour diagnostiquer les problèmes
    )

    # 11. Vérification des résultats
    print("\nStatistiques de jointure:")
    print(iris_complet['_merge'].value_counts())

    # 12. Remplacer les valeurs NULL par 0 pour les colonnes numériques
    for col in colonnes_pop[1:]:
        iris_complet[col] = iris_complet[col].fillna(0).astype(int)

    # 13. Export
    exporter_parquet(iris_complet.drop(columns='_merge'), "iris_epci_donnees")
    exporter_gpkg(iris_complet.drop(columns='_merge'), "iris_epci_donnes")

    print("\nExport réussi. Colonnes disponibles:")
    print(iris_complet.columns.tolist())

# Exécution
recuperer_population_iris()


# ##### 2.8.2.3. Téléchargement des donnéees contenant les revenus déclarés et disponibles par IRIS et jointure avec la géométrie
# ---

# In[36]:


# Ajout des revenus disponibles et déclarés aux IRIS
def recuperer_donnnes_eco_iris():
# 1. Configuration des sources
    sources = {
        'dec': {
            'nom': "insee_rev_dec_2021_iris",
            'fichier': "BASE_TD_FILO_IRIS_2021_DEC.csv",
            'prefixe': "DEC_"
        },
        'dis': {
            'nom': "insee_rev_dis_2021_iris",
            'fichier': "BASE_TD_FILO_IRIS_2021_DISP.csv",
            'prefixe': "DISP_"
        }
    }

    # 2. Charger les données existantes
    iris_complet = charger_fichier_parquet("iris_epci_donnees", crs=2154)

    # 3. Traitement pour chaque type de revenu
    for key, source in sources.items():
        print(f"\nTraitement des données {source['nom']}...")

        # Téléchargement et extraction
        url = trouver_source_url(source['nom'])
        zip_path, extract_path, _ = definir_chemins(source['nom'], ext="zip")
        telecharger_fichier(url, zip_path)
        os.makedirs(extract_path, exist_ok=True)
        extraire_zip(zip_path, extract_path)

        # Chargement du CSV avec gestion des types et formats
        csv_path = trouver_fichier(source['fichier'], extract_path)
        revenus = pd.read_csv(
            csv_path,
            sep=";",
            encoding='utf-8',
            dtype={'IRIS': str},  # Force le code IRIS en texte
            decimal=",",          # Gère les nombres avec virgule
            low_memory=False
        )

        # Nettoyage des données
        revenus['IRIS'] = revenus['IRIS'].str.strip()

        # Renommage des colonnes avec préfixe
        colonnes_a_garder = [col for col in revenus.columns if col.startswith(source['prefixe']) or col == 'IRIS']
        revenus = revenus[colonnes_a_garder]

        # Conversion des nombres (remplacement des virgules déjà géré par decimal=",")

        # Jointure avec vérification
        print(f"Nombre de lignes avant jointure: {len(iris_complet)}")
        print(f"Nombre de codes IRIS uniques dans les revenus: {revenus['IRIS'].nunique()}")

        iris_complet = iris_complet.merge(
            revenus,
            on='IRIS',
            how='left',
            validate='one_to_one',
            indicator=f'merge_{key}'
        )

        # Vérification
        print(f"Statistiques de jointure ({source['nom']}):")
        print(iris_complet[f'merge_{key}'].value_counts())
        iris_complet = iris_complet.drop(columns=f'merge_{key}')

    # 4. Post-traitement
    # Remplacer les éventuelles valeurs manquantes par 0 pour les indicateurs numériques
    colonnes_numeriques = iris_complet.select_dtypes(include=['float64', 'int64']).columns
    iris_complet[colonnes_numeriques] = iris_complet[colonnes_numeriques].fillna(0)

    # 5. Export final
    exporter_parquet(iris_complet, "iris_epci_donnees")
    exporter_gpkg(iris_complet, "iris_epci_donnees")

    print("\nExport réussi. Statistiques finales :")
    print(f"- Nombre d'IRIS: {len(iris_complet)}")
    print(f"- Colonnes ajoutées:")
    print([col for col in iris_complet.columns if col.startswith('DEC_') or col.startswith('DISP_')])

# Exécution
recuperer_donnnes_eco_iris()


# ##### 2.8.2.4. Téléchargement des données filosofi au niveau communal pour les communes non-divisées au niveau IRIS et jointure avec la géométrie
# ---

# In[37]:


# Rajoute les données filosofi aux communes non-divisées au niveau IRIS
def recuperer_donnees_filosi_iris():
    # 1. Charger les données existantes
    iris_complet = charger_fichier_parquet("iris_epci_donnees", crs=2154)

    # 2. Identifier les communes sans données IRIS (left-only)
    # On vérifie d'abord si la colonne de merge existe
    if 'merge_dec' in iris_complet.columns and 'merge_dis' in iris_complet.columns:
        communes_sans_iris = iris_complet[
            (iris_complet['merge_dec'] == 'left_only') | 
            (iris_complet['merge_dis'] == 'left_only')
        ]['INSEE_COM'].unique()
    else:
        # Méthode alternative si pas d'indicateur de merge
        communes_sans_iris = iris_complet[
            iris_complet['DEC_MED21'].isna() | 
            iris_complet['DISP_MED21'].isna()
        ]['INSEE_COM'].unique()

    print(f"Nombre de communes sans données IRIS: {len(communes_sans_iris)}")

    # 3. Charger les données communales
    def charger_revenus_communes():
        # Configuration des sources
        sources = {
            'dec': {
                'nom': "insee_rev_2021_com",
                'fichier': "FILO2021_DEC_COM.csv",
                'prefixe': "COM_DEC_",
                'colonnes': ['CODGEO', 'NBMEN21', 'NBPERS21', 'NBUC21', 'PMIMP21', 'Q121', 'Q221', 'Q321', 'Q3_Q1']
            },
            'dis': {
                'nom': "insee_rev_2021_com",
                'fichier': "FILO2021_DISP_COM.csv",
                'prefixe': "COM_DISP_",
                'colonnes': ['CODGEO', 'NBMEN21', 'NBPERS21', 'NBUC21', 'Q121', 'Q221', 'Q321', 'Q3_Q1']
            }
        }

        revenus_com = {}

        for key, source in sources.items():
            print(f"\nTraitement des données {source['nom']} ({key})...")

            # Téléchargement et extraction
            url = trouver_source_url(source['nom'])
            zip_path, extract_path, _ = definir_chemins(source['nom'], ext="zip")
            telecharger_fichier(url, zip_path)
            os.makedirs(extract_path, exist_ok=True)
            extraire_zip(zip_path, extract_path)

            # Chargement du CSV
            csv_path = trouver_fichier(source['fichier'], extract_path)
            df = pd.read_csv(
                csv_path,
                sep=";",
                encoding='utf-8',
                dtype={'CODGEO': str},  # Code commune en texte
                decimal=",",            # Gestion des nombres français
                usecols=source['colonnes']  # Sélection des colonnes utiles
            )

            # Nettoyage
            df['CODGEO'] = df['CODGEO'].str.strip()

            # Renommage avec préfixe
            colonnes_rename = {col: f"{source['prefixe']}{col}" for col in source['colonnes'] if col != 'CODGEO'}
            df = df.rename(columns=colonnes_rename)

            revenus_com[key] = df

        # Fusion des deux sources
        revenus_communes = revenus_com['dec'].merge(
            revenus_com['dis'],
            on='CODGEO',
            how='outer'
        )

        return revenus_communes

    revenus_communes = charger_revenus_communes()

    # 4. Jointure pour compléter les données manquantes
    print("\nComplétion des données communales...")

    # a. Créer un dataframe des communes à compléter
    communes_a_completer = iris_complet[
        iris_complet['INSEE_COM'].isin(communes_sans_iris)
    ][['INSEE_COM']].drop_duplicates()

    # b. Jointure avec les revenus communaux
    communes_completees = communes_a_completer.merge(
        revenus_communes,
        left_on='INSEE_COM',
        right_on='CODGEO',
        how='left'
    )

    # c. Mise à jour des IRIS correspondants
    for index, row in communes_completees.iterrows():
        # Trouver tous les IRIS de cette commune
        mask = iris_complet['INSEE_COM'] == row['INSEE_COM']

        # Mettre à jour les colonnes de revenus déclarés
        iris_complet.loc[mask, 'DEC_MED21'] = row['COM_DEC_Q221']
        iris_complet.loc[mask, 'DEC_Q121'] = row['COM_DEC_Q121']
        iris_complet.loc[mask, 'DEC_Q321'] = row['COM_DEC_Q321']

        # Mettre à jour les colonnes de revenus disponibles
        iris_complet.loc[mask, 'DISP_MED21'] = row['COM_DISP_Q221']
        iris_complet.loc[mask, 'DISP_Q121'] = row['COM_DISP_Q121']
        iris_complet.loc[mask, 'DISP_Q321'] = row['COM_DISP_Q321']

    # 5. Vérification finale
    communes_completees = iris_complet[
        iris_complet['INSEE_COM'].isin(communes_sans_iris)
    ]
    print(f"\nVérification - communes complétées : {len(communes_completees['INSEE_COM'].unique())}")
    print("Exemple de données complétées :")
    print(communes_completees[['INSEE_COM', 'DEC_MED21', 'DISP_MED21']].head())

    # 6. Export final
    exporter_parquet(iris_complet, "iris_epci_donnees")
    exporter_gpkg(iris_complet, "iris_epci_donnees")

    print("\nOpération terminée avec succès. Données sauvegardées.")

# Exécution
recuperer_donnees_filosi_iris()


# ##### 2.8.2.5. Téléchargement des données sur l'activité professionnelle de la population et leur catégorie d'emploi (ouvriers, employés, apprentis/stagiaires, etc.) et jointure avec la géométrie
# ---

# In[38]:


def recuperer_activites_residents():
    # 1. Charger les données existantes
    iris_complet = charger_fichier_parquet("iris_epci_donnees", crs=2154)

    # 2. Configuration de la source des données d'activité
    source_activite = {
        'nom': "insee_activite_2021_iris",
        'fichier': "base-ic-activite-residents-2021.CSV",
        'colonnes_a_garder': [
            'IRIS',  # Code IRIS
            'P21_ACTOCC1564',  # Actifs occupés 15-64 ans
            'P21_SAL15P',  # Salariés 15 ans ou plus
            'P21_NSAL15P',  # Non-salariés 15 ans ou plus
            'P21_ACTOCC15P',  # Actifs occupés 15 ans ou plus
            'P21_SAL15P_CDI',  # Salariés en CDI ou fonction publique
            'P21_SAL15P_CDD',  # Salariés en CDD
            'P21_SAL15P_INTERIM',  # Salariés intérimaires
            'P21_SAL15P_APPR',  # Apprentis/stagiaires
            'C21_ACTOCC1564_CS1',  # Agriculteurs exploitants
            'C21_ACTOCC1564_CS2',  # Artisans, commerçants, chefs d'entreprise
            'C21_ACTOCC1564_CS3',  # Cadres et professions intellectuelles supérieures
            'C21_ACTOCC1564_CS4',  # Professions intermédiaires
            'C21_ACTOCC1564_CS5',  # Employés
            'C21_ACTOCC1564_CS6',  # Ouvriers
            'C21_ACTOCC15P', # Actifs occupés 15 ans ou plus
            'C21_ACTOCC15P_PAS', # Actifs occupés 15 ans ou plus qui n'utilisent pas de moyen de transport pour aller travailler
            'C21_ACTOCC15P_MAR', # Actifs occupés 15 ans ou plus qui vont travailler principalement à pied
            'C21_ACTOCC15P_VELO', # Actifs occupés 15 ou plus qui utilisent principalement un vélo pour aller travailler
            'C21_ACTOCC15P_2ROUESMOT', # Actifs occupés 15 ou plus qui utilisent principalement un deux-roues motorisé pour aller travailler
            'C21_ACTOCC15P_VOIT', # Actifs occupés 15 ou plus qui utilisent principalement une voiture pour aller travailler
            'C21_ACTOCC15P_TCOM' # Actifs occupés 15 ou plus qui utilisent principalement les transports en commun pour aller travailler
        ]
    }

    # 3. Téléchargement et extraction
    url = trouver_source_url(source_activite['nom'])
    zip_path, extract_path, _ = definir_chemins(source_activite['nom'], ext="zip")
    telecharger_fichier(url, zip_path)
    os.makedirs(extract_path, exist_ok=True)
    extraire_zip(zip_path, extract_path)

    # 4. Chargement du CSV
    csv_path = trouver_fichier(source_activite['fichier'], extract_path)
    activite_df = pd.read_csv(
        csv_path,
        sep=";",
        encoding='utf-8',
        dtype={'IRIS': str},  # Code IRIS en texte
        decimal=".",          # Format des nombres
        usecols=source_activite['colonnes_a_garder']  # Sélection des colonnes utiles
    )

    # 5. Nettoyage des données
    activite_df['IRIS'] = activite_df['IRIS'].str.strip()

    # 6. Jointure avec les données existantes
    print("\nJointure des données d'activité avec les données IRIS existantes...")

    # Vérifier si la colonne IRIS existe dans les données existantes
    if 'IRIS' not in iris_complet.columns:
        raise ValueError("La colonne 'IRIS' est absente des données existantes")

    # Effectuer la jointure
    iris_complet = iris_complet.merge(activite_df,on='IRIS',how='left')

    # 7. Vérification
    print("\nVérification des données ajoutées :")
    print(f"Nombre d'IRIS avec données d'activité : {iris_complet['P21_SAL15P'].notna().sum()}")
    print("Exemple de données ajoutées :")
    print(iris_complet[['IRIS', 'P21_SAL15P', 'P21_NSAL15P', 'C21_ACTOCC1564_CS3']].head())

    # 8. Export final
    exporter_parquet(iris_complet, "iris_epci_donnees")
    exporter_gpkg(iris_complet, "iris_epci_donnees")

    print("\nOpération terminée avec succès. Données d'activité des résidents ajoutées.")

# Exécution
recuperer_activites_residents()


# In[90]:


def afficher_population_iris(export = False):
    # 1. Chargement des données
    iris = charger_fichier_parquet("iris_epci_donnees", crs=3857)
    limites_epci = charger_fichier_parquet("limites_epci", crs=3857)

    # 2. Affichage
    fig, ax = plt.subplots(figsize=(10, 10))
    iris.plot(column="P21_POP",
             cmap="OrRd",
             legend=True,
             legend_kwds={'label': "Population par IRIS (2021)"},
             ax=ax,
             linewidth=0.1,
             edgecolor="grey")
    limites_epci.plot(ax=ax, alpha=1, edgecolor='black', facecolor='none')
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)

    ax.set_title("Population par IRIS (2021)")
    ax.axis('off')
    plt.tight_layout()

    # (Optionnel) Export PNG
    if export :
        images_export_dir = os.path.join(images_dir, "visualisation_population_iris.png")
        plt.savefig(images_export_dir, dpi=300, bbox_inches='tight')
        print(f"Carte exportée vers : {images_export_dir}")

    plt.show()
    plt.close()

# Exécution
afficher_population_iris(export = True)


# In[91]:


def afficher_revenu_median_iris(export = False):
    # 1. Chargement des données
    iris = charger_fichier_parquet("iris_epci_donnees", crs=3857)
    limites_epci = charger_fichier_parquet("limites_epci", crs=3857)

    # 2. Conversion des données et gestion des 'ns' (non-significatif)
    iris['DISP_MED21'] = iris['DISP_MED21'].replace('ns', np.nan)
    iris['DISP_MED21'] = pd.to_numeric(iris['DISP_MED21'], errors='coerce')

    # 3. Affichage
    fig, ax = plt.subplots(figsize=(10, 10), facecolor='white')

    norm = mcolors.Normalize(vmin=iris['DISP_MED21'].min(), vmax=iris['DISP_MED21'].max())
    cmap = cm.viridis

    iris.plot(column="DISP_MED21",
             cmap=cmap,
             norm=norm,
             legend=True,
             legend_kwds={
                 'label': "Revenu médian disponible (€/an)",
                 'orientation': "vertical"
             },
             missing_kwds={
                 'color': '#cccccc',
                 'label': "Non-significatives"
             },
             ax=ax,
             linewidth=0.2,
             edgecolor='white')

    limites_epci.plot(ax=ax, alpha=1, edgecolor='black', facecolor='none', linewidth=1)

    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
    ax.set_title("Revenu médian disponible par IRIS (2021)\nDonnées Filosofi - INSEE", fontsize=14)
    ax.axis('off')
    ns_patch = mpatches.Patch(color='#cccccc', label="Données non-significatives selon l'INSEE")
    ax.legend(handles=[ns_patch], loc='upper right')
    plt.tight_layout()

    # (Optionnel) Export PNG
    if export :
        images_export_dir = os.path.join(images_dir, "visualisation_iris_revenu_median.png")
        plt.savefig(images_export_dir, dpi=300, bbox_inches='tight')
        print(f"Carte exportée vers : {images_export_dir}")

    plt.show()
    plt.close()

# Exécution
afficher_revenu_median_iris(export = True)


# In[92]:


def afficher_population_active_iris(export = False):
    # 1. Chargement des données
    iris = charger_fichier_parquet("iris_epci_donnees", crs=3857)
    limites_epci = charger_fichier_parquet("limites_epci", crs=3857)

    # 2. Calcul de la proportion de population active
    iris['PROP_ACTIVE'] = iris['C21_ACTOCC15P'] / iris['P21_POP'] * 100

    # 3. Gestion des valeurs manquantes ou non calculables
    iris['PROP_ACTIVE'] = iris['PROP_ACTIVE'].replace([np.inf, -np.inf], np.nan)

    # 4. Affichage
    fig, ax = plt.subplots(figsize=(10, 10), facecolor='white')

    norm = mcolors.Normalize(vmin=iris['PROP_ACTIVE'].min(), vmax=iris['PROP_ACTIVE'].max())
    cmap = cm.plasma  # Choix d'une palette colorimétrique

    iris.plot(column="PROP_ACTIVE",
             cmap=cmap,
             norm=norm,
             legend=True,
             legend_kwds={
                 'label': "Part de population active (%)",
                 'orientation': "vertical",
                 'format': "%.0f%%"
             },
             missing_kwds={
                 'color': '#cccccc',
                 'label': "Données manquantes"
             },
             ax=ax,
             linewidth=0.2,
             edgecolor='white',
             alpha=0.9)

    limites_epci.plot(ax=ax, alpha=1, edgecolor='black', facecolor='none', linewidth=1)

    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)

    ax.set_title("Part de la population active (15 ans ou plus) par rapport à la population totale par IRIS (2021)\nDonnées INSEE - Activité des résidents", 
                fontsize=14)
    ax.axis('off')
    missing_patch = mpatches.Patch(color='#cccccc', label='Données manquantes')
    ax.legend(handles=[missing_patch], loc='upper right')
    plt.tight_layout()

    # (Optionnel) Export PNG
    if export :
        images_export_dir = os.path.join(images_dir, "visualisation_population_active.png")
        plt.savefig(images_export_dir, dpi=300, bbox_inches='tight')
        print(f"Carte exportée vers : {images_export_dir}")

    plt.show()
    plt.close()

# Exécuter la fonction
afficher_population_active_iris(export = True)


# ### 2.9. Données SIRENE sur les entreprises / établissements et leur localisation dans la zone étudiée via l'API
# ---
# Les données SIRENE servent à estimer le nombre d'emplois disponibles et à obtenir leur géolocalisation. Il existe une source d'erreurs potentielles : le nombre d'emplois liés aux sièges sociaux.
# 
# Dans le cas de l'EMS, le siège social de l'entreprise Herbalife est par exemple listé comme ayant entre 5000 et 9999 employés. Le problème, c'est qu'il est impossible qu'autant de personnes travailllent sur le site, qui se trouve dans la zone commerciale de Mundolsheim à coté d'un Castorama et d'un petit parking.
# 
# Ce genre d'entrées pose problème, là où d'autres sont légitimes (ex : le siège social de l'université de Strasbourg). Comme je n'ai pas trouvé de moyen fiable de corriger ces données de manière automatisée, il s'agit d'une source d'imprécision pour le calcul de certains indicateurs.
# 
# Documentation : 
# * Les données SIRENE nous servent à récupérer les entreprises dans la zone étudiée étudié : https://www.economie.gouv.fr/entreprises/repertoire-sirene-gratuit
# * Comment se créer un compte pour accéder à l'API : https://www.sirene.fr/static-resources/documentation/Insee_API_publique_modalites_connexion.pdf
# * Liste des variables utilisées et leur signification : https://www.sirene.fr/static-resources/documentation/v_sommaire_311.htm

# In[47]:


# Vérifie l'accès à l'API
def tester_acces_api_sirene():
    API_KEY = "98a89327-7fab-4416-a893-277fabe416d1"
    headers = {"X-INSEE-Api-Key-Integration": API_KEY}
    params = {"q": "codeCommuneEtablissement:67001", "nombre": 10, "curseur": "*"}
    response = requests.get("https://api.insee.fr/api-sirene/3.11/siret", headers=headers, params=params)

    if response.status_code == 200:
        print("Connexion à l'API réussie")
    else:
        print(f"Erreur {response.status_code} : {response.text}")

# Exécution
tester_acces_api_sirene()


# #### 2.9.1. Appel à l'API SIRENE
# ---

# In[71]:


# Temps d'exécution : ~50 minutes
def recuperer_donnees_api_sirene():
    # 1. Définition de la clé pour utiliser l'API
    API_KEY = "98a89327-7fab-4416-a893-277fabe416d1"
    headers = {"X-INSEE-Api-Key-Integration": API_KEY}

    # 2. Chargement des données
    limites_epci = charger_fichier_parquet("limites_epci", crs=2154)
    communes_path = trouver_fichier("COMMUNE.shp", extraction_dir)

    # 3. Récupérer les codes INSEE uniques des communes dans la zone étudiée
    gdf_communes = gpd.read_file(communes_path).to_crs(2154)
    communes_epci = gpd.sjoin(gdf_communes, limites_epci, how='inner', predicate='within')
    communes = sorted(communes_epci['INSEE_COM'].unique().tolist())
    print(f" Récupération des données SIRENE sur {len(communes)} communes : {communes}") 

    etablissements_total = []

    # 4. Appel à l'API
    for code_commune in communes:
        curseur = "*"
        previous_curseur = None
        total_commune = 0

        while True:
            params = {
                "q": f"codeCommuneEtablissement:{code_commune}",
                "nombre": 1000,
                "curseur": curseur
            }

            url = "https://api.insee.fr/api-sirene/3.11/siret"
            response = requests.get(url, headers=headers, params=params)

            if response.status_code != 200:
                print(f"Erreur {response.status_code} : {response.text}")
                break

            data = response.json()
            etablissements = data.get("etablissements", [])
            total_commune += len(etablissements)
            etablissements_total.extend(etablissements)

            curseur_suivant = data.get("header", {}).get("curseurSuivant")

            # Stop si boucle détectée
            if not curseur_suivant or curseur_suivant == previous_curseur:
                break

            previous_curseur = curseur
            curseur = curseur_suivant
            time.sleep(2.1)  # Pause de 2.1s entre les requêtes pour ne pas
                             # dépasser la limite de 30 requêtes par minute de l'API

    # 5. Export
    sirene_epci = pd.json_normalize(etablissements_total)
    export_path = os.path.join(exports_dir, "sirene_epci.csv")
    sirene_epci.to_csv(export_path, index=False)

    print(f"Fichier '{export_path}' exporté avec succès")

# Exécution
recuperer_donnees_api_sirene()


# #### 2.9.2. Récupération des établissements géolocalisées dans la zone étudiée
# ---
# Note : dans les données SIRENE, tout les établissement avec une adresse physique possèdent leur géolocalisation en Lambert 93 associée
# 
# Pour nos calculs d'indicateurs, on ne conserve que les établissements géolocalisés : ce choix est discutable, car il mène à une sous-estimation 
# 
# Ce script est séparé de l'appel à l'API précédent pour ne pas avoir à tout re-télécharger en cas d'erreurs

# In[48]:


def recuperer_etablissements_geolocalises():
    # 1. Chargement des données
    sirene_epci= pd.read_csv(os.path.join(exports_dir, "sirene_epci.csv"))

    # 2. Filtre : ne conserve que les établissements dont la géolocalisation est précisée
    sirene_epci_geo = sirene_epci[
        sirene_epci["adresseEtablissement.coordonneeLambertAbscisseEtablissement"].notnull() &
        sirene_epci["adresseEtablissement.coordonneeLambertOrdonneeEtablissement"].notnull() &
        (sirene_epci["adresseEtablissement.coordonneeLambertAbscisseEtablissement"] != "[ND]") &
        (sirene_epci["adresseEtablissement.coordonneeLambertOrdonneeEtablissement"] != "[ND]")
    ]

    # 3. Export en CSV pour gérer les champs en JSON
    export_path = os.path.join(exports_dir, "etablissements_avec_coordonnees_lambert.csv")
    export_path = os.path.join(exports_dir, "etablissements_avec_coordonnees_lambert.csv")
    sirene_epci_geo.to_csv(
        export_path,
        index=False,
        sep=';',
        quoting=csv.QUOTE_ALL, # Toutes les valeurs entre guillemets
        quotechar='"',
        encoding="utf-8"
    )

    print(f"Fichier '{export_path}' exporté avec succès")

# Exécution
recuperer_etablissements_geolocalises()


# In[365]:


def recuperer_etablissements_geolocalises():
    # 1. Chargement des données
    sirene_epci = pd.read_csv(os.path.join(exports_dir, "sirene_epci.csv"))

    # 2. Comptage total
    total_etablissements = len(sirene_epci)

    # 3. Filtre : ne conserve que les établissements dont la géolocalisation est précisée
    coord_x = "adresseEtablissement.coordonneeLambertAbscisseEtablissement"
    coord_y = "adresseEtablissement.coordonneeLambertOrdonneeEtablissement"

    mask_coordonnees_valides = (
        sirene_epci[coord_x].notnull() &
        sirene_epci[coord_y].notnull() &
        (sirene_epci[coord_x] != "[ND]") &
        (sirene_epci[coord_y] != "[ND]")
    )

    sirene_epci_geo = sirene_epci[mask_coordonnees_valides].copy()
    nb_geolocalises = len(sirene_epci_geo)
    nb_non_geolocalises = total_etablissements - nb_geolocalises

    # 4. Export CSV
    export_path = os.path.join(exports_dir, "etablissements_avec_coordonnees_lambert_test.csv")
    sirene_epci_geo.to_csv(
        export_path,
        index=False,
        sep=';',
        quoting=csv.QUOTE_ALL,
        quotechar='"',
        encoding="utf-8"
    )

    print(f" Total d'établissements dans le fichier : {total_etablissements}")
    print(f" Établissements géolocalisés (coord. Lambert valides) : {nb_geolocalises}")
    print(f" Établissements sans coordonnées utilisables : {nb_non_geolocalises}")
    print(f" Fichier exporté : '{export_path}'")

# Exécution
recuperer_etablissements_geolocalises()


# #### 2.9.3. Préparation des données SIRENE pour le calcul des indicateurs
# ---
# La fonction créée une géométrie point associée aux établisemments actifs, avec le nombre d'employés estimés.

# In[50]:


def recuperer_donnees_sirene():
    # 1. Charger le fichier CSV
    sirene_epci_geo = pd.read_csv(
        os.path.join(exports_dir, "etablissements_avec_coordonnees_lambert.csv"),
        sep=';',
        quotechar='"',
        low_memory=False
    )

    # 2. Extraire les données contenues dans le JSON
    def extraire_caractere_employeur(val):
        try:
            periodes = ast.literal_eval(val)
            if isinstance(periodes, list) and len(periodes) > 0:
                dernier = periodes[0]  # normalement, le plus récent est en tête
                return dernier.get("caractereEmployeurEtablissement", "N")
        except (ValueError, SyntaxError):
            pass
        return "N"

    def extraire_activite_principale(val):
        try:
            periodes = ast.literal_eval(val)
            if isinstance(periodes, list) and len(periodes) > 0:
                dernier = periodes[0]
                return dernier.get("activitePrincipaleEtablissement", None)
        except (ValueError, SyntaxError):
            pass
        return None

    def etablissement_actif(val):
        try:
            periodes = ast.literal_eval(val)
            if isinstance(periodes, list) and len(periodes) > 0:
                return periodes[0].get("etatAdministratifEtablissement", "F") == "A"
        except (ValueError, SyntaxError):
            pass
        return False

    sirene_epci_geo["caractereEmployeurEtablissement"] = sirene_epci_geo["periodesEtablissement"].apply(extraire_caractere_employeur)
    sirene_epci_geo["activitePrincipaleEtablissement"] = sirene_epci_geo["periodesEtablissement"].apply(extraire_activite_principale)
    sirene_epci_geo["etablissementActif"] = sirene_epci_geo["periodesEtablissement"].apply(etablissement_actif)

    # 3. Remplir les effectifs manquants
    sirene_epci_geo["trancheEffectifsEtablissement"] = sirene_epci_geo["trancheEffectifsEtablissement"].fillna("00")

    # 4. Filtrer les établissements actifs et employeurs
    sirene_epci_geo = sirene_epci_geo[
        (sirene_epci_geo["etablissementActif"] == True) &
        (sirene_epci_geo["caractereEmployeurEtablissement"] == "O")
    ].copy()

    # 5. Nettoyage des coordonnées
    sirene_epci_geo["x"] = pd.to_numeric(sirene_epci_geo["adresseEtablissement.coordonneeLambertAbscisseEtablissement"], errors="coerce")
    sirene_epci_geo["y"] = pd.to_numeric(sirene_epci_geo["adresseEtablissement.coordonneeLambertOrdonneeEtablissement"], errors="coerce")
    sirene_epci_geo = sirene_epci_geo[sirene_epci_geo["x"].notnull() & sirene_epci_geo["y"].notnull()]

    # 6. Création des géométries
    geometry = [Point(xy) for xy in zip(sirene_epci_geo["x"], sirene_epci_geo["y"])]
    sirene_epci_geo_gdf = gpd.GeoDataFrame(sirene_epci_geo, geometry=geometry, crs="EPSG:2154")

    # 7. Estimation des emplois par établissement
    # Chaque tranche correspond à un nombre d'employés : par exemple, "01" veut dire que l'établissement emploie 1 à 2 personnes
    # Par conquéquent, on estime le nombre d'emplois via une moyenne.
    tranche_to_estimation = {
        "00": 1,    # "00" : Autoentrepreneur : on compte un employé
        "01": 2,    # "01" : 1 à 2 
        "02": 4,    # "02" : 3 à 5
        "03": 7,    # "03" : 6 à 9
        "11": 15,   # "11" : 10 à 19
        "12": 35,   # "12" : 20 à 49
        "21": 75,   # "21" : 50 à 99
        "22": 150,  # "22" : 100 à 199
        "31": 225,  # "31" : 200 à 249
        "32": 375,  # "32" : 250 à 499
        "41": 750,  # "41" : 500 à 999
        "42": 1500, # "42" : 1000 à 1999
        "51": 3500, # "51" : 2000 à 4999
        "52": 7500, # "52" : 5000 à 9999
        "53": 12500 # "53" : 10000 et plus
    }

    sirene_epci_geo_gdf["emplois"] = sirene_epci_geo_gdf["trancheEffectifsEtablissement"].map(tranche_to_estimation).fillna(0)

    # 8. Sélection des colonnes utiles
    colonnes_utiles = ["siret", "activitePrincipaleEtablissement", "emplois", "geometry"]
    sirene_epci_geo_gdf = sirene_epci_geo_gdf[colonnes_utiles]

    # 9. Export
    exporter_parquet(sirene_epci_geo_gdf, "etablissements_emplois_epci")
    exporter_gpkg(sirene_epci_geo_gdf, "etablissements_emplois_epci")

    print(f"{len(sirene_epci_geo_gdf)} établissements employeurs géolocalisés exportés avec succès.")

# Exécution
recuperer_donnees_sirene()


# In[87]:


def afficher_donnees_sirene(export = False):
    # 1. Chargement des données
    limites_epci = charger_fichier_parquet("limites_epci", crs=3857)
    etablissements_emplois_epci = charger_fichier_parquet("etablissements_emplois_epci", crs=3857)

    # 2. Affichage
    fig, ax = plt.subplots(figsize=(10, 10))

    limites_epci.plot(ax=ax, alpha=1, edgecolor='black', facecolor='none')
    etablissements_emplois_epci.plot(ax=ax, color='blue', markersize=10, alpha=0.7, edgecolor='black')

    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)

    ax.set_axis_off()
    plt.title("Sources d'emplois (données SIRENE, juin 2025)", fontsize=14)

    # (Optionnel) Export PNG
    if export :
        images_export_dir = os.path.join(images_dir, "visualisation_sirene_emplois.png")
        plt.savefig(images_export_dir, dpi=300, bbox_inches='tight')
        print(f"Carte exportée vers : {images_export_dir}")

    plt.show()
    plt.close()

# Exécution
afficher_donnees_sirene(export = True)


# ### 2.10. Base Permanente des Équipements (BPE) 
# ---
# La base permanente des équipements (BPE) est une base à vocation statistique. Elle répertorie un large éventail d'équipements et de services, marchands ou non, accessibles au public sur l'ensemble de la France au 1ᵉʳ janvier de chaque année.
# 
# En 2023, elle porte sur 209 types de services et équipements différents, répartis en sept grands domaines"
# 
# Types de services listés dans la BPE :
# - A : Services pour les particuliers
# - B : Commerces
# - C : Enseignement
# - D : Santé et action sociale
# - E : Transports et déplacements
# - F : Sports, loisirs et culture
# - G : Tourisme. On ne conserve pas les données de ce dernier type, car on considère qu'il ne s'agit pas d'un service à la population locale.
#   
# Documentation : 
# * https://www.insee.fr/fr/metadonnees/source/serie/s1161
# * Liste des types d'équipements : https://www.insee.fr/fr/metadonnees/source/fichier/BPE23_liste_hierarchisee_TYPEQU.html

# In[53]:


def recuperer_services_bpe():
    # 1. Définir la source
    nom_source = "insee_bpe_2023"
    url = trouver_source_url(nom_source)

    # 2. Charger les limites de l'EPCI
    limites_epci = charger_fichier_parquet("limites_epci", crs=2154)

    # 3. Définir les chemins
    archive_path, extract_path, _ = definir_chemins(nom_source, ext="zip")

    # 4. Télécharger et extraire
    telecharger_fichier(url, archive_path)
    extraire_zip(archive_path, extract_path)

    # 5. Charger le CSV de la BPE avec coordonnées en Lambert 93
    csv_path = os.path.join(extract_path, "BPE23.csv")

    # TYPEQU : indique le type d'équipement
    colonnes_utiles = ["TYPEQU", "LAMBERT_X", "LAMBERT_Y"]
    df_bpe = pd.read_csv(
        csv_path,
        sep=';',
        dtype={"LAMBERT_X": float, "LAMBERT_Y": float},
        usecols=lambda c: c in colonnes_utiles,
        low_memory=False
    )

    # 6. Retire les lignes sans coordonnées et les équipements touristiques (type G)
    df_bpe_lambert = df_bpe.dropna(subset=["LAMBERT_X", "LAMBERT_Y"])
    df_bpe_lambert = df_bpe_lambert[~df_bpe_lambert['TYPEQU'].str.startswith('G')] 

    # 7. Conversion en GeoDataFrame
    geometry = [Point(xy) for xy in zip(df_bpe_lambert["LAMBERT_X"], df_bpe_lambert["LAMBERT_Y"])]
    gdf_bpe = gpd.GeoDataFrame(df_bpe_lambert, geometry=geometry, crs="EPSG:2154")

    # 8. Jointure spatiale
    carreaux = charger_fichier_parquet("maille_200m_epci", crs=2154)
    bpe_epci = gpd.sjoin(gdf_bpe, limites_epci, how="inner", predicate="within").drop(columns=["index_right"])

    # 9. On ne conserve que les colonnes utiles
    bpe_epci = bpe_epci[["TYPEQU", "geometry"]]

    # 10. Export
    exporter_gpkg(bpe_epci, "insee_bpe_2023_epci")
    exporter_parquet(bpe_epci, "insee_bpe_2023_epci")

    print(f"{len(bpe_epci)} équipements trouvées dans la zone étudiée")

# Exécution
recuperer_services_bpe()


# In[88]:


def affichage_services_bpe(export = False):
    # 1. Chargement des données
    limites_epci = charger_fichier_parquet("limites_epci", crs=3857)
    bpe_epci = charger_fichier_parquet("insee_bpe_2023_epci", crs=3857)

    # 2. Affichage
    fig, ax = plt.subplots(figsize=(10, 10))
    limites_epci.plot(ax=ax, alpha=1, edgecolor='black', facecolor='none')
    bpe_epci.plot(ax=ax, color='red', markersize=10, alpha=0.7, edgecolor='black')
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)

    ax.set_axis_off()
    plt.title("Localisation des services (BPE 2023)", fontsize=14)

    # (Optionnel) Export PNG
    if export :
        images_export_dir = os.path.join(images_dir, "visualisation_bpe_services.png")
        plt.savefig(images_export_dir, dpi=300, bbox_inches='tight')
        print(f"Carte exportée vers : {images_export_dir}")

    plt.show()
    plt.close()

# Exécution
affichage_services_bpe(export = True)


# ### 2.11. Bornes de recharge des véhicules électriques
# ---
# Note : plusieurs bornes peuvent partager la même géométrie
# 
# Documentation : 
# * https://www.data.gouv.fr/fr/datasets/fichier-consolide-des-bornes-de-recharge-pour-vehicules-electriques/

# In[56]:


def recuperer_bornes_ve():
    # 1. Récupérer l'URL depuis les sources
    nom_source = "bornes_ve"
    url = trouver_source_url(nom_source)

    # 2. Chargement des limites de la zone étudiée
    bornes_ve = gpd.read_file(geojson_path)
    limites_epci = charger_fichier_parquet("limites_epci", crs=4326)

    # 3. Télécharger le fichier source
    geojson_path = os.path.join(dir, f"{nom_source}.geojson")
    #telecharger_fichier(url, geojson_path)

    # 4. Découper selon l'EPCI
    bornes_ve_epci = decouper_par_epci(bornes_ve, limites_epci, predicate="intersects")

    # 5. Nettoyage, ne conserve que la géométrie, le reste n'est pas utilisé
    bornes_ve_epci = bornes_ve_epci[['geometry']].copy()

    # 6. Export
    exporter_gpkg(bornes_ve_epci, "bornes_ve_epci")
    exporter_parquet(bornes_ve_epci, "bornes_ve_epci")

    print(f"{len(bornes_ve_epci)} bornes de recharge pour véhicules électriques trouvées dans l'EPCI")

# Exécution
recuperation_bornes_ve()


# In[89]:


def afficher_bornes_ve(export = False):
    # 1. Chargement des données
    limites_epci = charger_fichier_parquet("limites_epci", crs=3857)
    bornes_ve_epci = charger_fichier_parquet("bornes_ve_epci", crs=3857)

    # 2. Affichage
    fig, ax = plt.subplots(figsize=(10, 10))
    limites_epci.plot(ax=ax, alpha=1, edgecolor='black', facecolor='none')
    bornes_ve_epci.plot(ax=ax, color='red', markersize=10, alpha=0.7, edgecolor='black')
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)

    ax.set_axis_off()
    plt.title("Bornes de recharge de véhicule électriques", fontsize=14)

    # (Optionnel) Export PNG
    if export :
        images_export_dir = os.path.join(images_dir, "visualisation_bornes_recharge_ve.png")
        plt.savefig(images_export_dir, dpi=300, bbox_inches='tight')
        print(f"Carte exportée vers : {images_export_dir}")

    plt.show()
    plt.close()

# Exécution
afficher_bornes_ve(export = True)


# ### 2.12. Zones à Faible Émissions (ZFE)
# ---
# "Le fichier aires.geojson contient les règles de limitation de circulation sur des zones concernant différents types de véhicules (véhicules particuliers, utilitaires, poids lourds etc) où il n'est plus possible de circuler en fonction de certificats Crit’Air.
# 
# voies.geojson contient des exceptions sur certains axes routiers où la Zone à Faible Émission a des règles différentes."
# 
# Documentation : 
# * https://www.data.gouv.fr/fr/datasets/base-nationale-consolidee-des-zones-a-faibles-emissions/

# In[58]:


def recuperer_zfe():
    # 1. Récupérer les URLs depuis les sources
    nom_source_aires = "zfe_aires"
    nom_source_voies = "zfe_voies"

    url_aires = trouver_source_url(nom_source_aires)
    url_voies = trouver_source_url(nom_source_voies)

    # 2. Télécharger les fichiers sources
    geojson_path_aires = os.path.join(dir, f"{nom_source_aires}.geojson")
    geojson_path_voies = os.path.join(dir, f"{nom_source_voies}.geojson")
    telecharger_fichier(url_aires, geojson_path_aires)
    telecharger_fichier(url_voies, geojson_path_voies)

    # 3. Charger les données et préparer les CRS
    zfe_aires = gpd.read_file(geojson_path_aires)
    zfe_voies = gpd.read_file(geojson_path_voies)
    limites_tampon = charger_fichier_parquet("limites_epci_tampon", crs=4326)

    """
    4. Découper selon l'EPCI. 
    Le fichier 'aires' contient des polygones représentant des EPCI, mais leur géométrie n'est pas exacte 
    à celles de la BD Topo : elle dépasse par endroit la géométrie de la BD Topo. Pour ne conserver que 
    l'EPCI étudiée, on effectue la jointure spatiale par rapport à la zone tampon, légèrement plus large
    """
    zfe_aires_epci = decouper_par_epci(zfe_aires, limites_tampon, predicate="intersects")
    zfe_voies_epci = decouper_par_epci(zfe_voies, limites_tampon, predicate="intersects")

    # 5. Nettoyage, ne conserve que les champs utiles
    zfe_aires_epci = zfe_aires_epci[['autobus_autocars_critair', 'autobus_autocars_horaires', 'creation_date', 
                                     'date_debut', 'date_fin', 'date_fin_init', 'date_maj', 'deux_rm_critair', 
                                     'deux_rm_horaires', 'etiquette', 'geo_point_2d', 'gid', 'gml_id', 'id', 
                                     'modification_date', 'nom', 'obj_datec', 'obj_datem', 'objectid', 
                                     'pl_critair', 'pl_horaires', 'taxis_critair', 'vp_critair', 
                                     'vp_horaires', 'vul_critair', 'vul_horaires', 'geometry']].copy()

    zfe_voies_epci = zfe_voies_epci[['autobus_autocars_critair', 'autobus_autocars_horaires', 'date_debut', 
                                     'date_fin', 'deux_rm_critair', 'deux_rm_horaires', 'geo_point_2d', 'gid',
                                     'id', 'one_way', 'osm_id', 'pl_critair', 'pl_horaires', 'vp_critair', 
                                     'vp_horaires', 'vul_critair', 'vul_horaires', 'zfe_derogation', 'geometry']].copy()
    # 6. Exports
    exporter_gpkg(zfe_aires_epci, "zfe_aires_epci")
    exporter_parquet(zfe_aires_epci, "zfe_aires_epci")
    exporter_gpkg(zfe_voies_epci, "zfe_voies_epci")
    exporter_parquet(zfe_voies_epci, "zfe_voies_epci")

# Exécution
recuperer_zfe()


# In[96]:


def afficher_zfe(export = False):
    # 1. Chargement des données
    zfe_aires = charger_fichier_parquet("zfe_aires_epci", crs=3857)
    zfe_voies = charger_fichier_parquet("zfe_voies_epci", crs=3857)
    limites_epci = charger_fichier_parquet("limites_epci", crs=3857)

    # 2. Filtrer les ZFE actives (date_debut <= aujourd'hui)
    aujourdhui = pd.to_datetime(datetime.now().date())
    zfe_actives = zfe_aires[pd.to_datetime(zfe_aires['date_debut']) <= aujourdhui].copy()

    # 3. Trier par niveau Crit'Air (du plus restrictif au moins restrictif)
    zfe_actives['critair_num'] = zfe_actives['vp_critair'].str.extract('(\d)').astype(int)
    zfe_actives = zfe_actives.sort_values('critair_num', ascending=False)

    # 4. Affichage
    fig, ax = plt.subplots(figsize=(10, 10))

    # 4.1. Palette de couleurs pour les restrictions Crit'Air
    cmap = ListedColormap(['#2b83ba', '#abdda4', '#fdae61', '#d7191c'])  # V2 à V5
    norm = plt.Normalize(vmin=2, vmax=5)

    # 4.2. Affichage des polygones ZFE (dans l'ordre de restriction)
    for critair_level in [5, 4, 3, 2]:
        subset = zfe_actives[zfe_actives['critair_num'] == critair_level]
        if not subset.empty:
            subset.plot(ax=ax, color=cmap(norm(critair_level)), 
                       edgecolor='white', linewidth=0.5,
                       label=f'V{critair_level}', alpha=0.7)

    # 4.3. Autres afficahges
    if not zfe_voies.empty:
        zfe_voies.plot(ax=ax, color='black', alpha=0.4, linewidth=1.5, label='Voies ZFE')
    limites_epci.plot(ax=ax, color='none', edgecolor='black', linewidth=1)
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, attribution_size=8)

    # 6. Légende
    legend_elements = [
        Patch(facecolor='#2b83ba', edgecolor='white', label="V2 (Interdiction aux Crit'Air 2)"),
        Patch(facecolor='#abdda4', edgecolor='white', label='V3'),
        Patch(facecolor='#fdae61', edgecolor='white', label='V4'),
        Patch(facecolor='#d7191c', edgecolor='white', label="V5 (Interdiction aux Crit'Air 5)"),
        Patch(facecolor='black', edgecolor='black', alpha=0.4, label='Voies ZFE (Dérogation)')
    ]

    ax.legend(handles=legend_elements, title="Niveaux Crit'Air", loc='upper right')
    ax.set_title(f"Zones à Faibles Émissions actives au {datetime.now().strftime('%d/%m/%Y')}", fontsize=16)
    ax.axis('off')
    plt.tight_layout()

    # (Optionnel) Export PNG
    if export :
        images_export_dir = os.path.join(images_dir, "visualisation_zfe.png")
        plt.savefig(images_export_dir, dpi=300, bbox_inches='tight')
        print(f"Carte exportée vers : {images_export_dir}")

    plt.show()
    plt.close()

# Exécution
afficher_zfe(export = True)


# ### 2.13. Stations-service
# ---
# Documentation :
# * https://www.data.gouv.fr/fr/datasets/prix-des-carburants-en-france-flux-quotidien-1/

# In[63]:


def recuperer_stations_service():
    # 1. Récupérer l'URL depuis les sources
    nom_source = "stations_service"
    url = trouver_source_url(nom_source)

    # 2. Télécharger le fichier source
    geojson_path = os.path.join(dir, f"{nom_source}.geojson")
    telecharger_fichier(url, geojson_path)

    # 3. Charger les données et préparer les CRS
    pompes_stations = gpd.read_file(geojson_path)
    limites_epci = charger_fichier_parquet("limites_epci", crs=2154)

    # 4. Découper selon l'EPCI
    pompes_stations_epci = decouper_par_epci(pompes_stations, limites_epci, predicate="intersects")

    # 5. Nettoyage, ne conserve que les champs utiles
    pompes_stations_epci = pompes_stations_epci[['id','geometry']].copy()

    # 6. Regrouper par station à l'aide du champ 'id' identique pour les pompes d'une même station
    stations_service_epci = (pompes_stations_epci.dissolve(by='id', as_index=False).reset_index(drop=True))

    # 7. Exports
    exporter_gpkg(stations_service_epci, "stations_service_epci")
    exporter_parquet(stations_service_epci, "stations_service_epci")

    print(f"{len(stations_service_epci)} stations-service trouvées dans l'EPCI")

# Exécution
recuperer_stations_service()


# In[90]:


def afficher_stations_service(export = False):
    # 1. Chargement des données
    limites_epci = charger_fichier_parquet("limites_epci", crs=3857)
    stations_service_epci = charger_fichier_parquet("stations_service_epci", crs=3857)

    # 2. Affichage
    fig, ax = plt.subplots(figsize=(10, 10))
    limites_epci.plot(ax=ax, alpha=1, edgecolor='black', facecolor='none')
    stations_service_epci.plot(ax=ax, color='red', markersize=10, alpha=0.7, edgecolor='black')
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)

    ax.set_axis_off()
    plt.title("Localisation des stations-service", fontsize=14)

    # (Optionnel) Export PNG
    if export :
        images_export_dir = os.path.join(images_dir, "visualisation_stations_service.png")
        plt.savefig(images_export_dir, dpi=300, bbox_inches='tight')
        print(f"Carte exportée vers : {images_export_dir}")

    plt.show()

# Exécution
afficher_stations_service(export = True)


# 

# 
# 

# ### 2.14. Accidents dûs aux transports
# ---
# Les données nationales sont disponibles par année sur https://www.data.gouv.fr/fr/datasets/bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-routiere-annees-de-2005-a-2023/#/resources. A chaque année, des fichiers .csv usagers-aaaa, vehicules-aaaa, lieux-aaaa et caract-aaaa sont disponibles. Comme l'API de data.gouv ne permet que de faire des requêtes sur un fichier à la fois, après avoir manuellement récupérer leur identifiant métadonnée sur leur site, il n'est pas possible de simplement faire une requête pour récupérer tout les accidents ayant eu lieu entre deux années sur la zone étudiée. Et comme les URLs sont légèrement différentes à chaque fois, il n'est pas possible de remplacer l'année dans les liens.
# 
# Ce script télécharge tout les fichiers disponibles pour 2023. Il s'agit de la dernière année disponible, et même si intégrer les données des années Covid est possible, en tirer des moyennes annuelles (comme la nombre d'accidents moyens par maille) serait peu représentatif du fait qu'il y avait beaucoup moins de transports. Et "les données sur la qualification de blessé hospitalisé depuis l’année 2018 ne peuvent être comparées aux années précédentes suite à des modifications de process de saisie des forces de l’ordre." https://www.data.gouv.fr/fr/datasets/r/8ef4c2a3-91a0-4d98-ae3a-989bde87b62a. Ce lien contient également la description des valeurs utilisées dans les colonnes.

# In[68]:


def recuperer_accidents_transports():
    # 1. Charger l'emprise spatiale
    limites_epci = charger_fichier_parquet("limites_epci", crs=4326)

    # 2. Créer une emprise pour la requête
    emprise = limites_epci.geometry.union_all()

    # 3. Définir les années sélectionnées. Modifier selon les besoins, par exemple : [2019, 2020, 2021, 2022, 2023]
    annees = [2023] 
    gdf_liste = []

    # 4. Récupération et traitement des données par année
    # Pour chaque année traitée, le fichier .csv des sources doit contenir le nom des données avec le même format
    # Exemple : 'accidents_2023_usagers', 'accidents_2022_usagers'
    for annee in annees:
        print(f"Traitement des données pour l'année {annee}...")

        # 4.1. Noms des sources
        nom_usagers = f"accidents_{annee}_usagers"
        nom_vehicules = f"accidents_{annee}_vehicules"
        nom_lieux = f"accidents_{annee}_lieux"
        nom_caract = f"accidents_{annee}_caract"

        # 4.2. URLs
        url_usagers = trouver_source_url(nom_usagers)
        url_vehicules = trouver_source_url(nom_vehicules)
        url_lieux = trouver_source_url(nom_lieux)
        url_caract = trouver_source_url(nom_caract)

        # 4.3. Téléchargement
        path_usagers = os.path.join(exports_dir, f"{nom_usagers}.csv")
        path_vehicules = os.path.join(exports_dir, f"{nom_vehicules}.csv")
        path_lieux = os.path.join(exports_dir, f"{nom_lieux}.csv")
        path_caract = os.path.join(exports_dir, f"{nom_caract}.csv")

        telecharger_fichier(url_usagers, path_usagers)
        telecharger_fichier(url_vehicules, path_vehicules)
        telecharger_fichier(url_lieux, path_lieux)
        telecharger_fichier(url_caract, path_caract)

        # 4.4. Lecture
        df_usagers = pd.read_csv(path_usagers, sep=";", encoding="utf-8", low_memory=False)
        df_vehicules = pd.read_csv(path_vehicules, sep=";", encoding="utf-8", low_memory=False)
        df_lieux = pd.read_csv(path_lieux, sep=";", encoding="utf-8", low_memory=False)
        df_caract = pd.read_csv(path_caract, sep=";", encoding="utf-8", low_memory=False)

        # 4.5. Fusion
        df_accidents = pd.merge(df_caract, df_lieux, on="Num_Acc", how="inner")
        df_accidents = pd.merge(df_accidents, df_vehicules, on="Num_Acc", how="left")
        df_accidents = pd.merge(df_accidents, df_usagers, on=["Num_Acc", "num_veh"], how="left")
        df_accidents["annee"] = annee  # Ajout de l'année

        # 4.6. Géométrie
        gdf = gpd.GeoDataFrame(
            df_accidents,
            geometry=gpd.points_from_xy(
                df_accidents["long"].str.replace(",", ".").astype(float),
                df_accidents["lat"].str.replace(",", ".").astype(float)
                ),
            crs=4326
        )

        # 4.7. Filtrage spatial
        gdf = gdf[gdf.within(emprise)]

        # 4.8. Stockage temporaire
        gdf_liste.append(gdf)

    # 5. Concaténation
    gdf_final = gpd.GeoDataFrame(pd.concat(gdf_liste, ignore_index=True), crs=4326)

    # 6. Export
    exporter_gpkg(gdf_final, "accidents_routiers_epci")
    exporter_parquet(gdf_final, "accidents_routiers_epci")
    print(f"Export réalisé avec {len(gdf_final)} accidents sur {len(annees)} année(s)")

# Exécution
recuperer_accidents_transports()


# In[104]:


def afficher_accidents_transports(export = False):
    # 1. Chargement des données
    limites_epci = charger_fichier_parquet("limites_epci", crs=3857)
    accidents_routiers = charger_fichier_parquet("accidents_routiers_epci", crs=3857)

    # 2. Affichage
    fig, ax = plt.subplots(figsize=(10, 10))
    limites_epci.plot(ax=ax, alpha=1, edgecolor='black', facecolor='none')
    accidents_routiers.plot(ax=ax, color='red', markersize=10, alpha=0.6, edgecolor='black')

    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)

    ax.set_axis_off()
    plt.title("Accidents routiers dans l'EPCI (2023)", fontsize=14)

    # (Optionnel) Export PNG
    if export :
        images_export_dir = os.path.join(images_dir, "visualisation_accidents_transport.png")
        plt.savefig(images_export_dir, dpi=300, bbox_inches='tight')
        print(f"Carte exportée vers : {images_export_dir}")

    plt.show()
    plt.close()

# Exécution
afficher_accidents_transports(export = True)


# ### 2.15. Stations d'autoréparation de vélos
# ---
# Note : plutôt que de récupérer les données mises à disposition par l'EMS (https://data.strasbourg.eu/explore/dataset/ateliers-d-autoreparation-de-velo/information/) qui ne font que redistribuer les données OSM à leur échelle, on récupère directement les données OSM à l'échelle française.
# 
# Documentation : 
# * https://osmnx.readthedocs.io/en/stable/
# * https://wiki.openstreetmap.org/wiki/Key:service:bicycle:diy

# In[70]:


def recuperer_stations_autoreparation_velos():
    # 1. Charger l'emprise spatiale
    limites_epci = charger_fichier_parquet("limites_epci", crs=4326)

    # 2. Créer une emprise pour la requête
    emprise = limites_epci.geometry.union_all()

    # 3. Définir les tags OSM à extraire
    tags = {"service:bicycle:diy": "yes"}

    # 4. Interrogation d’OSM via OSMnx
    stations_autoreparation_velos = ox.features_from_polygon(emprise, tags)

    # 5. Nettoyage, ne conserve que les champs utiles
    stations_autoreparation_velos = stations_autoreparation_velos[['geometry', 'name', 'amenity', 
                                                                   'shop', 'service:bicycle:diy']].copy()
    # 6. Export
    exporter_gpkg(stations_autoreparation_velos, "stations_autoreparation_velo")
    exporter_parquet(stations_autoreparation_velos, "stations_autoreparation_velo")

    print(f"{len(stations_autoreparation_velos)} stations d'autoréparation trouvées dans la zone étudiée")

# Exécution
recuperer_stations_autoreparation_velos()


# In[93]:


def afficher_stations_autoreparation_velos(export = False):
    # 1. Chargement des données
    limites_epci = charger_fichier_parquet("limites_epci", crs=3857)
    stations_autoreparation_velos = charger_fichier_parquet("stations_autoreparation_velo", crs=3857)

    # 2. Affichge
    fig, ax = plt.subplots(figsize=(10, 10))
    limites_epci.plot(ax=ax, alpha=1, edgecolor='black', facecolor='none')
    stations_autoreparation_velos.plot(ax=ax, color='pink', markersize=20, edgecolor='black')
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)

    ax.set_axis_off()
    plt.title("Localisation des stations d'autoréparation de vélos", fontsize=14)

    # (Optionnel) Export PNG
    if export :
        images_export_dir = os.path.join(images_dir, "visualisation_stations_autoreparation_velos.png")
        plt.savefig(images_export_dir, dpi=300, bbox_inches='tight')
        print(f"Carte exportée vers : {images_export_dir}")

    plt.show()
    plt.close()

# Exécution
afficher_stations_autoreparation_velos(export = True)


# ### 2.16. Parcs et jardins urbains
# --- 
# Il n'existe actuellement pas de données surfaciques à l'échelle française pour les parcs et jardins urbain, bien que l'IGN a commencé la production d'une telle base (https://www.ign.fr/agenda/contribuer-la-cartographie-des-parcs-et-jardins-publics)
# 
# L'EMS dispose d'un tel jeu de données, mais il s'agit de données ponctuelles pour représenter les parcs, qui ne sont pas utilisable pour des calculs de distance / d'accessibilité (https://data.strasbourg.eu/explore/dataset/lieux_parcs/information/)
# 
# En attendant, ce sont les données d'OSM qui ont été utilisées.

# In[75]:


def recuperer_parcs_jardins_urbains():
    # 1. Charger l'emprise spatiale
    limites_epci = charger_fichier_parquet("limites_epci", crs=4326)

    # 2. Créer une emprise pour la requête
    emprise = limites_epci.geometry.union_all()

    # 3. Définir les tags OSM pour les parcs et jardins urbains
    tags = {
        'leisure': 'park',
        'leisure': 'garden',
        'landuse': 'recreation_ground',
    }

    # 4. Interrogation d'OSM via OSMnx
    parcs_jardins = ox.features_from_polygon(emprise, tags=tags)

    # 5. Filtrage des polygons (données surfaciques)
    parcs_jardins = parcs_jardins[parcs_jardins.geometry.type.isin(['Polygon', 'MultiPolygon'])]

    # 6. Nettoyage, ne conserve que les champs utiles
    parcs_jardins = parcs_jardins[['name', 'leisure', 'landuse', 'opening_hours', 'access', 
                                   'operator', 'geometry']].copy()
    # 7. Export
    exporter_gpkg(parcs_jardins, "parcs_jardins_urbains_epci")
    exporter_parquet(parcs_jardins, "parcs_jardins_urbains_epci")

    print(f"{len(parcs_jardins)} parcs et jardins urbains trouvés.")

# Exécution
recuperer_parcs_jardins_urbains()


# In[107]:


def afficher_parcs_jardins_urbains(export = False):
    # 1. Chargement des données
    limites_epci = charger_fichier_parquet("limites_epci", crs=3857)
    parcs_jardins = charger_fichier_parquet("parcs_jardins_urbains_epci", crs=3857)

    # 2. Affichage
    fig, ax = plt.subplots(figsize=(10, 10))
    limites_epci.plot(ax=ax, alpha=1, edgecolor='black', facecolor='none')
    parcs_jardins.plot(ax=ax, color='green', alpha=0.7, edgecolor='black')
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)

    ax.set_axis_off()
    plt.title("Parcs et jarding urbains dans l'EPCI", fontsize=14)

    # (Optionnel) Export PNG
    if export :
        images_export_dir = os.path.join(images_dir, "visualisation_parcs_jardins_urbains.png")
        plt.savefig(images_export_dir, dpi=300, bbox_inches='tight')
        print(f"Carte exportée vers : {images_export_dir}")

    plt.show()
    plt.close()

# Exécution
afficher_parcs_jardins_urbains(export = False)


# ### 2.17. Lieux de stationnement hors-voirie pour les véhicules
# ---
# Les données de stationnement se divisent en deux catégories : celles qui portent sur les données en voirie, et les autres (parkings).
# La base nationale sur les données hors-voirie est considérée comme obsolète, et n'est plus mise à jour (https://transport.data.gouv.fr/datasets/base-nationale-des-lieux-de-stationnement), les données OSM sont récupérées à la place. A noter que si des données locales existent, il est sans doute préférable de les utiliser.

# In[77]:


def recuperer_stationnement_hors_voirie():
    # 1. Charger l'emprise spatiale
    limites_epci = charger_fichier_parquet("limites_epci", crs=4326)

    # 2. Créer une emprise pour la requête
    emprise = limites_epci.geometry.union_all()

    # 3. Définir les tags OSM pour le stationnement (complet)
    tags = {
        'amenity': ['parking', 'parking_entrance', 'parking_space'],
        'parking': ['surface', 'underground', 'multi-storey', 'lane', 'street_side'],
        'capacity': True,  # Nombre de places si disponible
        'capacity:disabled': True,
        'fee': True,  # Payant/gratuit
        'access': ['public', 'private', 'customers'],
        'parking:lane:both': ['parallel', 'perpendicular', 'diagonal'],
        'maxstay': True  # Durée max de stationnement
    }

    # 4. Interrogation d'OSM via OSMnx
    gdf_stationnement = ox.features_from_polygon(emprise, tags=tags)

    # 5. Filtrage par type de géométrie
    geometries_valides = gdf_stationnement.geometry.type.isin(['Point', 'Polygon', 'MultiPolygon', 'LineString'])
    gdf_stationnement = gdf_stationnement[geometries_valides].copy()

    # 6. Nettoyage, ne conserve que les colonnes utiles
    gdf_stationnement = gdf_stationnement[['amenity', 'parking', 'name', 'capacity', 'capacity:disabled', 'fee', 
                                           'access','parking:lane:both', 'maxstay', 'operator','geometry']].copy()   
    # 7. Standardisation des capacités
    if 'capacity' in gdf_stationnement.columns:
        gdf_stationnement['capacity'] = pd.to_numeric(
            gdf_stationnement['capacity'], 
            errors='coerce'
        )

    # 8. Classification des types
    def classifier_type(row):
        if row.get('amenity') == 'parking':
            if row.get('parking') in ['underground', 'multi-storey']:
                return 'parking_souterrain'
            return 'parking_surface'
        elif row.get('amenity') == 'parking_space':
            return 'place_isolée'
        elif 'parking:lane' in str(row):
            return 'stationnement_rue'
        return 'autre'

    gdf_stationnement['type_stationnement'] = gdf_stationnement.apply(classifier_type, axis=1)

    # 9. Export
    exporter_gpkg(gdf_stationnement, "osm_stationnement_epci")
    exporter_parquet(gdf_stationnement, "osm_stationnement_epci")

    print(f"Total éléments trouvés : {len(gdf_stationnement)}")
    print("--- Répartition par type ---")
    print(gdf_stationnement['type_stationnement'].value_counts())

    if 'capacity' in gdf_stationnement.columns:
        cap_totale = gdf_stationnement['capacity'].sum()
        print(f"\nCapacité totale estimée : {cap_totale:.0f} places")

# Exécution
recuperer_stationnement_hors_voirie()


# ### 2.18. NDVI (Sentinel-2) depuis l'API Copernicus
# ---
# Le NDVI (Indice de végétation par différence normalisée) est un indice allant de 0 à 1, permettant de déterminer si une surface est végétalisée ou non.
# 
# L'API Copernicus permet d'acquérir le NDVI avec différents traitements : 
# * Le L2A est monodate et les effets atmosphériques corrigés
# * Le L3 est une mosaïque, faisant une moyenne des passages sans nuages selon période à définir
# 
# Théoriquement, le L3 est plus précis et plus stable, mais dans mon cas, les effets de mosaïques (plusieurs images collées ensemble) sont trop visibles, ce qui ne donne pas un résultat cohérent. Il y a donc 2 versions du code : 
# 1. Le premier calcule la médiane du NDVI annuel. La moyenne a également été calculée, mais les résultats ne correspondaient pas à la réalité (NDVI max de 0,73). Et une valeur médiane est plus représentative de la végatation au cours de l'année. Comme la France est couverte par plusieurs "tuiles" Sentinel-2, chacune de ses tuiles possède un % de couvert nuageux selon la date d'acquisition. Pour limiter les erreurs, il faut des images avec un couvert nuageux faible. Pour éviter un effet "mosaïque" où l'on verrait des différences dans le NDVI selon les tuiles, la médiane doit être calculée sur les jours où toutes les tuiles sur la zone étudiée sont "valides" (couvert nuageux faible). C'est la 1ère étape. La seconde télécharge les images pour tout ces jours, puis calcule la médiane.
# 2. Le second récupère le NDVI monodate le plus récent avec une couverture nuageuse faible. Cette dernière est conservée, car elle peut être facilement modifiée pour récupérer le NDVI d'une date en particulier, si l'on considère qu'il est mieux de se baser une seule date de référence.
# 
# Pour obtenir l'accès à l'API :    
# 1. Créer un compte sur dataspace.copernicus.eu
# 2. Créer un OAuth clients à l'adresse https://shapps.dataspace.copernicus.eu/dashboard/#/account/settings
# 3. Copier les Client ID et Client secret
# 
# Documentation :
# * https://fr.wikipedia.org/wiki/Indice_de_v%C3%A9g%C3%A9tation_par_diff%C3%A9rence_normalis%C3%A9e
# * https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/UserGuides/BeginnersGuide.html#python
# * https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Overview/Authentication.html
# * https://documentation.dataspace.copernicus.eu/Data/SentinelMissions/Sentinel2.html
# * https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Catalog/Examples.html#simple-get-search
# * https://cnes.fr/projets/sentinel-2

# #### 2.18.1 Version médiane annuelle
# ---

# In[64]:


def verifier_access_api_copernicus():
    # Vérifie l'accès à l'API. Si le code renvoyé est 200 : fonctionne
    CLIENT_ID = "sh-98d7bd24-3853-4a67-bcf4-800d8158e7ce"
    CLIENT_SECRET = "pGkp3CPWkraevjKdNca2P0IWISaUCD1b"

    auth_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"

    response = requests.post(auth_url, data={
        "grant_type": "client_credentials",
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET
    })

    if response.status_code == 200:
        print("Connexion à l'API réussie")
    else:
        print(f"Erreur {response.status_code} : {response.text}")

# Exécution
verifier_access_api_copernicus()


# ##### 2.17.1.1. Récupère les jours de capture commune des différentes tuiles Sentinel-2 de la zone étudiée
# ---

# In[559]:


def recuperer_jours_communs_api_copernicus():
    # 1. Paramètres
    CLIENT_ID = "sh-98d7bd24-3853-4a67-bcf4-800d8158e7ce"
    CLIENT_SECRET = "pGkp3CPWkraevjKdNca2P0IWISaUCD1b"
    DATE_DEBUT = "2024-01-01" # Date de début 
    DATE_FIN = "2024-12-31" # Date de fin
    EXPORT_PATH = os.path.join(exports_dir, "dates_communes_ndvi_2024.txt")

    # 2. Charge l'emprise de la zone étudiée
    limites_epci = charger_fichier_parquet("limites_epci", crs=4326)
    geometry = mapping(limites_epci.geometry.geometry.union_all())

    # 3. Fonction d'authentification
    def get_token():
        url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
        payload = {
            "grant_type": "client_credentials",
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
        }
        resp = requests.post(url, data=payload)
        resp.raise_for_status()
        return resp.json()["access_token"]

    access_token = get_token()
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/geo+json"
    }

    # 4. Interrogation de l'API
    search_url = "https://sh.dataspace.copernicus.eu/api/v1/catalog/1.0.0/search"
    search_payload = {
        "collections": ["sentinel-2-l2a"],
        "datetime": f"{DATE_DEBUT}T00:00:00Z/{DATE_FIN}T23:59:59Z",
        "intersects": geometry,
        "limit": 100, # Sentinel-2 capte la même zone tout les 5 jours, donc la limite de 100 * 5 jours n'est 
    }                 # jamais atteinte sur 1 an. A changer si on calcule le NDVI médian sur plusieurs années

    tuile_to_dates = defaultdict(set)
    nb_total = 0
    pbar = tqdm(desc="Téléchargement des métadonnées Sentinel-2", unit="images")

    while True:
        try:
            resp = requests.post(search_url, headers=headers, json=search_payload)
            if resp.status_code == 401:
                # Rafraîchit le token si expiré
                access_token = get_token()
                headers["Authorization"] = f"Bearer {access_token}"
                continue

            resp.raise_for_status()
            data = resp.json()
            nb_features = len(data["features"])
            pbar.update(nb_features)
            nb_total += nb_features

            for feat in data["features"]:
                props = feat["properties"]
                mgrs_id = (
                    props.get("mgrs:utm_zone", "") +
                    props.get("mgrs:latitude_band", "") +
                    props.get("mgrs:grid_square", "")
                )
                date_iso = parse(props["datetime"]).date().isoformat()
                tuile_to_dates[mgrs_id].add(date_iso)

            # Suivant ?
            next_link = next((link["href"] for link in data.get("links", []) if link.get("rel") == "next"), None)
            if not next_link:
                break

            # Copie le body pour la requête suivante (nécessaire car méthode = POST)
            next_body = next((link.get("body") for link in data["links"] if link.get("rel") == "next"), None)
            if next_body:
                search_payload.update(next_body)
            else:
                break

        except requests.exceptions.RequestException as e:
            print(f"[ERREUR] Requête interrompue : {e}")
            break

    pbar.close()
    print(f"\nTotal d’images analysées : {nb_total}")

    # 5. Calcul des dates communes
    if tuile_to_dates:
        common_dates = sorted(set.intersection(*tuile_to_dates.values()))
        print(f"\n{len(common_dates)} dates communes trouvées (10 premières affichées:")
        for d in common_dates[:10]:
            print("–", d)
    else:
        print("Aucune date commune trouvée.")

    # 6. Export
    with open(EXPORT_PATH, "w") as f:
        for d in common_dates:
            f.write(d + "\n")

    print(f"\n Fichier exporté : {EXPORT_PATH}")

# Exécution
recuperer_jours_communs_api_copernicus()


# ##### 2.17.1.2. Télécharge et calcule le NDVI médian à l'année
# ---

# In[108]:


# Récupère le NDVI mdéian annuel depuis l'API. La taile de l'image demandée est calculée dynamiquement pour éviter le rééchantillonage
def recuperer_ndvi_api_copernicus():
    # 1. Paramètres
    CLIENT_ID = "sh-98d7bd24-3853-4a67-bcf4-800d8158e7ce"
    CLIENT_SECRET = "pGkp3CPWkraevjKdNca2P0IWISaUCD1b"
    RESOLUTION = 20  # mètres/pixel
    DATES_PATH = os.path.join(exports_dir, "dates_communes_ndvi_2024.txt")
    NDVI_OUTPUT_PATH = os.path.join(exports_dir, "ndvi_epci_2024_mediane.tif")

    # 2. Chargement des dates
    with open(DATES_PATH) as f:
        dates = [line.strip() for line in f.readlines()]
    assert dates, "Aucune date trouvée."

    # 3. Chargement des limites et de la géométrie
    limites_epci = charger_fichier_parquet("limites_epci", crs=3857)
    bounds = limites_epci.total_bounds
    width_px = int((bounds[2] - bounds[0]) / RESOLUTION)
    height_px = int((bounds[3] - bounds[1]) / RESOLUTION)
    limites_epci = limites_epci.to_crs(4326)
    geometry = mapping(limites_epci.geometry.union_all())

    # 4. Authentification
    def get_token():
        auth_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
        payload = {
            "grant_type": "client_credentials",
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET
        }
        resp = requests.post(auth_url, data=payload)
        resp.raise_for_status()
        return resp.json()["access_token"]

    access_token = get_token()
    headers = {"Authorization": f"Bearer {access_token}"}

    # 5. Evalscript NDVI
    evalscript = """
    //VERSION=3
    function setup() {
      return {
        input: [{bands: ["B04", "B08"], units: "REFLECTANCE"}],
        output: {bands: 1, sampleType: "FLOAT32"}
      };
    }
    function evaluatePixel(sample) {
      let ndvi = (sample.B08 - sample.B04) / (sample.B08 + sample.B04);
      return [isFinite(ndvi) ? ndvi : 0];
    }
    """

    # 6. Requêtes NDVI
    stack = []
    with TemporaryDirectory() as tmpdir:
        for date in tqdm(dates, desc="Téléchargement NDVI"):
            payload = {
                "input": {
                    "bounds": {
                        "properties": {"crs": "http://www.opengis.net/def/crs/OGC/1.3/CRS84"},
                        "geometry": geometry
                    },
                    "data": [{
                        "type": "sentinel-2-l2a",
                        "dataFilter": {
                            "timeRange": {
                                "from": f"{date}T00:00:00Z",
                                "to": f"{date}T23:59:59Z"
                            },
                            "maxCloudCoverage": 30
                        }
                    }]
                },
                "output": {
                    "width": width_px,
                    "height": height_px,
                    "responses": [{
                        "identifier": "default",
                        "format": {"type": "image/tiff"}
                    }]
                },
                "evalscript": evalscript
            }

            process_url = "https://sh.dataspace.copernicus.eu/api/v1/process"
            resp = requests.post(process_url, headers=headers, json=payload)
            if resp.status_code == 401:
                access_token = get_token()
                headers["Authorization"] = f"Bearer {access_token}"
                resp = requests.post(process_url, headers=headers, json=payload)
            resp.raise_for_status()

            tif_path = os.path.join(tmpdir, f"ndvi_{date}.tif")
            with open(tif_path, "wb") as f:
                f.write(resp.content)
            stack.append(tif_path)

            time.sleep(10)  # Attente de 10s entre les requêtes pour ne pas dépasser le plafond de l'API

        # 7. Calcul de la moyennne
        arrays = []
        for tif in stack:
            with rasterio.open(tif) as src:
                arr = src.read(1).astype(np.float32)
                arr[arr == 0] = np.nan
                arrays.append(arr)
            ref_meta = src.meta.copy()

        # ndvi_mean = np.nanmean(np.stack(arrays), axis=0) # Calcul de la moyenne
        ndvi_median = np.nanmedian(np.stack(arrays), axis=0) # Calcul de la médiane

        # 8. Masquage final : ne conserve que les données dans l'emprise de la zone étudiée
        with rasterio.open(stack[0]) as ref:
            out_img, out_transform = mask(ref, [mapping(limites_epci.geometry.union_all())], crop=True, nodata=np.nan)
            ref_meta.update({
                "height": out_img.shape[1],
                "width": out_img.shape[2],
                "transform": out_transform,
                "nodata": np.nan
            })

        with rasterio.open(NDVI_OUTPUT_PATH, "w", **ref_meta) as dst:
            dst.write(ndvi_median[np.newaxis, :, :])

    print(f"NDVI médian exporté vers : {NDVI_OUTPUT_PATH}")

# Exécution
recuperer_ndvi_api_copernicus()


# In[108]:


def afficher_ndvi_api_copernicus(export = False):
    # 1. Chargement des données
    ndvi_tif_path = os.path.join(exports_dir, "ndvi_epci_2024_mediane.tif")

    with rasterio.open(ndvi_tif_path) as src:
        ndvi_array = src.read(1)  # On lit la première bande

    # 2. Affichage
    plt.figure(figsize=(10, 10))
    im = plt.imshow(ndvi_array, cmap='RdYlGn', vmin=-1, vmax=1)
    plt.colorbar(im, label='NDVI')
    plt.title("NDVI médian annuel (2024) - Sentinel-2")
    plt.axis('off')

        # (Optionnel) Export PNG
    if export :
        images_export_dir = os.path.join(images_dir, "visualisation_ndvi_median.png")
        plt.savefig(images_export_dir, dpi=300, bbox_inches='tight')
        print(f"Carte exportée vers : {images_export_dir}")

    plt.show()
    plt.close()

# Exécution
afficher_ndvi_api_copernicus(export = True)


# #### 2.18.2. Version monodate
# ---

# In[71]:


# Note : idéalement, on récupèrerait les métadonnées suivante: date de prise de vue et % de couvert nuageux,
# mais je n'ai pas réussi à crééer une requête les téléchargeant
def recuperer_ndvi_api_copernicus_monodate():
    ndvi_tif_path = os.path.join(exports_dir, "ndvi_epci.tif")
    ndvi_geotiff_path = os.path.join(exports_dir, "ndvi_epci.geotiff")

    # 1. Lire les limites et reprojeter en EPSG:3857 (mètres)
    limites_epci = charger_fichier_parquet("limites_epci", crs=3857)
    bounds = limites_epci.total_bounds  # [minx, miny, maxx, maxy]
    width_m = bounds[2] - bounds[0]
    height_m = bounds[3] - bounds[1]

    # 2. Déduire taille image à 20 mètres/pixel
    resolution = 20  # mètres/pixel (résolution de B04)
    width_px = int(width_m / resolution)
    height_px = int(height_m / resolution)

    # 3. Reprojeter pour l'API en EPSG:4326
    limites_epci = limites_epci.to_crs(epsg=4326)
    geometry = mapping(limites_epci.geometry.union_all())

    # 4. Authentification
    CLIENT_ID = "sh-98d7bd24-3853-4a67-bcf4-800d8158e7ce"
    CLIENT_SECRET = "pGkp3CPWkraevjKdNca2P0IWISaUCD1b"

    auth_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    auth_payload = {
        "grant_type": "client_credentials",
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
    }
    auth_response = requests.post(auth_url, data=auth_payload)
    auth_response.raise_for_status()
    access_token = auth_response.json()["access_token"]
    headers = {"Authorization": f"Bearer {access_token}"}

    # 5. Script NDVI
    evalscript = """
    //VERSION=3
    function setup() {
      return {
        input: [{bands: ["B04", "B08"], units: "REFLECTANCE"}],
        output: {id: "default", bands: 1, sampleType: SampleType.FLOAT32}
      };
    }
    function evaluatePixel(sample) {
      let ndvi = (sample.B08 - sample.B04) / (sample.B08 + sample.B04);
      return [ndvi];
    }
    """

    # 6. Requête API
    process_url = "https://sh.dataspace.copernicus.eu/api/v1/process"
    request_payload = {
        "input": {
            "bounds": {
                "properties": {"crs": "http://www.opengis.net/def/crs/OGC/1.3/CRS84"},
                "geometry": geometry,
            },
            "data": [{
                "type": "sentinel-2-l2a",
                "dataFilter": {
                    "timeRange": {
                        # "from": "2024-01-01T00:00:00Z", # En fonction des besoins : déterminer la date de départ ici
                        # "to": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"), # Et la date de fin ici
                        # Pour obtenir un NDVI représentif de l'année, il faut choisir une période sans trop de végétation,
                        # ni trop peu. Ici, c'est le mois d'avril 2025 qui a été sélectionné (Avril 2024 renvoie une 
                        # image corrumpue)
                        "from": "2025-04-01T00:00:00Z",
                        "to": "2025-04-30T00:00:00Z"
                    },
                    "mosaickingOrder": "mostRecent",
                    "maxCloudCoverage": 20
                },
                "processing": {"extractMetadata": True}
            }],
        },
        "output": {
            "width": width_px,
            "height": height_px,
            "responses": [{
                "identifier": "default",
                "format": {"type": "image/tiff"}
            }],
        },
        "evalscript": evalscript,
    }

    # 7. Requête de téléchargement
    print(f"Téléchargement NDVI ({width_px}x{height_px}) à 20m/pixel...")
    resp = requests.post(process_url, json=request_payload, headers=headers)
    resp.raise_for_status()

    # 8. Export
    with open(ndvi_tif_path, "wb") as f:
        f.write(resp.content)

    print(f"Fichier '{ndvi_tif_path}' exporté avec succès.")

# Exécution
recuperer_ndvi_api_copernicus_monodate()


# In[112]:


def afficher_ndvi_api_copernicus_monodate(export = False):
    # 1. Chargement des données
    ndvi_tif_path = os.path.join(exports_dir, "ndvi_epci.tif")

    with rasterio.open(ndvi_tif_path) as src:
        ndvi_array = src.read(1)  # On lit la première bande

    # 2. Affichage
    plt.figure(figsize=(10, 10))
    im = plt.imshow(ndvi_array, cmap='RdYlGn', vmin=-1, vmax=1)
    plt.colorbar(im, label='NDVI')
    plt.title("NDVI monodate entre le 1er avril et le 30 avril 2024 \n(date précise inconnue), Sentinel-2")
    plt.axis('off')

    # (Optionnel) Export PNG
    if export :
        images_export_dir = os.path.join(images_dir, "visualisation_ndvi_monodate.png")
        plt.savefig(images_export_dir, dpi=300, bbox_inches='tight')
        print(f"Carte exportée vers : {images_export_dir}")

    plt.show()
    plt.close()

# Exécution
afficher_ndvi_api_copernicus_monodate(export = True)


# ## 3. Téléchargement et traitement des données à l'échelle de l'EMS
# ---
# Dans les cas où il n'existe pas de fichier consolidé ou à jour à l'échelle française. Si ce projet venait à être réutilisé pour d'autres territoires, les sources de ces données devraient être changées, ou bien certains indicateurs ne pourront plus être calculés.

# In[85]:


def recuperer_donnees_citiz():
    # 1. Récupérer l'URL de téléchargement du json contenant les adresses des flux GBFS
    nom_source = "citiz_gbfs"
    url = trouver_source_url(nom_source)

    # 2. Charger l'emprise spatiale
    limites_epci = charger_fichier_parquet("limites_epci", crs=4326) # crs utilisé par citiz

    # 3. Télécharger le fichier
    json_path = os.path.join(dir, f"{nom_source}.json")
    telecharger_fichier(url, json_path)

    # 4. Charger le fichier
    with open(json_path, "r", encoding="utf-8") as f:
        gbfs_root = json.load(f)

    # 5. Création d'un dictionnaire {nom_flux: URL}
    flux = gbfs_root["data"]["feeds"]
    flux_urls = {feed["name"]: feed["url"] for feed in flux}

    # 6. Extraction des coûts
    def extract_pricing(pricing_url):
        pricing_data = requests.get(pricing_url).json()
        plans = pricing_data['data']['plans']
        costs = []

        for plan in plans:
            plan_name = plan['name'][0]['text'] if isinstance(plan['name'], list) else plan.get('name', 'N/A')

            for km_pricing in plan.get('per_km_pricing', []):
                costs.append({
                    'plan_name': plan_name,
                    'type': 'distance',
                    'unit': 'km',
                    'start_km': km_pricing['start'],
                    'end_km': km_pricing['end'],
                    'rate': km_pricing['rate'],
                    'interval': km_pricing.get('interval', 1)
                })

            for min_pricing in plan.get('per_min_pricing', []):
                costs.append({
                    'plan_name': plan_name,
                    'type': 'time',
                    'unit': 'min',
                    'start_min': min_pricing['start'],
                    'end_min': min_pricing['end'],
                    'rate': min_pricing['rate'],
                    'interval': min_pricing.get('interval', 15)
                })

        return pd.DataFrame(costs)

    # 7. Récupération des flux
    pricing_url = flux_urls.get("system_pricing_plans")
    geofencing_url = flux_urls.get("geofencing_zones")
    station_info_url = flux_urls.get("station_information")
    vehicle_status_url = flux_urls.get("vehicle_status")

    costs_df = extract_pricing(pricing_url)
    geofencing_data = requests.get(geofencing_url).json()
    station_data = requests.get(station_info_url).json()
    vehicle_data = requests.get(vehicle_status_url).json()

    # 8. Zones de gardiennage
    features = geofencing_data["data"]["geofencing_zones"]["features"]
    geoms = [shape(feature["geometry"]) for feature in features]
    gdf_geofencing = gpd.GeoDataFrame(geometry=geoms, crs="EPSG:4326")

    # 9. Stations d'autopartage
    stations = station_data['data']['stations']
    df_stations = pd.DataFrame([{
        'station_id': s['station_id'],
        'lat': s['lat'],
        'lon': s['lon'],
        'parking_type': s['parking_type'],
        'capacity': s['capacity'],
        'is_virtual_station': s['is_virtual_station'],
        'name': s['name']
    } for s in stations])

    gdf_stations = gpd.GeoDataFrame(
        df_stations,
        geometry=gpd.points_from_xy(df_stations.lon, df_stations.lat),
        crs="EPSG:4326"
    )

    # 9. Véhicules
    vehicles = vehicle_data['data']['vehicles']
    df_vehicles = pd.DataFrame([
        {
            'vehicle_id': v['vehicle_id'],
            'lat': v['lat'],
            'lon': v['lon'],
            'current_range_meters': v.get('current_range_meters'),
            'is_reserved': v.get('is_reserved', False),
            'is_disabled': v.get('is_disabled', False),
            'station_id': v.get('station_id'),
            'last_reported': v.get('last_reported')
        }
        for v in vehicles
    ])

    gdf_vehicles = gpd.GeoDataFrame(
        df_vehicles,
        geometry=gpd.points_from_xy(df_vehicles.lon, df_vehicles.lat),
        crs="EPSG:4326"
    )

    # 11. Ne conserve que les données dans la limite de l'EPCI
    gdf_stations_epci = decouper_par_epci(gdf_stations, limites_epci, predicate="within")
    gdf_geofencing_epci = decouper_par_epci(gdf_geofencing, limites_epci, predicate="within")
    gdf_vehicles_epci = decouper_par_epci(gdf_vehicles, limites_epci, predicate="within")

    # 12. Correction des données géométriques invalides
    def make_valid(geom):
        if geom.is_valid:
            return geom
        try:
            return geom.buffer(0)
        except:
            return None

    gdf_geofencing_epci["geometry"] = gdf_geofencing_epci["geometry"].apply(make_valid)
    gdf_geofencing_epci = gdf_geofencing_epci[gdf_geofencing_epci["geometry"].notnull()]
    gdf_geofencing_epci = gdf_geofencing_epci[gdf_geofencing_epci.is_valid]

    # 13. Exports
    exporter_gpkg(gdf_stations_epci, "citiz_stations")
    exporter_parquet(gdf_stations_epci, "citiz_stations")
    exporter_gpkg(gdf_geofencing_epci, "citiz_geofencing")
    exporter_parquet(gdf_geofencing_epci, "citiz_geofencing")
    exporter_gpkg(gdf_vehicles_epci, "citiz_vehicles")
    exporter_parquet(gdf_vehicles_epci, "citiz_vehicles")

    print("Fichiers Citiz extraits et sauvegardés avec succès.")
    print(f"{len(gdf_stations_epci)} stations Citiz dans les limites de l’EPCI")
    print(f"{len(gdf_geofencing_epci)} zones de gardiennage Citiz dans les limites de l’EPCI")
    print(f"{len(gdf_vehicles_epci)} véhicules Citiz dans les limites de l’EPCI")

# Exécution
recuperer_donnees_citiz()


# ### 3.1. Données GBFS issues de Citiz pour l'autopartage
# ---
# Citiz est une Société Coopérative d’Intérêt Collectif qui gère les véhicules en autopartage dans l'EMS. 
# 
# Documentation : 
# * https://grand-est.citiz.coop/
# * Données récupérées depuis les flux GBFS de Citiz pour l'EMS : https://data.strasbourg.eu/explore/dataset/autopartage_gbfs/table/
# * https://gbfs.org/fr/documentation/
# * Le terme de geofencing fait références aux zones de gardiennage des voitures Citiz. Tout les véhicules sont équipés de puces GPS, le système semble utilisé vérifier que les utilisateurs rendent bien le véhicule dans une zone autorisée : https://fr.wikipedia.org/wiki/G%C3%A9orep%C3%A9rageg

# In[113]:


def afficher_donnees_citiz(export = False):
    # 1. Chargement des données
    limites_epci = charger_fichier_parquet("limites_epci", crs=3857)
    stations_citiz = charger_fichier_parquet("citiz_stations", crs=3857)
    geofencing_citiz = charger_fichier_parquet("citiz_geofencing", crs=3857)

    # 2. Affichage
    fig, ax = plt.subplots(figsize=(10, 10))
    limites_epci.plot(ax=ax, alpha=1, edgecolor='black', facecolor='none')
    stations_citiz.plot(ax=ax, color='red', markersize=50, alpha=0.7, edgecolor='black')
    geofencing_citiz.plot(ax=ax, alpha=0.2, edgecolor='blue')

    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)

    ax.set_axis_off()
    plt.title("Stations Citiz et zones de dépose-libre", fontsize=14)

    # (Optionnel) Export PNG
    if export :
        images_export_dir = os.path.join(images_dir, "visualisation_stations_citiz_depose_libre.png")
        plt.savefig(images_export_dir, dpi=300, bbox_inches='tight')
        print(f"Carte exportée vers : {images_export_dir}")


    plt.show()
    plt.close()

# Exécution
afficher_donnees_citiz(export = True)


# ### 3.2. Données GBFS issues de Velhop, qui gère les vélos en libre-service dans l'EMS
# ---
# Données récupérées depuis les flux GBFS de Velhop pour l'EMS : https://data.strasbourg.eu/explore/dataset/velhop_gbfs/information/

# In[87]:


def recuperer_donnees_velhop():
    # 1. Récupérer l'URL de téléchargement du json contenant les adresses des flux GBFS
    nom_source = "velhop_gbfs"
    url = trouver_source_url(nom_source)

    # 2. Charger les limites de l'EPCI
    limites_epci = charger_fichier_parquet("limites_epci", crs=4326)

    # 3. Télécharger le fichier
    json_path = os.path.join(dir, f"{nom_source}.json")
    telecharger_fichier(url, json_path)

    # 4. Charger le fichier
    with open(json_path, "r", encoding="utf-8") as f:
        gbfs_root = json.load(f)

    # 5. Création d'un dictionnaire {nom_flux: URL}
    flux = gbfs_root
    flux_urls = {feed["name"]: feed["url"] for feed in flux}

    # 6. Récupération des flux
    pricing_url = flux_urls.get("system_pricing_plans")
    station_info_url = flux_urls.get("station_information")
    station_status_url = flux_urls.get("station_status")
    free_bike_status_url = flux_urls.get("free_bike_status")
    vehicle_types_url = flux_urls.get("vehicle_types")

    # 7. Téléchargement des données
    pricing_data = requests.get(pricing_url).json()
    station_info_data = requests.get(station_info_url).json()
    station_status_data = requests.get(station_status_url).json()
    free_bike_status_data = requests.get(free_bike_status_url).json()
    vehicle_types_data = requests.get(vehicle_types_url).json()

    # 8. Traitement des données des stations
    if station_info_data:
        stations = station_info_data['data']['stations']
        df_stations = pd.DataFrame([{
            'station_id': s['station_id'],
            'lat': s['lat'],
            'lon': s['lon'],
            'capacity': s.get('capacity'),
            'name': s.get('name')
        } for s in stations])

        gdf_stations = gpd.GeoDataFrame(
            df_stations,
            geometry=gpd.points_from_xy(df_stations.lon, df_stations.lat),
            crs="EPSG:4326"
        )

    # 8. Traitement des données des vélos disponibles
    if free_bike_status_data:
        bikes = free_bike_status_data['data']['bikes']
        df_bikes = pd.DataFrame([{
            'bike_id': b['bike_id'],
            'lat': b['lat'],
            'lon': b['lon'],
            'is_reserved': b.get('is_reserved', False),
            'is_disabled': b.get('is_disabled', False),
            'vehicle_type_id': b.get('vehicle_type_id')
        } for b in bikes])

        gdf_bikes = gpd.GeoDataFrame(
            df_bikes,
            geometry=gpd.points_from_xy(df_bikes.lon, df_bikes.lat),
            crs="EPSG:4326"
        )

    # 9. Ne conserver que les données dans la limite de l'EPCI
    gdf_bikes_epci = decouper_par_epci(gdf_bikes, limites_epci, predicate="within")
    gdf_stations_epci = decouper_par_epci(gdf_stations, limites_epci, predicate="within")
    """
    A décommenter si on veut faire un indicateur sur le coût horaire des Velhop

    # 10. Traitement des plans tarifaires
    plans = pricing_data['data']['plans']
    df_pricing = pd.DataFrame(plans)

    # 11. Exports
    exporter_gpkg(df_pricing, "velhop_tarifs")
    exporter_parquet(df_pricing, "velhop_tarifs")
    """
    exporter_gpkg(gdf_stations_epci, "velhop_stations")
    exporter_parquet(gdf_stations_epci, "velhop_stations")
    exporter_gpkg(gdf_bikes_epci, "velhop_bikes")
    exporter_parquet(gdf_bikes_epci, "velhop_bikes")

    print("Fichiers Velhop extraits et sauvegardés avec succès.")
    print(f"{len(gdf_stations_epci)} stations Velhop dans les limites de l’EPCI")
    print(f"{len(gdf_bikes_epci)} vélos Velhop dans les limites de l’EPCI")

# Exécution
recuperer_donnees_velhop()


# In[114]:


def afficher_donnees_velhop(export = False):
# 1. Chargement des données
    limites_epci = charger_fichier_parquet("limites_epci", crs=3857)
    velhop = charger_fichier_parquet("velhop_stations", crs=3857)

    # 2. Affichage
    fig, ax = plt.subplots(figsize=(10, 10))
    limites_epci.plot(ax=ax, alpha=1, edgecolor='black', facecolor='none')
    velhop.plot(ax=ax, color='green', markersize=10, alpha=1, edgecolor='black')

    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)

    ax.set_axis_off()
    plt.title("Stations Velhop", fontsize=14)

    # (Optionnel) Export PNG
    if export :
        images_export_dir = os.path.join(images_dir, "visualisation_stations_velhop.png")
        plt.savefig(images_export_dir, dpi=300, bbox_inches='tight')
        print(f"Carte exportée vers : {images_export_dir}")


    plt.show()
    plt.close()

# Exécution
afficher_donnees_velhop(export = True)


# ### 3.3. Lignes de tram et de bus

# In[90]:


def recuperer_lignes_tram():
    # 1. Trouver l'URL pour 'lignes_tram'
    nom_source = "lignes_tram"
    url = trouver_source_url(nom_source)

    # 2. Télécharger le fichier directement dans le dossier d'exports (pas de traitements à effectuer)
    geojson_path = os.path.join(exports_dir, f"{nom_source}.geojson")
    telecharger_fichier(url, geojson_path)

    # 3. Charger les données
    lignes_tram = gpd.read_file(geojson_path)

    # 4. Exports
    exporter_gpkg(lignes_tram, f"{nom_source}")
    exporter_parquet(lignes_tram, f"{nom_source}")

# Exécution
recuperer_lignes_tram()


# In[91]:


def recuperer_lignes_bus():
    # 1. Trouver l'URL pour 'lignes_bus'
    nom_source = "lignes_bus"
    url = trouver_source_url(nom_source)

    # 2. Télécharger le fichier directement dans le dossier d'exports (pas de traitements à effectuer)
    geojson_path = os.path.join(exports_dir, f"{nom_source}.geojson")
    telecharger_fichier(url, geojson_path)

    # 3. Charger les données et les limites de l'EPCI
    lignes_bus = gpd.read_file(geojson_path)

    # 4. Exports
    exporter_gpkg(lignes_bus, f"{nom_source}")
    exporter_parquet(lignes_bus, f"{nom_source}")

# Exécution
recuperer_lignes_bus()


# In[116]:


def afficher_lignes_tram_bus(export = False):
    # 1. Chargement des données
    limites_epci = charger_fichier_parquet("limites_epci", crs=3857)
    lignes_tram_epci = charger_fichier_parquet("lignes_tram_epci", crs=3857)
    lignes_bus_epci = charger_fichier_parquet("lignes_bus_epci", crs=3857)

    # 2. Affichage
    fig, ax = plt.subplots(figsize=(10, 10))
    limites_epci.plot(ax=ax, alpha=1, edgecolor='black', facecolor='none')
    lignes_tram_epci.plot(ax=ax, alpha=0.5, edgecolor='blue', facecolor='none')
    lignes_bus_epci.plot(ax=ax, alpha=0.5, edgecolor='red', facecolor='none')

    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)

    ax.set_axis_off()
    plt.title("Lignes de tram et de bus", fontsize=14)

    legend_elements = [
        Patch(facecolor='blue', edgecolor='blue', label='Tram', alpha=0.8),
        Patch(facecolor='red', edgecolor='red', label='Bus', alpha=0.7),
    ]
    ax.legend(handles=legend_elements, loc='upper left')

    # (Optionnel) Export PNG
    if export :
        images_export_dir = os.path.join(images_dir, "visualisation_lignes_tram_bus.png")
        plt.savefig(images_export_dir, dpi=300, bbox_inches='tight')
        print(f"Carte exportée vers : {images_export_dir}")

    plt.show()
    plt.close()

# Exécution
afficher_lignes_tram_bus(export = True)


# ### 3.4. Feux de circulation
# 
# ---

# In[93]:


def recuperer_feux_circulation():
    # 1. Trouver l'URL pour 'signalisation_verticale'
    nom_source = "signalisation_verticale"
    url = trouver_source_url(nom_source)

    # 2. Charger les limites de l'EPCI
    limites_epci = charger_fichier_parquet("limites_epci", crs=4326)

    # 3. Télécharger le fichier
    geojson_path = os.path.join(dir, f"{nom_source}.geojson")
    #telecharger_fichier(url, geojson_path)

    # 4. Charger les données et les limites de l'EPCI
    signalisation_verticale = gpd.read_file(geojson_path)

    # 5. Ne conserver que les feux de circulation
    feux_circulation = signalisation_verticale[signalisation_verticale['registre'].str.lower() == "poteau feux de circulation"]

    # 6. Jointure spatiale
    feux_circulation_epci = decouper_par_epci(feux_circulation, limites_epci, predicate="intersects")

    # 7. Exports
    exporter_gpkg(feux_circulation_epci, "feux_circulation_epci")  
    exporter_parquet(feux_circulation_epci, "feux_circulation_epci")

    print(f"{len(feux_circulation_epci)} feux de circulation dans les limites de l’EPCI")

# Exécution
recuperer_feux_circulation()


# In[94]:


def afficher_feux_circulation(export = False):
    # 1. Chargement des données
    limites_epci = charger_fichier_parquet("limites_epci", crs=3857)
    feux_circulation_epci = charger_fichier_parquet("feux_circulation_epci", crs=3857)

    # 2. Affichage
    fig, ax = plt.subplots(figsize=(10, 10))
    feux_circulation_epci.plot(ax=ax, color='green', markersize=10, alpha=1, edgecolor='black')
    limites_epci.plot(ax=ax, alpha=1, edgecolor='black', facecolor='none')

    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)

    ax.set_axis_off()
    plt.title("Localisation des feux de circulation", fontsize=14)

    # (Optionnel) Export PNG
    if export :
        images_export_dir = os.path.join(images_dir, "visualisation_feux_circulation.png")
        plt.savefig(images_export_dir, dpi=300, bbox_inches='tight')
        print(f"Carte exportée vers : {images_export_dir}")

    plt.show()
    plt.close()

# Exécution
afficher_feux_circulation(export = True)


# ### 3.5. Données GTFS concernant les trajets et horaires pour les bus et les tram
# 
# Note : si ce code venaît à être réutilisé, il faudra adapter la ou les sources GTFS selon le territoire étudié grâce à l'adresse suivante : https://transport.data.gouv.fr/datasets?q=GTFS 
# 
# J'utilise ici le GTFS 'Schedule' qui est publié à intervalles réguliers par l'EMS, et non pas la version temps réél qui demande de les contacter par mail puis d'attendre leur réponse. Les données temps réel ne sont pas nécessaires pour générer le graphe des bus ou des trams.
# 
# Documentation : 
# * https://gtfs.org/documentation/overview/
# 
# ---

# In[95]:


def recuperer_flux_gtfs_bus_tram():
    # 1. Définir la source
    nom_source = "gtfs_ems"
    url = trouver_source_url(nom_source)

    # 2. Définir les chemins
    archive_path, extract_path, _ = definir_chemins(nom_source, ext="zip")

    # 3. Téléchargement et extraction du .zip
    telecharger_fichier(url, archive_path)
    extraire_zip(archive_path, extract_path)

    # 4. Conversion de tous les .txt en .csv, qui sont ensuite exportés
    fichiers_txt = [f for f in os.listdir(extract_path) if f.endswith(".txt")]
    csv_paths = {}

    for nom_fichier in fichiers_txt:
        chemin_txt = os.path.join(extract_path, nom_fichier)
        df = pd.read_csv(chemin_txt)

        nom_base = os.path.splitext(nom_fichier)[0]  # ex: stop_times
        chemin_csv = os.path.join(exports_dir, f"gtfs_{nom_base}.csv")
        df.to_csv(chemin_csv, index=False)
        csv_paths[nom_base] = chemin_csv
        print(f"Fichier converti : {chemin_csv}")

        if nom_base == "stop_times":
            df_stop_times = df
        elif nom_base == "stops":
            df_stops = df

    # 5. Chargement des fichiers nécessaires
    df_routes = pd.read_csv(csv_paths["routes"])
    df_trips = pd.read_csv(csv_paths["trips"])
    df_stop_times = pd.read_csv(csv_paths["stop_times"])
    df_stops = pd.read_csv(csv_paths["stops"])

    # 6. Fusion pour récupérer route_type par trip_id
    df_trips_routes = df_trips.merge(df_routes[["route_id", "route_type"]], on="route_id", how="left")
    df_stop_times_trips = df_stop_times.merge(df_trips_routes[["trip_id", "route_type"]], on="trip_id", how="left")

    # 7. Identifier tous les types de transport présents
    types_transport = df_routes["route_type"].unique()
    print("Types de transport trouvés :", types_transport)

    # 8. Pour chaque type, filtrer et sauvegarder les fichiers
    for type_code in types_transport:
        suffix = {
            0: "tram",
            1: "metro",
            2: "train",
            3: "bus",
            4: "ferry",
            5: "cable_car",
            6: "gondola",
            7: "funicular"
        }.get(type_code, f"type_{type_code}")

        # 9. Filtrage des stop_times
        df_stop_times_filtre = df_stop_times_trips[df_stop_times_trips["route_type"] == type_code]
        trips_ids = df_stop_times_filtre["trip_id"].unique()

        # 10. Stops associés
        stop_ids = df_stop_times_filtre["stop_id"].unique()
        df_stops_filtre = df_stops[df_stops["stop_id"].isin(stop_ids)]

        # 11. Export en .csv
        df_stop_times_filtre.to_csv(os.path.join(exports_dir, f"gtfs_stop_times_{suffix}.csv"), index=False)
        print(f"Exporté : gtfs_stop_times_{suffix}.csv")
        df_stops_filtre.to_csv(os.path.join(exports_dir, f"gtfs_stops_{suffix}.csv"), index=False)
        print(f"Exporté : gtfs_stops_{suffix}.csv")

        # Export des trips (lignes de bus et de tram)
        # Documentation : "A trip is a sequence of two or more stops that occur during a specific time period"
        df_trips_filtre = df_trips[df_trips["trip_id"].isin(trips_ids)]
        df_trips_filtre.to_csv(os.path.join(exports_dir, f"gtfs_trips_{suffix}.csv"), index=False)
        print(f"Exporté : gtfs_trips_{suffix}.csv")

        # 12. Export des arrêts
        geometry = [Point(xy) for xy in zip(df_stops_filtre["stop_lon"], df_stops_filtre["stop_lat"])]
        gdf_stops = gpd.GeoDataFrame(df_stops_filtre, geometry=geometry, crs="EPSG:4326")
        exporter_gpkg(gdf_stops, f"gtfs_stops_{suffix}")
        exporter_parquet(gdf_stops, f"gtfs_stops_{suffix}")

# Exécution
recuperer_flux_gtfs_bus_tram()


# ### 3.6. Stations de gonflage et d'outils pour vélos
# 
# ---

# In[96]:


def recuperer_stations_outils_velo():
    # 1. Trouver l'URL pour 'stations_velos'
    nom_source = "stations_velos"
    url = trouver_source_url(nom_source)

    # 2. Télécharger le fichier
    geojson_path = os.path.join(dir, f"{nom_source}.geojson")
    telecharger_fichier(url, geojson_path)

    # 3. Charger les données
    stations_velos = gpd.read_file(geojson_path)

    # 4. Exporter le résultat
    exporter_gpkg(stations_velos, "stations_velos")
    exporter_parquet(stations_velos, "stations_velos")

    print(f"{len(stations_velos)} stations de gonflage et d'outils pour vélos dans les limites de l’EPCI")

# Exécution
recuperer_stations_outils_velo()


# In[121]:


def afficher_stations_outils_velo(export = False):
    # 1. Chargement des données
    limites_epci = charger_fichier_parquet("limites_epci", crs=3857)
    stations_velos = charger_fichier_parquet("stations_velos", crs=3857)

    # 2. Affichage
    fig, ax = plt.subplots(figsize=(10, 10))

    limites_epci.plot(ax=ax, alpha=1, edgecolor='black', facecolor='none')
    stations_velos.plot(ax=ax, color='orange', markersize=10, alpha=1, edgecolor='black')

    # Création d'un dictionnaire de couleurs pour chaque type de station
    couleurs = {'station_gonflage': 'green', 'station_outils': 'orange'}

    # Tracé des stations avec des couleurs différentes selon leur type
    for type_station, color in couleurs.items():
        stations_velos[stations_velos['type'] == type_station].plot(
            ax=ax, color=color,
            markersize=20,
            alpha=0.7,
            edgecolor='black',
            label=type_station
        )

    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)

    ax.legend(title="Type de station")
    ax.set_axis_off()
    plt.title("Stations de gonflages et d'outils pour vélos", fontsize=14)

    # (Optionnel) Export PNG
    if export :
        images_export_dir = os.path.join(images_dir, "visualisation_stations_gonflage_outils_velos.png")
        plt.savefig(images_export_dir, dpi=300, bbox_inches='tight')
        print(f"Carte exportée vers : {images_export_dir}")

    plt.show()
    plt.close()

# Exécution
afficher_stations_outils_velo(export = True)


# ### 3.7. Zones de stationnement payant
# ---
# 
# Documentation :
# * https://data.strasbourg.eu/explore/dataset/stationnement-payant/information/

# In[98]:


def recuperer_zones_stationnement_payant():
    # 1. Trouver l'URL pour 'zones_stationnement_payant'
    nom_source = "zones_stationnement_payant"
    url = trouver_source_url(nom_source)

    # 2. Télécharger le fichier
    geojson_path = os.path.join(dir, f"{nom_source}.geojson")
    telecharger_fichier(url, geojson_path)

    # 3. Charger les données
    zones_stationnement_payant = gpd.read_file(geojson_path)

    # 4. Exporter le résultat
    exporter_gpkg(zones_stationnement_payant, "zones_stationnement_payant")
    exporter_parquet(zones_stationnement_payant, "zones_stationnement_payant")

# Exécution
recuperer_zones_stationnement_payant()


# In[122]:


def afficher_zones_stationnement_payant(export = False):
    # 1. Chargement des données
    limites_epci = charger_fichier_parquet("limites_epci", crs=3857)
    zones_stationnement_payant = charger_fichier_parquet("zones_stationnement_payant", crs=3857)

    # 2. Définir des couleurs selon le tarif
    def determiner_couleur(tarif_str):
        if "1h = 1€" in tarif_str:
            return 'green'  # Tarif vert
        elif "1h = 2.5€" in tarif_str:
            return 'orange'  # Tarif orange
        elif "1h = 3.5€" in tarif_str:
            return 'red'    # Tarif rouge
        else:
            return 'gray'   # Cas par défaut

    zones_stationnement_payant['couleur_plot'] = zones_stationnement_payant['tarif'].apply(determiner_couleur)

    fig, ax = plt.subplots(figsize=(10, 10))

    limites_epci.plot(ax=ax, alpha=1, edgecolor='black', facecolor='none', linewidth=1)
    zones_stationnement_payant.plot(
        ax=ax,
        color=zones_stationnement_payant['couleur_plot'],
        alpha=0.7,
        edgecolor='black',
        linewidth=0.5
    )

    legend_elements = [
        mpatches.Patch(color='green', label='Tarif bas (1€/h)'),
        mpatches.Patch(color='orange', label='Tarif moyen (2.5€/h)'),
        mpatches.Patch(color='red', label='Tarif élevé (3.5€/h)'),
        mpatches.Patch(color='gray', label='Autre tarif')
    ]

    ax.legend(handles=legend_elements, title='Tarifs de stationnement', loc='upper left')

    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)

    ax.set_title("Zones de stationnement payant par tarif horaire", fontsize=14)
    ax.set_axis_off()
    plt.tight_layout()

    # (Optionnel) Export PNG
    if export :
        images_export_dir = os.path.join(images_dir, "visualisation_stations_gonflage_outils_velos.png")
        plt.savefig(images_export_dir, dpi=300, bbox_inches='tight')
        print(f"Carte exportée vers : {images_export_dir}")

    plt.show()
    plt.close()

# Exécution
afficher_zones_stationnement_payant(export = True)


# ### 3.8. Occupation des parkings relais
# ---
# Récupère les données d'entrée-sortie sur les 30 derniers jours sur les parkings relais de l'EMS en open data : https://opendata.cts-strasbourg.eu/parkings/

# In[100]:


def recuperer_occupation_parkings_relais():
    # 1. Trouver l'URL pour 'entrees_sorties_parkings'
    nom_source = "entrees_sorties_parkings"
    url = trouver_source_url(nom_source)

    # 2. Fenêtre : les 30 derniers jours
    n_jours = 30
    dates = [(datetime.now() - timedelta(days=i)).strftime('%Y%m%d') for i in range(1, n_jours + 1)]

    # 3. Stockage : CODE_PARC → [durées] et CODE_PARC → nom
    parking_durees = defaultdict(list)
    parking_noms = {}

    # 4. Boucle sur les dates
    for date_str in dates:
        url_base = f"{url}{date_str}/"
        try:
            response = requests.get(url_base)
            if response.status_code != 200:
                print(f"Dossier {date_str} indisponible ({response.status_code})")
                continue

            soup = BeautifulSoup(response.text, 'html.parser')
            fichiers = [a['href'] for a in soup.find_all('a') if 'SORTIE_vs_ENTREE' in a['href']]

            for fichier in fichiers:
                try:
                    # print(f"Téléchargement : {fichier}")
                    # Extraire le nom depuis le fichier
                    match = re.search(r'ENTREE_(.+)\.csv', fichier)
                    nom_parking = match.group(1) if match else "INCONNU"

                    # Lire les données
                    response = requests.get(url_base + fichier)
                    df = pd.read_csv(io.StringIO(response.content.decode('utf-8')), sep=';')

                    if 'DATE_ENTREE' not in df.columns or 'DATE_SORTIE' not in df.columns or 'CODE_PARC' not in df.columns:
                        print(f"Fichier {fichier} ignoré (colonnes manquantes)")
                        continue

                    df['DATE_ENTREE'] = pd.to_datetime(df['DATE_ENTREE'], format='%d/%m/%Y %H:%M', errors='coerce')
                    df['DATE_SORTIE'] = pd.to_datetime(df['DATE_SORTIE'], format='%d/%m/%Y %H:%M', errors='coerce')
                    df = df.dropna(subset=['DATE_ENTREE', 'DATE_SORTIE'])

                    df['DUREE'] = (df['DATE_SORTIE'] - df['DATE_ENTREE']).dt.total_seconds() / 60

                    # Agrégation par parking
                    for code_parc, group in df.groupby("CODE_PARC"):
                        parking_durees[code_parc].extend(group["DUREE"].tolist())
                        if code_parc not in parking_noms:
                            parking_noms[code_parc] = nom_parking

                except Exception as e:
                    print(f"Erreur lecture fichier {fichier} : {e}")

        except Exception as e:
            print(f"Erreur d'accès à {url_base} : {e}")

    # 5. Calcul des moyennes
    resultats = []
    for code, durees in parking_durees.items():
        if durees:
            duree_moy = sum(durees) / len(durees)
            resultats.append({
                "CODE_PARC": code,
                "nom": parking_noms.get(code, "INCONNU"),
                "duree_moyenne_min": round(duree_moy, 2),
                "duree_moyenne_heure": round(duree_moy / 60, 2)
            })

    # 6. Export CSV
    df_resultats = pd.DataFrame(resultats)
    df_resultats = df_resultats.sort_values("duree_moyenne_min", ascending=False)
    df_resultats.to_csv(os.path.join(exports_dir, "temps_stationnement_parking.csv"), index=False, encoding='utf-8-sig')

    print(f"\n Fichier exporté : temps_stationnement_parking.csv")
    print(f"Parkings analysés : {len(df_resultats)}")

# Exécution
recuperer_occupation_parkings_relais()


# ### 3.9. Places de stationnement en voirie dans l'EMS
# ---
# En l'absence de données nationales (la Base Nationales des Lieux de Stationnement https://www.data.gouv.fr/fr/datasets/base-nationale-des-lieux-de-stationnement/ n'est plus mise à jour), les données locales (https://data.strasbourg.eu/explore/dataset/vo_st_stationmnt_vehi/information/?disjunctive.occupation) sont utilisées. Ces données ne sont pas exhaustives : de nombreuses rues ne possèdent pas de places de stationnement, alors que les voitures peuvent en réalité s'y garer.

# In[103]:


def recuperer_places_stationnement_voirie_ems():
    # 1. Trouver l'URL pour 'stationnement_vl'
    nom_source = "stationnement_vl"
    url = trouver_source_url(nom_source)

    # 2. Télécharger le fichier
    geojson_path = os.path.join(dir, f"{nom_source}.geojson")
    telecharger_fichier(url, geojson_path)

    # 3. Charger les données
    stationnement_vl = gpd.read_file(geojson_path)

    # 4. Exporter le résultat
    exporter_gpkg(stationnement_vl, "stationnement_vl")
    exporter_parquet(stationnement_vl, "stationnement_vl")

# Exécution
recuperer_places_stationnement_voirie_ems()


# In[124]:


def afficher_places_stationnement_voirie_ems(export = False):
    # 1. Chargement des données
    limites_epci = charger_fichier_parquet("limites_epci", crs=3857)
    stationnement_vl = charger_fichier_parquet("stationnement_vl", crs=3857)

    # 2. Prétraitement
    stationnement_vl = stationnement_vl.to_crs(epsg=3857)
    stationnement_vl["nbre_places"] = pd.to_numeric(stationnement_vl["nbre_places"], errors="coerce")

    # 3. Regrouper les données par type
    groupes = (
        stationnement_vl.groupby("occupation")["nbre_places"]
        .sum()
        .sort_values(ascending=False)
    )
    types_stationnement = groupes.index.tolist()

    # 4. Attribution des couleurs dynamiques
    cmap = plt.colormaps.get_cmap("tab10")
    couleurs_dynamiques = {type_: cmap(i) for i, type_ in enumerate(types_stationnement)}

    # 5. Tracé
    fig, ax = plt.subplots(figsize=(10, 10))
    limites_epci.plot(ax=ax, edgecolor='black', facecolor='none', linewidth=1)

    # 6. Affichage de chaque type trié
    for type_station in types_stationnement:
        color = couleurs_dynamiques[type_station]
        subset = stationnement_vl[stationnement_vl["occupation"] == type_station]
        if not subset.empty:
            total_places = int(subset["nbre_places"].sum())
            subset.plot(
                ax=ax,
                color=color,
                linewidth=2,
                label=f"{type_station} ({total_places} places)"
            )

    # 7. Ajout de la carte de fond
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)

    # 8. Mise en forme
    ax.set_title(
        "Places de stationnement pour les véhicules légers\n"
        f"Total places : {int(stationnement_vl['nbre_places'].sum())}",
        fontsize=14
    )
    ax.legend(title="Type de stationnement", loc='upper left', fontsize='small')
    ax.set_axis_off()
    plt.tight_layout()

    # (Optionnel) Export PNG
    if export :
        images_export_dir = os.path.join(images_dir, "visualisation_places_stationnement_vl.png")
        plt.savefig(images_export_dir, dpi=300, bbox_inches='tight')
        print(f"Carte exportée vers : {images_export_dir}")

    plt.show()
    plt.close()

# Exécution
afficher_places_stationnement_voirie_ems(export = True)


# ### 3.10. Bruits routiers dans l'EMS
# ---
# 
# Les données concernant les zones exposées au bruits routiers sur une journée de 24h (en 2022) d'après l'indicateur Lden.
# 
# Le Lden est une norme europééenne de mesure pour exprimer le niveau sonore sur une journée. Il est pondéré : une pénalité de 5 db est appliquée sur les mesures en soirée (19h - 23h), et 10 db la nuit (23h - 7h).
# 
# L'EMS propose également des cartes de bruits dues au trafic aérien, à l'industrie, et aux voies ferrées. Comme les  transports de la SNCF (TER) ni l'aviation ne sont comptés ici de la mobilité locale, ces données ne sont pas récupérées.
# 
# Note : cette donnée ne comporte pas le nombre de db précis par polygones, mais des champs de valeurs : '0-50', '50-55', '55-60', '60-65', '65-70', '70-75', '>=75'.
# 
# Documentation : 
# * https://en.wikipedia.org/wiki/Day%E2%80%93evening%E2%80%93night_noise_level (Lden)

# In[105]:


def recuperer_bruits_routier_ems():
    # 1. Trouver l'URL pour 'ems_bruits_2022'
    nom_source = "ems_bruits_2022"
    url = trouver_source_url(nom_source)

    # 2. Télécharger le fichier
    geojson_path = os.path.join(dir, f"{nom_source}.geojson")
    telecharger_fichier(url, geojson_path)

    # 3. Charger les données et les limites de l'EPCI
    bruits_route_2022 = gpd.read_file(geojson_path)
    limites_epci = charger_fichier_parquet("limites_epci", crs=4326)

    # 4. Filtrage spatial : parkings dans l’EPCI
    bruits_route_2022_ems = gpd.overlay(bruits_route_2022, limites_epci, how="intersection")

    # 5. Exporter le résultat
    exporter_gpkg(bruits_route_2022_ems, "bruits_2022_ems")
    exporter_parquet(bruits_route_2022_ems, "bruits_2022_ems")

# Exécution
recuperer_bruits_routier_ems()


# In[125]:


def afficher_bruits_routier_ems(export = False):
    # 1. Chargement des données
    limites_epci = charger_fichier_parquet("limites_epci", crs=3857)
    bruits_route = charger_fichier_parquet("bruits_2022_ems", crs=3857)

    # 2. Extraction de la valeur numérique maximale pour chaque plage (version corrigée)
    def extraire_valeur_max(plage):
        if plage.startswith('>='):
            return int(plage[2:])  # Pour '>=75' retourne 75
        elif '-' in plage:
            parts = plage.split('-')
            # Pour les plages 'A-B', retourne B-1 car B est exclu
            return int(parts[1]) - 1  
        else:
            return int(plage)

    bruits_route['db_max'] = bruits_route['valeur'].apply(extraire_valeur_max)

    # 3. Définition de l'échelle de couleurs (version corrigée)
    def determiner_couleur(db_max):
        if db_max < 50:  # 0-49
            return '#2b83ba'  # Bleu clair
        elif 50 <= db_max < 55:  # 50-54
            return '#abdda4'  # Vert clair
        elif 55 <= db_max < 60:  # 55-59
            return '#66c2a5'  # Vert
        elif 60 <= db_max < 65:  # 60-64
            return '#ffffbf'  # Jaune
        elif 65 <= db_max < 70:  # 65-69
            return '#fee08b'  # Jaune-orange
        elif 70 <= db_max < 75:  # 70-74
            return '#fdae61'  # Orange
        else:  # 75 et plus
            return '#d7191c'  # Rouge

    bruits_route['couleur'] = bruits_route['db_max'].apply(determiner_couleur)

    # 4. Affichage
    fig, ax = plt.subplots(figsize=(12, 10))
    limites_epci.plot(ax=ax, edgecolor='black', facecolor='none', linewidth=1.5)
    bruits_route.plot(ax=ax, color=bruits_route['couleur'], alpha=0.8, edgecolor='none', linewidth=0.1)

    # Légende mise à jour avec les bonnes plages
    legend_elements = [
        mpatches.Patch(color='#2b83ba', label='0-49 dB (Très faible)'),
        mpatches.Patch(color='#abdda4', label='50-54 dB (Faible)'),
        mpatches.Patch(color='#66c2a5', label='55-59 dB (Moyen-faible)'),
        mpatches.Patch(color='#ffffbf', label='60-64 dB (Modéré)'),
        mpatches.Patch(color='#fee08b', label='65-69 dB (Élevé)'),
        mpatches.Patch(color='#fdae61', label='70-74 dB (Très élevé)'),
        mpatches.Patch(color='#d7191c', label='≥75 dB (Le plus élevé)')
    ]

    ax.legend(
        handles=legend_elements,
        loc='upper left',
        framealpha=1
    )

    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
    ax.set_title("Niveaux de bruit routier moyen journalier (2022)", fontsize=14)
    ax.set_axis_off()
    plt.tight_layout()

    # (Optionnel) Export PNG
    if export :
        images_export_dir = os.path.join(images_dir, "visualisation_bruits_routier.png")
        plt.savefig(images_export_dir, dpi=300, bbox_inches='tight')
        print(f"Carte exportée vers : {images_export_dir}")

    plt.show()
    plt.close()

# Exécution
afficher_bruits_routier_ems(export = True)


# ## 4. Création des graphes pour nos calculs d'accessibilité
# ---
# A l'aide de la librairie networkx, on crée un graphe pour chaque moyen de transport étudié. Un graphe est composé de noeuds reliés par des arêtes. Dans notre cas, chaque noeud représente une des points d'entrées ou de sorties pour le moyen de transport en question. A partir de ces informations, il devient possible de faire des calculs d'accessibilités, par exemple en partant d'un point et de retourner tout les noeuds accessibles en 15 minutes autour du point, ainsi que les arêtes potentiellement empruntables.
# 
# ![schema_types_graphes](images/schema_types_graphes.png)
# 
# Documentation : 
# * https://networkx.org/documentation/stable/reference/classes/digraph.html
# * https://networkx.org/documentation/stable/reference/classes/multidigraph.html
# * https://networkx.org/documentation/stable/reference/algorithms/shortest_paths.html
# 
# Notes : 
# * Après de nombreux essais pour optimiser la recherche d'itinéraires, cette version reste la plus rapide pour l'instant (entre 2 et 4 trajets / s). Parmi les optimisations testées mais non-intégrées :
#     * Utiliser différents librairies, comme igraph ou graph-tool. Bien que la librairie graph-tool semble plus performante, elle est moins portable et ne peut être installé via la commande pip install, ce qui pose problème pour les pc sous windows. Elle n'a pas été utilisée pour cette raison. Quand à igraph, sa distribution python ne dispose pas de l'algorithme A* utilisé ici par networkx car plus rapide.
#     * Utiliser d'autres algorithmes de calculs d'itinéraires, comme A* avec différentes heuristiques sur les networkx et igraph. Elles sont plus lentes que les fonctions par défaut de Networkx (!)
#     * Utiliser les autres backends de joblib (loky, multithread) pour réaliser les calculs en parallèle : sont beaucoup plus lents.

# ### 4.1. Fonctions pour créer et vérifier les graphes

# L'image suivante permet de mieux expliquer les fonctions de tests utilisées pour vérifier la validité des graphes créés. On récupère tout les noeuds valides présents sur la surface du carreau, qui sont enregistrés dans une liste du plus proche au point central au plus éloigné. Cela est utile pour nos fonctions de test, qui calculent le temps et la distance nécessaire pour aller d'un carreau à un autre.
# 
# Il existe plusieurs moyens de calculer un itinéraire entre 2 mailles. Les points de départs et d'arrivée peuvent être : 
# 1. Les points les plus proches du centre des mailles
# 2. La paire de points les plus proches géographiquement l'un de l'autre
# 3. La paire de points est sélectionnée aléatoirement
# 4. On calcule tout les itinéraires possibles avec toutes les combinaisons de points pour ne conserver que l'itinéraire le plus rapide (méthode très lente si elle est appliquée à l'ensemble du graphe, revient à calculer tout les itinéraires possibles entre tout les points du graphe).*
# 
# Afin que les calculs d'itinéraires soient les plus comparables et équitables, la fonction 'calculer_itineraire' utilise la 1ère méthode. Elle tente d'abord de calculer un itinéraire valide entre les points les plus proches du centre des mailles d'arrivée et de destination. Si aucun chemin ne relie ces deux points, on réessaye ensuite avec des points de plus en plus éloignés jusquà tomber sur un itinéraire valide (si il existe). Actuellement, cette fonction n'est utilisée que pour tester les graphes et vérifier la cohérence des données.
# 
# ![schema_graphe_calculs_itineraires](images/schema_graphe_calculs_itineraires.png)
# 
# Par exemple, sur cette image, il est possible de partir de plusieurs points du carreau central. Dans le cas d'un trajet en véhicule légers, il y en a 4. On calcule l'itinéraire en partant des plus proches du centre. Dans cet exemple, un itinéraire sera invalide car l'un des points de départ est une "impasse", et ne permet pas de rejoindre d'autres noeuds du graphe.

# In[26]:


# Ajoute une colonne 'noeuds' aux carreaux contenant tous les ID des nœuds du graphe
# présents à l’intérieur, triés par leur disntance au centre
def ajouter_noeuds_aux_carreaux(nom_graphe="vl"):
    # 1. Chargement des données
    carreaux = charger_fichier_parquet("maille_200m_epci", crs=2154).copy()
    graphe, id_to_coord = charger_graphe(nom_graphe)
    noeuds = "noeuds_" + nom_graphe

    # Filtrer les nœuds existants dans le graphe et présents dans id_to_coord
    node_ids = [node for node in graphe.nodes if node in id_to_coord]
    coords = [id_to_coord[node] for node in node_ids]

    # Création d’un GeoDataFrame des nœuds
    nodes_gdf = gpd.GeoDataFrame(
        {"node_id": node_ids},
        geometry=[Point(coord) for coord in coords],
        crs="EPSG:2154"
    )

    # Jointure spatiale : associer chaque nœud à un carreau
    nodes_with_tiles = gpd.sjoin(nodes_gdf, carreaux, how="inner", predicate="within")
    nodes_with_tiles = nodes_with_tiles.rename(columns={"index_right": "tile_index"})

    # Index inverse : regrouper les nœuds par carreau
    tile_to_nodes = (
        nodes_with_tiles.groupby("tile_index")[["node_id", "geometry"]]
        .apply(lambda df: df.reset_index(drop=True))
    )

    # Préparation des listes de nœuds triées par distance au centre
    noeuds_par_carreau = []

    for i, row in carreaux.iterrows():
        if i not in tile_to_nodes.index:
            noeuds_par_carreau.append([])
            continue

        centre = row.geometry.centroid
        df_nodes = tile_to_nodes.loc[i]

        if isinstance(df_nodes, pd.Series):
            noeuds_par_carreau.append([df_nodes["node_id"]])
        else:
            df_nodes = df_nodes.copy()
            df_nodes["dist_centre"] = df_nodes.geometry.distance(centre)
            df_sorted = df_nodes.sort_values("dist_centre")
            noeuds_par_carreau.append(df_sorted["node_id"].tolist())

    # Ajout de la colonne au GeoDataFrame
    carreaux[noeuds] = noeuds_par_carreau

    # Résumé et export
    nb_non_vides = sum(1 for l in noeuds_par_carreau if l)
    print(f"{nb_non_vides} carreaux ont au moins un nœud associé dans '{noeuds}'")

    exporter_parquet(carreaux, "maille_200m_epci")
    exporter_gpkg(carreaux, "maille_200m_epci")

    return carreaux


# In[27]:


# Calcule le chemin le plus court entre deux mailles sur un graphe. Tente le calculer le chemin existe entre les 
# noeuds les plus au centre de chaque maille, puis le retourne si il existe. Si ce n'est pas le cas, réessaye avec
# d'autres noeuds de plus en plus éloignés dans la maille. Si aucun chemin n'existe , retourne None.
def calculer_itineraire(noeuds_dep, noeuds_arr, graphe, id_to_coord, max_combi=None):
    try:
        max_len = max(len(noeuds_dep), len(noeuds_arr))
        max_test = max_len if max_combi is None else min(max_combi, max_len)

        is_multi = isinstance(graphe, nx.MultiDiGraph)

        for i in range(max_test):
            for j in range(max_test):
                if i >= len(noeuds_dep) or j >= len(noeuds_arr):
                    continue
                u, v = noeuds_dep[i], noeuds_arr[j]
                if u == v:
                    continue

                try:
                    path = nx.shortest_path(graphe, source=u, target=v, weight="weight")
                    if len(path) < 2:
                        continue

                    total_time = 0
                    total_dist = 0
                    lignes = []
                    geom_segments = []

                    for k in range(len(path) - 1):
                        n1, n2 = path[k], path[k + 1]
                        if is_multi:
                            edges = graphe.get_edge_data(n1, n2)
                            if not edges:
                                raise nx.NetworkXNoPath(f"Pas d'arête entre {n1} et {n2}")
                            best_key = min(edges, key=lambda key: edges[key]["weight"])
                            edge_data = edges[best_key]
                        else:
                            edge_data = graphe.edges[n1, n2]

                        total_time += edge_data["weight"]
                        total_dist += edge_data["distance"]
                        lignes.append(edge_data.get("route_id", "inconnu"))

                        # Récupérer la géométrie de l'arête
                        segment = edge_data.get("geometry")
                        if segment:
                            geom_segments.append(segment)
                        else:
                            geom_segments.append(LineString([id_to_coord[n1], id_to_coord[n2]]))

                    full_geom = linemerge(geom_segments)

                    # Traitement spécifique tram si route_id présent
                    has_tram_lines = any(
                        "route_id" in graphe.edges[n1, n2]
                        if not is_multi else
                        any("route_id" in d for d in graphe.get_edge_data(n1, n2).values())
                        for n1, n2 in zip(path[:-1], path[1:])
                    )

                    if has_tram_lines and lignes:
                        lignes_simplifiees = [l.split("-")[0] if l else "?" for l in lignes]
                        changements = [
                            (i + 1, lignes_simplifiees[i], lignes_simplifiees[i + 1])
                            for i in range(len(lignes_simplifiees) - 1)
                            if lignes_simplifiees[i] != lignes_simplifiees[i + 1]
                        ]

                        ligne_depart = lignes_simplifiees[0]
                        total_time += graphe.nodes[path[0]].get(f"attente_{ligne_depart}", 0)

                        for idx, _, ligne_suivante in changements:
                            total_time += graphe.nodes[path[idx]].get(f"attente_{ligne_suivante}", 0)

                        return {
                            "start": id_to_coord[u],
                            "end": id_to_coord[v],
                            "distance": total_dist,
                            "time": total_time,
                            "geometry": full_geom,
                            "lignes": lignes_simplifiees,
                            "changements": changements
                        }

                    else:
                        return {
                            "start": id_to_coord[u],
                            "end": id_to_coord[v],
                            "distance": total_dist,
                            "time": total_time,
                            "geometry": full_geom
                        }

                except (nx.NetworkXNoPath, nx.NodeNotFound, KeyError):
                    continue

        return None

    except Exception as e:
        return {"error": str(e)}


# In[28]:


# Affiche les mailles selon leur intégration à différents réseaux
def afficher_accessibilite_reseau(nom_graphe="vl", nom_legende="routier", export = True):
    # 1. Chargement des données
    carreaux = charger_fichier_parquet("maille_200m_epci")
    graphe, id_to_coord = charger_graphe(nom_graphe)

    # 2. Préparation des données pour l'affichage
    noeuds = "noeuds_" + nom_graphe
    texte_legende = f"Accessibilité au réseau {nom_legende}"

    # 3. Création de deux sous-ensembles : accès / pas d’accès
    carreaux_oui = carreaux[carreaux[noeuds].apply(lambda x: isinstance(x, list) and len(x) > 0)].copy()
    carreaux_non = carreaux[carreaux[noeuds].apply(lambda x: not isinstance(x, list) or len(x) == 0)].copy()

    # 4. Affichage
    fig, ax = plt.subplots(figsize=(10, 10))

    carreaux_oui.plot(ax=ax, color="green", alpha=0.5)
    carreaux_non.plot(ax=ax, color="black", alpha=0.5)

    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)

    legend_elements = [
        mpatches.Patch(color='green', alpha=0.5, label="Accès au réseau"),
        mpatches.Patch(color='black', alpha=0.5, label="Pas d'accès")
    ]
    ax.legend(handles=legend_elements, title=texte_legende, loc='upper right')

    ax.set_title(texte_legende, fontsize=14)
    ax.set_axis_off()

    plt.tight_layout()

    # (Optionnel) Export PNG
    if export :
        images_export_dir = os.path.join(images_dir, f"visualisation_accessibilite_reseau_{nom_graphe}.png")
        plt.savefig(images_export_dir, dpi=300, bbox_inches='tight')
        print(f"Carte exportée vers : {images_export_dir}")

    plt.show()
    plt.close()


# In[29]:


# Fonction de test : sélectionne aléatoirement deux carreaux différents avec au moins un noeud chacun, retourne l'itinéraire (optionnel)
def calcul_itineraire_test(nom_graphe="vl", export=False):
    # 1. Charger les données
    carreaux = charger_fichier_parquet("maille_200m_epci", crs=2154)
    graphe, id_to_coord = charger_graphe(nom_graphe)
    noeuds = "noeuds_" + nom_graphe

    # 2. Sélection de 2 carreaux valides
    carreaux_valides = carreaux[carreaux[noeuds].apply(lambda x: isinstance(x, list) and len(x) > 0)].copy()
    carreau_dep, carreau_arr = carreaux_valides.sample(2).to_dict("records")

    # 3. Noeuds triés par distance au centre
    noeuds_dep = carreau_dep[noeuds]
    noeuds_arr = carreau_arr[noeuds]

    # 4. Calcul de l’itinéraire
    itineraire = calculer_itineraire(noeuds_dep, noeuds_arr, graphe, id_to_coord)

    if not itineraire or "error" in itineraire:
        print("Aucun itinéraire trouvé ou erreur rencontrée.")
        return None

    # 5. Formatage du temps
    total_s = int(round(itineraire["time"]))
    h, m, s = total_s // 3600, (total_s % 3600) // 60, total_s % 60

    if h > 0:
        temps_str = f"{h} h {m} min {s} s"
    elif m > 0:
        temps_str = f"{m} min {s} s"
    else:
        temps_str = f"{s} s"

    print(f"Itinéraire calculé. Durée : {total_s} s ({temps_str}), Distance : {itineraire['distance']:.1f} m")

    # 6. Export optionnel
    if export:
        gdf_path = gpd.GeoDataFrame(geometry=[itineraire["geometry"]], crs="EPSG:2154")
        exporter_gpkg(gdf_path, "test_itineraire")
        exporter_parquet(gdf_path, "test_itineraire")

        gdf_tiles = gpd.GeoDataFrame(
            [carreau_dep, carreau_arr],
            geometry=[carreau_dep["geometry"], carreau_arr["geometry"]],
            crs="EPSG:2154"
        )
        exporter_gpkg(gdf_tiles, "test_carreaux_dep_arr")
        exporter_parquet(gdf_tiles, "test_carreaux_dep_arr")

    return itineraire


# In[30]:


# Fonction de test : affiche un itinéraire calculé
def afficher_itineraire_test(itineraire, est_tram = False, est_bus = False, export = False):
    # 1. Chargement des données
    limites_epci = charger_fichier_parquet("limites_epci", crs=3857)
    limites_epci_tampon = charger_fichier_parquet("limites_epci_tampon", crs=3857)
    bd_topo_routes_tampon = charger_fichier_parquet("bd_topo_routes_tampon", crs=3857)

    # 2. Récupère et affiche les lignes de tram ou bus si besoin
    if est_tram : lignes_tram_epci = charger_fichier_parquet("lignes_tram_epci", crs=3857)
    if est_bus : lignes_bus = charger_fichier_parquet("lignes_bus", crs=3857)

    # 2. Géométrie du chemin : on reconstruit les coordonnées à partir des IDs
    line_geom = itineraire['geometry']
    gdf_path = gpd.GeoDataFrame(geometry=[line_geom], crs="EPSG:2154")

    # 3. Départ et arrivée
    start_point = gpd.GeoDataFrame(geometry=[Point(itineraire['start'])], crs="EPSG:2154")
    end_point = gpd.GeoDataFrame(geometry=[Point(itineraire['end'])], crs="EPSG:2154")

    # 4. Affichage
    fig, ax = plt.subplots(figsize=(10, 10))
    gdf_path.to_crs(3857).plot(ax=ax, color='red', linewidth=2, label="Chemin")
    start_point.to_crs(3857).plot(ax=ax, color='green', markersize=50, label="Départ")
    end_point.to_crs(3857).plot(ax=ax, color='blue', markersize=50, label="Arrivée")

    if est_tram :
        lignes_tram_epci.plot(ax=ax, color="black", linewidth=1, alpha=0.3, label="Lignes de tram")
    elif est_bus :
        lignes_bus.plot(ax=ax, color="black", linestyle="--", linewidth=0.5, alpha=0.6, label="Lignes de bus")
    else: 
        limites_epci.plot(ax=ax, facecolor='none', edgecolor='black', alpha=0.5)
        limites_epci_tampon.plot(ax=ax, facecolor='none', edgecolor='orange', alpha=0.5)
        bd_topo_routes_tampon.plot(ax=ax, linewidth=0.3, edgecolor='green', alpha=0.5)

    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
    ax.set_title("Itinéraire le plus rapide entre deux points", fontsize=14)
    ax.axis('off')
    ax.legend()
    plt.tight_layout()

    # (Optionnel) Export PNG
    if export :
        images_export_dir = os.path.join(images_dir, "visualisation_itineraire.png") # A renommer manuellement après export
        plt.savefig(images_export_dir, dpi=300, bbox_inches='tight')
        print(f"Carte exportée vers : {images_export_dir}")

    plt.show()
    plt.close()

    return None


# In[31]:


# Fonction de test : mesure le temps d'exécution pour X itinéraires calculés
def temps_execution_itineraire(nom_graphe = "vl", nb_calculs = 100):
    # 1. Chargement des données
    carreaux = charger_fichier_parquet("maille_200m_epci")
    graphe, id_to_coord = charger_graphe(nom_graphe)
    noeuds = "noeuds_" + nom_graphe

    # Sélectionne X paires aléatoires de carreaux avec au moins un nœud valide
    carreaux_valides = carreaux[carreaux[noeuds].apply(lambda x: isinstance(x, list) and len(x) > 0)].copy()
    paires_carreaux = carreaux_valides.sample(nb_calculs*2, random_state=42)

    # Extraction directe des listes de nœuds
    carreaux_depart = paires_carreaux.iloc[:nb_calculs][noeuds].tolist()
    carreaux_arrivee = paires_carreaux.iloc[nb_calculs:][noeuds].tolist()
    paires = list(zip(carreaux_depart, carreaux_arrivee))

    # Chronométrage
    t0 = time.time()
    resultats = Parallel(n_jobs=-1, backend="threading", verbose=10)(
        delayed(calculer_itineraire)(noeuds_dep, noeuds_arr, graphe, id_to_coord, max_combi=None)
        for noeuds_dep, noeuds_arr in paires
    )
    t1 = time.time()

    # Filtrage des trajets valides
    trajets_valides = [r for r in resultats if r and "error" not in r]

    # Statistiques
    print(f"\n{len(trajets_valides)}/{len(paires)} itinéraires valides")
    print(f"Temps total : {t1 - t0:.2f} secondes")

    return None


# In[32]:


# Calcule un maximum d'itinéraires entre les mailles reliées au réseau, retourne le nombre d'utilisation de chaque tronçon de route
# Le but est de visualiser quelles routes ne sont jamais optimales, et déterminer la taille idéale de la zone tampon.
def utilisation_troncons(nom_graphe = "vl", max_combi=None, nb_carreaux=200):
    # 1. Chargement des données
    carreaux = charger_fichier_parquet("maille_200m_epci")
    graphe, id_to_coord = charger_graphe(nom_graphe)
    coord_to_id = {tuple(coord): nid for nid, coord in id_to_coord.items()}
    noeuds = "noeuds_" + nom_graphe

    # 2. Filtrage des carreaux
    carreaux_valides = carreaux[carreaux[noeuds].apply(lambda x: isinstance(x, list) and len(x) > 0)].copy()
    print(f"{len(carreaux_valides)} carreaux valides détectés")

    if max_combi is not None and max_combi < len(carreaux_valides):
        carreaux_valides = carreaux_valides.sample(n=max_combi)

    if len(carreaux_valides) > nb_carreaux:
        carreaux_valides = carreaux_valides.sample(n=nb_carreaux)
        print(f"{nb_carreaux} carreaux sélectionnés pour l'analyse")

    compteur_aretes = Counter()
    total = len(carreaux_valides)
    total_itineraires = total * (total - 1)
    liste_noeuds = carreaux_valides[noeuds].tolist()

    # 3. Calcul des itinéraires
    with tqdm(total=total_itineraires, desc="Calcul des itinéraires", unit="itin") as pbar:
        for i in range(total):
            noeuds_dep = liste_noeuds[i]
            for j in range(total):
                if i == j:
                    pbar.update(1)
                    continue

                result = calculer_itineraire(noeuds_dep, liste_noeuds[j], graphe, id_to_coord, max_combi)

                if result and "geometry" in result:
                    geom = result["geometry"]
                    try:
                        coords = []
                        if isinstance(geom, MultiLineString):
                            coords = [pt for line in geom.geoms for pt in line.coords]
                        elif isinstance(geom, LineString):
                            coords = list(geom.coords)
                        path_nodes = [coord_to_id[tuple(c)] for c in coords]
                        for u, v in zip(path_nodes[:-1], path_nodes[1:]):
                            compteur_aretes[tuple(sorted((u, v)))] += 1
                    except KeyError:
                        pass

                pbar.update(1)

    # 4. Création du geodataframe
    geometries, utilisations, u_list, v_list = [], [], [], []
    is_multi = isinstance(graphe, nx.MultiDiGraph)

    for edge in (graphe.edges(keys=True) if is_multi else graphe.edges()):
        u, v = edge[:2]
        key = tuple(sorted((u, v)))
        count = compteur_aretes.get(key, 0)

        try:
            geometries.append(LineString([id_to_coord[u], id_to_coord[v]]))
            utilisations.append(count)
            u_list.append(u)
            v_list.append(v)
        except KeyError:
            continue

    gdf_troncons = gpd.GeoDataFrame({
        "u": u_list,
        "v": v_list,
        "nb_utilisations": utilisations,
        "geometry": geometries
    }, crs="EPSG:2154")

    # 5. Export
    exporter_gpkg(gdf_troncons, "utilisation_troncons_routes")
    exporter_parquet(gdf_troncons, "utilisation_troncons_routes")

    print(f"Analyse terminée. {len(gdf_troncons)} tronçons analysés.")

    return gdf_troncons


# In[33]:


def exporter_noeuds_aretes_graphe(nom_graphe=""):
    # 1. Chargement du graphe
    graphe, id_to_coord = charger_graphe(nom_graphe)

    # 2. Création des GeoDataFrames
    # 2.1. Noeuds
    gdf_nodes = gpd.GeoDataFrame(
        {"node_id": list(id_to_coord.keys())},
        geometry=[Point(xy) for xy in id_to_coord.values()],
        crs=2154
    )

    # 2.2. Arêtes
    edges = []
    edge_attrs = []
    edge_geoms = []

    for u, v, data in graphe.edges(data=True):
        coord_u = id_to_coord[u]
        coord_v = id_to_coord[v]
        geom = data.get("geometry", LineString([coord_u, coord_v]))
        edges.append((u, v))
        edge_attrs.append(data)
        edge_geoms.append(geom)

    df_edges = pd.DataFrame(edge_attrs)
    df_edges["from_id"] = [u for u, v in edges]
    df_edges["to_id"] = [v for u, v in edges]
    gdf_edges = gpd.GeoDataFrame(df_edges, geometry=edge_geoms, crs=2154)

    # 3. Export
    gpkg_path = os.path.join(exports_dir, f"graphe_routier_{nom_graphe}_noeuds_aretes.gpkg")
    gdf_nodes.to_file(gpkg_path, layer="nodes", driver="GPKG")
    gdf_edges.to_file(gpkg_path, layer="edges", driver="GPKG")

    print(f"Export GPKG des noeuds et arêtes du graphe {nom_graphe} terminé : {gpkg_path}")
    print(f"{len(gdf_nodes)} nœuds | {len(gdf_edges)} arêtes exportés")


# In[34]:


# Visualisation du graphe, 1 couleur = 1 ligne
# ligne_a_afficher : "A", "B", etc. ou None pour tout afficher
def afficher_reseau_transport_commun(nom_graphe = "tram", ligne_a_afficher = None, export = False) :    
    # 1. Charger les données
    graphe, id_to_coord = charger_graphe(nom_graphe)
    lignes = charger_fichier_parquet(f"lignes_{nom_graphe}", crs=2154)

    # 2. Convertit les arêtes du graphe en GeoDataFrame
    def edges_to_gdf(graphe):
        data = []
        is_multi = isinstance(graphe, nx.MultiDiGraph)

        if is_multi:
            for u, v, key, d in graphe.edges(keys=True, data=True):
                if "geometry" in d:
                    # Modification ici: on ne garde que la partie avant le tiret
                    route_id = d.get("route_id", "inconnu")
                    if route_id != "inconnu":
                        route_id = route_id.split('-')[0]
                    data.append({
                        "u": u,
                        "v": v,
                        "key": key,
                        "route_id": route_id,
                        "geometry": d["geometry"]
                    })
        else:
            for u, v, d in graphe.edges(data=True):
                if "geometry" in d:
                    route_id = d.get("route_id", "inconnu")
                    if route_id != "inconnu":
                        route_id = route_id.split('-')[0]
                    data.append({
                        "u": u,
                        "v": v,
                        "key": None,
                        "route_id": route_id,
                        "geometry": d["geometry"]
                    })

        return gpd.GeoDataFrame(data, crs="EPSG:2154")

    gdf_edges = edges_to_gdf(graphe)

    # 3. Filtrage par ligne si demandé
    if ligne_a_afficher is not None:
        # Modification ici: comparaison avec la partie avant le tiret
        gdf_edges = gdf_edges[gdf_edges["route_id"] == ligne_a_afficher.split('-')[0]]

    # 4. Extraction des nœuds (arrêts)
    nodes = [
        Point(data["lon"], data["lat"])
        for _, data in graphe.nodes(data=True)
    ]
    noeuds = gpd.GeoDataFrame(geometry=nodes, crs="EPSG:2154")

    # 5. Génération d'une couleur par ligne
    lignes_uniques = gdf_edges["route_id"].unique()
    cmap = plt.get_cmap("tab20", len(lignes_uniques))
    colors = {rid: mcolors.rgb2hex(cmap(i)) for i, rid in enumerate(lignes_uniques)}

    fig, ax = plt.subplots(figsize=(10, 10))
    for rid, group in gdf_edges.groupby("route_id"):
        group.to_crs(3857).plot(ax=ax, linewidth=3, alpha=0.6, color=colors[rid], label=rid)
    lignes.to_crs(3857).plot(ax=ax, color="black", linestyle="--", linewidth=1, alpha=0.6, label="Lignes")
    noeuds.to_crs(3857).plot(ax=ax, color="black", markersize=15, alpha=0.5)

    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)

    titre = f"Ligne de {nom_graphe} : {ligne_a_afficher.split('-')[0]}" if ligne_a_afficher else f"Lignes de {nom_graphe} (GTFS)"
    ax.set_title(titre, fontsize=15)
    ax.axis("off")

    if ligne_a_afficher is None and len(lignes_uniques) > 1:
        ax.legend(title="Lignes", loc="upper right")

    plt.tight_layout()
    # (Optionnel) Export PNG
    if export :
        images_export_dir = os.path.join(images_dir, f"visualisation_{nom_graphe}.png")
        plt.savefig(images_export_dir, dpi=300, bbox_inches='tight')
        print(f"Carte exportée vers : {images_export_dir}")

    plt.show()
    plt.close()


# ### 4.2. Création du graphe pour les véhicules légers (VL, inclus les motos ou les voitures)
# ---
# Les tronçons de route de la BD Topo possèdent le champ 'VIT_MOY_VL' : il est utilisé pour créer les arêtes du graphe Voici sa méthodologie de calul : https://geoservices.ign.fr/documentation/services/services-geoplateforme/itineraire.
# 
# ![schema_graphe_vl](images/schema_graphe_vl.png)
# 
# Note : j'avais commencé à vouloir reproduire cette méthodologie pour calculer les vitesses avant de me rendre compte que le champ 'VIT_MOY_VL' était déjà calculé de cette manière.

# In[128]:


# Création du graphe représentant les routes empruntables par les véhicules légers
def creer_graphe_vl():
    # 1. Chargement des routes de la BD Topo
    routes = charger_fichier_parquet("bd_topo_routes_tampon", crs=2154) # Utiliser le 2154 pour calculer les bonnes distances cartésiennes

    # 2. Filtrage des tronçons accessibles aux VL
    """
    Voici les conditions pour déterminer si un tronçon est accessible aux VL :
    1. Son champ 'ETAT' doit être égal à 'En service'
    2. Son champ 'PRIVE' doit être égal à 'Non'
    3. Son champ 'ACCES_VL' doit être égal à 'libre' ou 'a péage'
    """
    routes = routes[
        (routes.geometry.notnull()) &
        (routes["ETAT"] == "En service") &
        (routes["PRIVE"].str.lower() == "non") &
        (routes["ACCES_VL"].str.lower().isin(["libre", "a péage"])) &
        (routes["VIT_MOY_VL"] != 0)
    ]
    routes = routes[routes.geometry.type.isin(["LineString", "MultiLineString"])]

    # 3. Liste des attributs conservés dans le graphe - Pas utilisé dans les calculs d'itinéraires, sert uniquement à vérifier
    """
    attributs = [
        'ID','LARGEUR', 'NATURE', 'NATURE_ITI',
        'NB_VOIES', 'RESTR_H', 'RESTR_LAR', 'RESTR_LON', 'RESTR_P',
        'RESTR_PPE'
    ]
    """
    attributs = []
    # 4. Initialisations
    graphe_vl = nx.DiGraph()
    edges, edge_attrs = [], []

    id_to_coord_vl = {}
    coord_to_id = {} # Temporaire, uniquement pour accélérer l'assignation
    next_id = 0

    def get_node_id(pt):
        nonlocal next_id
        if pt in coord_to_id:
            return coord_to_id[pt]
        node_id = next_id
        coord_to_id[pt] = node_id
        id_to_coord_vl[node_id] = pt
        next_id += 1
        return node_id

    # 5. Calcul des attributs sur toutes les routes à ajouter au graphe
    for _, row in routes.iterrows():
        geom = row.geometry
        lines = [geom] if isinstance(geom, LineString) else list(geom.geoms)

        """
        Par défaut, la BD Topo donne aux sentiers et autres petits chemins une vitesse 
        de 1 km/h pour les VL. En plus d'être irréaliste, cette valeur ralentit artificiellement
        les résultats de nos calculs d'itinéraires. On considère que si une route est accessible,
        alors sa vitesse moyenne ne peut pas être inférieure à 4 km / h, la vitesse de marche
        recommandée par le Cerema : doc.cerema.fr/Default/doc/SYRACUSE/16923/. Cette valeur 
        donne des calculs d'itinéraires très similaires au géoportail de l'IGN ou à google maps
        """
        vitesse = row.get("VIT_MOY_VL", 4)
        vitesse = max(vitesse, 4)

        sens = row.get("SENS", "Double sens")

        props = {
            attr: str(row[attr]) if pd.notnull(row[attr]) else "" for attr in attributs
        }
        props["VIT_MOY_VL"] = vitesse
        props["SENS"] = sens if sens else ""

        for line in lines:
            coords = list(line.coords)
            for i in range(len(coords) - 1):
                a, b = coords[i][:2], coords[i + 1][:2]
                dist = Point(a).distance(Point(b))
                # Calcul du temps nécessaire à la traversée d'une arrête
                time = (dist / 1000) / vitesse * 3600

                id_a = get_node_id(a)
                id_b = get_node_id(b)
                attrs = {"weight": time, "distance": dist, **props}

                # Le sens des arêtes du graphe est déterminé par l'attribut 'SENS' des rues de la BD Topo
                if sens == "Double sens":
                    edges.extend([(id_a, id_b), (id_b, id_a)])
                    edge_attrs.extend([attrs, attrs])
                elif sens == "Sens direct":
                    edges.append((id_a, id_b))
                    edge_attrs.append(attrs)
                elif sens == "Sens inverse":
                    edges.append((id_b, id_a))
                    edge_attrs.append(attrs)

    graphe_vl.add_nodes_from(id_to_coord_vl.keys())
    for edge, attr in zip(edges, edge_attrs):
        graphe_vl.add_edge(*edge, **attr)

    # 6. Exports
    chemin_graphe_graphml = os.path.join(exports_dir, "graphe_routier_vl.graphml")
    nx.write_graphml(graphe_vl, chemin_graphe_graphml)

    chemin_graphe_pkl = os.path.join(exports_dir, "graphe_routier_vl.pkl")
    with open(chemin_graphe_pkl, "wb") as f:
        pickle.dump((graphe_vl, id_to_coord_vl), f)

    print(f"Graphe VL exporté : {chemin_graphe_graphml}")
    print(f"Pickle sauvegardé : {chemin_graphe_pkl}")
    print(f"Le graphe se compose de {graphe_vl.number_of_nodes()} nœuds et {graphe_vl.number_of_edges()} arêtes")

    return graphe_vl, id_to_coord_vl


# In[139]:


def appel_creer_graphe_vl():
    # 1. Construction du graphe
    creer_graphe_vl()

    # 2. Exporte les noeuds et arêtes du graphe pour vérification manuelle dans un SIG
    exporter_noeuds_aretes_graphe(nom_graphe="vl")

    # 3. Rajoute aux carreaux une nouvelle colonne contenant tout les noeuds du graphe auxquels ils ont accès
    ajouter_noeuds_aux_carreaux(nom_graphe="vl")

    # 4. Affiche l'accessibilité des carreaux au réseau
    afficher_accessibilite_reseau(nom_graphe = "vl", nom_legende = "routier (VL)", export = True)

# Execution
appel_creer_graphe_vl()


# Les itinéraires peuvent être vérifiés depuis l'outil 'calculer un itinéraire du géoportail : https://www.geoportail.gouv.fr/carte

# In[170]:


# Calcule et affiche un itinéraire test
def afficher_itineraire_test_vl():
    itineraire = calcul_itineraire_test(nom_graphe = "vl", export = True)
    afficher_itineraire_test(itineraire, export = True)

# Exécution
afficher_itineraire_test_vl()


# In[63]:


# Affiche le temps nécessaire pour calculer X itinéraires, et le nombre d'itinéraires valides
temps_execution_itineraire(nom_graphe = "vl", nb_calculs = 100)


# In[229]:


# Calcul de l'utilisation des tronçons
def calcul_utilisation_troncons(): 
    carreaux = charger_fichier_parquet("maille_200m_epci")
    utilisation_troncons(nom_graphe = "vl", max_combi=None, nb_carreaux=15)

# Exécution
calcul_utilisation_troncons()


# ### 4.3. Création du graphe pour les vélos
# 
# ---

# In[171]:


# Création du graphe représentant les routes empruntables par les vélos
def creer_graphe_velos():
    # 1. Chargement des routes de la BD Topo
    routes = charger_fichier_parquet("bd_topo_routes_tampon", crs=2154)

    """
    Voici les conditions pour déterminer si un tronçon est accessible aux vélos :
    1. Son champ 'ETAT' doit être égal à 'En service'
    2. Son champ 'PRIVE' doit être égal à 'Non'
    3. Son champ 'NATURE' doit être différent de 'Type autoroutier'
    """
    # 2. Filtrage des tronçons accessibles aux vélos
    routes = routes[
        (routes.geometry.notnull()) &
        (routes["ETAT"] == "En service") &
        (routes["PRIVE"].str.lower() == "non") &
        (~routes["NATURE"].isin(["Type autoroutier"]))
    ]

    routes = routes[routes.geometry.type.isin(["LineString", "MultiLineString"])]

    # 3. Liste des attributs conservés dans le graphe
    attributs = [
        'ACCES_PED', 'ACCES_VL', 'BUS', 'CYCLABLE_D', 'CYCLABLE_G', 
        'ID', 'ITI_CYCL', 'IT_VERT', 'LARGEUR', 'NATURE', 'NATURE_ITI', 
        'NB_VOIES', 'RESTR_H', 'RESTR_LAR', 'RESTR_LON', 'RESTR_P', 
        'RESTR_PPE', 'SENS_CYC_D', 'SENS_CYC_G', 'VOIE_VERTE'
    ]

    # 4. Initialisations
    graphe_velos = nx.DiGraph()
    edges, edge_attrs = [], []

    coord_to_id = {}  # Temporaire, uniquement pour accélérer l'assignation
    id_to_coord_velos = {}
    next_id = 0

    def get_node_id(pt):
        nonlocal next_id
        if pt in coord_to_id:
            return coord_to_id[pt]
        node_id = next_id
        coord_to_id[pt] = node_id
        id_to_coord_velos[node_id] = pt
        next_id += 1
        return node_id

    # 5. Calcul des attributs sur toutes les routes à ajouter au graphe
    for _, row in routes.iterrows():
        geom = row.geometry
        lines = [geom] if isinstance(geom, LineString) else list(geom.geoms)

        # Assigne une vitesse de 14 km/h par défaut aux vélos. Chiffre issu 
        # des recommandations du CEREMA : doc.cerema.fr/Default/doc/SYRACUSE/16923/
        vitesse = 14 
        sens = row.get("SENS", "Double sens")

        props = {
            attr: str(row[attr]) if pd.notnull(row[attr]) else "" for attr in attributs
        }
        props["VIT_MOY_VL"] = vitesse
        props["SENS"] = sens if sens else ""

        for line in lines:
            coords = list(line.coords)
            for i in range(len(coords) - 1):
                a, b = coords[i][:2], coords[i + 1][:2]
                dist = Point(a).distance(Point(b))
                time = (dist / 1000) / vitesse * 3600

                id_a, id_b = get_node_id(a), get_node_id(b)
                attrs = {"weight": time, "distance": dist, **props}

                # L'orientation des arêtes pour les vélos suit celui pour les VL
                if sens == "Double sens":
                    edges.extend([(id_a, id_b), (id_b, id_a)])
                    edge_attrs.extend([attrs, attrs])
                elif sens == "Sens direct":
                    edges.append((id_a, id_b))
                    edge_attrs.append(attrs)
                elif sens == "Sens inverse":
                    edges.append((id_b, id_a))
                    edge_attrs.append(attrs)

    # 6. Ajout des éléments au graphe
    graphe_velos.add_nodes_from(id_to_coord_velos.keys())
    for edge, attr in zip(edges, edge_attrs):
        graphe_velos.add_edge(*edge, **attr)

    # 7. Exports
    chemin_graphe_graphml = os.path.join(exports_dir, "graphe_routier_velos.graphml")
    nx.write_graphml(graphe_velos, chemin_graphe_graphml)

    chemin_graphe_pkl = os.path.join(exports_dir, "graphe_routier_velos.pkl")
    with open(chemin_graphe_pkl, "wb") as f:
        pickle.dump((graphe_velos, id_to_coord_velos), f)

    # 8. Statistiques
    print(f"Graphe exporté : {chemin_graphe_graphml}")
    print(f"Pickle sauvegardé : {chemin_graphe_pkl}")
    print(f"Le graphe se compose de {graphe_velos.number_of_nodes()} nœuds et {graphe_velos.number_of_edges()} arêtes")

    return graphe_velos, id_to_coord_velos


# In[172]:


def appel_creer_graphe_velos():
    # 1. Construction du graphe
    creer_graphe_velos()

    # 2. Exporte les noeuds et arêtes du graphe pour vérification manuelle dans un SIG
    exporter_noeuds_aretes_graphe(nom_graphe="velos")

    # 3. Rajoute aux carreaux une nouvelle colonne contenant tout les noeuds du graphe auxquels ils ont accès
    ajouter_noeuds_aux_carreaux(nom_graphe="velos")

    # 4. Affiche l'accessibilité des carreaux au réseau
    afficher_accessibilite_reseau(nom_graphe = "velos", nom_legende = "cyclable", export = True)

# Execution
appel_creer_graphe_velos()


# Note : avec les valeurs par défaut du Cerema (14 km/h de vitesse moyenne pour les vélos), on semble un peu en dessous par rapports aux calculs de google maps (ex : 41 min pour nous, 35 min pour maps)

# In[184]:


# Calcule et affiche un itinéraire test
def afficher_itineraire_test_velos():
    itineraire = calcul_itineraire_test(nom_graphe = "velos", export = True)
    afficher_itineraire_test(itineraire, export = True)

# Exécution
afficher_itineraire_test_velos()


# In[119]:


# Affiche le temps nécessaire pour calculer X itinéraires, et le nombre d'itinéraires valides
temps_execution_itineraire(nom_graphe = "velos", nb_calculs = 100)


# ### 4.4. Création du graphe pour la marche
# ---
# 
# Note : pour le moment, les conditions pour qu'une route soit considérée comme accessible aux vélos et aux piétons sont les mêmes. Mais les graphes ne sont pas identiques, à cause des vitesse de déplacement différentes des piétons et des cyclistes.

# In[185]:


# Création du graphe représentant les routes empruntables par les piétons
def creer_graphe_marche():
    # 1. Chargement des données
    routes = charger_fichier_parquet("bd_topo_routes_tampon", crs=2154)

    """
    Voici les conditions pour déterminer si un tronçon est accessible aux vélos :
    1. Son champ 'ETAT' doit être égal à 'En service'
    2. Son champ 'PRIVE' doit être égal à 'Non'
    3. Son champ 'NATURE' doit être différent de 'Type autoroutier'
    Note : les données de la BD Topo étant assez incomplètes, le champ 'ACCES_PED'
    n'a pas été utilisé. Si l'on filtre dessus, le graphe est très réduit
    """
    # 2. Filtrage des tronçons accessibles aux piétons
    routes = routes[
        (routes.geometry.notnull()) &
        (routes["ETAT"] == "En service") &
        (routes["PRIVE"].str.lower() == "non") &
        (~routes["NATURE"].isin(["Type autoroutier"]))
    ]
    routes = routes[routes.geometry.type.isin(["LineString", "MultiLineString"])]

    # 3. Liste des attributs conservés dans le graphe
    attributs = [
        'ID','LARGEUR', 'NATURE', 'NATURE_ITI','NB_VOIES', 'RESTR_H', 
        'RESTR_LAR', 'RESTR_LON', 'RESTR_P','RESTR_PPE'
    ]

    # 4. Initialisations
    graphe_marche = nx.DiGraph()
    edges, edge_attrs = [], []

    coord_to_id = {}
    id_to_coord_marche = {}
    next_id = 0

    def get_node_id(pt):
        nonlocal next_id
        if pt not in coord_to_id:
            coord_to_id[pt] = next_id
            id_to_coord_marche[next_id] = pt
            next_id += 1
        return coord_to_id[pt]

    # 5. Calcul des attributs sur toutes les routes à ajouter au graphe
    for _, row in routes.iterrows():
        geom = row.geometry
        lines = [geom] if isinstance(geom, LineString) else list(geom.geoms)

        # Assigne une vitesse de 4 km/h par défaut aux piétons. Chiffre issu 
        # des recommandations du CEREMA : doc.cerema.fr/Default/doc/SYRACUSE/16923/
        vitesse = 4
        props = {
            attr: str(row[attr]) if pd.notnull(row[attr]) else "" for attr in attributs
        }
        props["VIT_MOY_VL"] = vitesse

        for line in lines:
            coords = list(line.coords)
            for i in range(len(coords) - 1):
                a, b = coords[i][:2], coords[i + 1][:2]
                dist = Point(a).distance(Point(b))
                time = (dist / 1000) / vitesse * 3600

                id_a, id_b = get_node_id(a), get_node_id(b)
                attrs = {"weight": time, "distance": dist, **props}

                # Les trajets à pied ne respectent pas l'attribut 'SENS' de la BD Topo
                # donc toutes les arêtes existent sont traversables dans les deux sens
                edges.extend([(id_a, id_b), (id_b, id_a)])
                edge_attrs.extend([attrs, attrs])

    # 6. Ajout des éléments au graphe
    graphe_marche.add_nodes_from(id_to_coord_marche.keys())
    for edge, attr in zip(edges, edge_attrs):
        graphe_marche.add_edge(*edge, **attr)

    # 7. Exports
    graph_path = os.path.join(exports_dir, "graphe_routier_marche.graphml")
    pickle_path = os.path.join(exports_dir, "graphe_routier_marche.pkl")

    nx.write_graphml(graphe_marche, graph_path)
    with open(pickle_path, "wb") as f:
        pickle.dump((graphe_marche, id_to_coord_marche), f)

    # 8. Statistiques
    print(f"Graphe exporté : {graph_path}")
    print(f"Pickle sauvegardé : {pickle_path}")
    print(f"Le graphe se compose de {graphe_marche.number_of_nodes()} nœuds et {graphe_marche.number_of_edges()} arêtes")

    return graphe_marche, id_to_coord_marche


# In[186]:


def appel_creer_graphe_marche():
    # 1. Construction du graphe
    creer_graphe_marche()

    # 2. Exporte les noeuds et arêtes du graphe pour vérification manuelle dans un SIG
    exporter_noeuds_aretes_graphe(nom_graphe="marche")

    # 3. Rajoute aux carreaux une nouvelle colonne contenant tout les noeuds du graphe auxquels ils ont accès
    ajouter_noeuds_aux_carreaux(nom_graphe="marche")

    # 4. Affiche l'accessibilité des carreaux au réseau
    afficher_accessibilite_reseau(nom_graphe = "marche", nom_legende = "piéton", export = True)

# Execution
appel_creer_graphe_marche()


# In[198]:


# Calcule et affiche un itinéraire test
def afficher_itineraire_test_marche():
    itineraire = calcul_itineraire_test(nom_graphe = "marche", export = True)
    afficher_itineraire_test(itineraire, export = True)

# Exécution
afficher_itineraire_test_marche()


# In[218]:


# Affiche le temps nécessaire pour calculer X itinéraires, et le nombre d'itinéraires valides
temps_execution_itineraire(nom_graphe = "marche", nb_calculs = 100)


# ### 4.5. Création du graphe pour les trams
# ---
# 
# Les deux sources de données pour créer ce graphe sont : 
# 1. Les flux GTFS de l'EMS
# 2. Les lignes de tram de l'EMS (dernière maj : 2022) : https://data.strasbourg.eu/explore/dataset/lignes_tram/information/?disjunctive.ligne&sort=ligne
# https://www.data.gouv.fr/fr/datasets/itineraires-de-tramways-dans-openstreetmap/
# 
# Contrairement aux autres graphes qui étaient construits en se basant sur les routes de la BD Topo, on ne peux pas faire cela pour les trams. Il est construit de la manière suivante : 
# 1. En se basant sur les flux GTFS, on récupère les différentes lignes et les points d'arrêts qu'elles empruntent. Cela sert à créer les noeuds.
# 2. En utilisant le fichier 'lignes de tram' de l'EMS, on récupère la géométrie réelle, dont on récupère la longueur pour l'appliquer aux arêtes : on connaît la longueur des trajets entre chaque arrêts de chaque ligne.
# 3. On connaît également le temps nécessaire pour le trajet du tram entre chaque arrêt : on peut calculer la vitesse, et l'appliquer au graphe. L'orientation des arêtes (le sens) est également déterminé de cette façon. Comme il peux y avoir de nombreux trajets sur une même ligne entre 2 noeuds, on calcule pour chaque arête et pour chaque ligne sa vitesse moyenne selon tout les trajets en semaine (donc on exclut le samedi et le dimanche). Cela donne une moyenne représentative. Les temps d'arrêts moyens sont calculés de la même manière.
# 
# Le graphe final possède des arêtes fidèles à la réalité, et prends en compte les différentes lignes. Sinon : aurait permis de naviguer entre tout les points dans n'importe quel sens, même ceux qui ne sont pas liés ensemble par une ligne.
# 
# Comme pour les routes, on conserve les arrêts de tram hors de l'EPCI, afin de mieux simuler nos calculs d'itinéraires.
# 
# Les fichiers GTFS de l'EMS : https://opendata.cts-strasbourg.eu/ peuvent être visulisés grâce au site suivant : https://gtfs.pleasantprogrammer.com/gtfs.html
# 
# Notes : pour un même 'route_id', on peut avoir plusieurs itinéraires. Par exemple, la 'route_id' 'E-1609' possède différents itinéraires, avec des points de départ / d'arrivée et un nombre d'arrêts différents.
# 
# ![schema_graphe_tram](images/schema_graphe_tram.png)

# In[189]:


# Création du graphe représentant les routes empruntables par le tram
def creer_graphe_tram():
    # 1. Chargement des données GTFS
    df_stops = pd.read_csv(os.path.join(exports_dir, "gtfs_stops_tram.csv"))
    df_stop_times = pd.read_csv(os.path.join(exports_dir, "gtfs_stop_times_tram.csv"))
    df_trips = pd.read_csv(os.path.join(exports_dir, "gtfs_trips_tram.csv"))

    # 2. Géométrie des arrêts projetée
    gdf_stops = gpd.GeoDataFrame(
        df_stops,
        geometry=gpd.points_from_xy(df_stops["stop_lon"], df_stops["stop_lat"]),
        crs="EPSG:4326"
    ).to_crs("EPSG:2154")
    gdf_stops["x"] = gdf_stops.geometry.x
    gdf_stops["y"] = gdf_stops.geometry.y
    stop_coords = {row["stop_id"]: (row["x"], row["y"]) for _, row in gdf_stops.iterrows()}
    stop_id_to_name = dict(zip(df_stops["stop_id"], df_stops["stop_name"]))

    # 3. Chargement des lignes de tram
    lignes_tram = charger_fichier_parquet("lignes_tram", crs=2154)
    ligne_geoms = {row["ligne"]: row.geometry for _, row in lignes_tram.iterrows()}

    # 4. Fusion des données GTFS
    df = df_stop_times.merge(df_trips[["trip_id", "route_id"]], on="trip_id")
    df["arrival_dt"] = df["arrival_time"].apply(safe_parse_time)
    df = df[df["arrival_dt"].notnull()]

    """
    Dans les flux GTFS de l'EMS, chaque ligne de tram (de A à F) est divisée 
    en plusieurs lignes selon un code interne, qui comprend 'A-1581', 'A-1606', 
    'A-1635', 'B-1581', 'C-1581', etc. jusqu'à 'F-1635'. Pour récupérer le n° de 
    ligne, on ne garde que les caractères présents avant le -. Si ce code venaît 
    à être réutilisé pour d'autres agglomérations, il est possible que leur flux
    GTFS soit structuré différemment. Il faudra donc modifier cette logique.
    """
    df["ligne"] = df["route_id"].str.split("-").str[0]

    # 5. Regroupement des trajets
    trajets_data = defaultdict(list)
    for route_id, group_route in df.groupby("route_id"):
        for trip_id, group_trip in group_route.groupby("trip_id"):
            group_trip = group_trip.sort_values("stop_sequence")
            stops = group_trip["stop_id"].tolist()
            arrivals = group_trip["arrival_dt"].tolist()
            for i in range(len(stops) - 1):
                u, v = stops[i], stops[i + 1]
                t1, t2 = arrivals[i], arrivals[i + 1]
                if t2 > t1:
                    delta = (t2 - t1).total_seconds()
                    key = (u, v, route_id, stops[0], stops[-1], len(stops))
                    trajets_data[key].append(delta)

    # 6. Création du graphe
    G = nx.MultiDiGraph()
    for stop_id, (x, y) in stop_coords.items():
        G.add_node(stop_id, lon=x, lat=y, stop_name=stop_id_to_name.get(stop_id, "inconnu"))

    for (u, v, route_id, start_stop, end_stop, nb_stops), deltas in trajets_data.items():
        if u not in stop_coords or v not in stop_coords:
            continue
        pt_u, pt_v = Point(stop_coords[u]), Point(stop_coords[v])
        ligne = route_id.split("-")[0]
        temps_moyen = sum(deltas) / len(deltas)

        geom = ligne_geoms.get(ligne)
        if geom is not None:
            try:
                proj_u = geom.project(pt_u)
                proj_v = geom.project(pt_v)
                start, end = sorted([proj_u, proj_v])
                segment = substring(geom, start, end)
                if segment.is_empty or not isinstance(segment, LineString):
                    raise ValueError("Segment vide ou invalide")
            except Exception as e:
                print(f"[!] Géométrie non trouvée pour {u}->{v} ({route_id}): {e}")
                segment = LineString([pt_u, pt_v])
        else:
            segment = LineString([pt_u, pt_v])

        dist = segment.length
        vitesse = (dist / 1000) / (temps_moyen / 3600) if temps_moyen > 0 else 10

        G.add_edge(
            u, v,
            key=f"{route_id}_{start_stop}_{end_stop}_{nb_stops}",
            route_id=route_id,
            distance=dist,
            weight=temps_moyen,
            vitesse=vitesse,
            geometry=segment
        )

    # 7. Remappage des identifiants
    coord_to_id = {}
    id_to_coord = {}
    remap = {}
    next_id = 0
    for stop_id in G.nodes:
        coord = (G.nodes[stop_id]["lon"], G.nodes[stop_id]["lat"])
        if coord not in coord_to_id:
            coord_to_id[coord] = next_id
            id_to_coord[next_id] = coord
            next_id += 1
        remap[stop_id] = coord_to_id[coord]

    graphe_remap = nx.MultiDiGraph()
    for old_id, new_id in remap.items():
        graphe_remap.add_node(new_id, **G.nodes[old_id])
    for u, v, key, data in G.edges(keys=True, data=True):
        graphe_remap.add_edge(remap[u], remap[v], key=key, **data)

    # 8. Export
    pickle_path = os.path.join(exports_dir, "graphe_routier_tram.pkl")
    with open(pickle_path, "wb") as f:
        pickle.dump((graphe_remap, id_to_coord), f)

    print(f"Graphe tram créé avec {graphe_remap.number_of_nodes()} nœuds et {graphe_remap.number_of_edges()} arêtes")
    print(f"Exporté vers : {pickle_path}")
    return graphe_remap, id_to_coord


# In[190]:


def appel_creer_graphe_tram():
    # 1. Construction du graphe
    creer_graphe_tram()

    # 2. Exporte les noeuds et arêtes du graphe pour vérification manuelle dans un SIG
    exporter_noeuds_aretes_graphe(nom_graphe="tram")

    # 3. Rajoute aux carreaux une nouvelle colonne contenant tout les noeuds du graphe auxquels ils ont accès
    ajouter_noeuds_aux_carreaux(nom_graphe="tram")

    # 4. Affiche l'accessibilité des carreaux au réseau
    afficher_accessibilite_reseau(nom_graphe = "tram", nom_legende = "tram", export = True)

# Execution
appel_creer_graphe_tram()


# In[116]:


afficher_reseau_transport_commun(nom_graphe = "tram", ligne_a_afficher = None)


# Les itinéraires peuvent être vérifies sur https://www.cts-strasbourg.eu/fr/se-deplacer/recherche-itineraires/
# Pour rappel : on explore uniquement le graphe des trams, on n'effectue pas de trajets à pied, et il existe typiquement deux stations proches les unes des autres selon le sens du tram. Cela explique que les itinéraires semblent faire des "boucles" en repassant sur le même trajet. L'itinéraire a comme point d'arrivée une station atteignable uniquement dans l'autre sens de circulation du tram.

# In[193]:


# Calcule et affiche un itinéraire test
def afficher_itineraire_test_tram():
    itineraire = calcul_itineraire_test(nom_graphe = "tram", export = True)
    afficher_itineraire_test(itineraire, est_tram = True, export = True)

# Exécution
afficher_itineraire_test_tram()


# In[87]:


# Affiche le temps nécessaire pour calculer X itinéraires, et le nombre d'itinéraires valides
temps_execution_itineraire(nom_graphe = "tram", nb_calculs = 50)


# ### 4.6. Création du graphe pour les bus
# ---

# SOLUTION : plaquer les arrêts de bus sur la route la plus proche et calculer l'itinéraire grace au graphe routier ?
# But : créer un graphe représentant les trajets routiers des bus, où 
# 1. L'itinéraire ne peux se faire que sur les lignes existantes, et naviguer entre les arrêts associés à chaque ligne uniquement
# 2. Les arrêts de bus doivent posséder leur grille horaire pour chaque ligne (utiliser des dictionnaires python pour stocker {id_arret_bus, liste des lignes qui l'utilise} et {id_ligne, liste des horaires pour les lignes que connaît l'arrêt} ?)
# 3. Besoin de repositionner les arrêts de bus directement sur la route (au plus proche) et de les considérer comme de nouvelles arêtes. Etape nécessaire car nos routes sont simplifiées et sont des lignes : écart entre nos arrêts et nos routes.
# 4. On considère uniquement les trajets réalisables en semaine (ex : un lundi ?) matin et après-midi sur le graphe. Sinon : devient ingérable pour tout gérer entre les lignes et les horaires qui changent selon le mois, le jour, les trajets de nuit, etc. Simuler les trajets possibles lors de l'utilisation la plus commune.
# 
# Comme pour les trams, on se base sur les flux GTFS pour créer le graphe. En revanche, les données sur les lignes de bus dans l'EMS sont de mauvaise qualité.
# Exemple d'erreurs dans les lignes de bus : 
# ![lignes_bus_incorrect_exemple](images/lignes_bus_incorrect_exemple.png)
# 
# De plus, les producteurs de ce fichier se sont trompés sur les lignes de bus elle-mêmes. Par exemple, ce qui devrait être la ligne C1 devient la ligne L1. L'entrée de la ligne C1 dans le fichier représente complètement autre chose.
# ![exemple_erreur_lignes](images/exemple_erreur_lignes.png)
# 
# Pour corriger ça, la méthode suivante est appliquée :
# ![schema_graphe_bus](images/schema_graphe_bus.png)

# In[129]:


# Correction des données issues des lignes de bus : on leur applique une zone tampon, et l'on fait une jointure
# spatiales sur les route de la BD Topo intersectant avec le tampon. On peut ensuite utiliser des routes sans erreurs 
def extraire_routes_lignes_bus():
    # 1. Chargement des données
    lignes_bus = charger_fichier_parquet("lignes_bus", crs=2154)
    routes_bd_topo = charger_fichier_parquet("bd_topo_routes_tampon", crs=2154)

    # 2. Création du tampon de X mètres autour des lignes de bus
    # tampon = lignes_bus.buffer(7) # 7 mètres : déterminé après plusieurs essais et vérificaiton visuelle
    # tampon_union = tampon.geometry.union_all()

    # 3. Filtrage des routes intersectées
    # routes_filtrees = routes_bd_topo[routes_bd_topo.geometry.intersects(tampon_union)].copy()
    routes_filtrees

    # 4. Filtre des routes accessibles aux bus
    routes_filtrees = routes_filtrees[
        (routes_filtrees.geometry.notnull()) &
        (routes_filtrees["ETAT"] == "En service") &
        (routes_filtrees["PRIVE"].str.lower() == "non") &
        (routes_filtrees["ACCES_VL"].str.lower().isin(["libre", "a péage"])) &
        (routes_filtrees["VIT_MOY_VL"] != 0)
    ]

    # 5. Export
    exporter_parquet(routes_filtrees, "lignes_bus_fusion_bd_topo")
    exporter_gpkg(routes_filtrees, "lignes_bus_fusion_bd_topo")

    print(f"{len(routes_filtrees)} tronçons extraits et exportés sous 'lignes_bus_fusion_bd_topo'")

# Exécution
extraire_routes_lignes_bus()


# In[ ]:


# Création du graphe représentant les routes potentiellement empruntables par les bus
def creer_graphe_routes_bus():
    # 1. Chargement des routes de la BD Topo
    routes = charger_fichier_parquet("bd_topo_routes_tampon", crs=2154) # Utiliser le 2154 pour calculer les bonnes distances cartésiennes

    # 2. Filtrage des tronçons accessibles aux VL
    """
    Voici les conditions pour déterminer si un tronçon est accessible aux VL :
    1. Son champ 'ETAT' doit être égal à 'En service'
    2. Son champ 'PRIVE' doit être égal à 'Non'
    3. Son champ 'ACCES_VL' doit être égal à 'libre' ou 'a péage'
    """
    routes = routes[
        (routes.geometry.notnull()) &
        (routes["ETAT"] == "En service") &
        (routes["PRIVE"].str.lower() == "non") &
        (routes["ACCES_VL"].str.lower().isin(["libre", "a péage"])) &
        (routes["VIT_MOY_VL"] != 0)
    ]
    routes = routes[routes.geometry.type.isin(["LineString", "MultiLineString"])]

    # 3. Liste des attributs conservés dans le graphe - Pas utilisé dans les calculs d'itinéraires, sert uniquement à vérifier
    """
    attributs = [
        'ID','LARGEUR', 'NATURE', 'NATURE_ITI',
        'NB_VOIES', 'RESTR_H', 'RESTR_LAR', 'RESTR_LON', 'RESTR_P',
        'RESTR_PPE'
    ]
    """
    attributs = []
    # 4. Initialisations
    graphe_vl = nx.DiGraph()
    edges, edge_attrs = [], []

    id_to_coord_vl = {}
    coord_to_id = {} # Temporaire, uniquement pour accélérer l'assignation
    next_id = 0

    def get_node_id(pt):
        nonlocal next_id
        if pt in coord_to_id:
            return coord_to_id[pt]
        node_id = next_id
        coord_to_id[pt] = node_id
        id_to_coord_vl[node_id] = pt
        next_id += 1
        return node_id

    # 5. Calcul des attributs sur toutes les routes à ajouter au graphe
    for _, row in routes.iterrows():
        geom = row.geometry
        lines = [geom] if isinstance(geom, LineString) else list(geom.geoms)

        """
        Par défaut, la BD Topo donne aux sentiers et autres petits chemins une vitesse 
        de 1 km/h pour les VL. En plus d'être irréaliste, cette valeur ralentit artificiellement
        les résultats de nos calculs d'itinéraires. On considère que si une route est accessible,
        alors sa vitesse moyenne ne peut pas être inférieure à 4 km / h, la vitesse de marche
        recommandée par le Cerema : doc.cerema.fr/Default/doc/SYRACUSE/16923/. Cette valeur 
        donne des calculs d'itinéraires très similaires au géoportail de l'IGN ou à google maps
        """
        vitesse = row.get("VIT_MOY_VL", 4)
        vitesse = max(vitesse, 4)

        sens = row.get("SENS", "Double sens")

        props = {
            attr: str(row[attr]) if pd.notnull(row[attr]) else "" for attr in attributs
        }
        props["VIT_MOY_VL"] = vitesse
        props["SENS"] = sens if sens else ""

        for line in lines:
            coords = list(line.coords)
            for i in range(len(coords) - 1):
                a, b = coords[i][:2], coords[i + 1][:2]
                dist = Point(a).distance(Point(b))
                # Calcul du temps nécessaire à la traversée d'une arrête
                time = (dist / 1000) / vitesse * 3600

                id_a = get_node_id(a)
                id_b = get_node_id(b)
                attrs = {"weight": time, "distance": dist, **props}

                # Le sens des arêtes du graphe est déterminé par l'attribut 'SENS' des rues de la BD Topo
                if sens == "Double sens":
                    edges.extend([(id_a, id_b), (id_b, id_a)])
                    edge_attrs.extend([attrs, attrs])
                elif sens == "Sens direct":
                    edges.append((id_a, id_b))
                    edge_attrs.append(attrs)
                elif sens == "Sens inverse":
                    edges.append((id_b, id_a))
                    edge_attrs.append(attrs)

    graphe_vl.add_nodes_from(id_to_coord_vl.keys())
    for edge, attr in zip(edges, edge_attrs):
        graphe_vl.add_edge(*edge, **attr)

    # 6. Exports
    chemin_graphe_graphml = os.path.join(exports_dir, "graphe_routier_routes_bus.graphml")
    nx.write_graphml(graphe_vl, chemin_graphe_graphml)

    chemin_graphe_pkl = os.path.join(exports_dir, "graphe_routier_routes_bus.pkl")
    with open(chemin_graphe_pkl, "wb") as f:
        pickle.dump((graphe_vl, id_to_coord_vl), f)

    print(f"Graphe VL exporté : {chemin_graphe_graphml}")
    print(f"Pickle sauvegardé : {chemin_graphe_pkl}")
    print(f"Le graphe se compose de {graphe_vl.number_of_nodes()} nœuds et {graphe_vl.number_of_edges()} arêtes")

    return graphe_vl, id_to_coord_vl


# In[416]:


import os
import pickle
from collections import defaultdict
from shapely.geometry import Point, LineString
from shapely.ops import linemerge
import networkx as nx
from joblib import Parallel, delayed
import threading
import pandas as pd
import geopandas as gpd
from tqdm import tqdm
from joblib import Parallel, delayed
from joblib.externals.loky import set_loky_pickler
from multiprocessing import Manager
from tqdm.contrib.concurrent import thread_map 
from shapely.ops import nearest_points

# On suppose que safe_parse_time, charger_fichier_parquet, charger_graphe,
# exports_dir, etc. sont déjà définis dans ton environnement.

def creer_graphe_bus():
    # 1. Chargement des GTFS
    df_stops = pd.read_csv(os.path.join(exports_dir, "gtfs_stops_bus.csv"))
    df_stop_times = pd.read_csv(os.path.join(exports_dir, "gtfs_stop_times_bus.csv"))
    df_trips = pd.read_csv(os.path.join(exports_dir, "gtfs_trips_bus.csv"))

    # 2. Géométrie des arrêts (projection en EPSG:2154)
    gdf_stops = gpd.GeoDataFrame(
        df_stops,
        geometry=gpd.points_from_xy(df_stops["stop_lon"], df_stops["stop_lat"]),
        crs="EPSG:4326"
    ).to_crs("EPSG:2154")
    gdf_stops["x"] = gdf_stops.geometry.x
    gdf_stops["y"] = gdf_stops.geometry.y
    stop_coords = {row["stop_id"]: (row["x"], row["y"]) for _, row in gdf_stops.iterrows()}
    stop_id_to_name = dict(zip(df_stops["stop_id"], df_stops["stop_name"]))

    # 3. Chargement du graphe routier pour les bus depuis la BD Topo (routes_bus)
    print("Chargement du graphe routier routes_bus...")
    graphe_routes_bus, id_to_coord = charger_graphe("routes_bus")
    # On suppose que graphe_routes_bus est un DiGraph avec poids "weight" et géométries.

    # 4. Fusion des données GTFS
    df = df_stop_times.merge(df_trips[["trip_id", "route_id"]], on="trip_id")
    df["arrival_dt"] = df["arrival_time"].apply(safe_parse_time)
    df = df[df["arrival_dt"].notnull()]
    # On extrait la "ligne" à partir du code route_id (avant "-")
    df["ligne"] = df["route_id"].str.split("-").str[0]

    trajets_data = defaultdict(list)
    for route_id, group_route in df.groupby("route_id"):
        for trip_id, group_trip in group_route.groupby("trip_id"):
            group_trip = group_trip.sort_values("stop_sequence")
            stops = group_trip["stop_id"].tolist()
            arrivals = group_trip["arrival_dt"].tolist()
            for i in range(len(stops) - 1):
                u, v = stops[i], stops[i + 1]
                t1, t2 = arrivals[i], arrivals[i + 1]
                if t2 > t1:
                    delta = (t2 - t1).total_seconds()
                    key = (u, v, route_id, stops[0], stops[-1], len(stops))
                    trajets_data[key].append(delta)

    # 5. Initialisation du graphe bus (résultat)
    G = nx.MultiDiGraph()
    for stop_id, (x, y) in stop_coords.items():
        G.add_node(stop_id, lon=x, lat=y, stop_name=stop_id_to_name.get(stop_id, "inconnu"))

    # 6. Mise en cache partagée et fonction utilitaire pour trouver le nœud le plus proche dans graphe_routes_bus.
    def find_n_closest_nodes(pt: Point, n=5):
        dists = []
        for nid, coord in id_to_coord.items():
            d = pt.distance(Point(coord))
            dists.append((d, nid))
        return [nid for _, nid in sorted(dists)[:n]]

    # Cache partagé pour les itinéraires. Clé : (n1, n2), valeur : geometry (LineString)
    path_cache = {}
    cache_lock = threading.Lock()

    # 7. Fonction de traitement d'un trajet (à paralléliser)
    def process_trajet(key, deltas):
        u, v, route_id, start_stop, end_stop, nb_stops = key
        if u not in stop_coords or v not in stop_coords:
            return None

        pt_u = Point(stop_coords[u])
        pt_v = Point(stop_coords[v])
        temps_moyen = sum(deltas) / len(deltas)

        noeuds_u = find_n_closest_nodes(pt_u, 5)
        noeuds_v = find_n_closest_nodes(pt_v, 5)

        # Essaye toutes les combinaisons de nœuds proches
        segment = None
        for n1 in noeuds_u:
            for n2 in noeuds_v:
                if n1 == n2:
                    continue
                cache_key = (n1, n2)
                with cache_lock:
                    if cache_key in path_cache:
                        segment = path_cache[cache_key]
                        break
                try:
                    path = nx.shortest_path(graphe_routes_bus, source=n1, target=n2, weight="weight")
                    segments = []
                    for i in range(len(path) - 1):
                        edge_data = graphe_routes_bus.get_edge_data(path[i], path[i + 1])
                        edge_geom = None
                        if edge_data and isinstance(edge_data, dict):
                            values = list(edge_data.values())
                            if values and isinstance(values[0], dict) and "geometry" in values[0]:
                                edge_geom = values[0]["geometry"]
                        if edge_geom is None:
                            coord_a = id_to_coord[path[i]]
                            coord_b = id_to_coord[path[i + 1]]
                            edge_geom = LineString([coord_a, coord_b])
                        segments.append(edge_geom)
                    segment = linemerge(segments)
                    with cache_lock:
                        path_cache[cache_key] = segment
                    break  # On a trouvé un chemin, on sort
                except Exception:
                    continue  # Essaie une autre combinaison
            if segment is not None:
                break

        # Aucun itinéraire trouvé : fallback géométrie directe
        if segment is None:
            print(f"[!] Itinéraire non trouvé pour {u}->{v} (aucun chemin entre les 5 nœuds proches)")
            segment = LineString([pt_u, pt_v])

        dist = segment.length
        vitesse = (dist / 1000) / (temps_moyen / 3600) if temps_moyen > 0 else 15

        return (u, v, route_id, start_stop, end_stop, nb_stops, temps_moyen, dist, vitesse, segment)

    # 8. Traitement parallèle
    print("Création des arêtes bus avec itinéraires routiers probables...")

    trajets_items = list(trajets_data.items())

    # tqdm + threading backend → thread_map simplifie la gestion
    processed = thread_map(
        lambda item: process_trajet(*item),
        trajets_items,
        max_workers=os.cpu_count(),
        desc="Trajets traités"
    )

    # 9. Ajout des arêtes au graphe bus
    for (u, v, route_id, start_stop, end_stop, nb_stops, temps_moyen, dist, vitesse, segment) in processed:
        G.add_edge(
            u, v,
            key=f"{route_id}_{start_stop}_{end_stop}_{nb_stops}",
            route_id=route_id,
            distance=dist,
            weight=temps_moyen,
            vitesse=vitesse,
            geometry=segment
        )

    # 10. Remappage des identifiants pour avoir des nœuds numériques
    coord_to_id = {}
    id_to_coord_bus = {}
    remap = {}
    next_id = 0
    for stop_id in G.nodes:
        coord = (G.nodes[stop_id]["lon"], G.nodes[stop_id]["lat"])
        if coord not in coord_to_id:
            coord_to_id[coord] = next_id
            id_to_coord_bus[next_id] = coord
            next_id += 1
        remap[stop_id] = coord_to_id[coord]

    G_remap = nx.MultiDiGraph()
    for old_id, new_id in remap.items():
        G_remap.add_node(new_id, **G.nodes[old_id])
    for u, v, key, data in G.edges(keys=True, data=True):
        G_remap.add_edge(remap[u], remap[v], key=key, **data)

    # 11. Export du graphe
    pickle_path = os.path.join(exports_dir, "graphe_routier_bus.pkl")
    with open(pickle_path, "wb") as f:
        pickle.dump((G_remap, id_to_coord_bus), f)

    print(f"Graphe bus créé avec {G_remap.number_of_nodes()} nœuds et {G_remap.number_of_edges()} arêtes")
    print(f"Exporté vers : {pickle_path}")
    return G_remap, id_to_coord_bus


# In[422]:


def appel_creer_graphe_bus():
    # 1. Construction du graphe
    creer_graphe_bus()

    # 2. Exporte les noeuds et arêtes du graphe pour vérification manuelle dans un SIG
    exporter_noeuds_aretes_graphe(nom_graphe="bus")

    # 3. Rajoute aux carreaux une nouvelle colonne contenant tout les noeuds du graphe auxquels ils ont accès
    ajouter_noeuds_aux_carreaux(nom_graphe="bus")

    # 4. Affiche l'accessibilité des carreaux au réseau
    afficher_accessibilite_reseau(nom_graphe = "bus", nom_legende = "bus", export = True)

# Execution
appel_creer_graphe_bus()


# In[428]:


afficher_reseau_transport_commun(nom_graphe = "bus", ligne_a_afficher =None)


# In[127]:


# Calcule et affiche un itinéraire test
def afficher_itineraire_test_bus():
    itineraire = calcul_itineraire_test(nom_graphe = "bus", export = True)
    afficher_itineraire_test(itineraire, est_bus = True, export = True)

# Exécution
afficher_itineraire_test_bus()


# ### 4.7. Création du "super-graphe" multimodal
# ---
# NOTE : ancien code, à refaire entièrement
# Ce graphe n'est pas utilisé pour le calcul des indicateurs, mais permet de calculer le trajet le plus court entre 2 points avec tout les moyens de transport disponibles

# In[287]:


def construire_index_coords(graphes, noeuds_par_mode=None):
    """
    Construit l’index global (id_to_coord) et une table de remappage (node_map)
    pour le graphe multimodal, en incluant tous les nœuds présents dans les graphes modaux.
    Si `noeuds_par_mode` est fourni, on ajoute un attribut 'interne' = True uniquement pour eux.
    """
    id_to_coord = {}
    coord_to_id = {}
    node_map = {}
    next_id = 0

    for mode, (G, id_to_coord_mode) in graphes.items():
        noeuds_internes = set(noeuds_par_mode.get(mode, [])) if noeuds_par_mode else None

        for node_id in G.nodes:
            coord = id_to_coord_mode[node_id]

            if coord not in coord_to_id:
                coord_to_id[coord] = next_id
                id_to_coord[next_id] = coord
                nid = next_id
                next_id += 1
            else:
                nid = coord_to_id[coord]

            node_map[(mode, node_id)] = nid

    return id_to_coord, node_map


# In[280]:


def creer_graphe_multimodal(graphes, noeuds_par_mode=None):
    graphe_multimodal = nx.MultiDiGraph()

    # Appliquer le filtre de nœuds valides dans la maille
    id_to_coord_multimodal, node_map = construire_index_coords(graphes, noeuds_par_mode)

    for mode, (G, id_to_coord_mode) in graphes.items():
        for node_id in G.nodes:
            key = (mode, node_id)
            if key not in node_map:
                continue

            nid = node_map[key]
            coord = id_to_coord_mode[node_id]
            graphe_multimodal.add_node(nid, coord=coord)

        for u, v, data in G.edges(data=True):
            key_u = (mode, u)
            key_v = (mode, v)
            if key_u not in node_map or key_v not in node_map:
                continue

            uid = node_map[key_u]
            vid = node_map[key_v]

            graphe_multimodal.add_edge(
                uid, vid,
                weight=data["weight"],
                distance=data["distance"],
                geometry=data.get("geometry"),
                mode=mode,
                route_id=data.get("route_id")
            )

    # Export
    chemin_graphe_pkl = os.path.join(exports_dir, "graphe_routier_multimodal.pkl")
    with open(chemin_graphe_pkl, "wb") as f:
        pickle.dump((graphe_multimodal, id_to_coord_multimodal), f)

    print(f"Pickle sauvegardé : {chemin_graphe_pkl}")
    print(f"Graphe multimodal : {graphe_multimodal.number_of_nodes()} nœuds / {graphe_multimodal.number_of_edges()} arêtes")

    return graphe_multimodal, id_to_coord_multimodal


# In[281]:


def creer_graphe_multimodal(graphes, noeuds_par_mode=None):
    graphe_multimodal = nx.MultiDiGraph()

    # Index global coord -> id et remapping
    id_to_coord_multimodal, node_map = construire_index_coords(graphes, noeuds_par_mode)

    for mode, (G, id_to_coord_mode) in graphes.items():
        for node_id in G.nodes:
            key = (mode, node_id)
            if key not in node_map:
                continue  # nœud hors des mailles
            nid = node_map[key]
            coord = id_to_coord_mode[node_id]
            graphe_multimodal.add_node(nid, coord=coord)

        for u, v, data in G.edges(data=True):
            key_u = (mode, u)
            key_v = (mode, v)

            # On ajoute l’arête si les deux extrémités ont un identifiant global
            if key_u in node_map and key_v in node_map:
                uid = node_map[key_u]
                vid = node_map[key_v]

                graphe_multimodal.add_edge(
                    uid, vid,
                    weight=data["weight"],
                    distance=data["distance"],
                    geometry=data.get("geometry"),
                    mode=mode,
                    route_id=data.get("route_id")
                )

    # Export
    chemin_graphe_pkl = os.path.join(exports_dir, "graphe_routier_multimodal.pkl")
    with open(chemin_graphe_pkl, "wb") as f:
        pickle.dump((graphe_multimodal, id_to_coord_multimodal), f)

    print(f"Pickle sauvegardé : {chemin_graphe_pkl}")
    print(f"Graphe multimodal : {graphe_multimodal.number_of_nodes()} nœuds / {graphe_multimodal.number_of_edges()} arêtes")

    return graphe_multimodal, id_to_coord_multimodal


# In[272]:


graphe_vl, id_to_coord_vl = charger_graphe("vl")
graphe_velos, id_to_coord_velos = charger_graphe("velos")
graphe_marche, id_to_coord_marche = charger_graphe("marche")
graphe_tram, id_to_coord_tram = charger_graphe("tram")
graphe_bus, id_to_coord_bus = charger_graphe("bus")

graphes = {
    "vl": (graphe_vl, id_to_coord_vl),
    "velos": (graphe_velos, id_to_coord_velos),
    "marche": (graphe_marche, id_to_coord_marche),
    "tram": (graphe_tram, id_to_coord_tram),
    "bus": (graphe_bus, id_to_coord_bus),
}


# In[288]:


carreaux = charger_fichier_parquet("maille_200m_epci", crs=3857)

# Construction du dictionnaire de noeuds autorisés
noeuds_par_mode = {
    mode: set(n for lst in carreaux[f"noeuds_{mode}"] if isinstance(lst, list) for n in lst)
    for mode in graphes
}
# Construction du graphe
graphe_multimodal, id_to_coord_multimodal = creer_graphe_multimodal(graphes, noeuds_par_mode=noeuds_par_mode)
# Rajoute aux carreaux de la maille une nouvelle colonne contenant tout les noeuds du graphe auxquels ils ont accès
carreaux = ajouter_noeuds_aux_carreaux("noeuds_multimodal", id_to_coord_multimodal, graphe_multimodal)


# In[289]:


# A MODIFIER POUR AFFICHER LES LIGNES DE DIFFERENTES COULEURS SELON LE MOYEN DE TRANSPORT UTILISE
# ET CORRIGER NOEUDS_MULTIMODAL NON COMPLET

# Code de test : sélectionne deux points centres du maillage au hasard, et calcule leur itinéraire sur le graphe
# Sélectionne aléatoirement deux carreaux différents avec au moins un noeud chacun
carreaux = charger_fichier_parquet("maille_200m_epci", crs=3857)
carreaux_valides = carreaux[carreaux["noeuds_multimodal"].apply(lambda x: isinstance(x, list) and len(x) > 0)].copy()
carreau_dep, carreau_arr = carreaux_valides.sample(2).to_dict("records")

# Listes de nœuds triés par distance au centre
noeuds_dep = carreau_dep["noeuds_multimodal"]
noeuds_arr = carreau_arr["noeuds_multimodal"]

# Nœuds centraux 
u, v = noeuds_dep[0], noeuds_arr[0]
print(f"Nœud central départ : {u}")
print(f"Nœud central arrivée : {v}")

# Calcul de l’itinéraire selon la nouvelle logique (par paires ordonnées)
itineraire = calculer_itineraire(noeuds_dep, noeuds_arr, graphe_multimodal, id_to_coord_multimodal)

if not itineraire or "error" in itineraire:
    print("Aucun itinéraire trouvé ou erreur rencontrée.")
else:
    try:
        print(f"Itinéraire calculé. Durée : {itineraire['time']:.1f} s, Distance : {itineraire['distance']:.1f} m")

        # Exports
        gdf_path = gpd.GeoDataFrame(geometry=[itineraire["geometry"]], crs="EPSG:3857")

        gdf_tiles = gpd.GeoDataFrame(
            [carreau_dep, carreau_arr],
            geometry=[carreau_dep["geometry"], carreau_arr["geometry"]],
            crs="EPSG:3857"
        )

        gdf_nodes = gpd.GeoDataFrame(
            geometry=[Point(id_to_coord[u]), Point(id_to_coord[v])],
            crs="EPSG:3857"
        )
        gdf_nodes["type"] = ["départ", "arrivée"]

    except (nx.NetworkXNoPath, nx.NodeNotFound):
        print("Chemin impossible entre les deux nœuds centraux.")
    except AssertionError as ae:
        print(str(ae))


# In[290]:


# 1. Géométrie du chemin : on reconstruit les coordonnées à partir des IDs
# Note : échoue si l'un des points est inaccessible via le réseau routier (typiquement : est dans une surface en eau)
line_geom = itineraire['geometry']
gdf_path = gpd.GeoDataFrame(geometry=[line_geom], crs="EPSG:3857")

# 2. Départ et arrivée
start_point = gpd.GeoDataFrame(geometry=[Point(itineraire['start'])], crs="EPSG:3857")
end_point = gpd.GeoDataFrame(geometry=[Point(itineraire['end'])], crs="EPSG:3857")

# 3. Charger fichiers
limites_epci = charger_fichier_parquet("limites_epci", crs=3857)
bd_topo_routes_epci = charger_fichier_parquet("bd_topo_routes_epci", crs=3857)

# 4. Affichage
fig, ax = plt.subplots(figsize=(10, 10))
limites_epci.to_crs(epsg=3857).plot(ax=ax, facecolor='none', edgecolor='black', alpha=0.5)
bd_topo_routes_epci.to_crs(epsg=3857).plot(ax=ax, linewidth=0.3, edgecolor='green', alpha=0.5)
gdf_path.to_crs(epsg=3857).plot(ax=ax, color='red', linewidth=2, label="Chemin")
start_point.to_crs(epsg=3857).plot(ax=ax, color='green', markersize=50, label="Départ")
end_point.to_crs(epsg=3857).plot(ax=ax, color='blue', markersize=50, label="Arrivée")

ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
ax.set_title("Itinéraire le plus rapide entre deux points", fontsize=14)
ax.axis('off')
ax.legend()
plt.tight_layout()
plt.show()
plt.close()


# In[291]:


# Code de test : mesure le temps d'exécution pour X itinéraires calculés pour le graphe multimodal
# Sélectionne 100 paires de carreaux avec au moins un nœud valide
carreaux = charger_fichier_parquet("maille_200m_epci")
carreaux_valides = carreaux[carreaux["noeuds_vl"].apply(lambda x: isinstance(x, list) and len(x) > 0)].copy()
paires_carreaux = carreaux_valides.sample(n=200, random_state=42)

# Extraction directe des listes de nœuds
carreaux_depart = paires_carreaux.iloc[:100]["noeuds_vl"].tolist()
carreaux_arrivee = paires_carreaux.iloc[100:]["noeuds_vl"].tolist()
paires = list(zip(carreaux_depart, carreaux_arrivee))

# Chronométrage
t0 = time.time()
resultats = Parallel(n_jobs=-1, backend="threading", verbose=10)(
    delayed(calculer_itineraire)(noeuds_dep, noeuds_arr, graphe_multimodal, id_to_coord_multimodal, max_combi=None)
    for noeuds_dep, noeuds_arr in paires
)
t1 = time.time()

# Statistiques
trajets_valides = [r for r in resultats if r and "error" not in r]
print(f"\n{len(trajets_valides)}/{len(paires)} itinéraires valides")
print(f"Temps total : {t1 - t0:.2f} secondes")

# (Optionnel) Supression des variables pour libérer de la ram
# del carreaux, carreaux_valides, paires_carreaux, carreaux_depart, carreaux_arrivee, paires, trajets_valides, graphe_multimodal, id_to_coord_multimodal


# ### 4.8. Création du "super-graphe" tram + marche
# ---

# In[260]:


# + récente
def combiner_graphe_tram_marche():
    # 1. Chargement des données
    graphe_tram, id_to_coord_tram = charger_graphe("tram")
    graphe_marche, id_to_coord_marche = charger_graphe("marche")
    gdf_arrets_tram = charger_fichier_parquet("arrets_tram_avec_donnees", crs=2154)
    df_stop_times = pd.read_csv(os.path.join(exports_dir, "gtfs_stop_times_tram.csv"))
    df_trips = pd.read_csv(os.path.join(exports_dir, "gtfs_trips_tram.csv"))

    # Préparation de l’index itinéraires uniques : (départ, arrivée, nbre d'arrêts)
    df_full = df_stop_times.merge(df_trips[['trip_id', 'route_id']], on='trip_id')
    df_grouped = df_full.groupby(['trip_id', 'route_id'])['stop_id'].agg(list).reset_index()
    df_grouped['itineraire_id'] = df_grouped['stop_id'].apply(lambda lst: (lst[0], lst[-1], len(lst)))

    itineraire_par_arret = df_grouped.explode('stop_id').groupby('stop_id')['itineraire_id'].apply(set).to_dict()

    # 2. Initialisation
    graphe_combine = nx.MultiDiGraph()
    id_to_coord_combine, coord_to_id = {}, {}
    next_id = 0

    def obtenir_ou_creer_id(coord):
        nonlocal next_id
        if coord not in coord_to_id:
            coord_to_id[coord] = next_id
            id_to_coord_combine[next_id] = coord
            next_id += 1
        return coord_to_id[coord]

    # 3. Ajout graphe marche
    for node, data in graphe_marche.nodes(data=True):
        coord = id_to_coord_marche[node]
        nid = obtenir_ou_creer_id(coord)
        graphe_combine.add_node(nid, **data)

    for u, v, data in graphe_marche.edges(data=True):
        cu, cv = id_to_coord_marche[u], id_to_coord_marche[v]
        nid_u, nid_v = coord_to_id[cu], coord_to_id[cv]
        graphe_combine.add_edge(nid_u, nid_v, **data)

    # 4. Ajout graphe tram
    for node, data in graphe_tram.nodes(data=True):
        coord = id_to_coord_tram[node]
        nid = obtenir_ou_creer_id(coord)
        graphe_combine.add_node(nid, **data)

    for u, v, key, data in graphe_tram.edges(keys=True, data=True):
        cu, cv = id_to_coord_tram[u], id_to_coord_tram[v]
        nid_u, nid_v = coord_to_id[cu], coord_to_id[cv]
        graphe_combine.add_edge(nid_u, nid_v, key=key, **data)

    # 5. Connexions marche -> tram (une arête par itinéraire)
    coords_marche = np.array(list(id_to_coord_marche.values()))
    tree_marche = cKDTree(coords_marche)
    rayon_connexion = 50  # m
    vitesse_marche_m_s = 3 / 3.6
    connexions = []

    for _, row in gdf_arrets_tram.iterrows():
        pt = row.geometry
        coord_tram = (pt.x, pt.y)
        id_tram = coord_to_id.get(coord_tram)
        if id_tram is None:
            continue

        stop_id = row['stop_id']
        itineraires = itineraire_par_arret.get(stop_id, set())
        attente = row.get("att_secondes_moy_semaine")
        if not itineraire_par_arret or pd.isna(attente):
            continue

        indices = tree_marche.query_ball_point(coord_tram, r=rayon_connexion)
        for i in indices:
            coord_marche = tuple(coords_marche[i])
            id_marche = coord_to_id[coord_marche]
            if id_marche == id_tram:
                continue

            dist = np.linalg.norm(np.array(coord_marche) - np.array(coord_tram))
            temps_marche = dist / vitesse_marche_m_s

            for itin in itineraires:
                poids_total = attente + temps_marche
                graphe_combine.add_edge(
                    id_marche, id_tram,
                    weight=poids_total,
                    distance=dist,
                    temps_attente=attente,
                    temps_marche=temps_marche,
                    itineraire_id=itin,
                    mode="connexion_tram"
                )
                connexions.append({
                    "from_id": id_marche,
                    "to_id": id_tram,
                    "weight": poids_total,
                    "temps_attente": attente,
                    "temps_marche": temps_marche,
                    "distance": dist,
                    "itineraire_id": str(itin),
                    "geometry": LineString([coord_marche, coord_tram])
                })

            # Connexion tram -> marche : 1 seule arête avec poids = temps_marche
            graphe_combine.add_edge(
                id_tram, id_marche,
                weight=temps_marche,
                distance=dist,
                mode="retour_marche"
            )

    # 6. Export
    pickle_path = os.path.join(exports_dir, "graphe_routier_tram_marche.pkl")
    with open(pickle_path, "wb") as f:
        pickle.dump((graphe_combine, id_to_coord_combine), f)

    print(f"\nGraphe combiné : {graphe_combine.number_of_nodes()} nœuds / {graphe_combine.number_of_edges()} arêtes")
    print(f"Export pickle : {pickle_path}")

    if connexions:
        gdf_connexions = gpd.GeoDataFrame(connexions, geometry="geometry", crs=2154)
        geojson_path = os.path.join(exports_dir, "connexions_tram_marche.geojson")
        gdf_connexions.to_file(geojson_path, driver="GeoJSON")
        print(f"Connexions exportées : {geojson_path}")

    return graphe_combine, id_to_coord_combine


# In[203]:


def combiner_graphe_tram_marche():
    """
    Combine les graphes tram et marche.
    Ajoute des arêtes bidirectionnelles entre les nœuds tram et les nœuds du graphe marche proches,
    avec poids = temps d'attente moyen + temps de marche (3 km/h).
    """
    # 1. Chargement des données
    graphe_tram, id_to_coord_tram = charger_graphe("tram")
    graphe_marche, id_to_coord_marche = charger_graphe("marche")
    gdf_arrets_tram = charger_fichier_parquet("arrets_tram_avec_donnees", crs=2154)

    # 2. Initialisation du graphe combiné
    graphe_combine = nx.MultiDiGraph()
    id_to_coord_combine = {}
    coord_to_id = {}
    next_id = 0

    def obtenir_ou_creer_id(coord):
        nonlocal next_id
        if coord not in coord_to_id:
            coord_to_id[coord] = next_id
            id_to_coord_combine[next_id] = coord
            next_id += 1
        return coord_to_id[coord]

    # 3. Ajout des nœuds et arêtes marche
    for node, data in graphe_marche.nodes(data=True):
        coord = id_to_coord_marche[node]
        nid = obtenir_ou_creer_id(coord)
        graphe_combine.add_node(nid, **data)

    for u, v, data in graphe_marche.edges(data=True):
        cu, cv = id_to_coord_marche[u], id_to_coord_marche[v]
        nid_u, nid_v = coord_to_id[cu], coord_to_id[cv]
        graphe_combine.add_edge(nid_u, nid_v, **data)

    # 4. Ajout des nœuds et arêtes tram
    for node, data in graphe_tram.nodes(data=True):
        coord = id_to_coord_tram[node]
        nid = obtenir_ou_creer_id(coord)
        graphe_combine.add_node(nid, **data)

    for u, v, key, data in graphe_tram.edges(keys=True, data=True):
        cu, cv = id_to_coord_tram[u], id_to_coord_tram[v]
        nid_u, nid_v = coord_to_id[cu], coord_to_id[cv]
        graphe_combine.add_edge(nid_u, nid_v, key=key, **data)

    # 5. Connexions tram ↔ marche
    coords_marche = np.array(list(id_to_coord_marche.values()))
    tree_marche = cKDTree(coords_marche)
    rayon_connexion = 50  # mètres
    vitesse_marche_m_s = 3 / 3.6  # ≈ 0.833 m/s

    connexions = []
    nb_connexions = 0

    for _, row in gdf_arrets_tram.iterrows():
        pt = row.geometry
        coord_tram = (pt.x, pt.y)
        id_tram = coord_to_id.get(coord_tram)
        if id_tram is None:
            continue

        attente = row.get("att_secondes_moy_semaine", None)
        if attente is None or np.isnan(attente):
            continue

        indices = tree_marche.query_ball_point(coord_tram, r=rayon_connexion)
        for i in indices:
            coord_marche = tuple(coords_marche[i])
            id_marche = coord_to_id[coord_marche]
            if id_marche == id_tram:
                continue

            dist = np.linalg.norm(np.array(coord_marche) - np.array(coord_tram))
            temps_marche = dist / vitesse_marche_m_s
            poids_total = attente + temps_marche

            for source, target in [(id_marche, id_tram), (id_tram, id_marche)]:
                graphe_combine.add_edge(
                    source, target,
                    weight=poids_total,
                    distance=dist,
                    temps_attente=attente,
                    temps_marche=temps_marche,
                    mode="connexion_tram"
                )
                connexions.append({
                    "from_id": source,
                    "to_id": target,
                    "weight": poids_total,
                    "temps_attente": attente,
                    "temps_marche": temps_marche,
                    "distance": dist,
                    "geometry": LineString([id_to_coord_combine[source], id_to_coord_combine[target]])
                })
                nb_connexions += 1

    # 6. Export du graphe combiné
    pickle_path = os.path.join(exports_dir, "graphe_routier_tram_marche.pkl")
    with open(pickle_path, "wb") as f:
        pickle.dump((graphe_combine, id_to_coord_combine), f)

    print(f"\nGraphe combiné : {graphe_combine.number_of_nodes()} nœuds / {graphe_combine.number_of_edges()} arêtes")
    print(f"Connexions tram ↔ marche ajoutées : {nb_connexions}")
    print(f"Export pickle : {pickle_path}")

    # 7. Export GeoDataFrame des connexions
    if connexions:
        gdf_connexions = gpd.GeoDataFrame(connexions, geometry="geometry", crs=2154)
        geojson_path = os.path.join(exports_dir, "connexions_tram_marche.geojson")
        gdf_connexions.to_file(geojson_path, driver="GeoJSON")
        print(f"Connexions exportées : {geojson_path}")
    else:
        print("Aucune connexion tram-marche exportée.")

    return graphe_combine, id_to_coord_combine


# In[204]:


def appel_creer_graphe_tram_marche():
    # 1. Construction du graphe
    combiner_graphe_tram_marche()

    # 2. Exporte les noeuds et arêtes du graphe pour vérification manuelle dans un SIG
    exporter_noeuds_aretes_graphe(nom_graphe="tram_marche")

    # 3. Rajoute aux carreaux une nouvelle colonne contenant tout les noeuds du graphe auxquels ils ont accès
    ajouter_noeuds_aux_carreaux(nom_graphe="tram_marche")

    # 4. Affiche l'accessibilité des carreaux au réseau
    afficher_accessibilite_reseau(nom_graphe = "tram_marche", nom_legende = "tram + piéton", export = True)

# Execution
appel_creer_graphe_tram_marche()


# In[211]:


# Calcule et affiche un itinéraire test
def afficher_itineraire_test_tram_marche():
    itineraire = calcul_itineraire_test(nom_graphe = "tram_marche", export=True)
    afficher_itineraire_test(itineraire, export = True)

# Exécution
afficher_itineraire_test_tram_marche()


# ### 4.9. Création du "super-graphe" bus + marche
# ---

# In[261]:


# + récente
def combiner_graphe_bus_marche():
    # 1. Chargement des données
    graphe_bus, id_to_coord_bus = charger_graphe("bus")
    graphe_marche, id_to_coord_marche = charger_graphe("marche")
    gdf_arrets_bus = charger_fichier_parquet("arrets_bus_avec_donnees", crs=2154)
    df_stop_times = pd.read_csv(os.path.join(exports_dir, "gtfs_stop_times_bus.csv"))
    df_trips = pd.read_csv(os.path.join(exports_dir, "gtfs_trips_bus.csv"))

    # Préparation de l’index itinéraires uniques : (départ, arrivée, nbre d'arrêts)
    df_full = df_stop_times.merge(df_trips[['trip_id', 'route_id']], on='trip_id')
    df_grouped = df_full.groupby(['trip_id', 'route_id'])['stop_id'].agg(list).reset_index()
    df_grouped['itineraire_id'] = df_grouped['stop_id'].apply(lambda lst: (lst[0], lst[-1], len(lst)))

    itineraire_par_arret = df_grouped.explode('stop_id').groupby('stop_id')['itineraire_id'].apply(set).to_dict()

    # 2. Initialisation
    graphe_combine = nx.MultiDiGraph()
    id_to_coord_combine, coord_to_id = {}, {}
    next_id = 0

    def obtenir_ou_creer_id(coord):
        nonlocal next_id
        if coord not in coord_to_id:
            coord_to_id[coord] = next_id
            id_to_coord_combine[next_id] = coord
            next_id += 1
        return coord_to_id[coord]

    # 3. Ajout graphe marche
    for node, data in graphe_marche.nodes(data=True):
        coord = id_to_coord_marche[node]
        nid = obtenir_ou_creer_id(coord)
        graphe_combine.add_node(nid, **data)

    for u, v, data in graphe_marche.edges(data=True):
        cu, cv = id_to_coord_marche[u], id_to_coord_marche[v]
        nid_u, nid_v = coord_to_id[cu], coord_to_id[cv]
        graphe_combine.add_edge(nid_u, nid_v, **data)

    # 4. Ajout graphe bus
    for node, data in graphe_bus.nodes(data=True):
        coord = id_to_coord_bus[node]
        nid = obtenir_ou_creer_id(coord)
        graphe_combine.add_node(nid, **data)

    for u, v, key, data in graphe_bus.edges(keys=True, data=True):
        cu, cv = id_to_coord_bus[u], id_to_coord_bus[v]
        nid_u, nid_v = coord_to_id[cu], coord_to_id[cv]
        graphe_combine.add_edge(nid_u, nid_v, key=key, **data)

    # 5. Connexions marche -> bus (une arête par itinéraire)
    coords_marche = np.array(list(id_to_coord_marche.values()))
    tree_marche = cKDTree(coords_marche)
    rayon_connexion = 50  # m
    vitesse_marche_m_s = 3 / 3.6
    connexions = []

    for _, row in gdf_arrets_bus.iterrows():
        pt = row.geometry
        coord_bus = (pt.x, pt.y)
        id_bus = coord_to_id.get(coord_bus)
        if id_bus is None:
            continue

        stop_id = row['stop_id']
        itineraires = itineraire_par_arret.get(stop_id, set())
        attente = row.get("att_secondes_moy_semaine")
        if not itineraire_par_arret or pd.isna(attente):
            continue

        indices = tree_marche.query_ball_point(coord_bus, r=rayon_connexion)
        for i in indices:
            coord_marche = tuple(coords_marche[i])
            id_marche = coord_to_id[coord_marche]
            if id_marche == id_bus:
                continue

            dist = np.linalg.norm(np.array(coord_marche) - np.array(coord_bus))
            temps_marche = dist / vitesse_marche_m_s

            for itin in itineraires:
                poids_total = attente + temps_marche
                graphe_combine.add_edge(
                    id_marche, id_bus,
                    weight=poids_total,
                    distance=dist,
                    temps_attente=attente,
                    temps_marche=temps_marche,
                    itineraire_id=itin,
                    mode="connexion_bus"
                )
                connexions.append({
                    "from_id": id_marche,
                    "to_id": id_bus,
                    "weight": poids_total,
                    "temps_attente": attente,
                    "temps_marche": temps_marche,
                    "distance": dist,
                    "itineraire_id": str(itin),
                    "geometry": LineString([coord_marche, coord_bus])
                })

            # Connexion bus -> marche : 1 seule arête avec poids = temps_marche
            graphe_combine.add_edge(
                id_bus, id_marche,
                weight=temps_marche,
                distance=dist,
                mode="retour_marche"
            )

    # 6. Export
    pickle_path = os.path.join(exports_dir, "graphe_routier_bus_marche.pkl")
    with open(pickle_path, "wb") as f:
        pickle.dump((graphe_combine, id_to_coord_combine), f)

    print(f"\nGraphe combiné : {graphe_combine.number_of_nodes()} nœuds / {graphe_combine.number_of_edges()} arêtes")
    print(f"Export pickle : {pickle_path}")

    if connexions:
        gdf_connexions = gpd.GeoDataFrame(connexions, geometry="geometry", crs=2154)
        geojson_path = os.path.join(exports_dir, "connexions_bus_marche.geojson")
        gdf_connexions.to_file(geojson_path, driver="GeoJSON")
        print(f"Connexions exportées : {geojson_path}")

    return graphe_combine, id_to_coord_combine


# In[262]:


def appel_creer_graphe_bus_marche():
    # 1. Construction du graphe
    combiner_graphe_bus_marche()

    # 2. Exporte les noeuds et arêtes du graphe pour vérification manuelle dans un SIG
    exporter_noeuds_aretes_graphe(nom_graphe="bus_marche")

    # 3. Rajoute aux carreaux une nouvelle colonne contenant tout les noeuds du graphe auxquels ils ont accès
    ajouter_noeuds_aux_carreaux(nom_graphe="bus_marche")

    # 4. Affiche l'accessibilité des carreaux au réseau
    afficher_accessibilite_reseau(nom_graphe = "bus_marche", nom_legende = "bus + piéton", export = True)

# Execution
appel_creer_graphe_bus_marche()


# In[51]:


# Calcule et affiche un itinéraire test
def afficher_itineraire_test_bus_marche():
    itineraire = calcul_itineraire_test(nom_graphe = "bus_marche", export=True)
    afficher_itineraire_test(itineraire, export = True)

# Exécution
afficher_itineraire_test_bus_marche()


# ## 5. Fonctions pour les calculs d'indicateurs
# ---

# ### 5.1. Calcul du ratio d'emplois / services accessibles en X minutes
# ---

# Il existe plusieurs manières de calculer l'accessibilité aux emplois / services depuis un point. On peut : 
# 1. Calculer tout les itinéraires possibles et, pour chaque carreau traversé par un itinéraire, récupérer le total des emplois présents dans la maille. Le problème, c'est que cette méthode est très imprécise, et sur-estime le nombre d'emplois / services réélement atteignables.
# 2. On récupère les emplois / services sur une zone tampon autour des trajets réalisables en X minutes. Cette méthode est plus précise, mais ne gère pas correctement les lieux géolocalisés assez loin des routes : elle mène à une sous-estimation.
# 
# On utilise une autre méthode, qui consiste à "plaquer" les emplois / services sur la route la plus proche.
# 
# ![schema_methodes_plaquage](images/schema_methodes_plaquage.png)

# In[59]:


# Cette fonction projette les emplois (données SIRENE) et les service (BPE) sur la route de la BD Topo
# la plus proche (équivalent de la fonction 'ligne la plus proche' dans QGIS)
def plaquer_emplois_services_routes():
    # 1. Charger les routes et forcer en 2D (une seule fois)
    routes = charger_fichier_parquet("bd_topo_routes_epci", crs=2154)
    routes = routes[routes.geometry.notnull() & routes.geometry.type.isin(["LineString", "MultiLineString"])].copy()
    routes["geometry"] = routes["geometry"].apply(force_2d)
    sindex = routes.sindex

    # 2. Fonction de projection générique
    def projeter_sur_routes(gdf):
        geometries_projetees = []
        for idx, point in enumerate(gdf.geometry):
            point_shapely = point if isinstance(point, shapely.geometry.base.BaseGeometry) else point.iloc[0]
            idx_possibles = list(sindex.nearest(point_shapely, return_all=False))
            nearest_dist = float("inf")
            nearest_proj = point_shapely

            for route_idx in idx_possibles:
                ligne = routes.geometry.iloc[route_idx]
                ligne_shapely = ligne if isinstance(ligne, shapely.geometry.base.BaseGeometry) else ligne.iloc[0]

                try:
                    proj = ligne_shapely.interpolate(ligne_shapely.project(point_shapely))
                    dist = point_shapely.distance(proj)
                    if dist < nearest_dist:
                        nearest_proj = proj
                        nearest_dist = dist
                except:
                    continue

            nearest_proj = force_2d(nearest_proj)
            geometries_projetees.append(nearest_proj if nearest_proj.geom_type == "Point" else point_shapely)

        gdf["geometry"] = gpd.GeoSeries(geometries_projetees, crs=2154)
        return gdf

    # 3. Traitement des établissements employeurs
    emplois = charger_fichier_gpkg("etablissements_emplois_epci", crs=2154)
    emplois = emplois[emplois.geometry.notnull() & (emplois.geometry.type == "Point")].copy()
    emplois_projetes = projeter_sur_routes(emplois)

    # 4. Traitement des services BPE
    bpe = charger_fichier_gpkg("insee_bpe_2023_epci", crs=2154)
    bpe = bpe[bpe.geometry.notnull() & (bpe.geometry.type == "Point")].copy()
    bpe_projetes = projeter_sur_routes(bpe)

    # 5. Export
    exporter_parquet(emplois_projetes, "etablissements_emplois_epci_plaque_routes")
    exporter_gpkg(emplois_projetes, "etablissements_emplois_epci_plaque_routes")
    exporter_parquet(bpe_projetes, "bpe_services_plaque_routes")
    exporter_gpkg(bpe_projetes, "bpe_services_plaque_routes")

    print(f"{len(emplois_projetes)} établissements et {len(bpe_projetes)} services plaqués avec succès.")

# Exécution
plaquer_emplois_services_routes()


# In[215]:


"""
FONCTION DE TEST : calcule un itinéraire à la fois. Permet de vérifier que la fonction
est correcte avant de lancer les calculs sur l'ensemble du maillage.

Calcule le nombre d'emplois et services accessibles en S secondes depuis un carreau
Le départ se fait du point le plus proche du centre de la maille et accessible
depuis le graphe entré en paramètre. Les emplois et services ont précédemment été
plaqués sur les routes de la BD Topo : pour les récupérer, on fait une petite jointure 
spatiale (1 mètres) autour des itinéraires calculés. 
"""
def calculer_nb_emplois_services_proches_test(nom_graphe = "vl", secondes=900):
    # 1. Charger les données
    carreaux = charger_fichier_parquet("maille_200m_epci", crs=2154)
    emplois = charger_fichier_parquet("etablissements_emplois_epci_plaque_routes", crs=2154)
    services = charger_fichier_parquet("bpe_services_plaque_routes", crs=2154)
    graphe, id_to_coord = charger_graphe(nom_graphe)
    noeuds = "noeuds_" + nom_graphe

    # 2. Sélection d'un carreau aléatoire et préparation des données
    carreau = carreaux.sample(1).iloc[0]
    idx_carreau = carreau.name
    noeuds_dep = carreau[noeuds][:1]  # Prend seulement le premier noeud

    # 3. Calcul des noeuds accessibles
    arrets_atteints = set()
    for noeud in noeuds_dep:
        try:
            lengths = nx.single_source_dijkstra_path_length(graphe, source=noeud, cutoff=secondes, weight="weight")
            arrets_atteints.update(lengths.keys())
        except Exception:
            continue

    # 4. Extraction des géométries des arêtes utilisées (optimisé)
    lignes_geom = []
    for u in arrets_atteints:
        for v in graphe.neighbors(u):
            if v in arrets_atteints and graphe.has_edge(u, v):
                edge_data = graphe.get_edge_data(u, v)

                # Gestion unifiée des différents types de graphes
                if isinstance(edge_data, dict):
                    # Cas MultiDiGraph
                    edges = edge_data.values() if isinstance(next(iter(edge_data.values())), dict) else [edge_data]
                else:
                    # Cas DiGraph simple
                    edges = [edge_data]

                for edge in edges:
                    geom = edge.get("geometry") if hasattr(edge, "get") else None
                    if geom:
                        lignes_geom.append(force_2d(geom))
                    else:
                        lignes_geom.append(LineString([id_to_coord[u], id_to_coord[v]]))

    # 5. Création de la zone tampon
    if not lignes_geom:
        print("Aucun itinéraire trouvé - zone tampon vide")
        zone_buffer = None
    else:
        union_geom = unary_union(lignes_geom)
        zone_buffer = gpd.GeoSeries([union_geom], crs=2154).buffer(1).geometry.union_all()

    # 6. Calcul des emplois et services accessibles
    nb_emplois = 0
    services_par_type = {}

    if zone_buffer is not None:
        # Services accessibles
        services_accessibles = services[services.intersects(zone_buffer)]
        services_par_type = services_accessibles["TYPEQU"].value_counts().to_dict()

        # Emplois accessibles
        emplois_accessibles = emplois[emplois.intersects(zone_buffer)]
        nb_emplois = emplois_accessibles["emplois"].sum()

    # 7. Préparation des données pour mise à jour (évite la fragmentation)
    data_update = {"nb_emplois_acces": nb_emplois}

    # Ajout des comptages par type de service
    if services_par_type:
        for code, count in services_par_type.items():
            data_update[f"nb_service_acces_{code}"] = count

    # Initialisation des colonnes manquantes à 0
    for code in services["TYPEQU"].unique():
        if f"nb_service_acces_{code}" not in data_update:
            data_update[f"nb_service_acces_{code}"] = 0

    # 8. Création d'un nouveau DataFrame pour éviter la fragmentation
    # Construction propre d'un GeoDataFrame sans fragmentation
    base = carreaux.loc[[idx_carreau]].copy()

    # Création d’un dictionnaire complet (colonnes existantes + data_update)
    colonnes_finales = {col: base.iloc[0][col] for col in base.columns}
    colonnes_finales.update(data_update)

    # Transformation en GeoDataFrame (en une passe)
    carreau_export = gpd.GeoDataFrame([colonnes_finales], crs=base.crs)

    # 9. Export du carreau de départ
    exporter_parquet(carreau_export, "accessibilite_test_carreau_depart")
    exporter_gpkg(carreau_export, "accessibilite_test_carreau_depart")

    # 10. Export des itinéraires
    if lignes_geom:
        gdf_itineraire = gpd.GeoDataFrame(geometry=lignes_geom, crs=2154)
        exporter_parquet(gdf_itineraire, "accessibilite_test_itineraires")
        exporter_gpkg(gdf_itineraire, "accessibilite_test_itineraires")
    else:
        gdf_itineraire = None

    print(f"Carreau {idx_carreau} : {nb_emplois} emplois accessibles et {sum(services_par_type.values())} services trouvés.")

# Exécution
calculer_nb_emplois_services_proches_test(nom_graphe = "velos", secondes=900)


# In[216]:


def afficher_nb_emplois_services_proches_test(export = False):
    # 1. Charger les données
    limites_epci = charger_fichier_parquet("limites_epci", crs=3857)
    bd_topo_routes_epci = charger_fichier_parquet("bd_topo_routes_epci", crs=3857)
    itineraires = charger_fichier_parquet("accessibilite_test_itineraires", crs=3857)
    carreaux = charger_fichier_parquet("accessibilite_test_carreau_depart", crs=3857)

    # 2. Affichage
    fig, ax = plt.subplots(figsize=(10, 10))

    limites_epci.plot(ax=ax, alpha=1, edgecolor='black', facecolor='none')
    carreaux.plot(ax=ax, edgecolor='black', facecolor='yellow', label="Carreau de départ")
    bd_topo_routes_epci.plot(ax=ax, linewidth=0.3, color='green', alpha=0.4)
    itineraires.plot(ax=ax, color='red', linewidth=0.3, label="Itinéraires")

    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)

    ax.set_title("Itinéraires accessibles en 15 minutes\net carreaux avec emplois atteints (velos)", fontsize=14)
    ax.axis('off')
    plt.tight_layout()

    # (Optionnel) Export PNG
    if export :
        images_export_dir = os.path.join(images_dir, "visualisation_itineraires_accessibles.png")
        plt.savefig(images_export_dir, dpi=300, bbox_inches='tight')
        print(f"Carte exportée vers : {images_export_dir}")

    plt.show()
    plt.close()

# Exécution
afficher_nb_emplois_services_proches_test(export = True)


# In[265]:


# Retourne pour chaque carreau les services et emplois accessibles en X secondes depuis un graphe
# Cette fonction est appellée en boucle, donc pas besoin de charger les graphes et autres données à chaque exécution de la fonction
def traiter_carreau(idx_carreau, carreaux, emplois, services, graphe, id_to_coord,
                    secondes, noeuds, nom_graphe,
                    nb_total_emplois, nb_total_services_par_type,
                    emplois_sindex, services_sindex, rayon=1.5):
    try:
        carreau = carreaux.loc[idx_carreau]
        noeuds_dep = carreau[noeuds][:1]
        if not noeuds_dep:
            return None

        # 1. Obtenir les noeuds atteignables
        arrets_atteints = set()
        for noeud in noeuds_dep:
            try:
                lengths = nx.single_source_dijkstra_path_length(
                    graphe, source=noeud, cutoff=secondes, weight="weight"
                )
                arrets_atteints.update(lengths.keys())
            except Exception:
                return None

        if not arrets_atteints:
            return None

        # 2. Construire un MultiPoint des coordonnées atteintes
        points_accessibles = [Point(id_to_coord[n]) for n in arrets_atteints if n in id_to_coord]
        if not points_accessibles:
            return None

        union_points = gpd.GeoSeries(points_accessibles, crs=2154)
        zone_buffer = union_points.buffer(rayon)
        zone = unary_union(zone_buffer)

        # 3. Requête spatiale
        emplois_candidats = emplois.iloc[list(emplois_sindex.intersection(zone.bounds))]
        services_candidats = services.iloc[list(services_sindex.intersection(zone.bounds))]

        emplois_accessibles = emplois_candidats[emplois_candidats.intersects(zone)]
        services_accessibles = services_candidats[services_candidats.intersects(zone)]

        # 4. Comptage
        nb_emplois = emplois_accessibles["emplois"].sum()
        services_par_domaine = {k: 0 for k in "ABCDEF"}
        for domain in services_accessibles["TYPEQU"]:
            prefix = str(domain)[:1]
            if prefix in services_par_domaine:
                services_par_domaine[prefix] += 1

        # 5. Calcul des ratios
        data_update = {
            f"{nom_graphe}_ratio_emplois": nb_emplois / nb_total_emplois if nb_total_emplois else 0
        }
        for domain_code in "ABCDEF":
            total = nb_total_services_par_type.get(domain_code, 1)
            ratio = services_par_domaine[domain_code] / total
            data_update[f"{nom_graphe}_ratio_services_dom_{domain_code}"] = ratio

        # 6. Résultat
        return pd.DataFrame({**{"idINSPIRE": [carreau["idINSPIRE"]]},
                              **{k: [v] for k, v in data_update.items()}})

    except Exception as e:
        print(f"[ERREUR] Carreau {idx_carreau} : {e}")
        return None


# In[266]:


# Ancienne version sans timeout
def calculer_ratio_emplois_services_proches(nom_graphe="vl", secondes=900, n_jobs=-1):
    # 1. Chargement du graphe
    vrai_nom_graphe = "vl" if nom_graphe == "autopartage" else nom_graphe
    graphe, id_to_coord = charger_graphe(vrai_nom_graphe)
    noeuds = "noeuds_" + vrai_nom_graphe

    # 2. Chargement des données
    carreaux = charger_fichier_parquet("maille_200m_avec_donnees", crs=2154)

    emplois = charger_fichier_parquet("etablissements_emplois_epci_plaque_routes", crs=2154)
    emplois = emplois.drop(columns=["siret", "activitePrincipaleEtablissement"], errors="ignore")

    services = charger_fichier_parquet("bpe_services_plaque_routes", crs=2154)
    services = services.drop(columns=["fid"], errors="ignore")

    # 3. Supprimer les anciennes colonnes si elles existent
    suffixes = ["emplois"] + [f"services_dom_{c}" for c in "ABCDEF"]
    colonnes_a_supprimer = [f"{nom_graphe}_ratio_{s}" for s in suffixes]
    carreaux = nettoyer_colonnes(carreaux, colonnes_a_supprimer)

    # 4. Totaux globaux dans la zone
    nb_total_emplois = emplois["emplois"].sum()
    nb_total_services_par_type = services["TYPEQU"].str[0].value_counts().to_dict()
    nb_total_services_par_type = {k: v for k, v in nb_total_services_par_type.items() if k in "ABCDEF"}

    # 5. Index spatial
    emplois_sindex = emplois.sindex
    services_sindex = services.sindex

    # 6. Sélection des index à traiter (selon le mode)
    if nom_graphe == "autopartage":
        idxs = carreaux[carreaux["nb_stations_autopartage"] > 0].index.tolist()
    else:
        idxs = carreaux.index.tolist()

    # (Optionnel) Pour tester les résulats avant de lancer le calcul sur l'ensemble de la maille
    #idxs = idxs[:2]

    print(f"Lancement du calcul sur {len(idxs)} carreaux avec {n_jobs} threads pour '{nom_graphe}'...")

    # 7. Traitement parallèle
    results = Parallel(n_jobs=n_jobs)(
        delayed(traiter_carreau)(
            idx, carreaux, emplois, services, graphe, id_to_coord,
            secondes, noeuds, vrai_nom_graphe,
            nb_total_emplois, nb_total_services_par_type,
            emplois_sindex, services_sindex
        )
        for idx in idxs
    )

    resultats_valides = [res for res in results if res is not None]
    if not resultats_valides:
        print("Aucun résultat valide.")
        return None

    df_resultats = pd.concat(resultats_valides)

    # 8. Renommage des colonnes si autopartage
    if nom_graphe == "autopartage":
        df_resultats.rename(
            columns=lambda c: c.replace("vl_ratio_", "autopartage_ratio_") if c.startswith("vl_ratio_") else c,
            inplace=True
        )

    # 9. Fusion avec la maille complète
    colonnes_a_ajouter = [col for col in df_resultats.columns if col.startswith(f"{nom_graphe}_ratio_")]
    carreaux = carreaux.merge(df_resultats[["idINSPIRE"] + colonnes_a_ajouter], on="idINSPIRE", how="left")

    # 10. Remplir les valeurs manquantes par -1 (non traités)
    for col in colonnes_a_ajouter:
        carreaux[col] = carreaux[col].fillna(-1)

    # 11. Export
    exporter_parquet(carreaux, "maille_200m_avec_donnees")
    exporter_gpkg(carreaux, "maille_200m_avec_donnees")
    exporter_geojson(carreaux, "maille_200m_avec_donnees")

    print(f"{len(df_resultats)} carreaux mis à jour avec les ratios d'accessibilité ({nom_graphe}).")

# (Test) Exécution
# calculer_ratio_emplois_services_proches(nom_graphe="bus_marche", secondes=900, n_jobs=-1)


# In[264]:


from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback

def calculer_ratio_emplois_services_proches(nom_graphe="vl", secondes=900, n_jobs=4, timeout_par_tache=300):
    import time

    vrai_nom_graphe = "vl" if nom_graphe == "autopartage" else nom_graphe
    graphe, id_to_coord = charger_graphe(vrai_nom_graphe)
    noeuds = "noeuds_" + vrai_nom_graphe

    carreaux = charger_fichier_parquet("maille_200m_epci", crs=2154)
    emplois = charger_fichier_parquet("etablissements_emplois_epci_plaque_routes", crs=2154).drop(columns=["siret", "activitePrincipaleEtablissement"], errors="ignore")
    services = charger_fichier_parquet("bpe_services_plaque_routes", crs=2154).drop(columns=["fid"], errors="ignore")

    suffixes = ["emplois"] + [f"services_dom_{c}" for c in "ABCDEF"]
    colonnes_a_supprimer = [f"{nom_graphe}_ratio_{s}" for s in suffixes]
    carreaux = nettoyer_colonnes(carreaux, colonnes_a_supprimer)

    nb_total_emplois = emplois["emplois"].sum()
    nb_total_services_par_type = services["TYPEQU"].str[0].value_counts().to_dict()
    nb_total_services_par_type = {k: v for k, v in nb_total_services_par_type.items() if k in "ABCDEF"}

    emplois_sindex = emplois.sindex
    services_sindex = services.sindex

    if nom_graphe == "autopartage":
        idxs = carreaux[carreaux["nb_stations_autopartage"] > 0].index.tolist()
    else:
        idxs = carreaux.index.tolist()

    # (Optionnel) Pour tester les résulats avant de lancer le calcul sur l'ensemble de la maille
    idxs = idxs[:2]

    print(f"Lancement du calcul sur {len(idxs)} carreaux avec {n_jobs} threads (timeout {timeout_par_tache}s)...")

    resultats = []
    erreurs = []

    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        futures = {
            executor.submit(
                traiter_carreau,
                idx, carreaux, emplois, services,
                graphe, id_to_coord, secondes, noeuds, vrai_nom_graphe,
                nb_total_emplois, nb_total_services_par_type,
                emplois_sindex, services_sindex
            ): idx for idx in idxs
        }

        for future in as_completed(futures):
            idx = futures[future]
            try:
                res = future.result(timeout=timeout_par_tache)
                if res is not None:
                    resultats.append(res)
            except Exception as e:
                erreurs.append((idx, str(e)))
                print(f"[TIMEOUT ou ERREUR] Carreau {idx} : {e}")

    if not resultats:
        print("Aucun résultat valide.")
        return

    df_resultats = pd.concat(resultats)

    if nom_graphe == "autopartage":
        df_resultats.rename(columns=lambda c: c.replace("vl_ratio_", "autopartage_ratio_") if c.startswith("vl_ratio_") else c, inplace=True)

    colonnes_a_ajouter = [col for col in df_resultats.columns if col.startswith(f"{nom_graphe}_ratio_")]
    carreaux = carreaux.merge(df_resultats[["idINSPIRE"] + colonnes_a_ajouter], on="idINSPIRE", how="left")

    for col in colonnes_a_ajouter:
        carreaux[col] = carreaux[col].fillna(-1)

    exporter_parquet(carreaux, "maille_200m_epci")
    exporter_gpkg(carreaux, "maille_200m_epci")
    exporter_geojson(carreaux, "maille_200m_epci")

    print(f"{len(df_resultats)} carreaux traités. {len(erreurs)} erreurs/timeouts.")

# (Test) Exécution
calculer_ratio_emplois_services_proches(nom_graphe="bus_marche", secondes=900, n_jobs=12)


# In[22]:


# Retourne pour chaque carreau les services et emplois accessibles en X secondes depuis un graphe
# Cette fonction est appellée en boucle, donc pas besoin de charger les graphes et autres données à chaque exécution de la fonction
def traiter_carreau(idx_carreau, carreaux, emplois, services, graphe, id_to_coord, secondes, noeuds, nom_graphe, nb_total_emplois, nb_total_services_par_type):
    try:
        carreau = carreaux.loc[idx_carreau]
        noeuds_dep = carreau[noeuds][:1]

        arrets_atteints = set()
        for noeud in noeuds_dep:
            try:
                lengths = nx.single_source_dijkstra_path_length(graphe, source=noeud, cutoff=secondes, weight="weight")
                arrets_atteints.update(lengths.keys())
            except Exception:
                return None

        lignes_geom = []
        for u in arrets_atteints:
            for v in graphe.neighbors(u):
                if v in arrets_atteints and graphe.has_edge(u, v):
                    edge_data = graphe.get_edge_data(u, v)
                    if isinstance(edge_data, dict):
                        edges = edge_data.values() if isinstance(next(iter(edge_data.values())), dict) else [edge_data]
                    else:
                        edges = [edge_data]
                    for edge in edges:
                        geom = edge.get("geometry") if hasattr(edge, "get") else None
                        if geom:
                            lignes_geom.append(LineString(geom.coords[:2]))
                        else:
                            lignes_geom.append(LineString([id_to_coord[u], id_to_coord[v]]))

        if not lignes_geom:
            return None

        union_geom = unary_union(lignes_geom)
        zone_buffer = gpd.GeoSeries([union_geom], crs=3857).buffer(1).geometry.union_all()

        nb_emplois = 0
        services_par_domaine = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0, 'F': 0}
        if zone_buffer is not None:
            services_accessibles = services[services.intersects(zone_buffer)]
            for domain in services_accessibles["TYPEQU"]:
                prefix = str(domain)[:1]
                if prefix in services_par_domaine:
                    services_par_domaine[prefix] += 1
            emplois_accessibles = emplois[emplois.intersects(zone_buffer)]
            nb_emplois = emplois_accessibles["emplois"].sum()

        # === RATIO de chaque domaine (évite /0 avec .get(x,1))
        data_update = {
            f"{nom_graphe}_ratio_emplois": nb_emplois / nb_total_emplois if nb_total_emplois else 0
        }
        for domain_code in services_par_domaine:
            total_services_type = nb_total_services_par_type.get(domain_code, 1)
            ratio = services_par_domaine[domain_code] / total_services_type
            data_update[f"{nom_graphe}_ratio_services_dom_{domain_code}"] = ratio

        carreau_result = carreaux.loc[[idx_carreau]].copy()
        for col, val in data_update.items():
            carreau_result[col] = val

        return carreau_result

    except Exception as e:
        print(f"[ERREUR] Carreau {idx_carreau} : {e}")
        return None


# In[ ]:


def calculer_ratio_emplois_services_proches(nom_graphe="vl", secondes=900, n_jobs=-1):
    graphe, id_to_coord = charger_graphe(nom_graphe)
    noeuds = "noeuds_" + nom_graphe

    carreaux = charger_fichier_parquet("maille_200m_avec_donnees", crs=2154)
    emplois = charger_fichier_parquet("etablissements_emplois_epci_plaque_routes", crs=2154)
    services = charger_fichier_parquet("bpe_services_plaque_routes", crs=2154)

    # === Calcul des valeurs totales dans l’EPCI ===
    nb_total_emplois = emplois["emplois"].sum()
    nb_total_services_par_type = services["TYPEQU"].str[0].value_counts().to_dict()
    nb_total_services_par_type = {k: v for k, v in nb_total_services_par_type.items() if k in "ABCDEF"}

    idxs = carreaux.index.tolist()
    #idxs = carreaux.index.tolist()[:2] # Pour tester la fonction avant de l'exécuter sur les tout les carreaux

    print(f"Lancement du calcul sur {len(idxs)} carreaux avec {n_jobs} threads pour le graphe '{nom_graphe}'...")

    results = Parallel(n_jobs=n_jobs)(
        delayed(traiter_carreau)(
            idx, carreaux, emplois, services, graphe, id_to_coord,
            secondes, noeuds, nom_graphe,
            nb_total_emplois, nb_total_services_par_type
        )
        for idx in idxs
    )

    resultats_valides = [res for res in results if res is not None]
    if not resultats_valides:
        print("Aucun résultat valide.")
        return None

    df_final = pd.concat(resultats_valides)
    gdf_final = gpd.GeoDataFrame(df_final, geometry="geometry", crs=2154)

    exporter_parquet(gdf_final, f"accessibilite_ratios_emplois_services_{nom_graphe}")
    exporter_gpkg(gdf_final, f"accessibilite_ratios_emplois_services_{nom_graphe}")
    exporter_geojson(gdf_final, f"accessibilite_ratios_emplois_services_{nom_graphe}")

    print(f"{len(gdf_final)} carreaux traités avec succès pour le mode '{nom_graphe}'.")

# Exécution
calculer_ratio_emplois_services_proches(nom_graphe="bus_marche", secondes=900, n_jobs=10)


# ### 5.2. Calcul du ratio hauteur des bâtiments / largeur des rues
# --- 
# La BD Topo possède bien un champ 'LARGEUR' pour ses rues, mais il reste assez incomplet : 32.2% des rues n'ont pas cet attribut. De plus, de nombreuses
# rues sont indiquées comme ayant une largeur égale à 0. Le Cerema a développé une méthodologie pour déterminer la largeur d'une route : https://piece-jointe-carto.developpement-durable.gouv.fr/REG074B/FONCIER_SOL/N_OCCUPATION_SOL/L_EMPRISE_ROUTE_R74/Fiche1-7-1.pdf, mais son application dans notre cas n'est pas concluant. De nombreux tronçons n'ont pas d'attribut 'NB_VOIES', dont on ne peux leur attribuer une largeur de cette manière. 
# 
# Idéalement, on aurait une seule règle simple pour attribuer la largeur à toutes les rues. J'ai tenté de déterminé toutes les largeur en calculant, pour tout les tronçons francais, la moyenne de leur 
# largeur selon les attributs suivants : "NATURE", "IMPORTANCE", "NB_VOIES". Cela pose deux problèmes : les rues du même type ont toutes la même largeur, et une grande portion de routes n'avaient toujours pas de largeur attribuée.
# 
# Finalement, la largeur des rues est donc conservée pour celle qui possèdent déjà cet attribut, et des règles manuelles les rajoutent pour les autres :
# * Les sentiers prennent la valeur 1.5 (doc : http://bdtopoexplorer.ign.fr/?id_theme=72&id_classe=77#attribute_value_611 : "Voie étroite ne permettant pas le passage de véhicules automobiles et de largeur inférieure à 2 m. Un sentier peut être revêtu ou non.")
# * Les chemins prennent la valeur 2 (http://bdtopoexplorer.ign.fr/?id_theme=72&id_classe=77#attribute_value_606 : "Les chemins sont prévus pour la circulation de véhicules ou d’engins d’exploitation. Ils ne sont pas forcément carrossables pour tous les véhicules et par tous les temps (voir aussi la Nature="Route empierrée"").
# * Les pistes cyclables à double sens prennent la valeur 3, 2 pour les sens unique (doc : https://www.cerema.fr/system/files/documents/2019/08/4_-_boite_a_outils_amgts_cyclables.pdf : "Largeur préconisée : 2m pour piste unidirectionnelle, 3m pour piste bidirectionnelle")
# * Il semble ne pas y avoir de consensus pour la largueur des routes empierrées (doc : "	Route sommairement revêtue (pas de revêtement de surface ou revêtement très dégradé), mais permettant la circulation de véhicules automobiles de tourisme par tous les temps."), leur largeur a été mise à 2.5.
# * Tout les autres cas ont également été mis à 2.5
# 
# Documentation : 
# * https://web.mit.edu/nature/archive/student_projects/2009/jcalamia/Frame/03_urbastreetcanyon.html
# * https://pub-townofmammothlakes.escribemeetings.com/filestream.ashx?documentid=10069
# 
# ![schema_ratio_rues_HL](images/schema_ratio_rues_HL.png)

# In[222]:


"""
Méthode "classique" pour calculer le ratio. On récupère tout les bâtiments
dans une zone tampon de 50 mètres (sans extension aux extrémités) autour 
de des tronçons de route.
Problème : les bâtiments peuvent ne se trouver que d'un seul côté de la rue,
laissant l'autre côté complètement libre.
"""
def calculer_ratio_rues():
    routes = charger_fichier_parquet("bd_topo_routes_epci", crs=2154)
    batiments = charger_fichier_parquet("bd_topo_batiments_epci", crs=2154)

    # 1. Les bâtiments avec une hauteur inconnue sont ignorés
    batiments = batiments[batiments["HAUTEUR"].notnull()]
    batiments = batiments[batiments["HAUTEUR"] > 0]

    # 2. Création de la zone tampon de 50 mètres (25 à gauche, 25 à droite) autour des routes
    routes["buffer_50m"] = routes.geometry.buffer(25, cap_style=2, join_style=2)

    # 3. Rcupéreration des bâtiments dans chaque zone tampon
    buffers_gdf = gpd.GeoDataFrame(routes[["buffer_50m"]], geometry="buffer_50m", crs=2154)
    batiments = batiments.set_geometry("geometry")
    routes = routes.set_geometry("buffer_50m")
    jonction = gpd.sjoin(batiments, routes, how="inner", predicate="intersects")

    # 4. Moyenne des hauteurs par tronçon
    moyennes_hauteur = jonction.groupby(jonction.index_right)["HAUTEUR"].mean()

    # 5. Attribution des valeurs finales
    routes["hauteur_moyenne"] = routes.index.map(moyennes_hauteur)
    routes["ratio_hauteur_largeur"] = routes["hauteur_moyenne"] / routes["largeur_calculee"]

    # 6. On revient à la géométrie initiale
    routes = routes.set_geometry("geometry")
    routes = routes.drop(columns=["buffer_50m"])

    # 7. Export
    exporter_parquet(routes, "bd_topo_routes_epci_ratio_hauteur_largeur")
    exporter_gpkg(routes, "bd_topo_routes_epci_ratio_hauteur_largeur")


# In[90]:


"""
Méthode personnelle pour calculer le ratio. On créé 2 zones tampons de 25 m
chacunes, à gauche et à droite de la route. Les bâtiments y sont récupérés,
mais l'indicateur n'est calculé que si il y a au moins un bâtiment de chaque
côté.

AMELIORATION POSSIBLE : prendre en compte la superficie des bâtiments ? Evite
que la hauteur d'un petit bâtiment influe autant qu'un grand
"""
def calculer_ratio_rues():
    # 1. Charger les données
    routes = charger_fichier_parquet("bd_topo_routes_epci", crs=2154)
    batiments = charger_fichier_parquet("bd_topo_batiments_epci", crs=2154)

    # 2. Nettoyage des bâtiments
    batiments = batiments[batiments["HAUTEUR"].notnull() & (batiments["HAUTEUR"] > 0)]

    # 3. Forcer géométries en 2D
    def force_2d(geom):
        if geom is None:
            return None
        elif isinstance(geom, LineString):
            return LineString([(x, y) for x, y, *_ in geom.coords])
        elif isinstance(geom, MultiLineString):
            lignes_2d = [LineString([(x, y) for x, y, *_ in line.coords]) for line in geom.geoms]
            return MultiLineString(lignes_2d)
        else:
            return geom

    routes["geometry"] = routes.geometry.apply(force_2d)

    # 4. Création des zones tampon
    def create_side_polygons(geom, largeur=25):
        if not isinstance(geom, LineString) or len(geom.coords) < 2:
            return None, None

        # Créer un buffer complet autour de la ligne
        full_buffer = geom.buffer(largeur, cap_style=1, join_style=2)

        # Séparer en deux polygones gauche/droit en utilisant la ligne originale comme séparateur
        # Méthode alternative pour diviser le buffer
        left_poly = None
        right_poly = None

        try:
            # Créer une ligne décalée à gauche pour servir de limite
            left_line = geom.parallel_offset(largeur, 'left')
            # Créer un polygone temporaire avec la ligne originale et la ligne décalée
            left_poly = Polygon(list(geom.coords) + list(left_line.coords)[::-1])

            # Même chose pour le côté droit
            right_line = geom.parallel_offset(largeur, 'right')
            right_poly = Polygon(list(geom.coords) + list(right_line.coords)[::-1])
        except:
            # En cas d'erreur (géométrie complexe), utiliser une méthode simplifiée
            left_poly = full_buffer  # Solution temporaire
            right_poly = full_buffer

        return left_poly, right_poly

    # Application aux géométries
    buffers_g, buffers_d = [], []
    for geom in routes.geometry:
        g, d = create_side_polygons(geom)
        buffers_g.append(g)
        buffers_d.append(d)

    routes["buffer_gauche"] = buffers_g
    routes["buffer_droit"] = buffers_d

    gdf_gauche = gpd.GeoDataFrame(routes[["buffer_gauche"]].copy(), geometry="buffer_gauche", crs=2154)
    gdf_droit = gpd.GeoDataFrame(routes[["buffer_droit"]].copy(), geometry="buffer_droit", crs=2154)
    gdf_gauche["id_route"] = routes.index
    gdf_droit["id_route"] = routes.index

    # (Optionnel) Export buffers pour vérification manuelle dans QGIS
    # exporter_gpkg(gdf_gauche, "buffers_routes_epci_gauche")
    # exporter_gpkg(gdf_droit, "buffers_routes_epci_droit")

    # 5. Jointures spatiales
    batiments = batiments.set_geometry("geometry")
    print("Jointures spatiales gauche et droite...")
    jonction_g = gpd.sjoin(batiments, gdf_gauche, how="inner", predicate="intersects")
    jonction_d = gpd.sjoin(batiments, gdf_droit, how="inner", predicate="intersects")

    # 6. Moyennes de hauteur
    hauteur_g = jonction_g.groupby("id_route")["HAUTEUR"].mean()
    hauteur_d = jonction_d.groupby("id_route")["HAUTEUR"].mean()

    # 7. Filtrage des routes avec bâtiments des deux côtés
    index_commun = hauteur_g.index.intersection(hauteur_d.index)
    routes_valides = routes.loc[index_commun].copy()
    routes_valides["hauteur_moyenne"] = (hauteur_g.loc[index_commun] + hauteur_d.loc[index_commun]) / 2
    routes_valides["ratio_hauteur_largeur"] = routes_valides["hauteur_moyenne"] / routes_valides["largeur_calculee"]

    # 8. Nettoyage & export
    routes_valides = routes_valides.set_geometry("geometry")
    routes_valides.drop(columns=["buffer_gauche", "buffer_droit"], inplace=True)

    exporter_parquet(routes_valides, "bd_topo_routes_epci_ratio_hauteur_largeur")
    exporter_gpkg(routes_valides, "bd_topo_routes_epci_ratio_hauteur_largeur")

    print(f"Export terminé. {len(routes_valides)} tronçons ont un ratio hauteur/largeur valide.")


# ### 5.3. Calcul de la pente des rues
# ---
# 
# La pente maximale est limitée à 100%, et la pente minimale de -100 %.

# In[257]:


# FONCTION DE TEST, calcule la pente d'un seul tronçon de route
def calculer_pentes_bd_topo_test(troncon_id=None):
    # 1. Chargement des données
    routes = charger_fichier_parquet("bd_topo_routes_epci", crs=2154)

    # 2. Sélection du tronçon
    if troncon_id is None:
        troncon = routes[routes.geometry.type == 'LineString'].iloc[0]
    else:
        troncon = routes.loc[troncon_id]

    print(f"\n=== Tronçon {troncon.name or troncon_id} ===")
    print(f"Nature: {troncon.get('nature', 'Non spécifiée')}")
    print(f"Longueur: {troncon.geometry.length:.1f} m")

    # 3. Extraction des coordonnées 3D
    geom = troncon.geometry

    if not isinstance(geom, LineString):
        print("Erreur: La géométrie n'est pas une LineString")
        return None

    coords = list(geom.coords)

    if len(coords) < 2:
        print("Erreur: Tronçon trop court")
        return None

    # Vérification présence de l'altitude (Z)
    if len(coords[0]) < 3:
        print("Avertissement: Pas de données altimétriques (Z) dans ce tronçon")
        return None

    # 4. Calcul des statistiques altimétriques
    altitudes = [z for x, y, z in coords]
    z_min, z_max = min(altitudes), max(altitudes)
    denivele = z_max - z_min

    print("\nAltitudes extraites:")
    print(f"Points: {len(coords)}")
    print(f"Altitude min: {z_min:.1f} m, max: {z_max:.1f} m")
    print(f"Dénivelé: {denivele:.1f} m")

    # 5. Calcul de la pente moyenne (optionnel)
    distances = []
    pentes = []

    for i in range(len(coords)-1):
        x1, y1, z1 = coords[i]
        x2, y2, z2 = coords[i+1]
        distance_2d = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        if distance_2d > 0:
            pente = 100 * (z2 - z1) / distance_2d
            distances.append(distance_2d)
            pentes.append(pente)

    if distances:
        pente_moy = np.average(pentes, weights=distances)
        print(f"\nPente moyenne: {pente_moy:.1f}%")
    else:
        pente_moy = None
        print("\nImpossible de calculer la pente")

    # 6. Retour des résultats
    resultats = {
        'coordonnees_3d': coords,
        'altitudes': altitudes,
        'statistiques': {
            'z_min': z_min,
            'z_max': z_max,
            'denivele': denivele,
            'pente_moyenne': pente_moy
        },
        'metadata': {
            'troncon_id': troncon_id,
            'nature': troncon.get('nature'),
            'longueur_2d': troncon.geometry.length
        }
    }

    return resultats

# Exécution
resultat = calculer_pentes_bd_topo_test(troncon_id=None)

if resultat:
    print("\nExtraction réussie")
    print(f"Premier point (x,y,z): {resultat['coordonnees_3d'][0]}")
    print(f"Dernier point (x,y,z): {resultat['coordonnees_3d'][-1]}")

else:
    print("\nÉchec de l'extraction des altitudes")


# In[240]:


# Calcule les pentes pour tous les tronçons de route à partir des données altimétriques
# de la BD Topo et agrège par carreaux.
def calculer_pentes_bd_topo():
    # 1. Chargement des données
    routes = charger_fichier_parquet("bd_topo_routes_epci", crs=2154)
    carreaux = charger_fichier_parquet("maille_200m_avec_donnees", crs=2154)

    # 2. Initialisation des colonnes et compteurs
    routes["pente_pourcentage"] = np.nan
    routes["denivele"] = np.nan
    routes["altitude_moyenne"] = np.nan
    routes["a_altitude"] = False  # Booléen, indique si la route possède une altitude
    carreaux["pente_moyenne"] = np.nan

    # 3.. Compteurs pour statistiques
    total_troncons = len(routes)
    sans_geom_valide = 0
    sans_donnees_z = 0
    avec_donnees_z = 0
    troncons_trop_courts = 0
    geometries_non_linestring = 0

    # 4. Fonction de calcul pour un seul tronçon
    def calcul_pente_troncon(geom):
        nonlocal sans_donnees_z, avec_donnees_z, troncons_trop_courts, geometries_non_linestring

        if not isinstance(geom, LineString):
            geometries_non_linestring += 1
            return np.nan, np.nan, np.nan, False

        coords = list(geom.coords)
        if len(coords) < 2:
            troncons_trop_courts += 1
            return np.nan, np.nan, np.nan, False

        # 4.1. Vérification présence de Z
        if len(coords[0]) < 3:
            sans_donnees_z += 1
            return np.nan, np.nan, np.nan, False

        # 4.2. Extraction altitudes et calculs
        avec_donnees_z += 1
        altitudes = np.array([z for _, _, z in coords])
        denivele = np.max(altitudes) - np.min(altitudes)
        altitude_moy = np.mean(altitudes)

        # 4.3. Calcul pente moyenne pondérée
        distances = []
        pentes = []
        for i in range(len(coords)-1):
            x1, y1, z1 = coords[i]
            x2, y2, z2 = coords[i+1]
            dist = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            if dist > 0:
                pente = 100 * (z2 - z1) / dist
                distances.append(dist)
                pentes.append(pente)

        pente_moy = np.average(pentes, weights=distances) if distances else np.nan
        return pente_moy, denivele, altitude_moy, True

    # 5. Calcul pour tous les tronçons avec barre de progression
    print("Calcul des pentes pour tous les tronçons...")
    for idx, route in routes.iterrows():
        if route.geometry is None:
            sans_geom_valide += 1
            continue

        pente, denivele, alt_moy, a_altitude = calcul_pente_troncon(route.geometry)
        routes.at[idx, "pente_pourcentage"] = pente
        routes.at[idx, "denivele"] = denivele
        routes.at[idx, "altitude_moyenne"] = alt_moy
        routes.at[idx, "a_altitude"] = a_altitude

    if avec_donnees_z > 0:
        print("\nRépartition des pentes calculées (en %):")
        pentes_valides = routes["pente_pourcentage"].dropna()
        print(pentes_valides.describe(percentiles=[0.25, 0.5, 0.75, 0.95]))

    print("\nAgrégation par carreaux...")
    sindex = routes.sindex if hasattr(routes, 'sindex') else routes.sindex

    for idx, carreau in carreaux.iterrows():
        try:
            # 6. Sélection des tronçons intersectant le carreau
            bounds = carreau.geometry.bounds
            possible_matches = list(sindex.intersection(bounds))
            routes_carreau = routes.iloc[possible_matches]
            routes_carreau = routes_carreau[routes_carreau.intersects(carreau.geometry)]

            if not routes_carreau.empty:
                # 7. Calcul de la longueur si la colonne n'existe pas
                if 'longueur_calculee' not in routes_carreau.columns:
                    routes_carreau['longueur_utilisee'] = routes_carreau.geometry.length
                else:
                    routes_carreau['longueur_utilisee'] = routes_carreau['longueur_calculee'].fillna(routes_carreau.geometry.length)

                # 8. Filtre des valeurs valides
                valid = routes_carreau.dropna(subset=['pente_pourcentage', 'longueur_utilisee'])

                if len(valid) > 0:
                    total_len = valid['longueur_utilisee'].sum()
                    if total_len > 0:
                        weighted_pente = (valid['pente_pourcentage'] * valid['longueur_utilisee']).sum() / total_len
                        carreaux.at[idx, "pente_moyenne"] = weighted_pente
                        continue

            # Si on arrive ici, c'est qu'on n'a pas pu calculer
            carreaux.at[idx, "pente_moyenne"] = np.nan

        except Exception as e:
            print(f"\nErreur carreau {idx}: {str(e)[:200]}")
            carreaux.at[idx, "pente_moyenne"] = np.nan
            continue

    # 9. Nettoyage et formatage des pentes moyennes dans les carreaux
    print("\nNettoyage des valeurs de pente...")

    # Arrondir à 2 décimales
    carreaux["pente_moyenne"] = carreaux["pente_moyenne"].round(2)

    # Remplacer les NaN par -1
    carreaux["pente_moyenne"] = carreaux["pente_moyenne"].fillna(-1)

    # Limiter les valeurs entre -20 et 20
    carreaux["pente_moyenne"] = carreaux["pente_moyenne"].clip(lower=-20, upper=20)

    # 10. Calcul de la pente moyenne absolue
    print("\nCalcul de la pente moyenne absolue...")
    carreaux["pente_moyenne_absolue"] = (
        carreaux["pente_moyenne"].replace(-1, np.nan).abs().round(2).fillna(-1)
    )

    # 11. Affichage des statistiques finales
    print("\n--- Statistiques Finales ---")
    print(f"Total tronçons analysés: {total_troncons}")
    print(f"- Tronçons avec données Z: {avec_donnees_z} ({avec_donnees_z/total_troncons:.1%})")
    print(f"- Tronçons sans données Z: {sans_donnees_z} ({sans_donnees_z/total_troncons:.1%})")
    print(f"- Tronçons trop courts (<2 points): {troncons_trop_courts}")
    print(f"- Géométries non LineString: {geometries_non_linestring}")
    print(f"- Sans géométrie valide: {sans_geom_valide}")

    stats_pentes = carreaux["pente_moyenne"].describe()
    print("\nDistribution des pentes moyennes par carreau:")
    print(f"- Min: {stats_pentes['min']:.2f}%")
    print(f"- Max: {stats_pentes['max']:.2f}%")
    print(f"- Moyenne: {stats_pentes['mean']:.2f}%")
    print(f"- Médiane: {stats_pentes['50%']:.2f}%")
    print(f"- Carreaux sans données (valeur -1): {(carreaux['pente_moyenne'] == -1).sum()}")

    # 12. Export des résultats
    print("\nExport des résultats...")
    exporter_parquet(routes, "bd_topo_routes_epci")
    exporter_gpkg(routes, "bd_topo_routes_epci")
    exporter_parquet(carreaux, "maille_200m_avec_donnees")
    exporter_gpkg(carreaux, "maille_200m_avec_donnees")


# ### 5.4. Désagrégation des données IRIS au carroyage de 200m
# ---
# Pour désagréger les données et obtenir la population depuis l'IRIS vers nos carreaux de 200 m², la population est répartie proportionnellement à l'aire des bâtiments (issus de la BD Topo) sur les mailles de 200 m. Si le carreau est superposé sur plusieurs IRIS, on fait une moyenne pondérée selon le % de surface du carreau dans chaque IRIS. 
# 
# Quant aux données sociaux-economiques (comme le revenu médian disponible), elles sont simplement copiées depuis l'IRIS où est contenu le carreau. Evidemment, étant donné que l'on a plus la même population en désagrégeant au carreau, les données socio-économiques devraient logiquement être recalculées en conséquence. Comme c'est impossible à moins de posséder les revenus de toute la population, il s'agit ici d'une estimation.

# In[47]:


def desagreger_donnees_vers_carreaux():
    # 1. Chargement des données
    carreaux_200m = charger_fichier_parquet("maille_200m_epci", crs=2154).copy()
    iris_complet = charger_fichier_parquet("iris_epci_donnees", crs=2154).copy()
    carreaux_existants = charger_fichier_parquet("maille_200m_avec_donnees", crs=2154)
    batiments = charger_fichier_parquet("bd_topo_batiments_epci", crs=2154)

    # 9. Supprimer les colonnes déjà présentes et de jointure
    colonnes_a_supprimer = ['population_estimee_x', 'surface_batie_ponderee_x',
       'P21_ACTOCC1564_estime_x', 'P21_SAL15P_estime_x',
       'P21_NSAL15P_estime_x', 'C21_ACTOCC1564_CS3_estime_x',
       'C21_ACTOCC1564_CS4_estime_x', 'C21_ACTOCC1564_CS5_estime_x',
       'C21_ACTOCC1564_CS6_estime_x', 'C21_ACTOCC15P_estime_x',
       'C21_ACTOCC15P_PAS_estime_x', 'C21_ACTOCC15P_MAR_estime_x',
       'C21_ACTOCC15P_VELO_estime_x', 'C21_ACTOCC15P_2ROUESMOT_estime_x',
       'C21_ACTOCC15P_VOIT_estime_x', 'C21_ACTOCC15P_TCOM_estime_x',
       'DEC_MED21_x', 'DISP_MED21_x', 'DEC_Q121_x', 'DEC_Q321_x',
       'DISP_Q121_x', 'DISP_Q321_x', 'taux_activite_x', 'population_estimee_y',
       'surface_batie_ponderee_y', 'P21_ACTOCC1564_estime_y',
       'P21_SAL15P_estime_y', 'P21_NSAL15P_estime_y',
       'C21_ACTOCC1564_CS3_estime_y', 'C21_ACTOCC1564_CS4_estime_y',
       'C21_ACTOCC1564_CS5_estime_y', 'C21_ACTOCC1564_CS6_estime_y',
       'C21_ACTOCC15P_estime_y', 'C21_ACTOCC15P_PAS_estime_y',
       'C21_ACTOCC15P_MAR_estime_y', 'C21_ACTOCC15P_VELO_estime_y',
       'C21_ACTOCC15P_2ROUESMOT_estime_y', 'C21_ACTOCC15P_VOIT_estime_y',
       'C21_ACTOCC15P_TCOM_estime_y', 'DEC_MED21_y', 'DISP_MED21_y',
       'DEC_Q121_y', 'DEC_Q321_y', 'DISP_Q121_y', 'DISP_Q321_y',
       'taux_activite_y', 'population_estimee', 'surface_batie_ponderee',
       'P21_ACTOCC1564_estime', 'P21_SAL15P_estime', 'P21_NSAL15P_estime',
       'C21_ACTOCC1564_CS3_estime', 'C21_ACTOCC1564_CS4_estime',
       'C21_ACTOCC1564_CS5_estime', 'C21_ACTOCC1564_CS6_estime',
       'C21_ACTOCC15P_estime', 'C21_ACTOCC15P_PAS_estime',
       'C21_ACTOCC15P_MAR_estime', 'C21_ACTOCC15P_VELO_estime',
       'C21_ACTOCC15P_2ROUESMOT_estime', 'C21_ACTOCC15P_VOIT_estime',
       'C21_ACTOCC15P_TCOM_estime', 'DEC_MED21', 'DISP_MED21', 'DEC_Q121',
       'DEC_Q321', 'DISP_Q121', 'DISP_Q321', 'taux_activite'] # Utiliser la fonction .column

    carreaux_200m  = nettoyer_colonnes(carreaux_200m, colonnes_a_supprimer)
    carreaux_existants  = nettoyer_colonnes(carreaux_existants, colonnes_a_supprimer)

    # 2. Préparation des bâtiments
    batiments = batiments[batiments['ETAT'] == 'En service'].copy()
    batiments['HAUTEUR'] = batiments['HAUTEUR'].fillna(6)
    batiments['POIDS'] = batiments.geometry.area * batiments['HAUTEUR']

    # 3. Calcul des surfaces bâties pondérées
    print("Calcul des surfaces bâties par maille 200m...")
    batiments_sindex = batiments.sindex
    results = []
    for idx, carreau in carreaux_200m.iterrows():
        possible_matches_index = list(batiments_sindex.intersection(carreau.geometry.bounds))
        possible_matches = batiments.iloc[possible_matches_index]
        precise_matches = possible_matches[possible_matches.intersects(carreau.geometry)]
        surface_ponderee = sum(
            bat.geometry.intersection(carreau.geometry).area * bat['HAUTEUR']
            for _, bat in precise_matches.iterrows()
            if not bat.geometry.intersection(carreau.geometry).is_empty
        )
        results.append({
            'idINSPIRE': carreau['idINSPIRE'],
            'geometry': carreau.geometry,
            'surface_batie_ponderee': surface_ponderee
        })

    carreaux_surface = gpd.GeoDataFrame(results, crs=2154)

    # 4. Jointure avec les IRIS et calcul des proportions
    carreaux_avec_iris = gpd.sjoin(
        carreaux_surface,
        iris_complet[['IRIS', 'INSEE_COM', 'geometry']],
        how='left',
        predicate='intersects'
    )
    total_par_iris = carreaux_avec_iris.groupby('IRIS')['surface_batie_ponderee'].sum().reset_index()
    total_par_iris.columns = ['IRIS', 'total_surface_iris']
    carreaux_avec_iris = carreaux_avec_iris.merge(total_par_iris, on='IRIS', how='left')
    carreaux_avec_iris['proportion'] = carreaux_avec_iris['surface_batie_ponderee'] / carreaux_avec_iris['total_surface_iris'].replace(0, 1)

    # 5. Jointure avec les données IRIS
    colonnes_iris = [
        'IRIS', 'INSEE_COM', 'P21_POP',
        'P21_ACTOCC1564', 'P21_SAL15P', 'P21_NSAL15P',
        'C21_ACTOCC1564_CS3', 'C21_ACTOCC1564_CS4', 'C21_ACTOCC1564_CS5', 'C21_ACTOCC1564_CS6',
        'C21_ACTOCC15P', 'C21_ACTOCC15P_PAS', 'C21_ACTOCC15P_MAR', 'C21_ACTOCC15P_VELO',
        'C21_ACTOCC15P_2ROUESMOT', 'C21_ACTOCC15P_VOIT', 'C21_ACTOCC15P_TCOM',
        'DEC_MED21', 'DISP_MED21', 'DEC_Q121', 'DEC_Q321', 'DISP_Q121', 'DISP_Q321'
    ]
    carreaux_complets = carreaux_avec_iris.merge(iris_complet[colonnes_iris], on=['IRIS', 'INSEE_COM'], how='left')

    # 6. Répartition
    carreaux_complets['population_estimee'] = (
        carreaux_complets['P21_POP'] * carreaux_complets['proportion']
    ).round().fillna(0).astype(int)

    indicateurs_activite = [
        'P21_ACTOCC1564', 'P21_SAL15P', 'P21_NSAL15P',
        'C21_ACTOCC1564_CS3', 'C21_ACTOCC1564_CS4', 'C21_ACTOCC1564_CS5', 'C21_ACTOCC1564_CS6',
        'C21_ACTOCC15P', 'C21_ACTOCC15P_PAS', 'C21_ACTOCC15P_MAR', 'C21_ACTOCC15P_VELO',
        'C21_ACTOCC15P_2ROUESMOT', 'C21_ACTOCC15P_VOIT', 'C21_ACTOCC15P_TCOM'
    ]
    for col in indicateurs_activite:
        carreaux_complets[f"{col}_estime"] = (
            carreaux_complets[col] * carreaux_complets['proportion']
        ).round().fillna(0).astype(int)

    print("Gestion des carreaux intersectant plusieurs IRIS...")
    agg_functions = {
        'geometry': 'first',
        'population_estimee': 'sum',
        'surface_batie_ponderee': 'sum'
    }
    for col in indicateurs_activite:
        agg_functions[f"{col}_estime"] = 'sum'
    indicateurs_socio = [
        'DEC_MED21', 'DISP_MED21', 'DEC_Q121', 'DEC_Q321', 'DISP_Q121', 'DISP_Q321'
    ]
    for col in indicateurs_socio:
        agg_functions[col] = lambda x: pd.to_numeric(x, errors='coerce').mean()

    carreaux_finaux = carreaux_complets.groupby('idINSPIRE').agg(agg_functions).reset_index()
    carreaux_finaux = gpd.GeoDataFrame(carreaux_finaux, geometry='geometry', crs=2154)

    # 8. Indicateurs dérivés et arrondi des valeurs
    carreaux_finaux['taux_activite'] = (
        carreaux_finaux['C21_ACTOCC15P_estime'] /
        carreaux_finaux['population_estimee'].replace(0, 1) * 100
    ).round(1)

    # Arrondi des indicateurs socio-économiques à 2 décimales
    indicateurs_socio = [
        'DEC_MED21', 'DISP_MED21', 'DEC_Q121', 'DEC_Q321', 'DISP_Q121', 'DISP_Q321'
    ]
    for col in indicateurs_socio:
        carreaux_finaux[col] = carreaux_finaux[col].fillna(0).round(2)

    # Arrondi des proportions et surfaces
    carreaux_finaux['surface_batie_ponderee'] = carreaux_finaux['surface_batie_ponderee'].round(2)

    # 9. Jointure avec les données déjà présentes
    print("Jointure avec les données précédentes...")
    carreaux_finaux = carreaux_finaux.to_crs(2154)
    carreaux_final_merge = carreaux_existants.drop(columns='geometry').merge(
        carreaux_finaux.drop(columns='geometry'),
        on='idINSPIRE',
        how='left'
    )
    carreaux_final_merge['geometry'] = carreaux_existants['geometry']
    carreaux_final_merge = gpd.GeoDataFrame(carreaux_final_merge, geometry='geometry', crs=2154)

    # 10. Export
    exporter_parquet(carreaux_final_merge, "maille_200m_avec_donnees")
    exporter_gpkg(carreaux_final_merge, "maille_200m_avec_donnees")

    print(f"""
Résultats :
- {len(carreaux_final_merge)} carreaux traités
- Population totale estimée : {carreaux_final_merge['population_estimee'].sum()}
- Actifs occupés : {carreaux_final_merge['C21_ACTOCC15P_estime'].sum()} 
- Répartition professionnelle :
  * Cadres : {carreaux_final_merge['C21_ACTOCC1564_CS3_estime'].sum()}
  * Professions intermédiaires : {carreaux_final_merge['C21_ACTOCC1564_CS4_estime'].sum()}
  * Employés : {carreaux_final_merge['C21_ACTOCC1564_CS5_estime'].sum()}
  * Ouvriers : {carreaux_final_merge['C21_ACTOCC1564_CS6_estime'].sum()}
- Revenus médians :
  * Déclarés : {carreaux_final_merge['DEC_MED21'].mean():.2f} €
  * Disponibles : {carreaux_final_merge['DISP_MED21'].mean():.2f} €
""")

# Exécution
desagreger_donnees_vers_carreaux()


# Note : le revenu médian disponible moyen est légèrement surestimés par rapport aux chiffres directements fournis par l'INSEE : https://www.insee.fr/fr/statistiques/2011101?geo=EPCI-246700488, sans doute car fait ce calcul sur les carreaux, et non sur les IRIS. Mais les données semblent cohérentes.
# 
# ![ems_revenus_moyens](images/ems_revenus_moyens.png)

# ### 5.5. Calculs du temps effectif moyen 
# ---
# Voici la définition originale de l'indicateur tiré du document de travail Sumo-Rhine : 
# 
# Temps d'accès effectif à un véhicule (EATV) au cours d'une journée de travail. En supposant qu'un véhicule s'arrête pendant 1 minute par station, que le véhicule est accessible aux passagers pendant cet intervalle de temps et que le service parfait est celui qui est accessible en permanence, l'indicateur est calculé pour une station k de la manière suivante : EATVk = fk * dk, où : 
# o fk est la fréquence des voyages k pendant une heure pour un jour ouvrable (nombre de voyages par heure
# o dk est la durée de disponibilité du service en k (en heures h)
# 
# J'ai plusieurs problèmes avec cet indicateur. Déjà, il ne calcule en réalité que le nombre de voyages par arrêts :
# * fk = nb passages / dk (où dk est le nombre d'heures desservies)
# * EATV = fk * dk = (nb passages / heures) * heures = nb passages
# 
# Cela n'est pas un problème en soit, mais l'analyse des EATV calculés est ambivalente. Par exemple :
# * Arrêt A : 100 passages entre 6h et 22h
#     * dk = 16(h)
#     * fk = 100 / 16 = 6.25 
#     * EATV = 6.25 * 16 = 100
# * Arrêt B : 100 passages entre 7h et 9h
#     * dk = 2(h)
#     * fk = 100 / 2 = 50
#     * EATV = 50 * 2 = 100
#   
# Les EATV obtenus sont les mêmes, mais ne veulent pas du tout dire la même chose. A moins de complexifier la formule (surtout qu'idéalement, les indicateurs doivent être simples à comprendre), je ne pense pas que cet indicateur soit souhaitable ici. La fonction qui le calcule est présente mais n'est pas utilisée
# 
# ---
# Pour le remplacer, je calcule le temps d'attente moyen (en secondes) par arrêts.
# 
# En supposant qu'un véhicule s'arrête dans un arrêt pendant 1 minute, que le véhicule est accessible aux passagers pendant cet intervalle de temps et que le service parfait est celui qui est accessible en permanence, l'indicateur est calculé pour un arrêt k de la manière suivante : EATVk = 3600 / fk, où : 
# * fk = Nombre de voyages à à l'arrêt k / durée de disponibilité du service en k (en heures pour un jour ouvrable). fk est le nombre de voyages par heure pour l'arrêt k.
# 
# L'indicateur final retourne le temps(s) durant lequel un arrêt est desservi durant une journée.
# 
# Exemples : 
# * Arrêt A : 100 passages entre 6h à 22h
#     * 16 heures de service
#     * fk = 100 / 16 = 6.25 (fréquence de voyages pendant une heure)
#     * EATV : 3600 / 6.25 = 576 secondes d'attente moyenne entre chaque passage (soit 19 minutes et 12 secondes) à l'arrêt k
# * Arrêt B : 100 passages entre 7h et 9h
#     * 2 heures de service
#     * fk = 100 / 2 = 50 (fréquence de voyages pendant une heure)
#     * EATV : 3600 / 50 = 72 secondes d'attente moyenne entre chaque passage (soit 1 minute et 12 secondes) à l'arrêt k
# 
# Dans ce script, on calcule cet indicateur du lundi au dimanche pour chaque arrêt. On créé deux champs 'att_secondes_moy_semaine' (moyenne du lundi au vendredi) et 'att_secondes_moy_week_end' (moyenne du samedi et du dimanche)
# 
# Les résultats peuvent être vérifiés d'après ce site : https://cts-strasbourg.eu/fr/se-deplacer/fiches-horaires/recherche-des-fiches-horaires/. De ce que j'ai pu voir, les résultats obtenus par la fonction semblent bon.

# In[490]:


def calculer_temps_acces_effectif(suffixe="bus"):
    # 1. Chargement des données
    df_stop_times = pd.read_csv(os.path.join(exports_dir, f"gtfs_stop_times_{suffixe}.csv"))
    df_trips = pd.read_csv(os.path.join(exports_dir, f"gtfs_trips_{suffixe}.csv"))
    df_calendar = pd.read_csv(os.path.join(exports_dir, "gtfs_calendar.csv"))
    gdf_stops = gpd.read_parquet(os.path.join(exports_dir, f"gtfs_stops_{suffixe}.parquet"))[["stop_id", "geometry"]]
    limites_epci = charger_fichier_parquet("limites_epci", crs=4326)

    # 2. Associer chaque trip à son service_id
    df = df_stop_times.merge(df_trips[["trip_id", "service_id"]], on="trip_id")

    # 3. Associer les jours à chaque service_id
    jours = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    jours_dict = {j: df_calendar[df_calendar[j] == 1]["service_id"].unique() for j in jours}

    # 4. Parse des horaires
    def parse_time(h):
        try:
            return datetime.strptime(h, "%H:%M:%S").time()
        except:
            return None

    df["arrival_time_dt"] = df["arrival_time"].apply(parse_time)
    df = df[df["arrival_time_dt"].notnull()]
    df["arrival_hour"] = df["arrival_time_dt"].apply(lambda t: t.hour + t.minute / 60)

    # 5. Calcul EATV = fk × dk pour chaque jour
    df_result = pd.DataFrame({"stop_id": df["stop_id"].unique()})

    for jour in jours:
        sous_df = df[df["service_id"].isin(jours_dict[jour])]
        res = []

        for stop_id, group in sous_df.groupby("stop_id"):
            heures = sorted(group["arrival_hour"].tolist())
            if len(heures) < 2:
                continue
            dk = heures[-1] - heures[0]  # durée en heures
            if dk <= 0:
                continue
            fk = len(heures) / dk       # fréquence (voyages/h)
            eatv = fk * dk              # accessibilité effective
            res.append({"stop_id": stop_id, f"eatv_{jour}": eatv})

        df_jour = pd.DataFrame(res)
        df_result = df_result.merge(df_jour, on="stop_id", how="left")

    # 6. Calcul des moyennes semaine et week-end
    jours_semaine = ["monday", "tuesday", "wednesday", "thursday", "friday"]
    jours_weekend = ["saturday", "sunday"]

    df_result["acces_effectif_moy_semaine"] = df_result[[f"eatv_{j}" for j in jours_semaine]].mean(axis=1)
    df_result["acces_effectif_moy_week_end"] = df_result[[f"eatv_{j}" for j in jours_weekend]].mean(axis=1)

    # 7. Jointure avec géométrie et filtre EPCI
    gdf_result = gdf_stops.merge(df_result, on="stop_id", how="left")
    gdf_result = gdf_result.to_crs(4326)
    gdf_result = gpd.sjoin(gdf_result, limites_epci[["geometry"]], predicate="within", how="inner").drop(columns="index_right")

    # 8. Arrondi final
    gdf_result["acces_effectif_moy_semaine"] = gdf_result["acces_effectif_moy_semaine"].round(2)
    gdf_result["acces_effectif_moy_week_end"] = gdf_result["acces_effectif_moy_week_end"].round(2)

    # 9. Supprimer les colonnes de test 'att_monday' à 'att_sunday', servant à vérifier les données
    colonnes_a_supprimer = [f"eatv_{j}" for j in jours]
    gdf_result = nettoyer_colonnes(gdf_result, colonnes_a_supprimer)

    # 10. Export
    nom_base = f"arrets_{suffixe}_avec_donnees"
    exporter_parquet(gdf_result, nom_base)
    exporter_gpkg(gdf_result, nom_base)

    print(f"EATV moyen exporté pour les arrêts ({suffixe}) – semaine et week-end.")


# In[446]:


# Calcul du EATV
def calculer_temps_acces_effectif(suffixe = "bus"):
    # 1. Chargement des donnéees
    df_stop_times = pd.read_csv(os.path.join(exports_dir, f"gtfs_stop_times_{suffixe}.csv"))
    df_trips = pd.read_csv(os.path.join(exports_dir, f"gtfs_trips_{suffixe}.csv"))
    df_stops = pd.read_csv(os.path.join(exports_dir, f"gtfs_stops_{suffixe}.csv"))

    # 2. Filtrer les jours ouvrables (exclut samedi/dimanche dans service_id ou trip_id)
    jours_exclus = ["samedi", "dimanche"]
    df_trips = df_trips[~df_trips["trip_id"].str.lower().str.contains("|".join(jours_exclus))]

    # 3. Jointure stop_times <-> trips pour associer trip_id + stop_id + horaires
    df = df_stop_times.merge(df_trips[["trip_id", "route_id"]], on="trip_id", how="inner")

    # 4. Conversion horaire vers datetime.time
    def parse_time(h):
        try:
            return datetime.strptime(h, "%H:%M:%S").time()
        except:
            return None

    df["arrival_time_dt"] = df["arrival_time"].apply(parse_time)
    df = df[df["arrival_time_dt"].notnull()]

    # 5. Calcul fk et dk pour chaque (stop_id, route_id)
    df["arrival_hour"] = df["arrival_time_dt"].apply(lambda t: t.hour + t.minute / 60)

    resultats = []

    for (stop_id, route_id), group in df.groupby(["stop_id", "route_id"]):
        heures = sorted(group["arrival_hour"].tolist())
        if len(heures) < 2:
            continue
        dk = heures[-1] - heures[0]  # durée de service en heures
        fk = len(heures) / dk if dk > 0 else 0  # fréquence horaire
        acces = fk * dk
        resultats.append({
            "stop_id": stop_id,
            "route_id": route_id,
            "fk": fk,
            "dk": dk,
            "acces_effectif": acces
        })

    df_resultats = pd.DataFrame(resultats)

    # 6. Agrégation par arrêt (somme de tous les accès effectifs pour un même arrêt)
    df_stops = df_resultats.groupby("stop_id").agg({
        "acces_effectif": "sum"
    }).reset_index()

    # 7. Jointure avec la géométrie des arrêts
    gdf_stops = gpd.read_parquet(os.path.join(exports_dir, f"gtfs_stops_{suffixe}.parquet"))
    gdf_stops = gdf_stops[["stop_id", "geometry"]].copy()
    gdf_result = gdf_stops.merge(df_stops, on="stop_id", how="left")
    gdf_result["acces_effectif"] = gdf_result["acces_effectif"].fillna(0)

    # 8. Exports
    exporter_gpkg(gdf_result, f"gtfs_stops_{suffixe}_acces_effectif")
    exporter_parquet(gdf_result, f"gtfs_stops_{suffixe}_acces_effectif")


# ### 5.6. Calcul de la part des navetteurs selon le transport
# ---
# Ce calcul se fait en fonction du mode de transport majoritairement utilisé par les actifs pour aller travailler. Cela ne veut pas dire que les navetteurs utilisant principalement les transports en commun ne marchent pas.
# 
# Les actifs qui n'ont pas besoin de se déplacer pour travailler sont exclus de ce calcul.
# 
# Les données de base proviennent du niveau IRIS, qui a été désagrégé pour les carreaux de 200m : il s'agit donc d'une estimation.
# 
# Documentation :
# * https://www.insee.fr/fr/statistiques/8268843
# * https://www.insee.fr/fr/statistiques/5013868

# In[69]:


def calculer_part_navetteurs(nom_transport=""):
    # 1. Chargement des données
    carreaux = charger_fichier_parquet("maille_200m_avec_donnees", crs=2154)

    # 2. Définition des configurations pour chaque mode
    configs = {
        "marche": {
            "colonne_source": "C21_ACTOCC15P_MAR_estime",
            "colonne_resultat": "part_marche",
            "libelle": "marche",
            "calcul": lambda df: df["C21_ACTOCC15P_MAR_estime"]
        },
        "velos": {
            "colonne_source": "C21_ACTOCC15P_VELO_estime",
            "colonne_resultat": "part_velos",
            "libelle": "vélo",
            "calcul": lambda df: df["C21_ACTOCC15P_VELO_estime"]
        },
        "tcom": {
            "colonne_source": "C21_ACTOCC15P_TCOM_estime",
            "colonne_resultat": "part_communs",
            "libelle": "transports en commun",
            "calcul": lambda df: df["C21_ACTOCC15P_TCOM_estime"]
        },
        "vl": {
            "colonne_source": ["C21_ACTOCC15P_2ROUESMOT_estime", "C21_ACTOCC15P_VOIT_estime"],
            "colonne_resultat": "part_vl",
            "libelle": "véhicules légers",
            "calcul": lambda df: df["C21_ACTOCC15P_2ROUESMOT_estime"] + df["C21_ACTOCC15P_VOIT_estime"]
        }
    }

    # Vérification du mode saisi
    if nom_transport not in configs:
        modes_valides = ", ".join(configs.keys())
        raise ValueError(f"Mode de transport '{nom_transport}' invalide. Options: {modes_valides}")

    config = configs[nom_transport]

    # 3. Calcul du total des navetteurs
    colonnes_transport = [
        'C21_ACTOCC15P_MAR_estime',
        'C21_ACTOCC15P_VELO_estime',
        'C21_ACTOCC15P_2ROUESMOT_estime',
        'C21_ACTOCC15P_VOIT_estime',
        'C21_ACTOCC15P_TCOM_estime'
    ]

    carreaux['total_navetteurs'] = carreaux[colonnes_transport].sum(axis=1)

    # 4. Calcul de la part spécifique (convertie en pourcentage)
    carreaux[config["colonne_resultat"]] = np.where(
        carreaux['total_navetteurs'] > 0,
        (config["calcul"](carreaux) / carreaux['total_navetteurs']) * 100,  # Multiplication par 100
        -1  # Valeur de -1 si aucune donnée présente
    ).round(2)  # Arrondi à 2 décimales

    # 5. Retire la colonne 'total_navetteurs'
    carreaux = carreaux.drop(columns=["total_navetteurs"])

    # 6. Statistiques  
    valides = carreaux[carreaux[config["colonne_resultat"]] >= 0]
    part_moyenne = valides[config["colonne_resultat"]].mean()
    print(f"""
    Part des navetteurs en {config['libelle']} :
    - Carreaux avec données : {len(valides)}/{len(carreaux)}
    - Moyenne : {part_moyenne:.2f}%
    - Médiane : {valides[config["colonne_resultat"]].median():.2f}%
    - Max : {valides[config["colonne_resultat"]].max():.2f}%
    """)

    # 7. Export
    exporter_parquet(carreaux, "maille_200m_avec_donnees")
    exporter_gpkg(carreaux, "maille_200m_avec_donnees")


# In[434]:


def afficher_part_navetteurs(nom_transport="", export = False):
    # 1. Configurations selon le mode
    config = {
        "marche": {
            "colonne": "part_marche",
            "titre": "Part de navetteurs utilisant principalement la marche",
            "label": "% de navetteurs à pied"
        },
        "velos": {
            "colonne": "part_velos",
            "titre": "Part de navetteurs utilisant principalement le vélo",
            "label": "% de navetteurs à vélo"
        },
        "tcom": {
            "colonne": "part_communs",
            "titre": "Part de navetteurs utilisant principalement \nles transports en commun",
            "label": "% de navetteurs en transports en commun"
        },
        "vl": {
            "colonne": "part_vl",
            "titre": "Part de navetteurs utilisant principalement \nles véhicules légers",
            "label": "% de navetteurs en véhicules légers"
        }
    }

    if nom_transport not in config:
        raise ValueError(f"Mode inconnu : {nom_transport}. Choisir parmi : {', '.join(config.keys())}")

    params = config[nom_transport]

    # 2. Chargement des données
    limites_epci = charger_fichier_parquet("limites_epci", crs=3857)
    carreaux = charger_fichier_parquet("maille_200m_avec_donnees", crs=3857)

    # 3. Filtrage : exclure les valeurs à -1 (carreaux sans données)
    carreaux_filtrés = carreaux[carreaux[params["colonne"]] != -1]

    # 4. Affichage
    fig, ax = plt.subplots(figsize=(10, 10))

    limites_epci.plot(ax=ax, alpha=1, edgecolor='black', facecolor='none')

    carreaux_filtrés.plot(
        column=params["colonne"],
        cmap="YlGn",
        vmin=0, vmax=100,
        legend=True,
        #legend_kwds={'label': params["label"]},
        ax=ax,
        linewidth=0.1,
        edgecolor="grey"
    )

    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)

    ax.set_title(params["titre"], fontsize=18)
    ax.axis('off')
    plt.tight_layout()

    # (Optionnel) Export PNG
    if export :
        images_export_dir = os.path.join(images_dir, f"indicateur_part_navetteurs_{nom_transport}.png")
        plt.savefig(images_export_dir, dpi=300, bbox_inches='tight')
        print(f"Carte exportée vers : {images_export_dir}")

    plt.show()
    plt.close()


# ### 5.7. Calculs pour les transports en commun
# ---
# Note : actuellement, on calcule séparément les indicateurs pour les bus et les tram

# #### 5.7.1. Calcul du coût d'un trajet (€ / h)
# ---
# Note : possibilité de rapporter le coût du ticket aux revenus de la population de la maille
# 
# A comprendre dans le sens "pour le prix d'un ticket, combien de minutes de trajet peut-on effectuer en moyenne en partant d'un arrêt ?"

# ##### 5.7.1.1. Par arrêt
# ---

# In[208]:


def calculer_cout_trajet_par_arret(transport="bus"):
    # 1. Paramètres
    tarif_unitaire = 1.9  # € par trajet aller-simple

    # 2. Chargement des données
    df_stop_times = pd.read_csv(os.path.join(exports_dir, f"gtfs_stop_times_{transport}.csv"))
    limites_epci = charger_fichier_parquet("limites_epci", crs=2154)
    gdf_stops = charger_fichier_parquet(f"arrets_{transport}_avec_donnees", crs=2154)

    # Suppression des colonnes déjà calculées et de jointure (si présentes)
    colonnes_a_supprimer = ['cout_horaire', 'cout_horaire_x', 'cout_horaire_y']
    gdf_stops = nettoyer_colonnes(gdf_stops, colonnes_a_supprimer)

    # 3. Nettoyage & parsing des horaires
    def parse_time(t):
        try:
            return datetime.strptime(t, "%H:%M:%S")
        except:
            return None

    df_stop_times["arrival_dt"] = df_stop_times["arrival_time"].apply(parse_time)
    df_stop_times["departure_dt"] = df_stop_times["departure_time"].apply(parse_time)
    df_stop_times = df_stop_times[
        df_stop_times["arrival_dt"].notnull() & df_stop_times["departure_dt"].notnull()
    ]

    # 4. Durée des trajets
    trip_durations = df_stop_times.groupby("trip_id").agg(
        start=("arrival_dt", "min"),
        end=("departure_dt", "max")
    )
    trip_durations["duration_min"] = (
        (trip_durations["end"] - trip_durations["start"]).dt.total_seconds() / 60
    )
    trip_durations = trip_durations[trip_durations["duration_min"] > 0]

    # 5. Associer les durées aux arrêts
    df_durations = df_stop_times[["trip_id", "stop_id"]].merge(
        trip_durations[["duration_min"]], on="trip_id", how="left"
    )

    # 6. Moyenne par arrêt
    duree_par_stop = (
        df_durations.groupby("stop_id")["duration_min"]
        .mean()
        .reset_index()
    )
    duree_par_stop["duration_h"] = duree_par_stop["duration_min"] / 60
    duree_par_stop["cout_horaire"] = (tarif_unitaire / duree_par_stop["duration_h"]).round(2)

    # 7. Jointure avec les données d'arrêts
    gdf_result = gdf_stops.merge(duree_par_stop[["stop_id", "cout_horaire"]], on="stop_id", how="left")

    # 8. Export
    exporter_parquet(gdf_result, f"arrets_{transport}_avec_donnees")
    exporter_gpkg(gdf_result, f"arrets_{transport}_avec_donnees")

    # 9. Statistiques
    nb_total = len(gdf_result)
    nb_cout = gdf_result["cout_horaire"].notnull().sum()
    cout_moyen = gdf_result["cout_horaire"].mean()

    print(f"""
Résultats exportés :
- {nb_total} arrêts dans l'EPCI
- {nb_cout} arrêts avec coût horaire estimé
- Coût horaire moyen : {cout_moyen:.2f} €/h
""")


# In[217]:


def afficher_cout_trajet_par_arret(transport="bus", export = False):
    # 1. Chargement des données
    limites_epci = charger_fichier_parquet("limites_epci", crs=3857)
    gdf = charger_fichier_parquet(f"arrets_{transport}_avec_donnees", crs=3857)
    gdf = gdf[gdf["cout_horaire"].notnull()]

    # 2. Affichage
    cmap = plt.cm.viridis_r
    norm = mcolors.Normalize(vmin=gdf["cout_horaire"].min(), vmax=gdf["cout_horaire"].max())

    fig, ax = plt.subplots(figsize=(10, 10))
    limites_epci.plot(ax=ax, alpha=1, edgecolor='black', facecolor='none')

    gdf.plot(ax=ax, column="cout_horaire", cmap=cmap, markersize=10, legend=True, norm=norm)

    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)

    ax.set_title(f"Coût horaire moyen par arrêt de {transport} (€ / h)", fontsize=15)
    ax.axis("off")
    plt.tight_layout()

    # (Optionnel) Export PNG
    if export :
        images_export_dir = os.path.join(images_dir, f"indicateur_cout_trajet_arret_{transport}.png")
        plt.savefig(images_export_dir, dpi=300, bbox_inches='tight')
        print(f"Carte exportée vers : {images_export_dir}")

    plt.show()
    plt.close()


# ##### 5.7.1.2. Par maille
# ---

# In[210]:


def calculer_cout_trajet_par_maille(transport="bus"):
    # 1. Chargement des données
    gdf_arrets = charger_fichier_parquet(f"arrets_{transport}_avec_donnees", crs=2154)
    gdf_maille = charger_fichier_parquet("maille_200m_avec_donnees", crs=2154)

    # 2. Nettoyage des colonnes déjà calculées
    gdf_maille = nettoyer_colonnes(gdf_maille, [f"cout_horaire_{transport}"])

    # 3. Vérification de la présence de la colonne
    if "cout_horaire" not in gdf_arrets.columns:
        raise ValueError(f"La colonne 'cout_horaire' est absente de arrets_{transport}_avec_donnees. "
                         "Lance d'abord 'calculer_cout_trajet_par_arret'.")

    # 4. Filtrage
    gdf_arrets_valides = gdf_arrets[
        gdf_arrets["cout_horaire"].notnull() & gdf_arrets.geometry.notnull()
    ].copy()

    # 5. Jointure spatiale : rattachement aux carreaux
    jointure = gpd.sjoin(
        gdf_arrets_valides[["stop_id", "cout_horaire", "geometry"]],
        gdf_maille[["idINSPIRE", "geometry"]],
        how="inner",
        predicate="within"
    )

    # 6. Moyenne par maille
    cout_moyen_maille = (
        jointure.groupby("idINSPIRE")["cout_horaire"]
        .mean()
        .reset_index()
        .rename(columns={"cout_horaire": f"cout_horaire_{transport}"})
    )

    # 7. Arrondi à 2 chiffres significatifs
    cout_moyen_maille[f"cout_horaire_{transport}"] = (
        cout_moyen_maille[f"cout_horaire_{transport}"].round(2)
    )

    # 8. Fusion avec la maille
    gdf_maille = gdf_maille.merge(cout_moyen_maille, on="idINSPIRE", how="left")

    # 9. Remplacement des valeurs manquantes par -1
    gdf_maille[f"cout_horaire_{transport}"] = (
        gdf_maille[f"cout_horaire_{transport}"].fillna(-1)
    )

    # 10. Export
    exporter_parquet(gdf_maille, "maille_200m_avec_donnees")
    exporter_gpkg(gdf_maille, "maille_200m_avec_donnees")

    # 11. Statistiques
    valides = gdf_maille[gdf_maille[f"cout_horaire_{transport}"] >= 0]
    print(f"""
Résultats exportés :
- {len(gdf_maille)} carreaux
- {len(valides)} ont au moins un arrêt de {transport}
- Coût horaire moyen global : {valides[f"cout_horaire_{transport}"].mean():.2f} €/h
""")


# In[373]:


def afficher_cout_trajet_par_maille(transport="bus", export = False):
    # 1. Chargement des données
    limites_epci = charger_fichier_parquet("limites_epci", crs=3857)
    carreaux = charger_fichier_parquet("maille_200m_avec_donnees", crs=3857)
    carreaux = carreaux[carreaux[f"cout_horaire_{transport}"] != -1]

    # 2. Affichage
    fig, ax = plt.subplots(figsize=(10, 10))
    limites_epci.plot(ax=ax, alpha=1, edgecolor='black', facecolor='none')

    carreaux.plot(column=f"cout_horaire_{transport}",
             cmap="YlOrRd",
             legend=True,
             #legend_kwds={'label' : f"Coût horaire moyen en partant en {transport} depuis la maille"},
             ax=ax,
             linewidth=0.1,
             edgecolor="grey")

    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)

    ax.set_title(f"Coût horaire moyen (€ /h) en partant en {transport} depuis la maille", fontsize=18)
    ax.axis('off')
    plt.tight_layout()

    # (Optionnel) Export PNG
    if export :
        images_export_dir = os.path.join(images_dir, f"indicateur_cout_moyen_maille_{transport}.png")
        plt.savefig(images_export_dir, dpi=300, bbox_inches='tight')
        print(f"Carte exportée vers : {images_export_dir}")

    plt.show()
    plt.close()


# #### 5.7.2. Part des arrêts disposant de parking relais
# ---
# Ici, on considère qu'un arrêt possède un parking relais si il y en a un à moins de 400 mètres de l'arrêt (6 min à pied pour une marche à 4 km/h) 

# In[52]:


def calculer_part_arrets_avec_pr(transport="bus"):
    # 1. Chargement des données
    arrets_avec_donnees = charger_fichier_parquet(f"arrets_{transport}_avec_donnees", crs=2154).copy()
    gdf_parkings = charger_fichier_parquet("parkings_relais_epci", crs=2154).copy()

    # Suppression des colonnes déjà calculées et de jointure (si présentes)
    colonnes_a_supprimer = ['possede_parking_relais']
    arrets_avec_donnees = nettoyer_colonnes(arrets_avec_donnees, colonnes_a_supprimer)

    # 2. Nettoyage des géométries
    arrets = arrets_avec_donnees[
        arrets_avec_donnees.geometry.notnull() & 
        (arrets_avec_donnees.geometry.type == "Point")
    ].copy()
    gdf_parkings = gdf_parkings[
        gdf_parkings.geometry.notnull() & 
        (gdf_parkings.geometry.type.isin(["Polygon", "MultiPolygon"]))
    ].copy()

    # 3. Création d'un buffer de 400m autour des arrêts
    arrets["geometry"] = arrets.geometry.buffer(400)

    # 4. Jointure spatiale : arrêts (en buffer) contenant un parking relais (polygone dans buffer)
    jointure = gpd.sjoin(arrets, gdf_parkings, how="left", predicate="intersects")

    # 5. Création de l'indicateur numérique (0/1)
    jointure["possede_parking_relais"] = jointure["index_right"].notnull().astype(int)

    # 6. Agrégation par arrêt
    presences = jointure.groupby("stop_id")["possede_parking_relais"].max().reset_index()

    # 7. Fusion avec les données enrichies
    gdf_resultat = arrets_avec_donnees.merge(presences, on="stop_id", how="left")
    gdf_resultat["possede_parking_relais"] = gdf_resultat["possede_parking_relais"].fillna(0).astype(int)

    # 8. Statistiques
    total_arrets = len(gdf_resultat)
    arrets_avec_parking = gdf_resultat["possede_parking_relais"].sum()
    taux_couverture = (arrets_avec_parking / total_arrets) * 100

    print("\n--- Statistiques ---")
    print(f"Nombre total d'arrêts de {transport} dans l'EPCI : {total_arrets}")
    print(f"Arrêts avec au moins un parking relais à moins de 400m : {arrets_avec_parking}")
    print(f"Taux de couverture : {taux_couverture:.1f}%")

    # 9. Export
    exporter_parquet(gdf_resultat, f"arrets_{transport}_avec_donnees")
    exporter_gpkg(gdf_resultat, f"arrets_{transport}_avec_donnees")


# In[279]:


def calculer_part_arrets_avec_pr(transport="bus"):
    # 1. Chargement des données
    arrets_avec_donnees = charger_fichier_parquet(f"arrets_{transport}_avec_donnees", crs=2154).copy()
    gdf_parkings = charger_fichier_parquet("parkings_relais_epci", crs=2154).copy()

    # Suppression des colonnes déjà calculées et de jointure (si présentes)
    colonnes_a_supprimer = ['possede_parking_relais']
    arrets_avec_donnees = nettoyer_colonnes(arrets_avec_donnees, colonnes_a_supprimer)

    # 2. Nettoyage des géométries
    arrets = arrets_avec_donnees[
        arrets_avec_donnees.geometry.notnull() & 
        (arrets_avec_donnees.geometry.type == "Point")
    ].copy()
    gdf_parkings = gdf_parkings[
        gdf_parkings.geometry.notnull() & 
        (gdf_parkings.geometry.type == "Point")
    ].copy()

    # 3. Création d’un buffer de 400m autour des arrêts
    arrets["buffer_400m"] = arrets.geometry.buffer(400)
    gdf_buffers = gpd.GeoDataFrame(
        arrets[["stop_id", "buffer_400m"]],
        geometry="buffer_400m",
        crs=2154
    )

    # 4. Jointure spatiale : arrêts avec au moins un parking relais dans leur buffer
    jointure = gpd.sjoin(gdf_buffers, gdf_parkings, how="left", predicate="contains")

    # 5. Indicateur booléen
    jointure["possede_parking_relais"] = jointure["index_right"].notnull()

    # 6. Agrégation par arrêt
    presences = jointure.groupby("stop_id")["possede_parking_relais"].max().reset_index()

    # 7. Fusion avec les données enrichies
    gdf_resultat = arrets_avec_donnees.merge(presences, on="stop_id", how="left")
    gdf_resultat["possede_parking_relais"] = gdf_resultat["possede_parking_relais"].fillna(False)

    # 8. Statistiques
    total_arrets = len(gdf_resultat)
    arrets_avec_parking = gdf_resultat["possede_parking_relais"].sum()
    taux_couverture = (arrets_avec_parking / total_arrets) * 100

    print("\n--- Statistiques ---")
    print(f"Nombre total d'arrêts de {transport} dans l'EPCI : {total_arrets}")
    print(f"Arrêts avec au moins un parking relais à moins de 400m : {arrets_avec_parking}")
    print(f"Taux de couverture : {taux_couverture:.1f}%")

    # 9. Statistiques supplémentaires sur les parkings
    if arrets_avec_parking > 0:
        parkings_utilises = jointure[jointure["possede_parking_relais"]]["id_parking"].nunique()
        print(f"Parkings relais concernés : {parkings_utilises}/{len(gdf_parkings)}")
        parkings_par_arret = jointure.groupby("stop_id")["id_parking"].count().mean()
        print(f"Nombre moyen de parkings relais par arrêt couvert : {parkings_par_arret:.1f}")

    # 10. Export
    exporter_parquet(gdf_resultat, f"arrets_{transport}_avec_donnees")
    exporter_gpkg(gdf_resultat, f"arrets_{transport}_avec_donnees")


# In[373]:


def afficher_part_arrets_avec_pr(transport="bus", export = False):
    # 1. Chargement des données
    gdf = charger_fichier_parquet(f"arrets_{transport}_avec_donnees", crs=3857)
    limites_epci = charger_fichier_parquet("limites_epci", crs=3857)

    # 2. Affichage
    fig, ax = plt.subplots(figsize=(10, 10))

    limites_epci.plot(ax=ax, color='none', edgecolor='black', linewidth=1)
    gdf[gdf["possede_parking_relais"] == 0].plot(ax=ax, color='red', markersize=5, label='Sans parking relais')
    gdf[gdf["possede_parking_relais"] == 1].plot(ax=ax, color='green', markersize=8, label='Avec parking relais')

    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)

    ax.legend(title=f"Arrêts de {transport}")
    ax.set_title(f"Arrêts de {transport} avec parking relais à moins de 400m", fontsize=14)
    ax.axis('off')

    plt.tight_layout()

    # (Optionnel) Export PNG
    if export :
        images_export_dir = os.path.join(images_dir, f"indicateur_arrets_avec_pr_{transport}.png")
        plt.savefig(images_export_dir, dpi=300, bbox_inches='tight')
        print(f"Carte exportée vers : {images_export_dir}")

    plt.show()
    plt.close()


# In[193]:


def calculer_part_arrets_avec_pr_par_maille(transport="bus"):
    # 1. Chargement des données
    gdf_arrets = charger_fichier_parquet(f"arrets_{transport}_avec_donnees", crs=2154)
    gdf_maille = charger_fichier_parquet("maille_200m_avec_donnees", crs=2154)

    # Nettoyage préalable
    gdf_maille = nettoyer_colonnes(gdf_maille, [f"ratio_arrets_pr_{transport}"])

    # 2. Vérification des géométries
    gdf_arrets = gdf_arrets[gdf_arrets.geometry.notnull() & (gdf_arrets.geometry.type == "Point")].copy()

    # 3. Vérification de la colonne 'possede_parking_relais'
    if "possede_parking_relais" not in gdf_arrets.columns:
        raise ValueError("La colonne 'possede_parking_relais' est absente des données des arrêts.")

    # 4. Jointure spatiale arrêt → maille
    jointure = gpd.sjoin(
        gdf_arrets[["stop_id", "possede_parking_relais", "geometry"]],
        gdf_maille[["idINSPIRE", "geometry"]],
        how="inner",
        predicate="within"
    )

    # 5. Calcul du ratio d'arrêts avec PR par maille
    ratio_par_maille = (
        jointure.groupby("idINSPIRE")
        .agg(
            nb_total_arrets=("stop_id", "count"),
            nb_arrets_pr=("possede_parking_relais", "sum")
        )
        .assign(ratio=lambda df: (df["nb_arrets_pr"] / df["nb_total_arrets"]) * 100)
        .reset_index()[["idINSPIRE", "ratio"]]
        .rename(columns={"ratio": f"ratio_arrets_pr_{transport}"})
    )

    # 6. Arrondi et remplacement des valeurs manquantes
    ratio_par_maille[f"ratio_arrets_pr_{transport}"] = (
        ratio_par_maille[f"ratio_arrets_pr_{transport}"].round(2)
    )

    gdf_maille = gdf_maille.merge(ratio_par_maille, on="idINSPIRE", how="left")
    gdf_maille[f"ratio_arrets_pr_{transport}"] = gdf_maille[f"ratio_arrets_pr_{transport}"].fillna(-1)

    # 7. Export
    exporter_parquet(gdf_maille, "maille_200m_avec_donnees")
    exporter_gpkg(gdf_maille, "maille_200m_avec_donnees")

    # 8. Statistiques
    valides = gdf_maille[gdf_maille[f"ratio_arrets_pr_{transport}"] >= 0]
    print(f"""
Résultats exportés :
- {len(gdf_maille)} carreaux
- {len(valides)} contenant au moins un arrêt
- Taux moyen de couverture par PR : {valides[f"ratio_arrets_pr_{transport}"].mean():.2f}%
""")


# In[392]:


def afficher_part_arrets_avec_pr_par_maille(transport="bus", export = False):
    # 1. Chargement des données
    limites_epci = charger_fichier_parquet("limites_epci", crs=3857)
    carreaux = charger_fichier_parquet("maille_200m_avec_donnees", crs=3857)
    carreaux = carreaux[carreaux[f"ratio_arrets_pr_{transport}"] != -1]

    # 2. Affichage
    fig, ax = plt.subplots(figsize=(10, 10))
    limites_epci.plot(ax=ax, alpha=1, edgecolor='black', facecolor='none')

    carreaux.plot(column=f"ratio_arrets_pr_{transport}",
             cmap="YlGn",
             legend=True,
             # legend_kwds={'label' : f"% des arrêts de {transport} dans la maille possédant un parking relais à proximité"},
             ax=ax,
             linewidth=0.1,
             edgecolor="grey")

    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)

    ax.set_title(f"Part des arrêts de {transport} possédant un parking relais à proximité", fontsize=18)
    ax.axis('off')
    plt.tight_layout()

    # (Optionnel) Export PNG
    if export :
        images_export_dir = os.path.join(images_dir, f"indicateur_arrets_avec_pr_par_maille_{transport}.png")
        plt.savefig(images_export_dir, dpi=300, bbox_inches='tight')
        print(f"Carte exportée vers : {images_export_dir}")

    plt.show()
    plt.close()


# #### 5.7.3. Part des arrêts disposant de stations d'autopartage
# ---
# Ici, on considère qu'un arrêt possède une station d'autopartage si il y en a une à moins de 400 mètres de l'arrêt (6 min à pied pour une marche à 4 km/h)

# In[281]:


def calculer_part_arrets_avec_autopartage(transport="bus"):
    # 1. Chargement des données
    limites_epci = charger_fichier_parquet("limites_epci", crs=2154)
    gdf_citiz = charger_fichier_parquet("citiz_stations", crs=2154).copy()
    arrets_avec_donnees = charger_fichier_parquet(f"arrets_{transport}_avec_donnees", crs=2154).copy()

    # 2. Nettoyage des colonnes déjà présentes et de jointure
    colonnes_a_supprimer = ['possede_autopartage']
    arrets_avec_donnees = nettoyer_colonnes(arrets_avec_donnees, colonnes_a_supprimer)

    # 3. Nettoyage des géométries
    gdf_citiz = gdf_citiz[gdf_citiz.geometry.notnull() & (gdf_citiz.geometry.type == "Point")].copy()
    gdf_arrets = arrets_avec_donnees[arrets_avec_donnees.geometry.notnull() & (arrets_avec_donnees.geometry.type == "Point")].copy()

    # 4. Création de buffers de 400m autour des arrêts
    gdf_arrets["buffer_400m"] = gdf_arrets.geometry.buffer(400)
    gdf_buffers = gpd.GeoDataFrame(gdf_arrets[["stop_id", "buffer_400m"]], geometry="buffer_400m", crs=2154)

    # 5. Jointure spatiale : présence d'une station Citiz dans le buffer
    jointure = gpd.sjoin(
        gdf_buffers,
        gdf_citiz[["geometry"]],
        how="left",
        predicate="contains"
    )

    # 6. Création de l'indicateur numérique (0/1)
    jointure["possede_autopartage"] = jointure["index_right"].notnull().astype(int)  # Conversion en 0/1

    # 7. Agrégation par arrêt
    presences = jointure.groupby("stop_id")["possede_autopartage"].max().reset_index()

    # 8. Fusion avec les arrêts existants
    gdf_resultat = arrets_avec_donnees.merge(presences, on="stop_id", how="left")
    gdf_resultat["possede_autopartage"] = gdf_resultat["possede_autopartage"].fillna(0)  # 0 au lieu de False

    # 9. Statistiques
    nb_arrets = len(gdf_resultat)
    nb_avec_autopartage = gdf_resultat["possede_autopartage"].sum()
    taux_couverture = (nb_avec_autopartage / nb_arrets) * 100 if nb_arrets > 0 else 0

    print("\n--- Statistiques ---")
    print(f"Nombre total d'arrêts de {transport} analysés : {nb_arrets}")
    print(f"Arrêts avec au moins une station d'autopartage à moins de 400m : {nb_avec_autopartage}")
    print(f"Taux de couverture : {taux_couverture:.2f}%")

    # 10. Export
    exporter_parquet(gdf_resultat, f"arrets_{transport}_avec_donnees")
    exporter_gpkg(gdf_resultat, f"arrets_{transport}_avec_donnees")


# In[266]:


def afficher_part_arrets_avec_autopartage(transport="bus", export = False):
    # 1. Chargement des données
    gdf = charger_fichier_parquet(f"arrets_{transport}_avec_donnees", crs=3857)
    limites_epci = charger_fichier_parquet("limites_epci", crs=3857)

    # 2. Affichage
    couleurs = gdf["possede_autopartage"].map({1: "green", 0: "red"})
    fig, ax = plt.subplots(figsize=(10, 10))
    limites_epci.plot(ax=ax, alpha=1, edgecolor='black', facecolor='none')
    gdf.plot(ax=ax, color=couleurs, markersize=10, edgecolor="None", alpha=1)
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
    ax.set_title(f"Présence de stations Citiz autour des arrêts de {transport} (400 m)", fontsize=14)
    ax.axis("off")
    plt.tight_layout()

    # (Optionnel) Export PNG
    if export :
        images_export_dir = os.path.join(images_dir, f"indicateur_arrets_avec_autopartage{transport}.png")
        plt.savefig(images_export_dir, dpi=300, bbox_inches='tight')
        print(f"Carte exportée vers : {images_export_dir}")

    plt.show()
    plt.close()


# In[153]:


def calculer_part_arrets_avec_autopartage_par_maille(transport="bus"):
    # 1. Chargement des données
    gdf_arrets = charger_fichier_parquet(f"arrets_{transport}_avec_donnees", crs=2154)
    gdf_maille = charger_fichier_parquet("maille_200m_avec_donnees", crs=2154)

    # Nettoyage préalable
    gdf_maille = nettoyer_colonnes(gdf_maille, [f"ratio_arrets_autopartage_{transport}"])

    # 2. Vérification des géométries
    gdf_arrets = gdf_arrets[gdf_arrets.geometry.notnull() & (gdf_arrets.geometry.type == "Point")].copy()

    # 3. Vérification de la colonne 'possede_autopartage'
    if "possede_autopartage" not in gdf_arrets.columns:
        raise ValueError("La colonne 'possede_autopartage' est absente des données des arrêts.")

    # 4. Jointure spatiale arrêt → maille
    jointure = gpd.sjoin(
        gdf_arrets[["stop_id", "possede_autopartage", "geometry"]],
        gdf_maille[["idINSPIRE", "geometry"]],
        how="inner",
        predicate="within"
    )

    # 5. Calcul du ratio d'arrêts avec autopartage par maille
    ratio_par_maille = (
        jointure.groupby("idINSPIRE")
        .agg(
            nb_total_arrets=("stop_id", "count"),
            nb_arrets_autopartage=("possede_autopartage", "sum")
        )
        .assign(ratio=lambda df: (df["nb_arrets_autopartage"] / df["nb_total_arrets"]) * 100)
        .reset_index()[["idINSPIRE", "ratio"]]
        .rename(columns={"ratio": f"ratio_arrets_autopartage_{transport}"})
    )

    # 6. Arrondi et remplacement des valeurs manquantes
    ratio_par_maille[f"ratio_arrets_autopartage_{transport}"] = (
        ratio_par_maille[f"ratio_arrets_autopartage_{transport}"].round(2)
    )

    gdf_maille = gdf_maille.merge(ratio_par_maille, on="idINSPIRE", how="left")
    gdf_maille[f"ratio_arrets_autopartage_{transport}"] = gdf_maille[f"ratio_arrets_autopartage_{transport}"].fillna(-1)

    # 7. Export
    exporter_parquet(gdf_maille, "maille_200m_avec_donnees")
    exporter_gpkg(gdf_maille, "maille_200m_avec_donnees")

    # 8. Statistiques
    valides = gdf_maille[gdf_maille[f"ratio_arrets_autopartage_{transport}"] >= 0]
    print(f"""
Résultats exportés :
- {len(gdf_maille)} carreaux
- {len(valides)} contenant au moins un arrêt
- Taux moyen de couverture : {valides[f"ratio_arrets_autopartage_{transport}"].mean():.2f}%
""")


# In[395]:


def afficher_part_arrets_avec_autopartage_par_maille(transport="bus", export = False):
    # 1. Chargement des données
    limites_epci = charger_fichier_parquet("limites_epci", crs=3857)
    carreaux = charger_fichier_parquet("maille_200m_avec_donnees", crs=3857)
    carreaux = carreaux[carreaux[f"ratio_arrets_autopartage_{transport}"] != -1]

    # 2. Affichage
    fig, ax = plt.subplots(figsize=(10, 10))
    limites_epci.plot(ax=ax, alpha=1, edgecolor='black', facecolor='none')

    carreaux.plot(column=f"ratio_arrets_autopartage_{transport}",
             cmap="YlGn",
             legend=True,
             #legend_kwds={'label' : f"% des arrêts de {transport} dans la maille possédant une station d'autopartage Citiz à proximité"},
             ax=ax,
             linewidth=0.1,
             edgecolor="grey")

    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)

    ax.set_title(f"Part des arrêts de {transport} possédant une station \nd'autopartage Citiz à proximité", fontsize = 18)
    ax.axis('off')
    plt.tight_layout()

    # (Optionnel) Export PNG
    if export :
        images_export_dir = os.path.join(images_dir, f"indicateur_arrets_avec_autopartage_par_maille_{transport}.png")
        plt.savefig(images_export_dir, dpi=300, bbox_inches='tight')
        print(f"Carte exportée vers : {images_export_dir}")

    plt.show()
    plt.close()


# #### 5.7.4. Part de surface bâtie à moins de 400 mètres d'un arrêt
# ---

# In[507]:


def calculer_part_surface_batie_autour_arrets(transport="bus"):
    # 1. Chargement des données
    arrets = charger_fichier_parquet(f"arrets_{transport}_avec_donnees", crs=2154)
    bd_topo_batiments = charger_fichier_parquet("bd_topo_batiments_tampon", crs=2154)
    limites_france = charger_fichier_parquet("limites_france", crs=2154)

    # 2. Création du buffer 400m autour des arrêts
    arrets["buffer_400m"] = arrets.geometry.buffer(400)

    # 3. Détection des buffers partiellement hors de France
    buffer_unifie_france = limites_france.geometry.union_all()
    arrets["buffer_entierement_en_france"] = arrets["buffer_400m"].within(buffer_unifie_france)

    # 4. Calcul surface de chaque buffer
    surface_buffer_reference = arrets["buffer_400m"].iloc[0].area
    arrets["surface_buffer"] = arrets.apply(
        lambda row: row["buffer_400m"].area if not row["buffer_entierement_en_france"] else surface_buffer_reference,
        axis=1
    )

    # 5. Calcul de la surface bâtie par intersection
    sindex_bat = bd_topo_batiments.sindex
    surface_batie_list = []

    for idx, row in arrets.iterrows():
        buffer_geom = row["buffer_400m"]
        possibles = list(sindex_bat.intersection(buffer_geom.bounds))
        batiments_possibles = bd_topo_batiments.iloc[possibles]
        batiments_intersects = batiments_possibles[batiments_possibles.intersects(buffer_geom)]

        surface = batiments_intersects.intersection(buffer_geom).area.sum()
        surface_batie_list.append(surface)

    arrets["surface_batie"] = surface_batie_list

    # 6. Calcul du pourcentage
    arrets["part_bati_stations"] = (
        arrets["surface_batie"] / arrets["surface_buffer"] * 100
    ).round(2)

    # 7. Nettoyage
    arrets.drop(columns=["buffer_400m", "surface_batie", "surface_buffer", "buffer_entierement_en_france"], inplace=True)

    # 8. Export
    exporter_parquet(arrets, f"arrets_{transport}_avec_donnees")
    exporter_gpkg(arrets, f"arrets_{transport}_avec_donnees")
    print("Export terminé avec succès.")


# In[97]:


def afficher_part_surface_batie_autour_arrets(transport="bus", export = False):
    # 1. Chargement des données
    arrets = charger_fichier_parquet(f"arrets_{transport}_avec_donnees", crs=3857)
    limites_epci = charger_fichier_parquet("limites_epci", crs=3857)

    # 2. Création des buffers pour affichage
    arrets["buffer_400m"] = arrets.geometry.buffer(400)

    # 3. Construction du GeoDataFrame des buffers pour la visualisation
    gdf_buffers = gpd.GeoDataFrame(
        arrets[["stop_id", "part_bati_stations"]],
        geometry=arrets["buffer_400m"],
        crs=3857
    )

    # 4. Affichage
    fig, ax = plt.subplots(figsize=(10, 10))

    limites_epci.plot(ax=ax, color='none', edgecolor='black', linewidth=1)

    gdf_buffers.plot(
        ax=ax,
        column='part_bati_stations',
        cmap='YlGn',
        alpha=0.7,
        legend=True,
        legend_kwds={
            'label': "Part du bâti dans un rayon de 400m (%)",
            'shrink': 0.7,
            'format': "%.0f%%"
        }
    )

    arrets.plot(ax=ax, color='black', markersize=15, label=f"Arrêts de {transport}")

    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)

    ax.set_title(f"Densité du bâti autour des arrêts de {transport} (400m)", fontsize=14)
    ax.axis('off')
    plt.tight_layout()

    # (Optionnel) Export PNG
    if export :
        images_export_dir = os.path.join(images_dir, f"indicateur_surface_batie_autour_arret_{transport}.png")
        plt.savefig(images_export_dir, dpi=300, bbox_inches='tight')
        print(f"Carte exportée vers : {images_export_dir}")

    plt.show()
    plt.close()


# In[124]:


def calculer_part_surface_batie_autour_arrets_par_maille(transport="bus"):
    # 1. Chargement des données
    gdf_arrets = charger_fichier_parquet(f"arrets_{transport}_avec_donnees", crs=2154)
    gdf_maille = charger_fichier_parquet("maille_200m_avec_donnees", crs=2154)
    bd_topo_batiments = charger_fichier_parquet("bd_topo_batiments_tampon", crs=2154)
    limites_france = charger_fichier_parquet("limites_france", crs=2154)

    # 2. Nettoyage
    gdf_arrets = gdf_arrets[gdf_arrets.geometry.notnull() & (gdf_arrets.geometry.type == "Point")].copy()
    gdf_maille = nettoyer_colonnes(gdf_maille, [f"part_batie_autour_arrets_{transport}"])

    # 3. Buffer autour des arrêts
    gdf_arrets["buffer_400m"] = gdf_arrets.geometry.buffer(400)

    # 4. Détection des buffers partiellement hors de France
    union_france = limites_france.geometry.union_all()
    surface_buffer_ref = gdf_arrets["buffer_400m"].iloc[0].area
    gdf_arrets["buffer_entierement_en_france"] = gdf_arrets["buffer_400m"].within(union_france)
    gdf_arrets["surface_buffer"] = gdf_arrets.apply(
        lambda row: row["buffer_400m"].area if not row["buffer_entierement_en_france"] else surface_buffer_ref,
        axis=1
    )

    # 5. Calcul surface bâtie intersectée
    sindex_bat = bd_topo_batiments.sindex
    surface_batie_list = []

    for idx, row in gdf_arrets.iterrows():
        buffer_geom = row["buffer_400m"]
        possibles = list(sindex_bat.intersection(buffer_geom.bounds))
        candidats = bd_topo_batiments.iloc[possibles]
        intersectes = candidats[candidats.intersects(buffer_geom)]
        surface = intersectes.intersection(buffer_geom).area.sum()
        surface_batie_list.append(surface)

    gdf_arrets["surface_batie"] = surface_batie_list
    gdf_arrets["part_batie_stations"] = (
        gdf_arrets["surface_batie"] / gdf_arrets["surface_buffer"] * 100
    )

    # 6. Jointure spatiale arrêt -> maille
    jointure = gpd.sjoin(
        gdf_arrets[["stop_id", "part_batie_stations", "geometry"]],
        gdf_maille[["idINSPIRE", "geometry"]],
        how="inner",
        predicate="within"
    )

    # 7. Moyenne de la part bâtie par maille
    moyenne_par_maille = (
        jointure.groupby("idINSPIRE")["part_batie_stations"]
        .mean()
        .reset_index()
        .rename(columns={"part_batie_stations": f"part_batie_autour_arrets_{transport}"})
    )

    # 8. Intégration dans la maille
    gdf_maille = gdf_maille.merge(moyenne_par_maille, on="idINSPIRE", how="left")
    gdf_maille[f"part_batie_autour_arrets_{transport}"] = (
        gdf_maille[f"part_batie_autour_arrets_{transport}"]
        .round(2)
        .fillna(-1)
    )

    # 9. Export
    exporter_parquet(gdf_maille, "maille_200m_avec_donnees")
    exporter_gpkg(gdf_maille, "maille_200m_avec_donnees")

    # 10. Statistiques
    valides = gdf_maille[gdf_maille[f"part_batie_autour_arrets_{transport}"] >= 0]
    print(f"""
Résultats exportés :
- {len(gdf_maille)} carreaux
- {len(valides)} avec au moins un arrêt à proximité
- Moyenne des parts de surface bâtie autour des arrêts : {valides[f"part_batie_autour_arrets_{transport}"].mean():.2f}%
""")


# In[378]:


def afficher_part_surface_batie_autour_arrets_par_maille(transport="bus", export = False):
    # 1. Chargement des données
    limites_epci = charger_fichier_parquet("limites_epci", crs=3857)
    carreaux = charger_fichier_parquet("maille_200m_avec_donnees", crs=3857)
    carreaux = carreaux[carreaux[f"part_batie_autour_arrets_{transport}"] != -1]

    # 2. Affichage
    fig, ax = plt.subplots(figsize=(10, 10))
    limites_epci.plot(ax=ax, alpha=1, edgecolor='black', facecolor='none')

    carreaux.plot(column=f"part_batie_autour_arrets_{transport}",
             cmap="YlGn",
             legend=True,
             #legend_kwds={'label' : f"% de surface bâtie autour des arrêts de {transport}"},
             ax=ax,
             linewidth=0.1,
             edgecolor="grey",
             vmin = 0,
             vmax = 100)

    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)

    ax.set_title(f"Part de surface bâtie autour des arrêts de {transport}", fontsize=18)
    ax.axis('off')
    plt.tight_layout()

    # (Optionnel) Export PNG
    if export :
        images_export_dir = os.path.join(images_dir, f"indicateur_part_surface_batie_par_maille_{transport}.png")
        plt.savefig(images_export_dir, dpi=300, bbox_inches='tight')
        print(f"Carte exportée vers : {images_export_dir}")

    plt.show()
    plt.close()


# In[502]:


def calculer_nombre_accidents():
    # 1. Chargement des données
    carreaux = charger_fichier_parquet("maille_200m_avec_donnees", crs=3857).copy()
    accidents = charger_fichier_parquet("accidents_routiers_epci", crs=3857).copy()

    # 2. Nettoyage : s'assurer que la colonne catv est bien numérique
    accidents["catv"] = pd.to_numeric(accidents["catv"], errors="coerce")

    # 3. Filtrer les accidents impliquant un autobus
    accidents_bus = accidents[accidents["catv"] == 37].copy()

    # 4. Jointure spatiale : associer chaque accident à un carreau
    accidents_avec_carreaux = gpd.sjoin(
        accidents_bus,
        carreaux[["idINSPIRE", "geometry"]],
        how="inner",
        predicate="within"
    )

    # 5. Agrégation : nombre d'accidents par carreau
    nb_accidents_par_carreau = accidents_avec_carreaux.groupby("idINSPIRE").size().reset_index(name="nb_accidents_bus")

    # 6. Fusion avec les carreaux
    if "nb_accidents_bus" in carreaux.columns:
        carreaux = carreaux.drop(columns="nb_accidents_bus")

    carreaux = carreaux.merge(nb_accidents_par_carreau, on="idINSPIRE", how="left")
    carreaux["nb_accidents_bus"] = carreaux["nb_accidents_bus"].fillna(0).astype(int)

    # 7. Export
    exporter_parquet(carreaux, "maille_200m_avec_accidents")
    exporter_gpkg(carreaux, "maille_200m_avec_accidents")


# #### 5.7.5. Calcul du temps d'attente moyen
# ---

# ##### 5.7.5.1. Par arrêt
# ---

# In[283]:


def calculer_temps_attente_moyen(suffixe="bus"):
    # 1. Chargement des données
    df_stop_times = pd.read_csv(os.path.join(exports_dir, f"gtfs_stop_times_{suffixe}.csv"))
    df_trips = pd.read_csv(os.path.join(exports_dir, f"gtfs_trips_{suffixe}.csv"))
    df_calendar = pd.read_csv(os.path.join(exports_dir, "gtfs_calendar.csv"))
    gdf_stops = gpd.read_parquet(os.path.join(exports_dir, f"gtfs_stops_{suffixe}.parquet"))[["stop_id", "geometry"]]
    limites_epci = charger_fichier_parquet("limites_epci", crs=4326)

    # 2. Jointure stop_times + trips
    df = df_stop_times.merge(df_trips[["trip_id", "service_id"]], on="trip_id")

    # 3. Associer chaque service_id aux jours actifs
    jours = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    jours_dict = {jour: df_calendar[df_calendar[jour] == 1]["service_id"].unique() for jour in jours}

    # 4. Parsing des horaires
    def parse_time(h):
        try:
            return datetime.strptime(h, "%H:%M:%S").time()
        except:
            return None

    df["arrival_time_dt"] = df["arrival_time"].apply(parse_time)
    df = df[df["arrival_time_dt"].notnull()]
    df["arrival_hour"] = df["arrival_time_dt"].apply(lambda t: t.hour + t.minute / 60)

    # 5. Calcul EATV pour chaque jour
    df_result = pd.DataFrame({"stop_id": df["stop_id"].unique()})

    for jour in jours:
        sous_df = df[df["service_id"].isin(jours_dict[jour])]
        res = []

        for stop_id, group in sous_df.groupby("stop_id"):
            heures = sorted(group["arrival_hour"].tolist())
            if len(heures) < 2:
                continue
            dk = heures[-1] - heures[0]
            if dk <= 0:
                continue
            fk = len(heures) / dk
            eatv = 3600 / fk
            res.append({"stop_id": stop_id, f"att_{jour}": eatv})

        df_jour = pd.DataFrame(res)
        df_result = df_result.merge(df_jour, on="stop_id", how="left")

    # 6. Moyennes : semaine, week-end, hebdo
    jours_semaine = ["monday", "tuesday", "wednesday", "thursday", "friday"]
    jours_weekend = ["saturday", "sunday"]

    df_result["att_secondes_moy_semaine"] = df_result[[f"att_{j}" for j in jours_semaine]].mean(axis=1)
    df_result["att_secondes_moy_week_end"] = df_result[[f"att_{j}" for j in jours_weekend]].mean(axis=1)
    df_result["att_secondes_moy_hebdo"] = df_result[[f"att_{j}" for j in jours]].mean(axis=1)

    # 7. Jointure géométrique + filtre EPCI
    gdf_result = gdf_stops.merge(df_result, on="stop_id", how="left")
    gdf_result = gdf_result.to_crs(4326)
    gdf_result = gpd.sjoin(gdf_result, limites_epci[["geometry"]], predicate="within", how="inner").drop(columns="index_right")

    # 8. Arrondi
    for col in ["att_secondes_moy_semaine", "att_secondes_moy_week_end", "att_secondes_moy_hebdo"]:
        gdf_result[col] = gdf_result[col].round(2)

    # 9. Suppression des colonnes temporaires
    colonnes_a_supprimer = [f"att_{j}" for j in jours]
    gdf_result = nettoyer_colonnes(gdf_result, colonnes_a_supprimer)

    # 10. Export
    nom_base = f"arrets_{suffixe}_avec_donnees"
    exporter_parquet(gdf_result, nom_base)
    exporter_gpkg(gdf_result, nom_base)

    print(f"Temps d’attente moyen calculé (semaine, week-end, hebdo) pour {suffixe}")


# In[177]:


def calculer_temps_attente_moyen(suffixe="bus"):
    # 1. Chargement des données
    df_stop_times = pd.read_csv(os.path.join(exports_dir, f"gtfs_stop_times_{suffixe}.csv"))
    df_trips = pd.read_csv(os.path.join(exports_dir, f"gtfs_trips_{suffixe}.csv"))
    df_calendar = pd.read_csv(os.path.join(exports_dir, "gtfs_calendar.csv"))
    gdf_stops = gpd.read_parquet(os.path.join(exports_dir, f"gtfs_stops_{suffixe}.parquet"))[["stop_id", "geometry"]]
    limites_epci = charger_fichier_parquet("limites_epci", crs=4326)

    # 2. Associer chaque trip à son service_id
    df = df_stop_times.merge(df_trips[["trip_id", "service_id"]], on="trip_id")

    # 3. Associer les jours de la semaine à chaque service_id
    jours = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    jours_dict = {j: [] for j in jours}
    for jour in jours:
        ids = df_calendar[df_calendar[jour] == 1]["service_id"].unique()
        jours_dict[jour] = ids

    # 4. Parse les horaires
    def parse_time(h):
        try:
            return datetime.strptime(h, "%H:%M:%S").time()
        except:
            return None

    df["arrival_time_dt"] = df["arrival_time"].apply(parse_time)
    df = df[df["arrival_time_dt"].notnull()]
    df["arrival_hour"] = df["arrival_time_dt"].apply(lambda t: t.hour + t.minute / 60)

    # 5. Calcule EATV pour chaque jour
    df_result = pd.DataFrame({"stop_id": df["stop_id"].unique()})

    for jour in jours:
        sous_df = df[df["service_id"].isin(jours_dict[jour])]
        res = []

        for stop_id, group in sous_df.groupby("stop_id"):
            heures = sorted(group["arrival_hour"].tolist())
            if len(heures) < 2:
                continue
            dk = heures[-1] - heures[0]
            if dk <= 0:
                continue
            fk = len(heures) / dk
            eatv = 3600 / fk
            res.append({"stop_id": stop_id, f"att_{jour}": eatv})

        df_jour = pd.DataFrame(res)
        df_result = df_result.merge(df_jour, on="stop_id", how="left")

    # 6. Calcul de la moyenne semaine et week-end
    jours_semaine = ["monday", "tuesday", "wednesday", "thursday", "friday"]
    jours_weekend = ["saturday", "sunday"]

    df_result["att_secondes_moy_semaine"] = df_result[[f"att_{j}" for j in jours_semaine]].mean(axis=1)
    df_result["att_secondes_moy_week_end"] = df_result[[f"att_{j}" for j in jours_weekend]].mean(axis=1)

    # 7. Jointure avec géométrie + filtre EPCI
    gdf_result = gdf_stops.merge(df_result, on="stop_id", how="left")
    gdf_result = gdf_result.to_crs(4326)
    gdf_result = gpd.sjoin(gdf_result, limites_epci[["geometry"]], predicate="within", how="inner").drop(columns="index_right")

    # 8. Arrondi
    gdf_result["att_secondes_moy_semaine"] = gdf_result["att_secondes_moy_semaine"].round(2)
    gdf_result["att_secondes_moy_week_end"] = gdf_result["att_secondes_moy_week_end"].round(2)

    # 9. Supprimer les colonnes de test 'att_monday' à 'att_sunday', servant à vérifier les données
    colonnes_a_supprimer = ['att_monday', 'att_tuesday', 'att_wednesday', 'att_thursday',
                           'att_friday', 'att_saturday', 'att_sunday']

    gdf_result = nettoyer_colonnes(gdf_result, colonnes_a_supprimer)

    nom_base = f"arrets_{suffixe}_avec_donnees"
    exporter_parquet(gdf_result, nom_base)
    exporter_gpkg(gdf_result, nom_base)

    print(f"Temps d’attente moyen calculé pour tous les jours – semaine et week-end (suffixe = {suffixe})")


# In[285]:


def afficher_temps_attente_moyen_par_arret(transport="bus", export=False):
    # 1. Chargement des données
    limites_epci = charger_fichier_parquet("limites_epci", crs=3857)
    gdf = charger_fichier_parquet(f"arrets_{transport}_avec_donnees", crs=3857)
    gdf = gdf[gdf["att_secondes_moy_semaine"].notnull()].copy()

    # 2. Affichage
    cmap = plt.cm.viridis_r
    norm = mcolors.Normalize(vmin=gdf["att_secondes_moy_hebdo"].min(), vmax=gdf["att_secondes_moy_hebdo"].max())

    fig, ax = plt.subplots(figsize=(10, 10))
    limites_epci.plot(ax=ax, alpha=1, edgecolor='black', facecolor='none')

    gdf.plot(ax=ax,
             column="att_secondes_moy_hebdo",
             cmap=cmap,
             markersize=10,
             legend=True,
             norm=norm)

    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)

    ax.set_title(f"Temps d'attente moyen en semaine (secondes) – arrêts {transport}", fontsize=14)
    ax.axis("off")
    plt.tight_layout()

    # 3. Export PNG (optionnel)
    if export:
        images_export_path = os.path.join(images_dir, f"attente_moyenne_arrets_{transport}.png")
        plt.savefig(images_export_path, dpi=300, bbox_inches="tight")
        print(f"Carte exportée vers : {images_export_path}")

    plt.show()
    plt.close()


# ##### 5.7.5.2. Par maille
# ---

# In[296]:


# Calcul du lundi au vendredi, ou le week-end
def calculer_temps_attente_par_maille(transport="bus"):
    # 1. Chargement des données
    gdf_arrets = charger_fichier_parquet(f"arrets_{transport}_avec_donnees", crs=2154)
    gdf_maille = charger_fichier_parquet("maille_200m_avec_donnees", crs=2154)

    # 2. Nettoyage des colonnes existantes
    gdf_maille = nettoyer_colonnes(gdf_maille, [f"att_moy_hebdo_{transport}"])

    # 3. Filtrage des arrêts valides
    gdf_arrets_valides = gdf_arrets[
        gdf_arrets["att_secondes_moy_hebdo"].notnull() & gdf_arrets.geometry.notnull()
    ].copy()

    # 4. Jointure spatiale arrêt → maille
    jointure = gpd.sjoin(
        gdf_arrets_valides[["stop_id", "att_secondes_moy_hebdo", "geometry"]],
        gdf_maille[["idINSPIRE", "geometry"]],
        how="inner",
        predicate="within"
    )

    # 5. Moyenne par maille
    attente_moyenne_par_maille = (
        jointure.groupby("idINSPIRE")["att_secondes_moy_hebdo"]
        .mean()
        .reset_index()
        .rename(columns={"att_secondes_moy_hebdo": f"att_moy_hebdo_{transport}"})
    )

    # 6. Arrondi et fusion
    attente_moyenne_par_maille[f"att_moy_hebdo_{transport}"] = (
        attente_moyenne_par_maille[f"att_moy_hebdo_{transport}"].round(2)
    )
    gdf_maille = gdf_maille.merge(attente_moyenne_par_maille, on="idINSPIRE", how="left")
    gdf_maille[f"att_moy_hebdo_{transport}"] = gdf_maille[f"att_moy_hebdo_{transport}"].fillna(-1)

    # 7. Export
    exporter_parquet(gdf_maille, "maille_200m_avec_donnees")
    exporter_gpkg(gdf_maille, "maille_200m_avec_donnees")

    # 8. Statistiques
    valides = gdf_maille[gdf_maille[f"att_moy_hebdo_{transport}"] >= 0]
    print(f"""
Résultats exportés :
- {len(gdf_maille)} carreaux
- {len(valides)} avec au moins un arrêt de {transport}
- Temps d'attente moyen (en hebdo) : {valides[f"att_moy_hebdo_{transport}"].mean():.2f} secondes
""")


# In[371]:


def afficher_temps_attente_par_maille(transport="bus", export=False):
    # 1. Chargement des données
    limites_epci = charger_fichier_parquet("limites_epci", crs=3857)
    carreaux = charger_fichier_parquet("maille_200m_avec_donnees", crs=3857)

    # 2. Filtrage des valeurs valides
    col_attente = f"att_moy_hebdo_{transport}"
    carreaux = carreaux[carreaux[col_attente] != -1]

    # 3. Affichage
    fig, ax = plt.subplots(figsize=(10, 10))
    limites_epci.plot(ax=ax, alpha=1, edgecolor='black', facecolor='none')

    carreaux.plot(
        column=col_attente,
        cmap="YlOrRd",
        legend=True,
        # legend_kwds={'label': f"Temps d'attente moyen ({transport}) hebdomadaire (secondes)"},
        ax=ax,
        linewidth=0.1,
        edgecolor="grey"
    )

    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
    ax.set_title(f"Temps d'attente moyen hebdomadaire – {transport} (s)", fontsize=18)
    ax.axis('off')
    plt.tight_layout()

    # 4. Export PNG si demandé
    if export:
        images_export_path = os.path.join(images_dir, f"indicateur_temps_attente_maille_{transport}.png")
        plt.savefig(images_export_path, dpi=300, bbox_inches='tight')
        print(f"Carte exportée vers : {images_export_path}")

    plt.show()
    plt.close()


# ### 5.8. Nombre d'accidents routiers
# ---

# In[225]:


def calculer_nb_accidents(transport="total"):
    # 1. Chargement des données de base
    carreaux = charger_fichier_parquet("maille_200m_avec_donnees", crs=3857).copy()
    accidents = charger_fichier_parquet("accidents_routiers_epci", crs=3857).copy()

    # 2. Définition des paramètres selon le type de transport
    configs = {
        "pietons": {
            "colonne_filtre": "catu",
            "valeurs": [3],
            "colonne_resultat": "nb_accidents_pieton"
        },
        "velos": {
            "colonne_filtre": "catv",
            "valeurs": [1],
            "colonne_resultat": "nb_accidents_velo"
        },
        "bus": {
            "colonne_filtre": "catv",
            "valeurs": [37],
            "colonne_resultat": "nb_accidents_bus"
        },
        "tram": {
            "colonne_filtre": "catv",
            "valeurs": [40],
            "colonne_resultat": "nb_accidents_tram"
        },
        "vl": {
            "colonne_filtre": "catv",
            "valeurs": [7, 30, 31, 32, 33, 34, 80],
            "colonne_resultat": "nb_accidents_legers"
        },
        "total": {
            "colonne_resultat": "nb_accidents_circulation"
        }
    }

    if transport not in configs:
        raise ValueError(f"Mode '{transport}' non reconnu. Choisir parmi : {', '.join(configs.keys())}")

    config = configs[transport]

    # 3. Filtrage des accidents si nécessaire
    if transport != "total":
        colonne = config["colonne_filtre"]
        accidents[colonne] = pd.to_numeric(accidents[colonne], errors="coerce")
        accidents_filtrés = accidents[accidents[colonne].isin(config["valeurs"])].copy()
    else:
        accidents_filtrés = accidents.copy()

    # 4. Jointure spatiale avec les carreaux
    accidents_avec_carreaux = gpd.sjoin(
        accidents_filtrés,
        carreaux[["idINSPIRE", "geometry"]],
        how="inner",
        predicate="within"
    )

    # 5. Agrégation du nombre d'accidents par carreau
    nom_colonne = config["colonne_resultat"]
    nb_accidents_par_carreau = accidents_avec_carreaux.groupby("idINSPIRE").size().reset_index(name=nom_colonne)

    # 6. Suppression de la colonne si elle existe déjà
    if nom_colonne in carreaux.columns:
        carreaux = carreaux.drop(columns=nom_colonne)

    # 7. Fusion avec les carreaux
    carreaux = carreaux.merge(nb_accidents_par_carreau, on="idINSPIRE", how="left")
    carreaux[nom_colonne] = carreaux[nom_colonne].fillna(0).astype(int)

    # 8. Export
    if transport == "total":
        exporter_parquet(carreaux, "maille_200m_avec_donnees")
        exporter_gpkg(carreaux, "maille_200m_avec_donnees")
    else:
        nom_fichier = f"maille_200m_accidents_{transport}"
        exporter_parquet(carreaux[["idINSPIRE", "geometry", nom_colonne]], nom_fichier)
        exporter_gpkg(carreaux[["idINSPIRE", "geometry", nom_colonne]], nom_fichier)

    print(f"{nom_colonne} calculée et exportée pour le mode : {transport}")


# In[441]:


def afficher_nb_accidents(transport="", export = False):
    # 1. Paramètres
    config = {
        "pietons": {
            "colonne": "nb_accidents_pieton",
            "label": "Nombre d'accidents piétons (2023)",
            "titre": "Accidents impliquant des piétons (2023)"
        },
        "velos": {
            "colonne": "nb_accidents_velo",
            "label": "Nombre d'accidents vélos (2023)",
            "titre": "Accidents impliquant des vélos (2023)"
        },
        "bus": {
            "colonne": "nb_accidents_bus",
            "label": "Nombre d'accidents bus (2023)",
            "titre": "Accidents impliquant des bus (2023)"
        },
        "tram": {
            "colonne": "nb_accidents_tram",
            "label": "Nombre d'accidents tram (2023)",
            "titre": "Accidents impliquant des trams (2023)"
        },
        "vl": {
            "colonne": "nb_accidents_legers",
            "label": "Nombre d'accidents véhicules légers (2023)",
            "titre": "Accidents impliquant des véhicules légers (2023)"
        },
        "total": {
            "colonne": "nb_accidents_circulation",
            "label": "Nombre total d'accidents (2023)",
            "titre": "Nombre total des accidents de circulation (2023)"
        }
    }

    if transport not in config:
        raise ValueError(f"Transport non reconnu. Choisir parmi : {', '.join(config.keys())}")

    params = config[transport]

    # 2. Chargement des données
    limites_epci = charger_fichier_parquet("limites_epci", crs=3857)

    if transport == "total":
        carreaux = charger_fichier_parquet("maille_200m_avec_donnees", crs=3857)
    else:
        carreaux = charger_fichier_parquet(f"maille_200m_accidents_{transport}", crs=3857)

    # 3. Affichage
    fig, ax = plt.subplots(figsize=(10, 10))

    limites_epci.plot(ax=ax, alpha=1, edgecolor='black', facecolor='none')

    carreaux.plot(
        column=params["colonne"],
        cmap="YlOrRd",
        legend=True,
        #legend_kwds={'label': params["label"]},
        ax=ax,
        linewidth=0.1,
        edgecolor="grey"
    )

    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)

    ax.set_title(params["titre"], fontsize=18)
    ax.axis('off')
    plt.tight_layout()

    # (Optionnel) Export PNG
    if export :
        images_export_dir = os.path.join(images_dir, f"indicateur_nb_accidents_{transport}.png")
        plt.savefig(images_export_dir, dpi=300, bbox_inches='tight')
        print(f"Carte exportée vers : {images_export_dir}")

    plt.show()
    plt.close()


# ### 5.9. Affichage des % de routes réservées aux transport
# ---

# In[399]:


def afficher_part_routes_reservees(transport="", export = False):
    # 1. Configuration selon le mode
    config = {
        "bus": {
            "colonne": "part_routes_bus",
            "titre": "Part des routes réservées aux bus",
            "label_legende": "Part des routes réservées aux bus"
        },
        "tram": {
            "colonne": "part_routes_tram",
            "titre": "Part des routes réservées aux trams",
            "label_legende": "Part des voies réservées aux trams"
        }
    }

    if transport not in config:
        raise ValueError("Transport invalide. Options : 'bus' ou 'tram'")

    col = config[transport]["colonne"]

    # 2. Chargement des données
    limites_epci = charger_fichier_parquet("limites_epci", crs=3857)
    carreaux = charger_fichier_parquet("maille_200m_avec_donnees", crs=3857)

    if col not in carreaux.columns:
        raise ValueError(f"Colonne '{col}' manquante dans les données.")

    carreaux_valides = carreaux[carreaux[col] >= 0]

    # 3. Affichage
    fig, ax = plt.subplots(figsize=(10, 10))
    limites_epci.plot(ax=ax, alpha=1, edgecolor='black', facecolor='none')

    carreaux_valides.plot(
        column=col,
        cmap="YlGn",
        #vmin=0, vmax=100,
        legend=True,
        # legend_kwds={'label': config[transport]["label_legende"]},
        ax=ax,
        linewidth=0.1,
        edgecolor="grey"
    )

    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)

    ax.set_title(config[transport]["titre"], fontsize=18)
    ax.axis('off')
    plt.tight_layout()

    # (Optionnel) Export PNG
    if export :
        images_export_dir = os.path.join(images_dir, f"indicateur_part_routes_reservees_{transport}.png")
        plt.savefig(images_export_dir, dpi=300, bbox_inches='tight')
        print(f"Carte exportée vers : {images_export_dir}")

    plt.show()
    plt.close()


# ### 5.10. Affichage de la part des emplois et services disponibles
# ---

# In[350]:


def afficher_ratio_emplois_proches(nom_graphe = "marche", export = False):
    # 1. Chargement des données
    limites_epci = charger_fichier_parquet("limites_epci", crs=3857)
    carreaux = charger_fichier_parquet("maille_200m_avec_donnees", crs=3857)
    carreaux = carreaux[carreaux[f"{nom_graphe}_ratio_emplois"] != -1]
    # NOTE : temporaire
    carreaux[f"{nom_graphe}_ratio_emplois"] = carreaux[f"{nom_graphe}_ratio_emplois"]*100

    # 2. Affichage
    fig, ax = plt.subplots(figsize=(10, 10))
    limites_epci.plot(ax=ax, alpha=1, edgecolor='black', facecolor='none')

    carreaux.plot(
        column=(f"{nom_graphe}_ratio_emplois"),
        cmap="YlGn",
        # vmin=0, vmax=0.5,
        legend=True,
        ax=ax,
        edgecolor="none"
    )
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
    ax.set_axis_off()
    plt.title(f"Part d'emplois accessibles en 15 minutes \ndepuis la maille ({nom_graphe})", fontsize=18)
    plt.tight_layout()

    # (Optionnel) Export PNG
    if export :
        images_export_dir = os.path.join(images_dir, f"visualisation_ratio_emplois_{nom_graphe}.png")
        plt.savefig(images_export_dir, dpi=300, bbox_inches='tight')
        print(f"Carte exportée vers : {images_export_dir}")

    plt.show()
    plt.close()


# In[346]:


def afficher_ratio_services_proches(nom_graphe="marche", export=False):
    # 1. Chargement des données
    limites_epci = charger_fichier_parquet("limites_epci", crs=3857)
    carreaux = charger_fichier_parquet("maille_200m_avec_donnees", crs=3857)

    # 2. Filtrage des colonnes existantes et remplacement des -1 par NaN
    carreaux[f"{nom_graphe}_ratio_services_dom_moyenne"] = carreaux[f"{nom_graphe}_ratio_services_dom_moyenne"].replace(-1, np.nan)

        # NOTE : temporaire
    carreaux[f"{nom_graphe}_ratio_services_dom_moyenne"] = carreaux[f"{nom_graphe}_ratio_services_dom_moyenne"]*100

    # 3. Filtrage des lignes valides
    carreaux = carreaux[carreaux[f"{nom_graphe}_ratio_services_dom_moyenne"].notnull()]

    # 4. Affichage
    fig, ax = plt.subplots(figsize=(10, 10))
    limites_epci.plot(ax=ax, alpha=1, edgecolor='black', facecolor='none')

    carreaux.plot(
        column=f"{nom_graphe}_ratio_services_dom_moyenne",
        cmap="YlGn",
        # vmin=0, vmax=0.25,
        legend=True,
        ax=ax,
        edgecolor="none"
    )
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
    ax.set_axis_off()
    plt.title(f"Part moyenne des services (A à F) accessibles en \n15 min depuis la maille ({nom_graphe})", fontsize=18)
    plt.tight_layout()

    # (Optionnel) Export en .PNG
    if export:
        images_export_path = os.path.join(images_dir, f"visualisation_ratio_services_moyens_{nom_graphe}.png")
        plt.savefig(images_export_path, dpi=300, bbox_inches='tight')
        print(f"Carte exportée vers : {images_export_path}")

    plt.show()
    plt.close()


# ### 5.11. Calcul du ratio moyen des services accessibles
# ---

# In[183]:


def calculer_moyenne_ratio_services_proches(transport="marche"):
    # 1. Chargement des données
    carreaux = charger_fichier_parquet("maille_200m_avec_donnees", crs=3857)

    # 2. Construction des noms de colonnes de services
    colonnes_services = [f"{transport}_ratio_services_dom_{c}" for c in "ABCDEF"]
    colonne_moyenne = f"{transport}_ratio_services_dom_moyenne"

    # 3. Nettoyage préalable
    carreaux = nettoyer_colonnes(carreaux, [colonne_moyenne])

    # 4. Vérification des colonnes existantes
    colonnes_existantes = [col for col in colonnes_services if col in carreaux.columns]
    colonnes_absentes = [col for col in colonnes_services if col not in carreaux.columns]

    if colonnes_absentes:
        print(f"[INFO] Colonnes manquantes pour {transport} :")
        for col in sorted(colonnes_absentes):
            print(f" - {col}")

    if not colonnes_existantes:
        print(f"Aucune colonne valide trouvée pour le transport '{transport}'. Calcul abandonné.")
        return

    # 5. Remplacement des valeurs manquantes
    carreaux[colonne_moyenne] = (
        carreaux[colonnes_existantes]
        .replace(-1, np.nan)
        .mean(axis=1)
        .round(10)
    )
    carreaux[colonne_moyenne] = carreaux[colonne_moyenne].fillna(-1)

    # 6. Export
    exporter_parquet(carreaux, "maille_200m_avec_donnees")
    exporter_gpkg(carreaux, "maille_200m_avec_donnees")
    print(f"Colonne '{colonne_moyenne}' ajoutée avec succès ({transport})")


# ## 6. Calcul des indicateurs
# ---
# Dans ce notebook, les indicateurs peuvent être :
# 1. Spatialisés, et liés à un moyen de transport
# 2. Spatialisés, et non-liés à un moyen de transport
# 3. Calculés à l'échelle de la zone d'étude

# ### 6.1. Marche
# ---
# Indicateur de marchabilité à calculer à partir des indicateurs suivants :
# * 6.1.1. Part de la surface accessible par le réseau pédestre - 'part_surface_pietonne'
# * 6.1.2. Part de la surface occupée par la végétation (NDVI > 0,7) - 'part_vegetation'
# * 6.1.3. Rapport moyen entre la hauteur des bâtiments et la largeur des rues - 'ratio_hl_bati_rues'
# * 6.1.4. Part des rues dont la vitesse moyenne est de moins de 30 km/h - 'part_routes_lentes'
# * 6.1.5. Part des rues inaccessibles aux VL - 'part_routes_inaccessibles_vl"
# * 6.1.6. Part des parcs dans l'aire urbaine - 'part_parcs'
# * Sous-indicateur de mobilité :
#     * 6.1.7. Part des emplois accessibles en moins de T minutes - 'marche_ratio_emploi'
#     * 6.1.8. Part des services accessibles en moins de T minutes. Moyenne sur S types et/ou catégories de services - 'marche_ratio_services_dom_A' à 'marche_ratio_services_dom_F'

# #### 6.1.1. Part de la surface accessible par le réseau pédestre
# ----
# Méthode de calcul : surface accessible sur le carreau autour du réseau piéton (avec tampon de 50 m) / total de surface du carreau

# In[87]:


def calculer_part_surface_accessible_marche():
    # 1. Charger les données
    carreaux = charger_fichier_parquet("maille_200m_epci", crs=3857)
    # Les routes du tampon sont utilisées : leur tampon peut se trouver dans les bordures de la zone étudiée
    routes = charger_fichier_parquet("bd_topo_routes_tampon", crs=3857)

    # 2. Filtrer les routes accessibles aux piétons
    routes = routes[
        (routes.geometry.notnull()) &
        (routes["ETAT"] == "En service") &
        (routes["PRIVE"].str.lower() == "non") &
        (~routes["NATURE"].isin(["Type autoroutier"])) &
        (routes.geometry.type.isin(["LineString", "MultiLineString"]))
    ].copy()

    # 3. Créer un buffer de 50 m autour des routes
    routes["buffer_50m"] = routes.geometry.buffer(50)

    # 4. Fusion des buffers
    zone_pietonne = routes["buffer_50m"].geometry.union_all()

    # 5. Calcul des surfaces
    carreaux["surf_totale_m2"] = carreaux.geometry.area
    carreaux["surf_pieton_m2"] = carreaux.geometry.intersection(zone_pietonne).area
    carreaux["part_surface_pietonne"] = (carreaux["surf_pieton_m2"] / carreaux["surf_totale_m2"] * 100).round(2)

    # 6. Statistiques
    ems_ratio = (carreaux["surf_pieton_m2"].sum() / carreaux["surf_totale_m2"].sum()) * 100
    print(f"Part de surface accessible aux piétons dans l'EMS : {round(ems_ratio, 2)}%")

    # 7. Retire les colonnes 'surf_totale_m2' et 'surf_pieton_m2'
    carreaux = carreaux.drop(columns=["surf_totale_m2", "surf_pieton_m2"])

    # 8. Export
    exporter_parquet(carreaux, "maille_200m_avec_donnees")
    exporter_gpkg(carreaux, "maille_200m_avec_donnees")

# Exécution
calculer_part_surface_accessible_marche()


# In[318]:


def afficher_part_surface_accessible_marche(export = False):
    # 1. Chargement des données
    limites_epci = charger_fichier_parquet("limites_epci", crs=3857)
    carreaux = charger_fichier_parquet("maille_200m_avec_donnees", crs=3857)

    # 2. Affichage
    fig, ax = plt.subplots(figsize=(10, 10))
    limites_epci.plot(ax=ax, alpha=1, edgecolor='black', facecolor='none')

    carreaux.plot(column="part_surface_pietonne",
             cmap="YlGn",
             legend=True,
             #legend_kwds={'label': "% de surface accessible via le réseau pédestre"},
             ax=ax,
             linewidth=0.1,
             edgecolor="grey")

    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)

    ax.set_title("Part de surface accessible via le réseau pédestre", fontsize=18)
    ax.axis('off')
    plt.tight_layout()

    # (Optionnel) Export PNG
    if export :
        images_export_dir = os.path.join(images_dir, "indicateur_part_surface_accessible_marche.png")
        plt.savefig(images_export_dir, dpi=300, bbox_inches='tight')
        print(f"Carte exportée vers : {images_export_dir}")

    plt.show()
    plt.close()

# Exécution
afficher_part_surface_accessible_marche(export = True)


# #### 6.1.2. Part de la surface occupée par la végétation (NDVI > 0,7)
# ---
# Le NDVI (Indice de végétation par différence normalisée) est un indice allant de 0 à 1, permettant de déterminer si une surface est végétalisée ou non. Cette donnée est récupérée depuis l'imagerie satellite Sentinel-2. Le calcul se fait sur le NDVI médian à l'année (2024) de la partie 2.17.1.
# 
# Ici, on considère qu'une surface est végétalisée si son NDVI est supérieur à 0,7.
# 
# Documentation : 
# * https://fr.wikipedia.org/wiki/Indice_de_v%C3%A9g%C3%A9tation_par_diff%C3%A9rence_normalis%C3%A9e

# In[88]:


def calculer_part_surface_ndvi(): 
    # 1. Chargement des fichiers
    carreaux = charger_fichier_parquet("maille_200m_avec_donnees", crs=4326)
    # ndvi_path = os.path.join(exports_dir, "ndvi_epci_2024_mediane.tif") # crs : 4326, version moyenne annuelle
    ndvi_path = os.path.join(exports_dir, "ndvi_epci.tif") # Version monodate

    # 2. Charger le raster NDVI et calculer les pourcentages
    with rasterio.open(ndvi_path) as src:
        results = []
        total_veg = 0
        total_pixels = 0

        for _, row in carreaux.iterrows():
            try:
                out_image, _ = mask(src, [row.geometry], crop=True, filled=False)
                ndvi_values = out_image[0]
                valid = ndvi_values[~np.isnan(ndvi_values)]
                if valid.size == 0:
                    results.append(0.0)
                    continue
                veg_ratio = (valid > 0.7).sum() / valid.size
                results.append(round(veg_ratio * 100, 2))
                total_veg += (valid > 0.7).sum()
                total_pixels += valid.size
            except:
                results.append(0.0)

    # 3. Ajouter la colonne à la GeoDataFrame
    carreaux["part_vegetation"] = results

    # 4. Calcul global EMS
    ems_ratio = (total_veg / total_pixels) * 100 if total_pixels > 0 else 0
    indicateurs_ems = pd.DataFrame([{
        "nom": "part_vegatation",
        "valeur": round(ems_ratio, 2)
    }])

    # Export
    exporter_gpkg(carreaux, "maille_200m_avec_donnees")
    exporter_parquet(carreaux, "maille_200m_avec_donnees")

    print(indicateurs_ems)

# Exécution
calculer_part_surface_ndvi()


# In[320]:


def afficher_part_surface_ndvi(export = False):
    # 1. Charger les données
    limites_epci = charger_fichier_parquet("limites_epci", crs=3857)
    carreaux = charger_fichier_parquet("maille_200m_avec_donnees", crs=3857)

    # 2. Affichage
    fig, ax = plt.subplots(figsize=(10, 10))
    limites_epci.plot(ax=ax, alpha=1, edgecolor='black', facecolor='none')

    carreaux.plot(column="part_vegetation",
             cmap="YlGn",
             legend=True,
             #legend_kwds={'label': "% de surface avec NDVI > 0.7"},
             ax=ax,
             linewidth=0.1,
             edgecolor="grey",
             vmin=0,
             vmax=100)

    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)

    ax.set_title("Pourcentage de surface couverte de végétation (NDVI > 0.7)" , fontsize=18)
    ax.axis('off')
    plt.tight_layout()

    # (Optionnel) Export PNG
    if export :
        images_export_dir = os.path.join(images_dir, "visualisation_ndvi.png")
        plt.savefig(images_export_dir, dpi=300, bbox_inches='tight')
        print(f"Carte exportée vers : {images_export_dir}")

    plt.show()
    plt.close()

# Exécution
afficher_part_surface_ndvi(export = True)


# #### 6.1.3. Rapport moyen entre la hauteur des bâtiments et la largeur des rues
# ---

# In[92]:


def calculer_rapport_moyen_hl_bati_rues():
    # 1. Exécution des fonctions
    calculer_largeur_rues()
    calculer_ratio_rues()

    # 2. Chargement des données
    carreaux = charger_fichier_parquet("maille_200m_avec_donnees", crs=2154)
    routes_ratio = charger_fichier_parquet("bd_topo_routes_epci_ratio_hauteur_largeur", crs=2154)

    # 3. Nettoyage préalable des colonnes existantes dans carreaux
    colonnes_a_supprimer = ['ratio_rues_pondere', 'ratio_hl_bati_rues']
    carreaux = nettoyer_colonnes(carreaux, colonnes_a_supprimer)

    # 4. Filtrage des routes avec ratio valide
    routes_valides = routes_ratio[routes_ratio["ratio_hauteur_largeur"].notnull()].copy()

    # 5. Calcul de la longueur initiale des tronçons
    routes_valides["longueur_initiale"] = routes_valides.geometry.length

    # 6. Intersection avec les carreaux
    print("Calcul des intersections routes/carreaux...")
    routes_par_carreau = gpd.overlay(
        routes_valides[["ratio_hauteur_largeur", "longueur_initiale", "geometry"]],
        carreaux[["idINSPIRE", "geometry"]],
        how="intersection"
    )

    # 7. Calcul de la longueur des segments intersectés
    routes_par_carreau["longueur_segment"] = routes_par_carreau.geometry.length

    # 8. Calcul du ratio pondéré par longueur
    routes_par_carreau["ratio_pondere"] = (
        routes_par_carreau["ratio_hauteur_largeur"] * 
        routes_par_carreau["longueur_segment"]
    )

    # 9. Agrégation par carreau
    stats = routes_par_carreau.groupby("idINSPIRE").agg(
        somme_ratios_ponderes=("ratio_pondere", "sum"),
        somme_longueurs=("longueur_segment", "sum")
    ).reset_index()

    # 10. Calcul du ratio moyen pondéré
    stats["ratio_hl_bati_rues"] = (stats["somme_ratios_ponderes"] / stats["somme_longueurs"]).round(2)

    # 11. Fusion avec les carreaux
    carreaux = carreaux.merge(
        stats[["idINSPIRE", "ratio_hl_bati_rues"]],
        on="idINSPIRE",
        how="left"
    )

    # 12. Nettoyage des valeurs manquantes
    carreaux["ratio_hl_bati_rues"] = carreaux["ratio_hl_bati_rues"].fillna(-1)

    # 12. Export
    exporter_parquet(carreaux, "maille_200m_avec_donnees")
    exporter_gpkg(carreaux, "maille_200m_avec_donnees")

    print(f"Calcul terminé. Ratio moyen calculé pour {len(stats)} carreaux.")

# Exécution
calculer_rapport_moyen_hl_bati_rues()


# In[322]:


def afficher_rapport_moyen_hl_bati_rues(export = False):
    # 1. Chargement des données
    limites_epci = charger_fichier_parquet("limites_epci", crs=3857)
    carreaux = charger_fichier_parquet("maille_200m_avec_donnees", crs=3857)
    carreaux = carreaux[carreaux["ratio_hl_bati_rues"] >= 0]

    # 2. Affichage
    fig, ax = plt.subplots(figsize=(10, 10))
    limites_epci.plot(ax=ax, alpha=1, edgecolor='black', facecolor='none')

    carreaux.plot(column="ratio_hl_bati_rues",
             cmap="YlGn",
             legend=True,
            # legend_kwds={'label' : "Ratio Hauteur du bâti / Largeur des rues moyen par carreau"},
             ax=ax,
             linewidth=0.1,
             edgecolor="grey")

    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)

    ax.set_title("Ratio hauteur du bâti / largeur des rues moyen", fontsize=18)
    ax.axis('off')
    plt.tight_layout()

    # (Optionnel) Export PNG
    if export :
        images_export_dir = os.path.join(images_dir, "visualisation_ratio_hl.png")
        plt.savefig(images_export_dir, dpi=300, bbox_inches='tight')
        print(f"Carte exportée vers : {images_export_dir}")

    plt.show()
    plt.close()

# Exécution
afficher_rapport_moyen_hl_bati_rues(export = True)


# #### 6.1.4. Part des rues dont la vitesse moyenne est de moins de 30 km/h
# ---

# In[329]:


def calculer_part_rues_vmoy_30kmh():
    # 1. Chargement des données
    carreaux = charger_fichier_parquet("maille_200m_avec_donnees", crs=3857)
    routes = charger_fichier_parquet("bd_topo_routes_epci", crs=3857).copy()

    # 2. Nettoyage préalable des colonnes existantes dans carreaux
    colonnes_a_supprimer = ['longueur_totale', 'longueur_lente', 'longueur_totale_x', 'longueur_lente_x',
                           'longueur_totale_y', 'longueur_lente_y', 'part_routes_lentes']
    carreaux = nettoyer_colonnes(carreaux, colonnes_a_supprimer)

    # 3. Prétraitement des routes
    routes = routes[
        (routes.geometry.notnull()) &
        (routes["ETAT"] == "En service") &
        (routes["PRIVE"].str.lower() == "non")
    ].copy()

    # 4. Nettoyage et conversion des vitesses
    routes["VIT_MOY_VL"] = pd.to_numeric(routes["VIT_MOY_VL"], errors="coerce")
    routes = routes[routes.geometry.type.isin(["LineString", "MultiLineString"])]

    # 5. Calcul de la longueur initiale
    routes["longueur"] = routes.geometry.length
    routes["est_lente"] = routes["VIT_MOY_VL"] < 30

    # 6. Intersection spatiale
    routes_par_carreau = gpd.overlay(routes, carreaux[["idINSPIRE", "geometry"]], how="intersection")
    routes_par_carreau["longueur"] = routes_par_carreau.geometry.length
    routes_par_carreau["est_lente"] = routes_par_carreau["VIT_MOY_VL"] < 30

    # 7. Agrégation
    stats = routes_par_carreau.groupby("idINSPIRE").agg(
        longueur_totale=("longueur", "sum"),
        longueur_lente=("longueur", lambda x: x[routes_par_carreau.loc[x.index, "est_lente"]].sum())
    ).reset_index()

    # 8. Fusion dans les carreaux
    carreaux = carreaux.merge(stats, on="idINSPIRE", how="left")

    # 9. Calcul de la part
    carreaux["part_routes_lentes"] = carreaux["longueur_lente"] / carreaux["longueur_totale"]
    # Si aucune route n'est présente dans le carreau, prends la valeur -1.
    carreaux["part_routes_lentes"] = (carreaux["part_routes_lentes"] * 100).fillna(-1).round(2)

    # 10. Retire les colonnes 'longueur_totale' et 'longueur_lente'
    carreaux = carreaux.drop(columns=["longueur_totale", "longueur_lente"])

    # 11. Export
    exporter_parquet(carreaux, "maille_200m_avec_donnees")
    exporter_gpkg(carreaux, "maille_200m_avec_donnees")
    print("Colonne 'part_routes_lentes' ajoutée avec succès.")

# Exécution
calculer_part_rues_vmoy_30kmh()


# In[330]:


def afficher_part_rues_vmoy_30kmh(export = False):
    # 1. Chargement des données
    limites_epci = charger_fichier_parquet("limites_epci", crs=3857)
    carreaux = charger_fichier_parquet("maille_200m_avec_donnees", crs=3857)
    carreaux = carreaux[carreaux["part_routes_lentes"] >= 0]

    # 2. Affichage
    fig, ax = plt.subplots(figsize=(10, 10))
    limites_epci.plot(ax=ax, alpha=1, edgecolor='black', facecolor='none')

    carreaux.plot(column="part_routes_lentes",
             cmap="YlGn",
             legend=True,
             # legend_kwds={'label': "% de routes dont la vitessse moyenne est < à 30 km/h par carreau"},
             ax=ax,
             linewidth=0.1,
             edgecolor="grey")

    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)

    ax.set_title("Part des routes dont la vitessse moyenne est < à 30 km/h" , fontsize=18)
    ax.axis('off')
    plt.tight_layout()

    # (Optionnel) Export PNG
    if export :
        images_export_dir = os.path.join(images_dir, "visualisation_routes_lentes.png")
        plt.savefig(images_export_dir, dpi=300, bbox_inches='tight')
        print(f"Carte exportée vers : {images_export_dir}")

    plt.show()
    plt.close()

# Exécution
afficher_part_rues_vmoy_30kmh(export = True)


# #### 6.1.5. Part des routes inaccessibles aux VL
# ---

# In[94]:


def calculer_part_rues_inaccessibles_vl():
    # 1. Chargement des données
    carreaux = charger_fichier_parquet("maille_200m_avec_donnees", crs=2154)
    routes = charger_fichier_parquet("bd_topo_routes_epci", crs=2154)

    # 2. Prétraitement des routes
    routes = routes[
        (routes.geometry.notnull()) &
        (routes["ETAT"] == "En service") &
        (routes["PRIVE"].str.lower() == "non")
    ].copy()

    # 3. Filtrage géométrique
    routes = routes[routes.geometry.type.isin(["LineString", "MultiLineString"])]

    # 4. Calcul de la longueur initiale
    routes["longueur"] = routes.geometry.length
    routes["inaccessible_vl"] = routes["ACCES_VL"].str.lower().isin(["physiquement impossible", "restreint aux ayants droit"])

    # 5. Intersection avec les carreaux
    routes_par_carreau = gpd.overlay(routes, carreaux[["idINSPIRE", "geometry"]], how="intersection")
    routes_par_carreau["longueur"] = routes_par_carreau.geometry.length
    routes_par_carreau["inaccessible_vl"] = routes_par_carreau["ACCES_VL"].str.lower().isin(["physiquement impossible", "restreint aux ayants droit"])

    # 6. Agrégation
    stats = routes_par_carreau.groupby("idINSPIRE").agg(
        longueur_totale=("longueur", "sum"),
        longueur_inaccessible=("longueur", lambda x: x[routes_par_carreau.loc[x.index, "inaccessible_vl"]].sum())
    ).reset_index()

    # 7. Fusion dans les carreaux
    carreaux = carreaux.merge(stats, on="idINSPIRE", how="left")

    # 8. Calcul de la part (converti en pourcentage)
    carreaux["part_routes_inaccessibles_vl"] = (
        (carreaux["longueur_inaccessible"] / carreaux["longueur_totale"]) * 100  # Conversion en %
    ).round(2)  # Arrondi à 2 décimales
    carreaux["part_routes_inaccessibles_vl"] = carreaux["part_routes_inaccessibles_vl"].fillna(-1).clip(lower=-1, upper=100)

    # 9. Nettoyage
    carreaux.drop(columns=["longueur_totale", "longueur_inaccessible"], inplace=True)

    # 10. Export et statistiques
    exporter_parquet(carreaux, "maille_200m_avec_donnees")
    exporter_gpkg(carreaux, "maille_200m_avec_donnees")

    # Statistiques
    valides = carreaux[carreaux["part_routes_inaccessibles_vl"] >= 0]
    print(f"""
    Part des rues inaccessibles aux voitures :
    - Carreaux avec données : {len(valides)}/{len(carreaux)}
    - Moyenne : {valides["part_routes_inaccessibles_vl"].mean():.2f}%
    - Médiane : {valides["part_routes_inaccessibles_vl"].median():.2f}%
    - Max : {valides["part_routes_inaccessibles_vl"].max():.2f}%
    """)

# Exécution
calculer_part_rues_inaccessibles_vl()


# In[331]:


def afficher_part_rues_inaccessibles_vl(export = False):
    limites_epci = charger_fichier_parquet("limites_epci", crs=3857)
    carreaux = charger_fichier_parquet("maille_200m_avec_donnees", crs=3857)
    carreaux = carreaux[carreaux["part_routes_inaccessibles_vl"] != -1]

    # Affichage
    fig, ax = plt.subplots(figsize=(10, 10))
    limites_epci.plot(ax=ax, alpha=1, edgecolor='black', facecolor='none')

    carreaux.plot(column="part_routes_inaccessibles_vl",
             cmap="YlGn",
             legend=True,
             #legend_kwds={'label': "% de rues inaccessibles aux voitures"},
             ax=ax,
             linewidth=0.1,
             edgecolor="grey")

    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)

    ax.set_title("Part des rues inaccessibles aux voitures", fontsize=18)
    ax.axis('off')
    plt.tight_layout()

    # (Optionnel) Export PNG
    if export :
        images_export_dir = os.path.join(images_dir, "visualisation_routes_inacessibles_vl.png")
        plt.savefig(images_export_dir, dpi=300, bbox_inches='tight')
        print(f"Carte exportée vers : {images_export_dir}")

    plt.show()
    plt.close()

# Exécution
afficher_part_rues_inaccessibles_vl(export = True)


# #### 6.1.6. Part des parcs dans l'aire urbaine
# ---
# 
# Note : il est possible que les chiffres soient sous-estimés, à vérifier

# In[95]:


def calculer_part_parcs_aire_urbaine():
    # 1. Charger les données
    carreaux = charger_fichier_parquet("maille_200m_avec_donnees", crs=2154)
    parcs_jardins = charger_fichier_parquet("parcs_jardins_urbains_epci", crs=2154)

    # 2. Calcul pour chaque carreau
    results = []
    total_surface_parcs = 0
    total_surface_carreaux = 0

    for _, carreau in carreaux.iterrows():
        try:
            # 3. Surface du carreau
            surface_carreau = carreau.geometry.area

            # 4. Intersection avec les parcs
            parcs_in_carreau = parcs_jardins[parcs_jardins.intersects(carreau.geometry)]
            surface_parcs = 0

            for _, parc in parcs_in_carreau.iterrows():
                intersection = carreau.geometry.intersection(parc.geometry)
                if not intersection.is_empty:
                    surface_parcs += intersection.area

            # 5. Calcul du pourcentage
            pourcentage = (surface_parcs / surface_carreau) * 100 if surface_carreau > 0 else 0
            results.append(round(pourcentage, 2))

            # 6. Mise à jour des totaux
            total_surface_parcs += surface_parcs
            total_surface_carreaux += surface_carreau

        except Exception as e:
            print(f"Erreur sur le carreau {carreau.name}: {str(e)}")
            results.append(0.0)

    # 7. Ajout de la colonne au GeoDataFrame
    carreaux["part_parcs"] = results

    # 8. Calcul global pour l'ensemble du territoire
    pourcentage_global = (total_surface_parcs / total_surface_carreaux) * 100 if total_surface_carreaux > 0 else 0
    indicateurs_parcs = pd.DataFrame([{
        "indicateur": "part_parcs",
        "valeur": round(pourcentage_global, 2),
        "unite": "%"
    }])

    # 9. Export
    exporter_parquet(carreaux, "maille_200m_avec_donnees")
    exporter_gpkg(carreaux, "maille_200m_avec_donnees")

    print(f"Part moyenne des parcs et jardins : {round(pourcentage_global, 2)}%")
    print(indicateurs_parcs)

# Exécution
calculer_part_parcs_aire_urbaine()


# In[351]:


def afficher_part_parcs_aire_urbaine(export = False):
    limites_epci = charger_fichier_parquet("limites_epci", crs=3857)
    carreaux = charger_fichier_parquet("maille_200m_avec_donnees", crs=3857)
    carreaux = carreaux[carreaux["part_parcs"] >= 0]

    # 2. Affichage
    fig, ax = plt.subplots(figsize=(10, 10))
    limites_epci.plot(ax=ax, alpha=1, edgecolor='black', facecolor='none')

    carreaux.plot(column="part_parcs",
             cmap="YlGn",
             legend=True,
             #legend_kwds={'label': "% de la surface couverte par un parc ou un jardin urbain"},
             ax=ax,
             linewidth=0.1,
             edgecolor="grey",
             vmin = 0,
             vmax = 100)

    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)

    ax.set_title("Part de la surface couverte par un parc ou un jardin urbain", fontsize = 18)
    ax.axis('off')
    plt.tight_layout()

    # (Optionnel) Export PNG
    if export :
        images_export_dir = os.path.join(images_dir, "visualisation_part_parcs.png")
        plt.savefig(images_export_dir, dpi=300, bbox_inches='tight')
        print(f"Carte exportée vers : {images_export_dir}")

    plt.show()
    plt.close()

# Exécution
afficher_part_parcs_aire_urbaine(export = True)


# #### 6.1.7. Part des emplois accessibles en moins de X minutes
# ---

# In[185]:


# 900 secondes : 15 minutes
calculer_ratio_emplois_services_proches(nom_graphe="marche", secondes=900, n_jobs=-1)


# In[352]:


afficher_ratio_emplois_proches(nom_graphe = "marche", export = True)


# #### 6.1.8. Part des services accessibles en moins de T minutes. Moyenne sur S types et/ou catégories de services
# ---

# In[186]:


calculer_moyenne_ratio_services_proches(transport="marche")


# In[347]:


afficher_ratio_services_proches(nom_graphe = "marche", export = True)


# #### 6.1.9. Nombre d'accidents impliquant des piétons (2023)
# ---

# In[321]:


calculer_nb_accidents(transport="pietons")


# In[241]:


afficher_nb_accidents(transport="pietons", export = True)


# ### 6.2. Vélos
# ---
# Indicateur de cyclabilité à calculer à partir des indicateurs suivants :
# * 6.1.4. Part des rues dont la vitesse moyenne est de moins de 30 km/h - 'part_routes_lentes'
# * 6.1.5. Part des rues inaccessibles aux VL - 'part_inaccessibles_vl'
# * 6.1.6. Part des parcs dans l'aire urbaine - 'part_parcs'
# * 6.2.1. Part de la surface accessible par le réseau cyclable - 'part_surface_cyclable'
# * 6.2.2. Nombre de stations de vélos en libre-partage (Vélhop) - 'part_normalise_surface_couverte_stations_velos'
# * 6.2.3. Nombre de vélos en libre-service - 'nb_velos_libre_service' (difficile à normaliser, non utilisé pour l'instant)
# * 6.2.4. Part des rues interdites aux poids lourds - 'part_routes_inaccessibles_pl'
# * 6.2.5. Part de rues pavées - 'part_routes_pavees'
# * 6.2.6.8. Pente moyenne des rues - 'pente_moyenne'
# * 6.2.9. Nombre de stations de réparation de vélos - 'part_normalise_couverte_stations_rep_velo'
# * Sous-indicateur de mobilité :
#     * 6.2.8. Part des emplois accessibles en moins de T minutes - 'marche_ratio_emploi'
#     * 6.2.9. Part des services accessibles en moins de T minutes. Moyenne sur S types et/ou catégories de services - 'marche_ratio_services_dom_A' à 'marche_ratio_services_dom_F'

# #### 6.2.1. Part de la surface accessible par le réseau cyclable
# ---

# In[ ]:


def calculer_part_surface_accessible_velos():
    # 1. Chargement des données
    carreaux = charger_fichier_parquet("maille_200m_avec_donnees", crs=3857).copy()
    routes = charger_fichier_parquet("bd_topo_routes_tampon", crs=3857).copy()

    # 2. Filtrer les routes accessibles aux vélos
    routes = routes[
        (routes.geometry.notnull()) &
        (routes["ETAT"] == "En service") &
        (routes["PRIVE"].str.lower() == "non") &
        (~routes["NATURE"].isin(["Type autoroutier"])) &
        (routes.geometry.type.isin(["LineString", "MultiLineString"]))
    ].copy()

    # 3. Buffer de 50 m autour des routes
    routes["buffer_50m"] = routes.geometry.buffer(50)

    # 4. Union des buffers
    zone_cyclable = routes["buffer_50m"].geometry.union_all()

    # 5. Calcul des surfaces
    carreaux["surf_totale_m2"] = carreaux.geometry.area
    carreaux["surf_cyclable_m2"] = carreaux.geometry.intersection(zone_cyclable).area
    carreaux["part_surface_cyclable"] = (carreaux["surf_cyclable_m2"] / carreaux["surf_totale_m2"] * 100).round(2)

    # 6. Statistiques
    ems_ratio = (carreaux["surf_cyclable_m2"].sum() / carreaux["surf_totale_m2"].sum()) * 100
    print(f"Part de surface accessible aux cyclistes dans l'EMS : {round(ems_ratio, 2)}%")

    # 7. Retire les colonnes 'surf_totale_m2' et 'surf_cyclable_m2'
    carreaux = carreaux.drop(columns=["surf_totale_m2", "surf_cyclable_m2"])

    # 8. Export
    exporter_parquet(carreaux, "maille_200m_avec_donnees")
    exporter_gpkg(carreaux, "maille_200m_avec_donnees")

# Exécution
calculer_part_surface_accessible_velos()


# In[349]:


def afficher_part_surface_accessible_velos(export = False):
    limites_epci = charger_fichier_parquet("limites_epci", crs=3857)
    carreaux = charger_fichier_parquet("maille_200m_avec_donnees", crs=3857)

    fig, ax = plt.subplots(figsize=(10, 10))
    limites_epci.plot(ax=ax, alpha=1, edgecolor='black', facecolor='none')

    carreaux.plot(column="part_surface_cyclable",
             cmap="YlGn",
             legend=True,
             # legend_kwds={'label': "% de surface accessible via le réseau cyclable"},
             ax=ax,
             linewidth=0.1,
             edgecolor="grey")

    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)

    ax.set_title("Part de surface accessible via le réseau cyclable", fontsize=18)
    ax.axis('off')
    plt.tight_layout()

    # (Optionnel) Export PNG
    if export :
        images_export_dir = os.path.join(images_dir, "visualisation_part_surface_accessible_velos.png")
        plt.savefig(images_export_dir, dpi=300, bbox_inches='tight')
        print(f"Carte exportée vers : {images_export_dir}")

    plt.show()
    plt.close()

# Exécution
afficher_part_surface_accessible_velos(export = True)


# #### 6.2.2. Nombre de stations de vélos en libre-service (Vélhop)
# ---

# In[244]:


def calculer_nb_stations_libre_partage():
    # 1. Chargement des données
    carreaux = charger_fichier_parquet("maille_200m_avec_donnees", crs=3857).copy()
    velhop_stations = charger_fichier_parquet("velhop_stations", crs=3857).copy()

    # 2. Supprime les colonne 'index_right' si elle existe
    velhop_stations = nettoyer_colonnes(velhop_stations)

    # 3. Supprime l’ancienne colonne si elle existe pour éviter conflits de fusion
    carreaux = nettoyer_colonnes(carreaux, ['nb_stations_velos_libre_service'])

    # 4. Filtrer uniquement les points valides
    velhop_stations = velhop_stations[
        velhop_stations.geometry.notnull() & (velhop_stations.geometry.type == "Point")
    ].copy()

    # 5. Jointure spatiale : associer chaque station à un carreau
    stations_avec_carreaux = gpd.sjoin(
        velhop_stations,
        carreaux[["idINSPIRE", "geometry"]],
        how="inner",
        predicate="within"
    )

    # 6. Agrégation : nombre de stations par carreau
    nb_stations_par_carreau = stations_avec_carreaux.groupby("idINSPIRE").size().reset_index(name="nb_stations_velos_libre_service")

    # 7. Fusion avec les carreaux
    carreaux = carreaux.merge(nb_stations_par_carreau, on="idINSPIRE", how="left")
    carreaux["nb_stations_velos_libre_service"] = carreaux["nb_stations_velos_libre_service"].fillna(0).astype(int)

    # 8. Export
    exporter_parquet(carreaux, "maille_200m_avec_donnees")
    exporter_gpkg(carreaux, "maille_200m_avec_donnees")

    print(f"Colonne 'nb_stations_velos_libre_service' ajoutée avec succès à {nb_stations_par_carreau.shape[0]} carreaux.")

# Exécution
calculer_nb_stations_libre_partage()


# In[355]:


def afficher_nb_stations_libre_partage(export = False):
    # 1. Chargement des données
    limites_epci = charger_fichier_parquet("limites_epci", crs=3857)
    carreaux = charger_fichier_parquet("maille_200m_avec_donnees", crs=3857)

    # 2. Affichage
    fig, ax = plt.subplots(figsize=(10, 10))
    limites_epci.plot(ax=ax, alpha=1, edgecolor='black', facecolor='none')

    carreaux.plot(column="nb_stations_velos_libre_service",
             cmap="YlGn",
             legend=True,
             #legend_kwds={'label': "Nombre de stations de vélos en libre-service (Vélhop) par maille"},
             ax=ax,
             linewidth=0.1,
             edgecolor="grey")

    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)

    ax.set_title("Nombre de stations de vélos en libre-service (Vélhop) par maille", fontsize=18)
    ax.axis('off')
    plt.tight_layout()

    # (Optionnel) Export PNG
    if export :
        images_export_dir = os.path.join(images_dir, "indicateur_nb_stations_velos_libre_partage.png")
        plt.savefig(images_export_dir, dpi=300, bbox_inches='tight')
        print(f"Carte exportée vers : {images_export_dir}")

    plt.show()
    plt.close()

# Exécution
afficher_nb_stations_libre_partage(export = True)


# In[139]:


# Version normalisée, % de surface, zone tampon de 400 mètres
def calculer_part_surface_couverte_stations_velos():
    # 1. Chargement des données
    carreaux = charger_fichier_parquet("maille_200m_avec_donnees", crs=2154).copy()
    velhop_stations = charger_fichier_parquet("velhop_stations", crs=2154).copy()

    # 2. Nettoyage : géométries valides uniquement
    velhop_stations = velhop_stations[
        velhop_stations.geometry.notnull() & (velhop_stations.geometry.type == "Point")
    ].copy()
    carreaux = carreaux[carreaux.geometry.notnull()].copy()

    # 3. Création des buffers 400m autour des stations
    velhop_stations["buffer_400m"] = velhop_stations.geometry.buffer(400)

    # 4. Fusion des buffers (union géométrique) pour éviter de compter plusieurs fois les buffers
    zone_couverte = velhop_stations["buffer_400m"].geometry.union_all()

    # 5. Intersection avec les carreaux
    intersections = carreaux.geometry.intersection(zone_couverte)

    # 6. Calcul de la part de surface couverte
    surfaces_totales = carreaux.geometry.area
    surfaces_intersectees = intersections.area
    part_surface = (surfaces_intersectees / surfaces_totales).clip(upper=1.0) * 100

    # 7. Ajout à la table
    carreaux["part_normalise_surface_couverte_stations_velos"] = part_surface.round(2).fillna(0)

    # 8. Export
    exporter_parquet(carreaux, "maille_200m_avec_donnees")
    exporter_gpkg(carreaux, "maille_200m_avec_donnees")

    # 9. Résumé
    print("Colonne 'part_surface_couverte_stations_velos' ajoutée avec succès.")
    print(f"Valeur moyenne : {carreaux['part_normalise_surface_couverte_stations_velos'].mean():.2f}%")

# Exécution
calculer_part_surface_couverte_stations_velos()


# In[370]:


def afficher_nb_stations_libre_partage(export = False):
    # 1. Chargement des données
    limites_epci = charger_fichier_parquet("limites_epci", crs=3857)
    carreaux = charger_fichier_parquet("maille_200m_avec_donnees", crs=3857)

    # 2. Affichage
    fig, ax = plt.subplots(figsize=(10, 10))
    limites_epci.plot(ax=ax, alpha=1, edgecolor='black', facecolor='none')

    carreaux.plot(column="part_normalise_surface_couverte_stations_velos",
             cmap="YlGn",
             legend=True,
             #legend_kwds={'label': "% de surface des carreaux présente à moins de 400 mètres d'une station de vélos en libre-service (Vélhop)"},
             ax=ax,
             linewidth=0.1,
             edgecolor="grey")

    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)

    ax.set_title("% de surface des carreaux présente à moins de 400 mètres \nd'une station de vélos en libre-service (Vélhop)", fontsize=18)
    ax.axis('off')
    plt.tight_layout()

    # (Optionnel) Export PNG
    if export :
        images_export_dir = os.path.join(images_dir, "indicateur_normalise_surface_couverte_stations_velos.png")
        plt.savefig(images_export_dir, dpi=300, bbox_inches='tight')
        print(f"Carte exportée vers : {images_export_dir}")

    plt.show()
    plt.close()

# Exécution
afficher_nb_stations_libre_partage(export = True)


# #### 6.2.3. Nombre de vélos en libre-service
# ---
# Note : le nombre de vélos est tiré de l'attribut 'capacity' de chaque station Vélhop
# 
# En attendant de trouver un bon moyen de normaliser cette donnée, elle est ignorée dans la calcul de la cyclabilité

# In[247]:


def calculer_nombre_velos_libre_service():
    # 1. Chargement des données
    carreaux = charger_fichier_parquet("maille_200m_avec_donnees", crs=3857).copy()
    velhop_stations = charger_fichier_parquet("velhop_stations", crs=3857).copy()

    # 2. Supprime la colonne 'index_right' si elle existe
    velhop_stations = nettoyer_colonnes(velhop_stations)

    # 3. Supprime l’ancienne colonne si elle existe pour éviter conflits de fusion
    carreaux = nettoyer_colonnes(carreaux, ['velhop_nb_velos', 'nb_velos_libre_service'])

    # 4. Filtrer uniquement les points valides
    velhop_stations = velhop_stations[
        velhop_stations.geometry.notnull() & (velhop_stations.geometry.type == "Point")
    ].copy()

    # 5. Jointure spatiale : associer chaque station à un carreau
    stations_avec_carreaux = gpd.sjoin(
        velhop_stations,
        carreaux[["idINSPIRE", "geometry"]],
        how="inner",
        predicate="within"
    )

    # 6. Agrégation : somme des capacités par carreau
    capacite_par_carreau = stations_avec_carreaux.groupby("idINSPIRE")["capacity"].sum().reset_index()
    capacite_par_carreau.rename(columns={"capacity": "nb_velos_libre_service"}, inplace=True)

    # 6. Fusion avec les carreaux
    carreaux = carreaux.merge(capacite_par_carreau, on="idINSPIRE", how="left")
    carreaux["nb_velos_libre_service"] = carreaux["nb_velos_libre_service"].fillna(0).astype(int)

    # 7. Export
    exporter_parquet(carreaux, "maille_200m_avec_donnees")
    exporter_gpkg(carreaux, "maille_200m_avec_donnees")
    print("Colonne 'nb_velos_libre_service' ajoutée avec succès.")

# Exécution
calculer_nombre_velos_libre_service()


# In[357]:


def afficher_nombre_velos_libre_service(export = False):
    # 1. Chargement des données
    limites_epci = charger_fichier_parquet("limites_epci", crs=3857)
    carreaux = charger_fichier_parquet("maille_200m_avec_donnees", crs=3857)

    # 2. Affichage
    fig, ax = plt.subplots(figsize=(10, 10))
    limites_epci.plot(ax=ax, alpha=1, edgecolor='black', facecolor='none')

    carreaux.plot(column="nb_velos_libre_service",
             cmap="YlGn",
             legend=True,
             #legend_kwds={'label': "Nombre de vélos en libre-service par maille"},
             ax=ax,
             linewidth=0.1,
             edgecolor="grey")

    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)

    ax.set_title("Nombre de vélos en libre-service par maille", fontsize=18)
    ax.axis('off')
    plt.tight_layout()

    # (Optionnel) Export PNG
    if export :
        images_export_dir = os.path.join(images_dir, "indicateur_nb_velos_libre_partage.png")
        plt.savefig(images_export_dir, dpi=300, bbox_inches='tight')
        print(f"Carte exportée vers : {images_export_dir}")

    plt.show()
    plt.close()

# Exécution
afficher_nombre_velos_libre_service(export = True)


# #### 6.2.4. Part des routes inacessibles aux poids lourds
# ---
# Sont considérés comme poids louds les véhicules transportant plus de 3,5 tonnes : https://fr.wikipedia.org/wiki/Poids_lourd
# 
# Ici, une route est inaccessible à un poids lourd si : 
# * Le tronçon possède une restriction de poids strictment supérieure à 3,5 tonnes
# * Son accès est interdit aux VL
# * Il s'agit d'un sentier

# In[368]:


def calculer_part_rues_inaccessibles_pl(): 
    # 1. Chargement des données
    carreaux = charger_fichier_parquet("maille_200m_avec_donnees", crs=3857).copy()
    routes = charger_fichier_parquet("bd_topo_routes_epci", crs=3857).copy()

    # 2. Prétraitement des routes
    routes = routes[
        (routes.geometry.notnull()) &
        (routes["ETAT"] == "En service")
    ].copy()

    # 3. Nettoyage des colonnes existantes
    colonnes_a_supprimer = [
        'longueur_totale', 'longueur', 'longueur_totale_x', 'longueur_x',
        'longueur_totale_y', 'longueur_y', 'est_interdit_pl'
    ]
    carreaux = nettoyer_colonnes(carreaux, colonnes_a_supprimer)

    # 4. Nettoyage et filtrage
    routes["RESTR_P"] = pd.to_numeric(routes["RESTR_P"], errors="coerce").fillna(0)
    routes = routes[routes.geometry.type.isin(["LineString", "MultiLineString"])]

    # 5. Longueur initiale et statut d'inaccessibilité
    routes["longueur"] = routes.geometry.length
    routes["est_interdit_pl"] = (
        (routes["RESTR_P"] > 3.5) |
        (routes["ACCES_VL"] == "Physiquement impossible") |
        (routes["NATURE"] == "Sentier")
    )

    # 6. Découpage spatial
    routes_par_carreau = gpd.overlay(
        routes[["RESTR_P", "ACCES_VL", "PRIVE", "NATURE", "longueur", "est_interdit_pl", "geometry"]],
        carreaux[["idINSPIRE", "geometry"]],
        how="intersection"
    )

    # 7. Mise à jour des longueurs et du statut
    routes_par_carreau["longueur"] = routes_par_carreau.geometry.length
    routes_par_carreau["est_interdit_pl"] = (
        (routes_par_carreau["RESTR_P"] > 3.5) |
        (routes_par_carreau["ACCES_VL"] == "Physiquement impossible") |
        (routes_par_carreau["NATURE"] == "Sentier")
    )

    # 8. Agrégation
    stats = routes_par_carreau.groupby("idINSPIRE").agg(
        longueur_totale=("longueur", "sum"),
        longueur_interdite=("longueur", lambda x: x[routes_par_carreau.loc[x.index, "est_interdit_pl"]].sum())
    ).reset_index()

    # 9. Fusion
    carreaux = carreaux.merge(stats, on="idINSPIRE", how="left")

    # 10. Calcul de la part en pourcentage, arrondi à 2 décimales
    carreaux["part_routes_interdites_pl"] = (
        100 * carreaux["longueur_interdite"] / carreaux["longueur_totale"]
    ).round(2).fillna(-1)

    # 11. Nettoyage
    carreaux.drop(columns=["longueur_totale", "longueur_interdite"], inplace=True)

    # 12. Export
    exporter_parquet(carreaux, "maille_200m_avec_donnees")
    exporter_gpkg(carreaux, "maille_200m_avec_donnees")

    print("Colonne 'part_routes_interdites_pl' ajoutée avec succès.")

# Exécution
calculer_part_rues_inaccessibles_pl()


# In[369]:


def afficher_part_rues_inaccessibles_pl(export = False):
    # 1. Chargement des données
    limites_epci = charger_fichier_parquet("limites_epci", crs=3857)
    carreaux = charger_fichier_parquet("maille_200m_avec_donnees", crs=3857)
    carreaux = carreaux[carreaux["part_routes_interdites_pl"] >= 0]

    # 2. Affichage
    fig, ax = plt.subplots(figsize=(10, 10))
    limites_epci.plot(ax=ax, alpha=1, edgecolor='black', facecolor='none')

    carreaux.plot(column="part_routes_interdites_pl",
             cmap="YlGn",
             legend=True,
             #legend_kwds={'label': "% de routes interdites aux poids lourds"},
             ax=ax,
             linewidth=0.1,
             edgecolor="grey")

    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)

    ax.set_title("Part des routes interdites aux poids lourds", fontsize=18)
    ax.axis('off')
    plt.tight_layout()

    # (Optionnel) Export PNG
    if export :
        images_export_dir = os.path.join(images_dir, "indicateur_part_routes_interdites_pl.png")
        plt.savefig(images_export_dir, dpi=300, bbox_inches='tight')
        print(f"Carte exportée vers : {images_export_dir}")

    plt.show()
    plt.close()

# Exécution
afficher_part_rues_inaccessibles_pl(export = True)


# #### 6.2.5. Part de routes pavées
# ---

# In[359]:


def calculer_part_rues_pavees():
    # 1. Chargement des données
    carreaux = charger_fichier_parquet("maille_200m_avec_donnees", crs=3857).copy()
    routes = charger_fichier_parquet("bd_topo_routes_epci", crs=3857).copy()

    # 2. Prétraitement des routes
    routes = routes[
        (routes.geometry.notnull()) &
        (routes["ETAT"] == "En service") &
        (routes["PRIVE"].str.lower() == "non")
    ].copy()

    # 3. Sélection géométrique
    routes = routes[routes.geometry.type.isin(["LineString", "MultiLineString"])]

    # 4. Calcul de la longueur initiale
    routes["longueur"] = routes.geometry.length
    routes["est_pavee"] = routes["NATURE"] == "Route empierrée"

    # 5. Intersection avec les carreaux (ajout de l'identifiant de carreau)
    routes_par_carreau = gpd.overlay(routes, carreaux[["idINSPIRE", "geometry"]], how="intersection")
    routes_par_carreau["longueur"] = routes_par_carreau.geometry.length
    # Une route est considérée comme pavée si son champ 'NATURE' est égal à 'Route empierrée'
    routes_par_carreau["est_pavee"] = routes_par_carreau["NATURE"] == "Route empierrée"

    # 6. Agrégation
    stats = routes_par_carreau.groupby("idINSPIRE").agg(
        longueur_totale=("longueur", "sum"),
        longueur_pavee=("longueur", lambda x: x[routes_par_carreau.loc[x.index, "est_pavee"]].sum())
    ).reset_index()

    # 7. Fusion dans les carreaux
    carreaux = carreaux.merge(stats, on="idINSPIRE", how="left")

    # 8. Calcul de la part
    carreaux["part_routes_pavees"] = carreaux["longueur_pavee"] / carreaux["longueur_totale"]
    carreaux["part_routes_pavees"] = (100 * carreaux["part_routes_pavees"]).round(2).fillna(-1)

    # 9. Nettoyage
    carreaux.drop(columns=["longueur_totale", "longueur_pavee"], inplace=True)

    # 10. Export
    exporter_parquet(carreaux, "maille_200m_avec_donnees")
    exporter_gpkg(carreaux, "maille_200m_avec_donnees")
    print("Colonne 'part_routes_pavees' ajoutée avec succès.")

# Exécution
calculer_part_rues_pavees()


# In[362]:


def afficher_part_rues_pavees(export = False):
    # 1. Chargement des donnéees
    limites_epci = charger_fichier_parquet("limites_epci", crs=3857)
    carreaux = charger_fichier_parquet("maille_200m_avec_donnees", crs=3857)
    carreaux_valides = carreaux[carreaux["part_routes_pavees"] >= 0]

    # 2. Affichage
    fig, ax = plt.subplots(figsize=(10, 10))
    limites_epci.plot(ax=ax, alpha=1, edgecolor='black', facecolor='none')

    carreaux_valides.plot(column="part_routes_pavees",
             cmap="YlGn",
             legend=True,
             #legend_kwds={'label': "% de routes pavées sur la maille"},
             ax=ax,
             linewidth=0.1,
             edgecolor="grey")

    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)

    ax.set_title("Part des routes pavées sur la maille", fontsize=18)
    ax.axis('off')
    plt.tight_layout()

    # (Optionnel) Export PNG
    if export :
        images_export_dir = os.path.join(images_dir, "indicateur_afficher_part_rues_pavees.png")
        plt.savefig(images_export_dir, dpi=300, bbox_inches='tight')
        print(f"Carte exportée vers : {images_export_dir}")

    plt.show()
    plt.close()

# Exécution
afficher_part_rues_pavees(export = True)


# #### 6.2.6. Pente moyenne des routes
# ---

# In[241]:


calculer_pentes_bd_topo()


# In[242]:


def afficher_pentes_bd_topo(export = False):
    # 1. Chargement des données
    carreaux = charger_fichier_parquet("maille_200m_avec_donnees", crs=3857)
    limites_epci = charger_fichier_parquet("limites_epci", crs=3857)
    carreaux = carreaux[carreaux["pente_moyenne"] != -1]

    # 2. Affichage
    fig, ax = plt.subplots(figsize=(10, 10))
    limites_epci.plot(ax=ax, alpha=1, edgecolor='black', facecolor='none', linewidth=1.5)
    carreaux.plot(column="pente_moyenne",
        cmap="RdYlGn_r",
        legend=True,
        legend_kwds={
            'label': "Pente moyenne (%)",
            'orientation': "vertical",
            'shrink': 0.6
        },
        ax=ax,
        linewidth=0.1,
        edgecolor="grey",
        alpha=0.8
    )

    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron,attribution_size=6)
    ax.set_title("Pente moyenne (%) des routes par carreau", fontsize=18)
    ax.axis('off')
    plt.tight_layout()

    # (Optionnel) Export PNG
    if export :
        images_export_dir = os.path.join(images_dir, "indicateur_pentes_bd_topo.png")
        plt.savefig(images_export_dir, dpi=300, bbox_inches='tight')
        print(f"Carte exportée vers : {images_export_dir}")

    plt.show()
    plt.close()

# Exécution
afficher_pentes_bd_topo(export = True)


# In[363]:


def afficher_pentes_absolues_bd_topo(export = False):
    # 1. Chargement des données
    carreaux = charger_fichier_parquet("maille_200m_avec_donnees", crs=3857)
    limites_epci = charger_fichier_parquet("limites_epci", crs=3857)
    carreaux = carreaux[carreaux["pente_moyenne_absolue"] != -1]

    # 2. Affichage
    fig, ax = plt.subplots(figsize=(10, 10))
    limites_epci.plot(ax=ax, alpha=1, edgecolor='black', facecolor='none', linewidth=1.5)
    carreaux.plot(column="pente_moyenne_absolue",
        cmap="YlOrRd",
        legend=True,
        legend_kwds={
            #'label': "Pente moyenne (%)",
            'orientation': "vertical"
        },
        ax=ax,
        linewidth=0.1,
        edgecolor="grey",
        alpha=0.8
    )

    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
    ax.set_title("Pente moyenne absolue (%) des routes par carreau", fontsize=18)
    ax.axis('off')
    plt.tight_layout()

    # (Optionnel) Export PNG
    if export :
        images_export_dir = os.path.join(images_dir, "indicateur_pentes_absolue_bd_topo.png")
        plt.savefig(images_export_dir, dpi=300, bbox_inches='tight')
        print(f"Carte exportée vers : {images_export_dir}")

    plt.show()
    plt.close()

# Exécution
afficher_pentes_absolues_bd_topo(export = True)


# #### 6.2.7. Nombre de stations de réparation de vélos
# ---
# Données utilisées :
# * Stations d'autoréparation de vélos
# * Stations de gonflages et d'outils pour les vélos

# In[431]:


def calculer_nombre_stations_reparation_velos():
    # 1. Chargement des données
    carreaux = charger_fichier_parquet("maille_200m_avec_donnees", crs=3857)
    stations_osm = charger_fichier_parquet("stations_autoreparation_velo", crs=3857)
    stations_velos = charger_fichier_parquet("stations_velos", crs=3857)

    # 2. Supprime les colonnes de jointure et 'nb_stations_reparation_velo' si elles existent déjà
    carreaux = nettoyer_colonnes(carreaux, ["nb_stations_reparation_velo"])

    # 3. Fusion des deux sources
    stations_reparation = pd.concat([stations_osm, stations_velos], ignore_index=True)

    # 4. Jointure spatiale : associer chaque station à un carreau
    stations_avec_carreaux = gpd.sjoin(
        stations_reparation,
        carreaux[["idINSPIRE", "geometry"]],
        how="inner",
        predicate="within"
    )

    # 5. Décompte par carreau
    nb_stations_par_carreau = (
        stations_avec_carreaux.groupby("idINSPIRE")
        .size()
        .reset_index(name="nb_stations_reparation_velo")
    )

    # 6. Fusion avec les carreaux
    carreaux = carreaux.merge(nb_stations_par_carreau, on="idINSPIRE", how="left")
    carreaux["nb_stations_reparation_velo"] = carreaux["nb_stations_reparation_velo"].fillna(0).astype(int)

    # 7. Export
    exporter_parquet(carreaux, "maille_200m_avec_donnees")
    exporter_gpkg(carreaux, "maille_200m_avec_donnees")
    print("Colonne 'nb_stations_reparation_velo' ajoutée avec succès.")

# Exécution
calculer_nombre_stations_reparation_velos()


# In[254]:


def afficher_nombre_stations_reparation_velos(export = False):
    # 1. Chargement des données
    limites_epci = charger_fichier_parquet("limites_epci", crs=3857)
    carreaux = charger_fichier_parquet("maille_200m_avec_donnees", crs=3857)

    # 2.Affichage
    fig, ax = plt.subplots(figsize=(10, 10))
    limites_epci.plot(ax=ax, alpha=1, edgecolor='black', facecolor='none')

    carreaux.plot(column="nb_stations_reparation_velo",
             cmap="YlGn",
             legend=True,
             legend_kwds={'label': "Nombre de stations de réparation de vélos par maille"},
             ax=ax,
             linewidth=0.1,
             edgecolor="grey")

    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)

    ax.set_title("Nombre de stations de réparation de vélos par maille")
    ax.axis('off')
    plt.tight_layout()

    # (Optionnel) Export PNG
    if export :
        images_export_dir = os.path.join(images_dir, "indicateur_nb_stations_reparation_velos.png")
        plt.savefig(images_export_dir, dpi=300, bbox_inches='tight')
        print(f"Carte exportée vers : {images_export_dir}")

    plt.show()
    plt.close()

# Exécution
afficher_nombre_stations_reparation_velos(export = True)


# In[134]:


# Version normalisée : % de surface des carreaux, zone tampon de 400 mètres
def calculer_part_surface_couverte_stations_reparation():
    # 1. Chargement des données
    carreaux = charger_fichier_parquet("maille_200m_avec_donnees", crs=2154).copy()
    stations_osm = charger_fichier_parquet("stations_autoreparation_velo", crs=2154)
    stations_velos = charger_fichier_parquet("stations_velos", crs=2154)

    # 2. Nettoyage des colonnes existantes
    carreaux = nettoyer_colonnes(carreaux, ["part_normalise_couverte_stations_rep_velos"])

    # 3. Fusion des deux sources
    stations_reparation = pd.concat([stations_osm, stations_velos], ignore_index=True)
    stations_reparation = stations_reparation[
        stations_reparation.geometry.notnull() & (stations_reparation.geometry.type == "Point")
    ].copy()

    # 4. Création des buffers 400 m
    stations_reparation["buffer_400m"] = stations_reparation.geometry.buffer(400)

    # 5. Union des buffers
    zone_couverte = stations_reparation["buffer_400m"].geometry.union_all()

    # 6. Intersection avec les carreaux
    intersections = carreaux.geometry.intersection(zone_couverte)

    # 7. Calcul de la part de surface
    surfaces_totales = carreaux.geometry.area
    surfaces_intersectees = intersections.area
    part_surface = (surfaces_intersectees / surfaces_totales).clip(upper=1.0) * 100

    # 8. Ajout à la table
    carreaux["part_normalise_couverte_stations_rep_velos"] = part_surface.round(2).fillna(0)

    # 9. Export
    exporter_parquet(carreaux, "maille_200m_avec_donnees")
    exporter_gpkg(carreaux, "maille_200m_avec_donnees")

    # 10. Résumé
    print("Colonne 'part_normalise_couverte_stations_rep_velos' ajoutée avec succès.")
    print(f"Valeur moyenne : {carreaux['part_normalise_couverte_stations_rep_velos'].mean():.2f}%")

# Exécution
calculer_part_surface_couverte_stations_reparation()


# In[367]:


def afficher_part_surface_couverte_stations_reparation(export = False):
    # 1. Chargement des données
    limites_epci = charger_fichier_parquet("limites_epci", crs=3857)
    carreaux = charger_fichier_parquet("maille_200m_avec_donnees", crs=3857)

    # 2. Affichage
    fig, ax = plt.subplots(figsize=(10, 10))
    limites_epci.plot(ax=ax, alpha=1, edgecolor='black', facecolor='none')

    carreaux.plot(column="part_normalise_couverte_stations_rep_velos",
             cmap="YlGn",
             legend=True,
             #legend_kwds={'label': "% de surface des carreaux présente à moins de 400 mètres d'une station de réparation de vélos"},
             ax=ax,
             linewidth=0.1,
             edgecolor="grey")

    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)

    ax.set_title("Part de surface présente à moins de 400 mètres \nd'une station de réparation de vélos", fontsize=18)
    ax.axis('off')
    plt.tight_layout()

    # (Optionnel) Export PNG
    if export :
        images_export_dir = os.path.join(images_dir, "indicateur_part_normalise_couverte_stations_rep_velos.png")
        plt.savefig(images_export_dir, dpi=300, bbox_inches='tight')
        print(f"Carte exportée vers : {images_export_dir}")

    plt.show()
    plt.close()

# Exécution
afficher_part_surface_couverte_stations_reparation(export = True)


# #### 6.2.8. Part des emplois accessibles en moins de T minutes
# ---

# In[39]:


# 900 secondes : 15 minutes
calculer_ratio_emplois_services_proches(nom_graphe="velos", secondes=900, n_jobs=-1)


# In[364]:


afficher_ratio_emplois_proches(nom_graphe = "velos", export = True)


# #### 6.2.9. Part des services accessibles en moins de T minutes. Moyenne sur S types et/ou catégories de services
# ---

# In[191]:


calculer_moyenne_ratio_services_proches(transport="velos")


# In[365]:


afficher_ratio_services_proches(nom_graphe = "velos", export = True)


# #### 6.2.10. Nombre d'accidents impliquant des vélos (2023)
# ---

# In[325]:


calculer_nb_accidents(transport="velos")


# In[258]:


afficher_nb_accidents(transport="velos", export = True)


# ### 6.3. Bus
# ---
# Indicateur de qualité de service des bus à calculer à partir des indicateurs suivants :
# * 6.3.1. Temps d'accès effectif à un véhicule - 'att_moy_semaine_bus'
# * 6.3.2. Coût d'un trajet (€/h) - 'cout_horaire_bus'
# * 6.3.3. Part des arrêts de bus disposant de parking relais - 'ratio_arrets_pr_bus'
# * 6.3.4. Part des arrêts de bus disposant de stations d'autopartage - 'ratio_arrets_autopartage_bus'
# * 6.3.5. Part des routes réservées aux bus - 'part_routes_bus'
# * 6.3.6. Part de surface bâtie à moins de 400m d'une station - 'part_batie_autour_arrets_bus'
# * Sous-indicateur de mobilité :
#     * 6.3.7. Part des emplois accessibles en moins de T minutes - 'marche_bus_ratio_emploi'
#     * 6.3.8. Part des services accessibles en moins de T minutes. Moyenne sur S types et/ou catégories de services - 'marche_bus_ratio_services_dom_A' à 'marche_bus_ratio_services_dom_F'

# #### 6.3.1. Temps d'attente moyen
# ---

# In[284]:


calculer_temps_attente_moyen(suffixe="bus")


# In[286]:


afficher_temps_attente_moyen_par_arret(transport="bus", export=True)


# In[297]:


calculer_temps_attente_par_maille(transport="bus")


# In[372]:


afficher_temps_attente_par_maille(transport="bus", export=True)


# In[491]:


# Exécution - EATV
# calculer_temps_acces_effectif(suffixe="bus")


# #### 6.3.2. Coût d'un trajet (€/h)
# ---
# Pour l'EMS, le coût d'un ticket de bus / tram en aller-simple n'est pas disponible en open data. L'information a donc été récupérée depuis le site d la CTS et inscrite en dur dans le script d'après https://www.cts-strasbourg.eu/fr/Titres-de-transport/tarifs/Tickets/ : 
# * Un ticket acheté à bord des bus coûte 2,5 €
# * Acheté à l'avance ou rechargé, il coûte 1,9 €. Comme les tickets ne peuvent être achetés à bord des trams et pour mieux comparer les coûts entre ces deux moyens de transports, c'est la valeur conservé ici.
# 
# Comme il ne semble pas y avoir de limites de temps sur les tickets de bus, un trajet peut durer aussi bien quelques minutes que plus d'une heure. Le calcul de cet indicateur est basé sur le temps de trajet moyen sur les lignes d'après dans les flux GTFS.
# 
# On récupère pour chaque arrêt ses trajets réalisables et leur durée : on peut ensuite calculer le coût horaire moyen (€ / h) de cet arrêt.
# 
# Comme on calcule la mobilité locale, les éventuels arrêts hors de la zone étudiée ne sont pas pris en compte.

# In[359]:


calculer_cout_trajet_par_arret(transport="bus")


# In[360]:


afficher_cout_trajet_par_arret(transport="bus", export = True)


# In[367]:


calculer_cout_trajet_par_maille(transport="bus")


# In[374]:


afficher_cout_trajet_par_maille(transport="bus", export = True)


# #### 6.3.3. Part des arrêts de bus disposant de parking relais
# ---
# Ici, on considère qu'un arrêt possède un parking relais si il y en a un à moins de 400 mètres de l'arrêt (6 min à pied pour une marche à 4 km/h) 

# In[377]:


calculer_part_arrets_avec_pr(transport="bus")


# In[378]:


afficher_part_arrets_avec_pr(transport="bus", export = True)


# In[398]:


calculer_part_arrets_avec_pr_par_maille(transport="bus")


# In[393]:


afficher_part_arrets_avec_pr_par_maille(transport="bus", export = True)


# #### 6.3.4. Part des arrêts de bus disposant de stations d'autopartage
# ---

# In[497]:


calculer_part_arrets_avec_autopartage(transport="bus")


# In[279]:


afficher_part_arrets_avec_autopartage("bus", export = True)


# In[391]:


calculer_part_arrets_avec_autopartage_par_maille(transport="bus")


# In[396]:


afficher_part_arrets_avec_autopartage_par_maille(transport="bus", export = True)


# #### 6.3.5. Part des routes réservées aux bus
# ---
# Dans la BD Topo, une valeur différente de 'NULL' dans l'attribut 'BUS' signifie que la voie est réservé aux bus : https://bdtopoexplorer.ign.fr/?id_theme=72&id_classe=77#attribute_563
# 
# Si l'attribut 'ACESS_VL' est égal à 'Restreint aux ayants droit', cela peut signifier que la voie est réservée aux bus, ou bien d'autres interdictions : "http://bdtopoexplorer.ign.fr/?id_theme=72&id_classe=77#attribute_value_655". Comme le spectre des restrictions est trop large, il n'est pas utilisé ici.

# In[382]:


def calculer_part_routes_reservees_bus():
    # 1. Chargement des données
    carreaux = charger_fichier_parquet("maille_200m_avec_donnees", crs=3857).copy()
    routes = charger_fichier_parquet("bd_topo_routes_epci", crs=3857).copy()

    # 2. Filtrage : routes valides et en service
    routes = routes[
        routes.geometry.notnull() &
        (routes["ETAT"] == "En service") &
        routes.geometry.type.isin(["LineString", "MultiLineString"])
    ].copy()

    # 3. Calcul de la longueur initiale
    routes["longueur"] = routes.geometry.length

    # 4. Indicateur "réservé bus" (si champ BUS non nul)
    routes["reserve_bus"] = routes["BUS"].notnull()

    # 5. Découpage spatial des routes par maille
    routes_par_carreau = gpd.overlay(
        routes[["longueur", "reserve_bus", "geometry"]],
        carreaux[["idINSPIRE", "geometry"]],
        how="intersection"
    )

    # 6. Longueur réelle après découpage
    routes_par_carreau["longueur"] = routes_par_carreau.geometry.length

    # 7. Agrégation : longueur totale et longueur réservée aux bus par carreau
    stats = routes_par_carreau.groupby("idINSPIRE").agg(
        longueur_totale=("longueur", "sum"),
        longueur_bus=("longueur", lambda x: x[routes_par_carreau.loc[x.index, "reserve_bus"]].sum())
    ).reset_index()

    # 8. Fusion avec les carreaux
    carreaux = carreaux.merge(stats, on="idINSPIRE", how="left")

    # 9. Calcul de la part des routes réservées aux bus (en pourcentage)
    carreaux["part_routes_bus"] = (
        (carreaux["longueur_bus"] / carreaux["longueur_totale"]) * 100  # Conversion en %
    ).round(2)  # Arrondi à 2 décimales
    carreaux["part_routes_bus"] = carreaux["part_routes_bus"].fillna(-1) # -1 = pas de route

    # 10. Nettoyage
    carreaux.drop(columns=["longueur_totale", "longueur_bus"], inplace=True)

    # 11. Export
    exporter_parquet(carreaux, "maille_200m_avec_donnees")
    exporter_gpkg(carreaux, "maille_200m_avec_donnees")

    # Statistiques
    valides = carreaux[carreaux["part_routes_bus"] >= 0]
    print(f"""
    Part des routes réservées aux bus :
    - Carreaux avec données : {len(valides)}/{len(carreaux)}
    - Moyenne : {valides["part_routes_bus"].mean():.2f}%
    - Médiane : {valides["part_routes_bus"].median():.2f}%
    - Max : {valides["part_routes_bus"].max():.2f}%
    """)

# Exécution
calculer_part_routes_reservees_bus()


# In[385]:


afficher_part_routes_reservees(transport="bus", export = True)


# #### 6.3.6. Part de surface bâtie à moins de 400m d'un arrêt de bus
# ---

# In[509]:


calculer_part_surface_batie_autour_arrets(transport="bus")


# In[98]:


afficher_part_surface_batie_autour_arrets(transport="bus", export = True)


# In[126]:


calculer_part_surface_batie_autour_arrets_par_maille(transport="bus")


# In[379]:


afficher_part_surface_batie_autour_arrets_par_maille(transport="bus", export = True)


# #### 6.3.7. Part des emplois accessibles en moins de T minutes
# ---

# In[267]:


# 900 secondes : 15 minutes
calculer_ratio_emplois_services_proches(nom_graphe="bus_marche", secondes=900, n_jobs=-1)


# In[ ]:


afficher_ratio_emplois_proches(nom_graphe = "bus", export = True)


# #### 6.3.8. Part des services accessibles en moins de T minutes. Moyenne sur S types et/ou catégories de services.
# ---

# In[ ]:


afficher_ratio_services_proches(nom_graphe = "bus", export = True)


# #### 6.3.9. Nombre d'accidents liés aux bus (2023)
# ---

# In[327]:


calculer_nb_accidents(transport="bus")


# In[274]:


afficher_nb_accidents(transport="bus", export = True)


# ### 6.4. Tram
# ---
# Indicateur de qualité de service des tram à calculer à partir des indicateurs suivants :
# * 6.4.1. Temps d'accès effectif à un véhicule - 'att_secondes_moy_semaine' et 'att_secondes_moy_week_end'
# * 6.4.2. Coût d'un trajet (€/h) - 'cout_horaire'
# * 6.4.3. Part des arrêts de tram disposant de parking relais - 'ratio_arrets_pr_tram'
# * 6.4.4. Part des arrêts de tram disposant de stations d'autopartage - 'ratio_arrets_autopartage_tram'
# * 6.4.5. Part des routes réservées aux tram - 'part_routes_tram'
# * 6.4.6. Part de surface bâtie à moins de 400m d'une station - 'part_batie_autour_arrets_tram'
# * Sous-indicateur de mobilité :
#     * 6.4.7. Part des emplois accessibles en moins de T minutes - 'marche_ratio_emploi'
#     * 6.4.8. Part des services accessibles en moins de T minutes. Moyenne sur S types et/ou catégories de services - 'marche_tram_ratio_services_dom_A' à 'marche_tram_ratio_services_dom_F'

# #### 6.4.1. Temps d'attente moyen
# ---

# In[302]:


calculer_temps_attente_moyen(suffixe="tram")


# In[303]:


afficher_temps_attente_moyen_par_arret(transport="tram", export=True)


# In[304]:


calculer_temps_attente_par_maille(transport="tram")


# In[386]:


afficher_temps_attente_par_maille(transport="tram", export=True)


# In[492]:


# Exécution - EATV
# calculer_temps_acces_effectif("tram")


# #### 6.4.2. Coût d'un trajet (€/h)
# ---

# In[212]:


calculer_cout_trajet_par_arret(transport="tram")


# In[218]:


afficher_cout_trajet_par_arret(transport="tram", export = True)


# In[371]:


calculer_cout_trajet_par_maille(transport="tram")


# In[387]:


afficher_cout_trajet_par_maille(transport="tram", export = True)


# #### 6.4.3. Part des arrêts de tram disposant de parking relais
# ---

# In[55]:


calculer_part_arrets_avec_pr(transport="tram")


# In[276]:


afficher_part_arrets_avec_pr(transport="tram", export = True)


# In[400]:


calculer_part_arrets_avec_pr_par_maille(transport="tram")


# In[394]:


afficher_part_arrets_avec_pr_par_maille(transport="tram", export = True)


# #### 6.4.4. Part des arrêts de tram disposant de stations d'autopartage
# ---
# Ici, on considère qu'un arrêt possède une station d'autopartage si il y en a une à moins de 400 mètres de l'arrêt (6 min à pied pour une marche à 4 km/h)

# In[282]:


calculer_part_arrets_avec_autopartage(transport="tram")


# In[283]:


afficher_part_arrets_avec_autopartage("tram", export = True)


# In[154]:


calculer_part_arrets_avec_autopartage_par_maille(transport="tram")


# In[397]:


afficher_part_arrets_avec_autopartage_par_maille(transport="tram", export = True)


# #### 6.4.5. Part des routes réservées aux tram
# ---
# Calcul pour chaque maille : longueur des lignes de tram / (longueur des lignes de tram + longueur routes de la BD Topo)

# In[117]:


def calculer_part_routes_reservees_tram():
    # 1. Chargement des données
    print("Chargement des données...")
    carreaux = charger_fichier_parquet("maille_200m_avec_donnees", crs=2154).copy()
    routes = charger_fichier_parquet("bd_topo_routes_epci", crs=2154).copy()
    lignes_tram = charger_fichier_parquet("lignes_tram", crs=2154).copy()

    # 2. Prétraitement : routes valides
    routes = routes[
        routes.geometry.notnull() &
        (routes["ETAT"] == "En service") &
        routes.geometry.type.isin(["LineString", "MultiLineString"])
    ].copy()
    routes["longueur_route"] = routes.geometry.length

    # 3. Prétraitement : lignes de tram valides
    lignes_tram = lignes_tram[
        lignes_tram.geometry.notnull() &
        lignes_tram.geometry.type.isin(["LineString", "MultiLineString"])
    ].copy()
    lignes_tram["longueur_tram"] = lignes_tram.geometry.length

    # 4. Intersections avec les carreaux
    print("Découpage spatial des routes...")
    routes_par_carreau = gpd.overlay(
        routes[["geometry", "longueur_route"]],
        carreaux[["idINSPIRE", "geometry"]],
        how="intersection"
    )
    routes_par_carreau["longueur_route"] = routes_par_carreau.geometry.length

    print("Découpage spatial des lignes de tram...")
    trams_par_carreau = gpd.overlay(
        lignes_tram[["geometry", "longueur_tram"]],
        carreaux[["idINSPIRE", "geometry"]],
        how="intersection"
    )
    trams_par_carreau["longueur_tram"] = trams_par_carreau.geometry.length

    # 5. Agrégation par maille
    print("Agrégation par carreau...")
    stats_routes = routes_par_carreau.groupby("idINSPIRE")["longueur_route"].sum().reset_index()
    stats_tram = trams_par_carreau.groupby("idINSPIRE")["longueur_tram"].sum().reset_index()

    stats = stats_routes.merge(stats_tram, on="idINSPIRE", how="outer").fillna(0)

    # 6. Calcul de la part des voies réservées aux trams (converti en pourcentage)
    stats["part_routes_tram"] = (
        (stats["longueur_tram"] / (stats["longueur_route"] + stats["longueur_tram"])) * 100  # Multiplication par 100
    ).round(2)  # Arrondi à 2 décimales
    stats["part_routes_tram"] = stats["part_routes_tram"].fillna(-1).clip(lower=-1, upper=100)  # -1 si aucune voie du tout

    # 7. Fusion avec les carreaux
    carreaux = carreaux.drop(columns=["part_routes_tram"], errors="ignore")
    carreaux = carreaux.merge(stats[["idINSPIRE", "part_routes_tram"]], on="idINSPIRE", how="left")
    carreaux["part_routes_tram"] = carreaux["part_routes_tram"].fillna(-1)

    # 8. Statistiques
    valides = carreaux[carreaux["part_routes_tram"] >= 0]
    print(f"""
    Part des voies réservées au tram :
    - Carreaux avec données : {len(valides)}/{len(carreaux)}
    - Moyenne : {valides["part_routes_tram"].mean():.2f}%
    - Médiane : {valides["part_routes_tram"].median():.2f}%
    - Max : {valides["part_routes_tram"].max():.2f}%
    """)

    # 9. Export
    exporter_parquet(carreaux, "maille_200m_avec_donnees")
    exporter_gpkg(carreaux, "maille_200m_avec_donnees")

    print("Colonne 'part_routes_tram' ajoutée avec succès.")

# Exécution
calculer_part_routes_reservees_tram()


# In[151]:


def calculer_part_routes_reservees_tram():
    # 1. Chargement des données
    print("Chargement des données...")
    carreaux = charger_fichier_parquet("maille_200m_avec_donnees", crs=3857).copy()
    routes = charger_fichier_parquet("bd_topo_routes_epci", crs=3857).copy()
    lignes_tram = charger_fichier_parquet("lignes_tram", crs=3857).copy()

    # 2. Prétraitement : routes valides
    routes = routes[
        routes.geometry.notnull() &
        (routes["ETAT"] == "En service") &
        routes.geometry.type.isin(["LineString", "MultiLineString"])
    ].copy()
    routes["longueur_route"] = routes.geometry.length

    # 3. Prétraitement : lignes de tram valides
    lignes_tram = lignes_tram[
        lignes_tram.geometry.notnull() &
        lignes_tram.geometry.type.isin(["LineString", "MultiLineString"])
    ].copy()
    lignes_tram["longueur_tram"] = lignes_tram.geometry.length

    # 4. Intersections avec les carreaux
    print("Découpage spatial des routes...")
    routes_par_carreau = gpd.overlay(
        routes[["geometry", "longueur_route"]],
        carreaux[["idINSPIRE", "geometry"]],
        how="intersection"
    )
    routes_par_carreau["longueur_route"] = routes_par_carreau.geometry.length

    print("Découpage spatial des lignes de tram...")
    trams_par_carreau = gpd.overlay(
        lignes_tram[["geometry", "longueur_tram"]],
        carreaux[["idINSPIRE", "geometry"]],
        how="intersection"
    )
    trams_par_carreau["longueur_tram"] = trams_par_carreau.geometry.length

    # 5. Agrégation par maille
    print("Agrégation par carreau...")
    stats_routes = routes_par_carreau.groupby("idINSPIRE")["longueur_route"].sum().reset_index()
    stats_tram = trams_par_carreau.groupby("idINSPIRE")["longueur_tram"].sum().reset_index()

    stats = stats_routes.merge(stats_tram, on="idINSPIRE", how="outer").fillna(0)

    # 6. Calcul de la part des voies réservées aux trams
    stats["part_routes_tram"] = (
        stats["longueur_tram"] / (stats["longueur_route"] + stats["longueur_tram"])
    ).fillna(-1).clip(lower=-1, upper=1)  # -1 si aucune voie du tout

    # 7. Fusion avec les carreaux
    carreaux = carreaux.drop(columns=["part_routes_tram"], errors="ignore")
    carreaux = carreaux.merge(stats[["idINSPIRE", "part_routes_tram"]], on="idINSPIRE", how="left")
    carreaux["part_routes_tram"] = carreaux["part_routes_tram"].fillna(-1)

    # 8. Export
    exporter_parquet(carreaux, "maille_200m_avec_donnees")
    exporter_gpkg(carreaux, "maille_200m_avec_donnees")

    print("Colonne 'part_routes_tram' ajoutée avec succès.")

# Exécution
calculer_part_routes_reservees_tram()


# In[400]:


afficher_part_routes_reservees(transport="tram", export = True)


# #### 6.4.6. Part de surface bâtie à moins de 400m d'un arrêt de tram
# ---
# 
# Données utilisées : 
# * Les flux GTFS contenant les informations sur la géolocalisation des arrêts de tram
# * Couche 'BATIMENTS.shp' de la BD Topo. On utilise les bâtiements présents dans la zone tampon de l'EPCI dans le cas où les stations sont en bordure.
# 
# Si une partie du tampon de 400 mètres est hors de la france, cette surface est ignorée lors du calcul de l'indicateur (car on se base sur la BD Topo, donc on ne possède pas le bâti hors de la France). Ainsi, l'indicateur reste correct.

# In[511]:


calculer_part_surface_batie_autour_arrets(transport="tram")


# In[287]:


afficher_part_surface_batie_autour_arrets(transport="tram", export = True)


# In[128]:


calculer_part_surface_batie_autour_arrets_par_maille(transport="tram")


# In[401]:


afficher_part_surface_batie_autour_arrets_par_maille(transport="tram", export = True)


# #### 6.4.7. Part des emplois accessibles en moins de T minutes
# ---

# In[ ]:


calculer_ratio_emplois_services_proches(nom_graphe="tram", secondes=900, n_jobs=-1)


# In[106]:


afficher_ratio_emplois_proches(nom_graphe = "tram", export = True)


# #### 6.4.8. Part des services accessibles en moins de T minutes. Moyenne sur S types et/ou catégories de services.
# ---

# In[ ]:


afficher_ratio_services_proches(nom_graphe = "tram", export = True)


# #### 6.4.9. Nombre d'accidents liés aux tram (2023)
# ---

# In[329]:


calculer_nb_accidents(transport="tram")


# In[290]:


afficher_nb_accidents(transport="tram", export = True)


# ### 6.5. Véhicules légers
# ---
# Indicateur de qualité de service des véhicules légers à calculer à partir des indicateurs suivants :
# * 6.5.1. Part des rues à sens unique - 'part_sens_unique'
# * 6.5.2. Part couverte par des réglementations antipollutions - 'part_zfe'
# * 6.5.3. Nombre de feux de circulation - 'nb_feux_circulation'
# * 6.5.4. Nombre de stations service - 'nb_stations_service' (ne pas utiliser pour l'instant)
# * 6.5.5. Nombre de places de stationnement - 'places_stationnement_par_habitant'
# * 6.5.6. Coût du stationnement (€/h) - 'cout_moyen_stationnement' 
# * 6.5.7. Durée moyenne de stationnement - 'duree_moyenne_stationnement' (ne pas utiliser pour l'instant)
# * 6.5.8. Nombre de bornes de recharge électriques - 'bornes_ve_par_habitant'
# * Sous-indicateur de mobilité :
#     * 6.5.9. Part des emplois accessibles en moins de T minutes - 'vl_ratio_emplois'
#     * 6.5.10. Part des services accessibles en moins de T minutes. Moyenne sur S types et/ou catégories de services - 'vl_ratio_services_dom_A' à 'vl_ratio_services_dom_F'

# #### 6.5.1. Part des routes à sens unique
# ---

# In[307]:


def calculer_part_rues_sens_unique():
    # 1. Chargement des données
    carreaux = charger_fichier_parquet("maille_200m_avec_donnees", crs=3857)
    routes = charger_fichier_parquet("bd_topo_routes_epci", crs=3857)

    # 2. Prétraitement des routes
    routes = routes[
        (routes.geometry.notnull()) &
        (routes["ETAT"] == "En service") &
        (routes["PRIVE"].str.lower() == "non")
    ].copy()

    # 3. Sélection géométrique
    routes = routes[routes.geometry.type.isin(["LineString", "MultiLineString"])]

    # 4. Calcul de la longueur initiale
    routes["longueur"] = routes.geometry.length
    routes["est_sens_unique"] = routes["SENS"].isin(["Sens direct", "Sens inverse"])

    # 5. Intersection avec les carreaux
    routes_par_carreau = gpd.overlay(routes, carreaux[["idINSPIRE", "geometry"]], how="intersection")
    routes_par_carreau["longueur"] = routes_par_carreau.geometry.length
    routes_par_carreau["est_sens_unique"] = routes_par_carreau["SENS"].isin(["Sens direct", "Sens inverse"])

    # 6. Agrégation
    stats = routes_par_carreau.groupby("idINSPIRE").agg(
        longueur_totale=("longueur", "sum"),
        longueur_sens_unique=("longueur", lambda x: x[routes_par_carreau.loc[x.index, "est_sens_unique"]].sum())
    ).reset_index()

    # 7. Fusion dans les carreaux
    carreaux = carreaux.merge(stats, on="idINSPIRE", how="left")

    # 8. Calcul de la part (converti en pourcentage)
    carreaux["part_sens_unique"] = (
        (carreaux["longueur_sens_unique"] / carreaux["longueur_totale"]) * 100  # Conversion en %
    ).round(2)  # Arrondi à 2 décimales
    carreaux["part_sens_unique"] = carreaux["part_sens_unique"].fillna(-1).clip(lower=-1, upper=100)

    # 9. Nettoyage
    carreaux.drop(columns=["longueur_totale", "longueur_sens_unique"], inplace=True)

    # 10. Export et statistiques
    exporter_parquet(carreaux, "maille_200m_avec_donnees")
    exporter_gpkg(carreaux, "maille_200m_avec_donnees")

    # Statistiques
    valides = carreaux[carreaux["part_sens_unique"] >= 0]
    print(f"""
    Part des rues à sens unique :
    - Carreaux avec données : {len(valides)}/{len(carreaux)}
    - Moyenne : {valides["part_sens_unique"].mean():.2f}%
    - Médiane : {valides["part_sens_unique"].median():.2f}%
    - Max : {valides["part_sens_unique"].max():.2f}%
    """)

# Exécution
calculer_part_rues_sens_unique()


# In[402]:


def afficher_part_rues_sens_unique(export = False):
    # 1. Chargment des données
    limites_epci = charger_fichier_parquet("limites_epci", crs=3857)
    carreaux = charger_fichier_parquet("maille_200m_avec_donnees", crs=3857)
    carreaux_valides = carreaux[carreaux["part_sens_unique"] != -1]

    # 2. Affichage
    fig, ax = plt.subplots(figsize=(10, 10))
    limites_epci.plot(ax=ax, alpha=1, edgecolor='black', facecolor='none')

    carreaux_valides.plot(column="part_sens_unique",
             cmap="YlOrRd",
             legend=True,
             # legend_kwds={'label': "% de routes à sens unique dans la maille"},
             ax=ax,
             linewidth=0.1,
             edgecolor="grey")

    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)

    ax.set_title("Part des routes à sens unique dans la maille", fontsize = 18)
    ax.axis('off')
    plt.tight_layout()

    # (Optionnel) Export PNG
    if export :
        images_export_dir = os.path.join(images_dir, "indicateur_part_rues_sens_unique.png")
        plt.savefig(images_export_dir, dpi=300, bbox_inches='tight')
        print(f"Carte exportée vers : {images_export_dir}")

    plt.show()
    plt.close()

# Exécution
afficher_part_rues_sens_unique(export = True)


# #### 6.5.2. Part couverte par des réglementations antipollutions
# ---
# Note : ne pas utiliser dans le calcul de la qualité de service des VL ? Est quasiment uniforme

# In[293]:


def calculer_part_couverte_reg_anti_pollution():
    # 1. Chargement
    carreaux = charger_fichier_parquet("maille_200m_avec_donnees", crs=2154)
    zfe_aires = charger_fichier_parquet("zfe_aires_epci", crs=2154)
    zfe_voies = charger_fichier_parquet("zfe_voies_epci", crs=2154)
    routes = charger_fichier_parquet("bd_topo_routes_epci", crs=2154)

    # 2. Nettoyage
    zfe_aires = nettoyer_colonnes(zfe_aires)
    zfe_voies = nettoyer_colonnes(zfe_voies)
    routes = nettoyer_colonnes(routes)

    # 3. ZFE actives uniquement
    aujourdhui = pd.to_datetime(datetime.now().date())
    zfe_aires = zfe_aires[pd.to_datetime(zfe_aires['date_debut']) <= aujourdhui].copy()

    # 4. Fusionner les ZFE en un seul MultiPolygon (pour éviter les doublons)
    geom_zfe_union = unary_union(zfe_aires.geometry)

    # 5. Préparer les voies de dérogation
    # Associer les largeurs
    zfe_voies = gpd.sjoin(zfe_voies, routes[['geometry', 'largeur_calculee']], how='left', predicate='intersects')
    zfe_voies = nettoyer_colonnes(zfe_voies)
    zfe_voies['largeur_calculee'] = zfe_voies['largeur_calculee'].fillna(3)  # défaut : 3m

    # Buffer des voies exclues (ZFE non applicables)
    zfe_voies['geometry'] = zfe_voies.geometry.buffer(zfe_voies['largeur_calculee'] / 2)
    geom_voies_exclues = unary_union(zfe_voies.geometry)

    # 6. Calcul des surfaces
    carreaux["surface_totale"] = carreaux.geometry.area

    # 6.1. Surface ZFE par carreau
    carreaux["surface_zfe"] = carreaux.geometry.intersection(geom_zfe_union).area

    # 6.2. Surface exclue (voies dérogatoires)
    carreaux["surface_voies_derog"] = carreaux.geometry.intersection(geom_voies_exclues).area

    # 6.3. Surface réglementée et pourcentage (arrondi à 2 décimales)
    carreaux["surface_reglementee"] = (carreaux["surface_zfe"] - carreaux["surface_voies_derog"]).clip(lower=0)
    carreaux["part_zfe"] = ((carreaux["surface_reglementee"] / carreaux["surface_totale"]) * 100).round(2)

    # 7. Statistiques globales
    surface_totale_epci = carreaux["surface_totale"].sum()
    surface_reglementee_epci = carreaux["surface_reglementee"].sum()
    pourcentage_global = round((surface_reglementee_epci / surface_totale_epci) * 100, 2)

    print("\n--- Statistiques ---")
    print(f"Surface totale analysée : {surface_totale_epci / 1e6:.1f} km²")
    print(f"Surface réglementée (ZFE - voies exclues) : {surface_reglementee_epci / 1e6:.1f} km²")
    print(f"Pourcentage couvert : {pourcentage_global:.2f} %")  # Format à 2 décimales

    # 8. Nettoyage avant export
    colonnes_a_supprimer = ['surface_totale', 'surface_zfe', 'surface_voies_derog', 'surface_reglementee']
    carreaux = nettoyer_colonnes(carreaux, colonnes_a_supprimer)

    # 9. Export
    exporter_parquet(carreaux, "maille_200m_avec_donnees")
    exporter_gpkg(carreaux, "maille_200m_avec_donnees")

# Exécution
calculer_part_couverte_reg_anti_pollution()


# In[403]:


def afficher_part_couverte_reg_anti_pollution(export = False):
    # 1. Chargement des données
    limites_epci = charger_fichier_parquet("limites_epci", crs=3857)
    carreaux_zfe = charger_fichier_parquet("maille_200m_avec_donnees", crs=3857)

    # 2. Affichage
    fig, ax = plt.subplots(figsize=(10, 10))

    limites_epci.plot(ax=ax, color='none', edgecolor='black', linewidth=1)

    carreaux_zfe.plot(
        column='part_zfe',
        cmap='YlOrRd',
        legend=True,
        ax=ax,
        edgecolor='none',
        alpha=0.7
    )

    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)

    ax.set_title("Couverture par les réglementations ZFE (%)", fontsize=18)
    ax.axis('off')
    plt.tight_layout()

    # (Optionnel) Export PNG
    if export :
        images_export_dir = os.path.join(images_dir, "indicateur_part_couverte_reg_anti_pollution.png")
        plt.savefig(images_export_dir, dpi=300, bbox_inches='tight')
        print(f"Carte exportée vers : {images_export_dir}")

    plt.show()
    plt.close()

# Exécution
afficher_part_couverte_reg_anti_pollution(export = True)


# #### 6.5.3. Nombre de feux de circulation
# ---
# 
# NOTE : à normaliser

# In[515]:


def calculer_nombre_feux_circulation():
    # 1. Chargement des données
    carreaux = charger_fichier_parquet("maille_200m_avec_donnees", crs=3857).copy()
    feux_circulation_epci = charger_fichier_parquet("feux_circulation_epci", crs=3857).copy()

    # 2. Supprime la colonne 'index_right' si elle existe
    feux_circulation_epci = nettoyer_colonnes(feux_circulation_epci)

    # 3. Supprime l’ancienne colonne si elle existe pour éviter conflits de fusion
    carreaux = nettoyer_colonnes(carreaux, ['nb_feux_circulation'])

    # 4. Filtrer uniquement les points valides
    feux_circulation_epci = feux_circulation_epci[
        feux_circulation_epci.geometry.notnull() & (feux_circulation_epci.geometry.type == "Point")
    ].copy()

    # 5. Jointure spatiale : associer chaque station à un carreau
    feux_avec_carreaux = gpd.sjoin(
        feux_circulation_epci,
        carreaux[["idINSPIRE", "geometry"]],
        how="inner",
        predicate="within"
    )

    # 6. Agrégation : nombre de stations par carreau
    nb_feux_par_carreau = feux_avec_carreaux.groupby("idINSPIRE").size().reset_index(name="nb_feux_circulation")

    # 7. Fusion avec les carreaux
    carreaux = carreaux.merge(nb_feux_par_carreau, on="idINSPIRE", how="left")
    carreaux["nb_feux_circulation"] = carreaux["nb_feux_circulation"].fillna(0).astype(int)

    # 8. Export
    exporter_parquet(carreaux, "maille_200m_avec_donnees")
    exporter_gpkg(carreaux, "maille_200m_avec_donnees")
    print("Colonne 'nb_feux_circulation' ajoutée avec succès.")

# Exécution
calculer_nombre_feux_circulation()


# In[404]:


def afficher_nombre_feux_circulation(export = False):
    # 1. Chargement des données
    limites_epci = charger_fichier_parquet("limites_epci", crs=3857)
    carreaux = charger_fichier_parquet("maille_200m_avec_donnees", crs=3857)

    # 2. Affichage
    fig, ax = plt.subplots(figsize=(10, 10))
    limites_epci.plot(ax=ax, alpha=1, edgecolor='black', facecolor='none')

    carreaux.plot(column="nb_feux_circulation",
             cmap="YlOrRd",
             legend=True,
             #legend_kwds={'label': "Nombre de feux de circulation par maille"},
            ax=ax,
             linewidth=0.1,
             edgecolor="grey")

    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)

    ax.set_title("Nombre de feux de circulation par maille", fontsize=18)
    ax.axis('off')
    plt.tight_layout()

    # (Optionnel) Export PNG
    if export :
        images_export_dir = os.path.join(images_dir, "indicateur_nombre_feux_circulation.png")
        plt.savefig(images_export_dir, dpi=300, bbox_inches='tight')
        print(f"Carte exportée vers : {images_export_dir}")

    plt.show()
    plt.close()

# Exécution
afficher_nombre_feux_circulation(export = True)


# #### 6.5.4. Nombre de stations service
# ---

# In[40]:


def calculer_nombre_stations_service():
    # 1. Chargement des données
    carreaux = charger_fichier_parquet("maille_200m_avec_donnees", crs=3857).copy()
    stations_service_epci = charger_fichier_parquet("stations_service_epci", crs=3857).copy()

    # 2. Supprime la colonne 'index_right' si elle existe
    stations_service_epci = nettoyer_colonnes(stations_service_epci)

    # 3. Supprime l’ancienne colonne si elle existe pour éviter conflits de fusion
    carreaux = nettoyer_colonnes(carreaux, ['nb_stations_service'])

    # 4. Filtrer uniquement les points valides
    stations_service_epci = stations_service_epci[
        stations_service_epci.geometry.notnull() & (stations_service_epci.geometry.type == "Point")
    ].copy()

    # 5. Jointure spatiale : associer chaque station à un carreau
    stations_avec_carreaux = gpd.sjoin(
        stations_service_epci,
        carreaux[["idINSPIRE", "geometry"]],
        how="inner",
        predicate="within"
    )

    # 6. Agrégation : nombre de stations par carreau
    nb_stations_par_carreau = stations_avec_carreaux.groupby("idINSPIRE").size().reset_index(name="nb_stations_service")

    # 7. Fusion avec les carreaux
    carreaux = carreaux.merge(nb_stations_par_carreau, on="idINSPIRE", how="left")
    carreaux["nb_stations_service"] = carreaux["nb_stations_service"].fillna(0).astype(int)

    # 8. Export
    exporter_parquet(carreaux, "maille_200m_avec_donnees")
    exporter_gpkg(carreaux, "maille_200m_avec_donnees")

# Exécution
calculer_nombre_stations_service()


# In[154]:


def afficher_nombre_stations_service(export = False):
    # 1. Chargement des données
    limites_epci = charger_fichier_parquet("limites_epci", crs=3857)
    carreaux = charger_fichier_parquet("maille_200m_avec_donnees", crs=3857)

    # 2. Affichage
    fig, ax = plt.subplots(figsize=(10, 10))
    limites_epci.plot(ax=ax, alpha=1, edgecolor='black', facecolor='none')

    carreaux.plot(column="nb_stations_service",
             cmap="YlGn",
             legend=True,
             legend_kwds={'label': "Nombre de stations services par maille"},
             ax=ax,
             linewidth=0.1,
             edgecolor="grey")

    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)

    ax.set_title("Nombre de stations services par maille")
    ax.axis('off')
    plt.tight_layout()

    # (Optionnel) Export PNG
    if export :
        images_export_dir = os.path.join(images_dir, "indicateur_nombre_stations_service.png")
        plt.savefig(images_export_dir, dpi=300, bbox_inches='tight')
        print(f"Carte exportée vers : {images_export_dir}")

    plt.show()
    plt.close()

# Exécution
afficher_nombre_stations_service(export = True)


# In[406]:


# Version normalisée, avec zones tampons de 4 km. 4 km à 50 km /h : quasiment 5 minutes (4 min 48)
def calculer_part_surface_couverte_stations_service():
    # 1. Chargement des données
    carreaux = charger_fichier_parquet("maille_200m_avec_donnees", crs=2154).copy()
    stations_service = charger_fichier_parquet("stations_service_epci", crs=2154)

    # 2. Nettoyage des colonnes existantes
    carreaux = nettoyer_colonnes(carreaux, ["part_normalise_couverte_stations_service"])

    # 3. Nettoyage géométrie des stations
    stations_service = stations_service[
        stations_service.geometry.notnull() & (stations_service.geometry.type == "Point")
    ].copy()

    # 4. Création des buffers 4000 m
    stations_service["buffer_4km"] = stations_service.geometry.buffer(4000)

    # 5. Union des buffers
    zone_couverte = stations_service["buffer_4km"].geometry.union_all()

    # 6. Intersection avec les carreaux
    intersections = carreaux.geometry.intersection(zone_couverte)

    # 7. Calcul de la part de surface (max 100 %)
    surfaces_totales = carreaux.geometry.area
    surfaces_intersectees = intersections.area
    part_surface = (surfaces_intersectees / surfaces_totales).clip(upper=1.0) * 100

    # 8. Ajout à la table
    carreaux["part_normalise_couverte_stations_service"] = part_surface.round(2).fillna(0)

    # 9. Export
    exporter_parquet(carreaux, "maille_200m_avec_donnees")
    exporter_gpkg(carreaux, "maille_200m_avec_donnees")

    # 10. Résumé
    print("Colonne 'part_normalise_couverte_stations_service' ajoutée avec succès.")
    print(f"Valeur moyenne : {carreaux['part_normalise_couverte_stations_service'].mean():.2f}%")

# Exécution
calculer_part_surface_couverte_stations_service()


# In[408]:


def afficher_part_surface_couverte_stations_service(export = False):
    # 1. Chargement des données
    limites_epci = charger_fichier_parquet("limites_epci", crs=3857)
    carreaux = charger_fichier_parquet("maille_200m_avec_donnees", crs=3857)

    # 2. Affichage
    fig, ax = plt.subplots(figsize=(10, 10))
    limites_epci.plot(ax=ax, alpha=1, edgecolor='black', facecolor='none')

    carreaux.plot(column="part_normalise_couverte_stations_service",
             cmap="YlGn",
             legend=True,
             #legend_kwds={'label': "% de surface des carreaux présente à moins de 4 km d'une station service"},
             ax=ax,
             linewidth=0.1,
             edgecolor="grey")

    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)

    ax.set_title("Part de surface présente à moins de 4 km d'une station-service", fontsize=18)
    ax.axis('off')
    plt.tight_layout()

    # (Optionnel) Export PNG
    if export :
        images_export_dir = os.path.join(images_dir, "indicateur_part_normalise_couverte_stations_service.png")
        plt.savefig(images_export_dir, dpi=300, bbox_inches='tight')
        print(f"Carte exportée vers : {images_export_dir}")

    plt.show()
    plt.close()

# Exécution
afficher_part_surface_couverte_stations_service(export = True)


# #### 6.5.5. Nombre de places de stationnement
# ---
# Note : le nombre de places de stationnements est ici sous-estimé. Les données sur les places en voiries de l'EMS oublient de nombreuses routes où des voitures peuvent être garées.
# 
# Données utilisées : 
# * Places de stationnement dans l'EMS (https://data.strasbourg.eu/explore/dataset/vo_st_stationmnt_vehi/information/?disjunctive.occupation) pour les données de voirie
# * OSM pour les données hors-voiries (parkings accessibles aux véhicules légers)

# In[298]:


def calculer_nombre_places_stationnement_vl():
    # 1. Chargement des données
    carreaux = charger_fichier_parquet("maille_200m_avec_donnees", crs=3857)
    stationnement_vl = charger_fichier_parquet("stationnement_vl", crs=3857).copy()
    osm_stationnement = charger_fichier_parquet("osm_stationnement_epci", crs=3857).copy()

    # Supression des colonnes de jointures et de celles déjà calculées (si présentes) 
    colonnes_a_supprimer = ['nb_places_voirie', 'nb_places_hors_voirie', 'nb_places_stationnement']
    carreaux = nettoyer_colonnes(carreaux, colonnes_a_supprimer)

    # 2. Suppression des colonnes inutiles
    for df in [carreaux, stationnement_vl, osm_stationnement]:
        df.drop(columns=[col for col in df.columns if "index_right" in col], inplace=True, errors="ignore")

    # 3. Nettoyage des données voirie
    stationnement_vl["nbre_places"] = pd.to_numeric(stationnement_vl["nbre_places"], errors="coerce")
    voirie_valide = stationnement_vl[stationnement_vl.geometry.notnull() & (stationnement_vl["nbre_places"] > 0)]

    # 4. Nettoyage des données OSM pour ne conserver que les parkings des VL
    osm_stationnement["capacity"] = pd.to_numeric(osm_stationnement.get("capacity", pd.NA), errors="coerce")
    osm_valides = osm_stationnement[
        osm_stationnement["geometry"].notnull() &
        osm_stationnement["capacity"].notnull() &
        osm_stationnement["type_stationnement"].isin(["parking_surface", "parking_souterrain"])
    ]

    # 5. Jointure spatiale avec les carreaux
    voirie_par_carreau = gpd.sjoin(
        voirie_valide[["nbre_places", "geometry"]],
        carreaux[["idINSPIRE", "geometry"]],
        how="inner", predicate="intersects"
    )
    osm_par_carreau = gpd.sjoin(
        osm_valides[["capacity", "geometry"]],
        carreaux[["idINSPIRE", "geometry"]],
        how="inner", predicate="intersects"
    )

    # 6. Agrégation par carreau
    places_voirie = voirie_par_carreau.groupby("idINSPIRE")["nbre_places"].sum().reset_index()
    places_voirie.rename(columns={"nbre_places": "nb_places_voirie"}, inplace=True)

    places_hors_voirie = osm_par_carreau.groupby("idINSPIRE")["capacity"].sum().reset_index()
    places_hors_voirie.rename(columns={"capacity": "nb_places_hors_voirie"}, inplace=True)

    # 7. Fusion des résultats avec les carreaux
    carreaux = carreaux.merge(places_voirie, on="idINSPIRE", how="left")
    carreaux = carreaux.merge(places_hors_voirie, on="idINSPIRE", how="left")

    # 8. Nettoyage final
    carreaux["nb_places_voirie"] = carreaux["nb_places_voirie"].fillna(0).astype(int)
    carreaux["nb_places_hors_voirie"] = carreaux["nb_places_hors_voirie"].fillna(0).astype(int)
    carreaux["nb_places_stationnement"] = carreaux["nb_places_voirie"] + carreaux["nb_places_hors_voirie"]
    # Pour supprimer les colonnes de test servant à vérifier les résultats
    carreaux.drop(columns=["nb_places_voirie", "nb_places_hors_voirie"], inplace=True)

    # 9. Export
    exporter_parquet(carreaux, "maille_200m_avec_donnees")
    exporter_gpkg(carreaux, "maille_200m_avec_donnees")

    print("Colonne 'nb_places_stationnement' ajoutée avec succès.")

# Exécution
calculer_nombre_places_stationnement_vl()


# In[409]:


def afficher_nombre_places_stationnement_vl(export = False):
    # 1. Chargement des données
    carreaux = charger_fichier_parquet("maille_200m_avec_donnees", crs=3857)
    limites_epci = charger_fichier_parquet("limites_epci", crs=3857)

    # 2. Affichage
    fig, ax = plt.subplots(figsize=(10, 10))

    limites_epci.plot(ax=ax, alpha=1, edgecolor='black', facecolor='none')

    carreaux.plot(
        column="nb_places_stationnement",
        cmap="YlGn",
        legend=True,
        #legend_kwds={'label': "Nombre de places de stationnement par maille"},
        ax=ax,
        linewidth=0.1,
        edgecolor="grey"
    )

    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)

    ax.set_title("Nombre de places de stationnement par maille", fontsize=18)
    ax.axis('off')
    plt.tight_layout()

    # (Optionnel) Export PNG
    if export :
        images_export_dir = os.path.join(images_dir, "indicateur_nombre_places_stationnement_vl.png")
        plt.savefig(images_export_dir, dpi=300, bbox_inches='tight')
        print(f"Carte exportée vers : {images_export_dir}")

    plt.show()
    plt.close()

# Exécution
afficher_nombre_places_stationnement_vl(export = True)


# In[149]:


# Version normalisée
def calculer_places_stationnement_vl_par_habitant():
    # 1. Chargement des données
    carreaux = charger_fichier_parquet("maille_200m_avec_donnees", crs=2154).copy()

    # 2. Suppression des anciennes colonnes si elles existent
    carreaux = nettoyer_colonnes(carreaux, ["places_stationnement_par_habitant"])

    # 3. Calcul des indicateurs normalisés
    def calcul_ratio(nb_places, pop):
        if pop is None or pop <= 0:
            return -1  # Valeur par défaut si population inconnue ou nulle
        return round(nb_places / pop, 4)

    carreaux["places_stationnement_par_habitant"] = carreaux.apply(
        lambda row: calcul_ratio(row["nb_places_stationnement"], row["population_estimee"]), 
        axis=1
    )

    # 4. Export
    exporter_parquet(carreaux, "maille_200m_avec_donnees")
    exporter_gpkg(carreaux, "maille_200m_avec_donnees")

    # 5. Statistiques
    valides = carreaux[carreaux["places_stationnement_par_habitant"] >= 0]
    print(f"""
Indicateurs de stationnement par habitant ajoutés :
- {len(carreaux)} carreaux traités
- {len(valides)} avec population connue
- Moyenne (dans et hors voirie: {valides['places_stationnement_par_habitant'].mean():.4f} places/hab
""")

# Exécution
calculer_places_stationnement_vl_par_habitant()


# In[410]:


def afficher_places_stationnement_vl_par_habitant(export = False):
    # 1. Chargement des données
    carreaux = charger_fichier_parquet("maille_200m_avec_donnees", crs=3857)
    limites_epci = charger_fichier_parquet("limites_epci", crs=3857)
    carreaux = carreaux[carreaux["places_stationnement_par_habitant"] != 0]

    # 2. Affichage
    fig, ax = plt.subplots(figsize=(10, 10))

    limites_epci.plot(ax=ax, alpha=1, edgecolor='black', facecolor='none')

    carreaux.plot(
        column="places_stationnement_par_habitant",
        cmap="YlGn",
        legend=True,
        # legend_kwds={'label': "Nombre de places de stationnement par habitant et par maille"},
        ax=ax,
        linewidth=0.1,
        edgecolor="grey"
    )

    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)

    ax.set_title("Nombre de places de stationnement par habitant et par maille", fontsize=18)
    ax.axis('off')
    plt.tight_layout()

    # (Optionnel) Export PNG
    if export :
        images_export_dir = os.path.join(images_dir, "indicateur_nombre_places_stationnement_vl.png")
        plt.savefig(images_export_dir, dpi=300, bbox_inches='tight')
        print(f"Carte exportée vers : {images_export_dir}")

    plt.show()
    plt.close()

# Exécution
afficher_places_stationnement_vl_par_habitant(export = True)


# #### 6.5.6. Coût du stationnement (€/h)
# ---
# La partie 3.7. affichait déjà les zones de stationnement payant dans l'EMS. Idéalement, on calculerait le coût de stationnement moyen pour les VL par maille, mais les données issues d'OSM (pour les parkings, partie 2.17) et de data.strasbourg (en voirie, partie 3.10) possèdent un attribut 'payant' mais sous la forme d'un booléen. Le prix du stationnement lui-même n'est pas précisé.
# 
# C'est pourquoi cet indicateur ne calcule que les coûts de stationnement issus de cette donnée.

# In[301]:


def calculer_cout_stationnement():
    # 1. Chargement des fichiers
    carreaux = charger_fichier_parquet("maille_200m_avec_donnees", crs=3857)
    zones = charger_fichier_parquet("zones_stationnement_payant", crs=3857)

    # Suppression de la colonne existante si elle est présente
    carreaux = nettoyer_colonnes(carreaux, ['cout_moyen_stationnement'])

    # 2. Nettoyage et parsing du tarif
    def extraire_tarif(tarif_str):
        if not isinstance(tarif_str, str):
            return None
        tarif_str = tarif_str.replace(",", ".").replace("€", "").replace(" ", "")
        match = re.search(r"1h=([\d.]+)", tarif_str)
        if match:
            try:
                return float(match.group(1))
            except:
                return None
        return None

    zones["tarif_horaire"] = zones["tarif"].apply(extraire_tarif)
    zones = zones[zones["tarif_horaire"].notnull()].copy()

    # 3. Intersection spatiale entre zones tarifées et carreaux
    intersections = gpd.overlay(zones[["geometry", "tarif_horaire"]], carreaux[["idINSPIRE", "geometry"]], how="intersection")
    intersections["surface"] = intersections.geometry.area

    # 4. Calcul de la somme pondérée des tarifs par carreau
    ponderee = (
        intersections.groupby("idINSPIRE")
        .apply(lambda df: (df["tarif_horaire"] * df["surface"]).sum() / df["surface"].sum())
        .reset_index(name="cout_moyen_stationnement")
    )

    # 5. Fusion avec les carreaux d'origine
    carreaux = carreaux.merge(ponderee, on="idINSPIRE", how="left")

    # 6. Remplacer les valeurs manquantes par 0 et arrondi
    carreaux["cout_moyen_stationnement"] = carreaux["cout_moyen_stationnement"].fillna(0).round(2)

    # 7. Export
    exporter_parquet(carreaux, "maille_200m_avec_donnees")
    exporter_gpkg(carreaux, "maille_200m_avec_donnees")

# Exécution
calculer_cout_stationnement()


# In[419]:


def afficher_cout_stationnement(export = False):
    # 1. Chargement des données
    carreaux = charger_fichier_parquet("maille_200m_avec_donnees", crs=3857)
    limites_epci = charger_fichier_parquet("limites_epci", crs=3857)

    # 2. Attribution des couleurs selon le tarif
    def get_tarif_couleur(val):
        if val == 0:
            return 'white'  # Pas dans une zone tarifée
        elif val <= 1.0:
            return 'green'  # Tarif bas (≤ 1€)
        elif 1.0 < val <= 2.5:
            return 'orange'  # Tarif moyen (>1€ à 2.5€)
        elif val > 2.5:
            return 'red'  # Tarif élevé (≥ 2.5€)

    carreaux["couleur"] = carreaux["cout_moyen_stationnement"].apply(get_tarif_couleur)

    # 3. Affichage
    fig, ax = plt.subplots(figsize=(10, 10))
    carreaux.plot(
        ax=ax,
        color=carreaux["couleur"],
        edgecolor="lightgrey",
        linewidth=0.1,
        alpha = 0.8
    )

    limites_epci.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1)

    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)

    legend_elements = [
        mpatches.Patch(color='green', label='Tarif bas (≤ 1€/h)'),
        mpatches.Patch(color='orange', label='Tarif moyen (>1€ à 2.5€/h)'),
        mpatches.Patch(color='red', label='Tarif élevé (≥ 2.5€/h)'),
        mpatches.Patch(color='gray', label='Autre tarif'),
        mpatches.Patch(color='white', label='Non tarifé')
    ]

    ax.legend(handles=legend_elements, title="Tarifs de stationnement", loc="upper left")
    ax.set_title("Coût horaire moyen par maille", fontsize=18)
    ax.set_axis_off()
    plt.tight_layout()

    # (Optionnel) Export PNG
    if export :
        images_export_dir = os.path.join(images_dir, "indicateur_cout_stationnement.png")
        plt.savefig(images_export_dir, dpi=300, bbox_inches='tight')
        print(f"Carte exportée vers : {images_export_dir}")

    plt.show()
    plt.close()

# Exécution
afficher_cout_stationnement(export = True)


# #### 6.5.7. Durée moyenne de stationnement
# ---
# Le calcul se base sur les 11 parkings relais de l'EMS, qui mettent à disposition tout leurs entrées et sorties
# 
# Seuls 7 parkings ont pu être joints avec leur géolocalisation sur les 11 disposant de données en temps rééls sur le portail open data de la CTS. Le seul moyen de faire une jointure est sur leur nom, et le formatage diffère entre les fichiers, d'où les multiples combinaisons pour remplacer les _ par des espaces, etc. Des parkings présents dans les données temps réél (exemple : KIBITZENAU) ne sont pas présents dans l'autre fichier : ceux-là ne peuvent pas être joints.
# 
# NOTE : à normaliser

# In[44]:


def calculer_duree_moyenne_stationnement():
    import unidecode

    # 1. Charger les données
    df_temps = pd.read_csv(os.path.join(exports_dir, "temps_stationnement_parking.csv"))
    gdf_park = charger_fichier_parquet("parkings_relais_epci", crs=2154)

    # 2. Normalisation : Supprime les accents et met en minuscule
    def normaliser_nom(nom):
        return unidecode.unidecode(str(nom)).lower().strip()

    df_temps["nom_brut"] = df_temps["nom"]
    df_temps["nom_base"] = df_temps["nom"].apply(normaliser_nom)

    # 3. Générer des variantes de nom pour plus de souplesse
    def generer_variantes(nom_base):
        return list({
            nom_base.replace("_", " "),
            nom_base.replace("_", "'"),
            nom_base.replace("_", " d'"),
            nom_base.replace("_", " de "),
            nom_base.replace("_", "-"),
            nom_base.replace("_", ""),
        })

    df_temps["variantes"] = df_temps["nom_base"].apply(generer_variantes)

    # 4. Préparer les noms du GeoDataFrame, avec toponyme nettoyé
    def nettoyer_toponyme(topo):
        nom = str(topo)
        # Supprime les préfixes comme "Parking Relais" ou "Parking Relais Tram"
        for prefixe in ["Parking Relais Tram", "Parking Relais", "Parking"]:
            if nom.lower().startswith(prefixe.lower()):
                nom = nom[len(prefixe):]
        return normaliser_nom(nom)

    gdf_park["nom_normalise"] = gdf_park["TOPONYME"].apply(nettoyer_toponyme)

    # 5. Construction d'une liste pour la recherche
    liste_noms_gdf = list(zip(gdf_park.index, gdf_park["nom_normalise"]))

    # 6. Tentative de correspondance souple
    matchs = {}
    for idx, row in df_temps.iterrows():
        for variante in row["variantes"]:
            for idx_gdf, nom_gdf in liste_noms_gdf:
                if variante in nom_gdf:
                    matchs[idx] = idx_gdf
                    break
            if idx in matchs:
                break

    # 7. Fusion des correspondances
    rows_associes = []
    for idx_csv, idx_gdf in matchs.items():
        ligne_csv = df_temps.loc[idx_csv]
        ligne_gdf = gdf_park.loc[idx_gdf]

        ligne_combinee = ligne_csv.to_dict()
        ligne_combinee.update({
            "geometry": ligne_gdf.geometry,
            "nom_officiel": ligne_gdf["TOPONYME"],
            "adresse": ligne_gdf.get("address", "")
        })
        rows_associes.append(ligne_combinee)

    # 8. Création du GeoDataFrame
    gdf_resultat = gpd.GeoDataFrame(rows_associes, crs="EPSG:2154")

    # 9. Export
    exporter_parquet(gdf_resultat, "duree_moyenne_stationnement_geolocalise")
    exporter_gpkg(gdf_resultat, "duree_moyenne_stationnement_geolocalise")

    # 10. Affichage
    print(f"{len(gdf_resultat)} parkings associés sur {len(df_temps)}")

# Exécution
calculer_duree_moyenne_stationnement()


# In[411]:


def afficher_duree_moyenne_stationnement(export = False):
    # 1. Chargement des donneés
    limites_epci = charger_fichier_parquet("limites_epci", crs = 3857)
    gdf = charger_fichier_parquet("duree_moyenne_stationnement_geolocalise", crs=3857)

    # 2. Affichage
    fig, ax = plt.subplots(figsize=(10, 10))
    # limites_epci.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1)

    gdf.plot(
        ax=ax,
        column='duree_moyenne_min',
        cmap='viridis',
        legend=True,
        markersize=60,
        edgecolor='black',
        linewidth=0.3
    )

    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
    ax.set_axis_off()
    ax.set_title("Durée moyenne de stationnement par parking (en minutes)", fontsize=18)
    plt.tight_layout()

    # (Optionnel) Export PNG
    if export :
        images_export_dir = os.path.join(images_dir, "indicateur_duree_moyenne_stationnement.png")
        plt.savefig(images_export_dir, dpi=300, bbox_inches='tight')
        print(f"Carte exportée vers : {images_export_dir}")

    plt.show()
    plt.close()

# Exécution
afficher_duree_moyenne_stationnement(export = True)


# In[529]:


# Exporte durée moyenne de stationnement sur tout les parkings sans faire de jointure, donc fait la moyenne sur toutes les données
def calculer_duree_moyenne_stationnement_non_geo():
    # 1. Charger le fichier CSV
    df_temps = pd.read_csv(os.path.join(exports_dir, "temps_stationnement_parking.csv"))

    # 2. Calcul de la durée moyenne globale
    duree_moyenne = df_temps["duree_moyenne_min"].mean()

    # 3. Création d’un DataFrame résultat
    df_resultat = pd.DataFrame([{
        "nom": "duree_moyenne_stationnement",
        "valeur": round(duree_moyenne, 2)
    }])

    # 4. Export
    df_resultat.to_parquet(os.path.join(exports_dir, "duree_moyenne_stationnement.parquet"), index=False)
    print(df_resultat)

# Exécution
calculer_duree_moyenne_stationnement_non_geo()


# #### 6.5.8. Nombre de bornes de recharge électriques
# ---

# In[91]:


def calculer_nombre_bornes_ve():
    # 1. Chargement des données
    carreaux = charger_fichier_parquet("maille_200m_avec_donnees", crs=3857).copy()
    bornes_ve = charger_fichier_parquet("bornes_ve_epci", crs=3857).copy()

    # 2. Supprime la colonne 'index_right' si elle existe
    bornes_ve = nettoyer_colonnes(bornes_ve)

    # 3. Supprime l’ancienne colonne si elle existe pour éviter conflits de fusion
    carreaux = nettoyer_colonnes(carreaux, ['nb_bornes_ve'])

    # 4. Filtrer uniquement les points valides
    bornes_ve = bornes_ve[
        bornes_ve.geometry.notnull() & (bornes_ve.geometry.type == "Point")
    ].copy()

    # 5. Jointure spatiale : associer chaque station à un carreau
    bornes_ve_avec_carreaux = gpd.sjoin(
        bornes_ve,
        carreaux[["idINSPIRE", "geometry"]],
        how="inner",
        predicate="within"
    )

    # 6. Agrégation : nombre de stations par carreau
    nb_bornes_ve_par_carreau = bornes_ve_avec_carreaux.groupby("idINSPIRE").size().reset_index(name="nb_bornes_ve")

    # 7. Fusion avec les carreaux
    carreaux = carreaux.merge(nb_bornes_ve_par_carreau, on="idINSPIRE", how="left")
    carreaux["nb_bornes_ve"] = carreaux["nb_bornes_ve"].fillna(0).astype(int)

    # 8. Export
    exporter_parquet(carreaux, "maille_200m_avec_donnees")
    exporter_gpkg(carreaux, "maille_200m_avec_donnees")

# Exécution
calculer_nombre_bornes_ve()


# In[311]:


def afficher_nombre_bornes_ve(export = False):
    # 1. Chargement des données
    limites_epci = charger_fichier_parquet("limites_epci", crs=3857)
    carreaux = charger_fichier_parquet("maille_200m_avec_donnees", crs=3857)

    # 2. Affichage
    fig, ax = plt.subplots(figsize=(10, 10))
    limites_epci.plot(ax=ax, alpha=1, edgecolor='black', facecolor='none')

    carreaux.plot(column="nb_bornes_ve",
             cmap="YlGn",
             legend=True,
             legend_kwds={'label': "Nombre de bornes de recharge pour véhicules électriques par maille"},
             ax=ax,
             linewidth=0.1,
             edgecolor="grey")

    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)

    ax.set_title("Nombre de bornes de recharge pour véhicules électriques par maille")
    ax.axis('off')
    plt.tight_layout()

    # (Optionnel) Export PNG
    if export :
        images_export_dir = os.path.join(images_dir, "indicateur_nombre_bornes_ve.png")
        plt.savefig(images_export_dir, dpi=300, bbox_inches='tight')
        print(f"Carte exportée vers : {images_export_dir}")

    plt.show()
    plt.close()

# Exécution
afficher_nombre_bornes_ve(export = True)


# In[119]:


# Version normalisée
def calculer_bornes_ve_par_habitant():
    # 1. Chargement des données
    carreaux = charger_fichier_parquet("maille_200m_avec_donnees", crs=3857).copy()

    # 2. Suppression de l'ancienne colonne si elle existe
    carreaux = nettoyer_colonnes(carreaux, ["bornes_ve_par_habitant"])

    # 3. Calcul de l'indicateur
    def calcul_ratio(nb_bornes, pop):
        if pop is None or pop <= 0:
            return -1  # Valeur par défaut si population inconnue ou nulle
        return round(nb_bornes / pop, 4)

    carreaux["bornes_ve_par_habitant"] = carreaux.apply(
        lambda row: calcul_ratio(row["nb_bornes_ve"], row["population_estimee"]), axis=1
    )

    # 4. Export
    exporter_parquet(carreaux, "maille_200m_avec_donnees")
    exporter_gpkg(carreaux, "maille_200m_avec_donnees")

    # 5. Statistiques
    valides = carreaux[carreaux["bornes_ve_par_habitant"] >= 0]
    print(f"""
Indicateur 'bornes_ve_par_habitant' ajouté :
- {len(carreaux)} carreaux traités
- {len(valides)} avec population connue
- Moyenne : {valides['bornes_ve_par_habitant'].mean():.4f} borne(s) VE par habitant
""")

# Exécution
calculer_bornes_ve_par_habitant()


# In[414]:


def afficher_bornes_ve_par_habitant(export = False):
    # 1. Chargement des données
    limites_epci = charger_fichier_parquet("limites_epci", crs=3857)
    carreaux = charger_fichier_parquet("maille_200m_avec_donnees", crs=3857)
    carreaux = carreaux[carreaux["bornes_ve_par_habitant"] != -1]

    # 2. Affichage
    fig, ax = plt.subplots(figsize=(10, 10))
    limites_epci.plot(ax=ax, alpha=1, edgecolor='black', facecolor='none')

    carreaux.plot(column="bornes_ve_par_habitant",
             cmap="YlGn",
             legend=True,
             #legend_kwds={'label': "Nombre de bornes de recharge pour véhicules électriques par maille et par habitant"},
             ax=ax,
             linewidth=0.1,
             edgecolor="grey")

    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)

    ax.set_title("Nombre de bornes de recharge pour \nvéhicules électriques par habitant", fontsize=18)
    ax.axis('off')
    plt.tight_layout()

    # (Optionnel) Export PNG
    if export :
        images_export_dir = os.path.join(images_dir, "indicateur_normalise_nombre_bornes_ve.png")
        plt.savefig(images_export_dir, dpi=300, bbox_inches='tight')
        print(f"Carte exportée vers : {images_export_dir}")

    plt.show()
    plt.close()

# Exécution
afficher_bornes_ve_par_habitant(export = True)


# In[247]:


def afficher_bornes_ve_par_habitant(export=False):
    # 1. Chargement des données
    limites_epci = charger_fichier_parquet("limites_epci", crs=3857)
    carreaux = charger_fichier_parquet("maille_200m_avec_donnees", crs=3857)
    carreaux = carreaux[carreaux["bornes_ve_par_habitant"] != -1].copy()

    # 2. Création des catégories et couleurs
    classes = [-0.1, 0.1, 1.1, 5.1, float('inf')]
    etiquettes= ['Aucune', '<1', '1-5', '5+']
    couleurs = ['#ffffcc', '#b6f267', '#41ab5d', '#006837']

    # 3. Création de la colonne de catégories
    carreaux['categorie'] = pd.cut(
        carreaux['bornes_ve_par_habitant'],
        bins=classes,
        labels=etiquettes,
        right=False
    )

    # 4. Création de la carte
    fig, ax = plt.subplots(figsize=(12, 10))

    # Contour EPCI
    limites_epci.plot(ax=ax, alpha=0.5, edgecolor='black', facecolor='none', linewidth=0.5)

    # Tracé des carreaux avec couleurs discrètes
    for i, (etiquette, couleur) in enumerate(zip(etiquettes, couleurs)):
        subset = carreaux[carreaux['categorie'] == etiquette]
        subset.plot(
            ax=ax,
            color=couleur,
            edgecolor='lightgrey',
            linewidth=0.2,
            label=etiquette
        )

    # Ajout du fond de carte
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)

    # 5. Personnalisation de la légende
    patches = [
        mpatches.Patch(color=couleur, label=etiquette)
        for couleur, etiquette in zip(couleurs, etiquettes)
    ]

    ax.legend(
        handles=patches,
        title="Bornes VE/habitant",
        loc='upper right',
        frameon=True,
        facecolor='white',
        framealpha=0.8
    )

    # 6. Titre et mise en page
    ax.set_title("Nombre de bornes de recharge pour véhicules électriques par maille et par habitant", fontsize=18)
    ax.axis('off')
    plt.tight_layout()

    # 7. Export
    if export:
        images_export_dir = os.path.join(images_dir, "indicateur_nombre_bornes_ve.png")
        plt.savefig(
            images_export_dir,
            dpi=300,
            bbox_inches='tight',
            facecolor='white',
            pad_inches=0.1
        )
        print(f"Carte exportée vers : {images_export_dir}")

    plt.show()
    plt.close()

# Exécution
afficher_bornes_ve_par_habitant(export=True)


# #### 6.5.9. Part des emplois accessibles en moins de T minutes
# ---
# 
# NOTE : nom des colonnes à changer de '_ratio_emplois' à '_part_emplois'

# In[ ]:


calculer_ratio_emplois_services_proches(nom_graphe="vl", secondes=900, n_jobs=-1)


# In[ ]:


afficher_ratio_emplois_proches(nom_graphe = "vl", export = True)


# #### 6.5.10. Part des services accessibles en moins de T minutes. Moyenne sur S types et/ou catégories de services.
# 
# ---

# In[ ]:


afficher_ratio_services_proches(nom_graphe = "vl", export = True)


# #### 6.5.11. Nombre d'accidents liés aux véhicules légers (2023)
# ---

# In[331]:


calculer_nb_accidents(transport="vl")


# In[305]:


afficher_nb_accidents(transport="vl", export = True)


# ### 6.6. Véhicules en autopartage
# ---
# Indicateur de qualité de service des véhicules en autopartage à calculer à partir des indicateurs suivants :
# * 6.6.1. Nombre de stations d'autopartage de véhicules - 'part_normalise_couverte_autopartage'
# * 6.6.2. Nombre de voitures en autopartage - 'nb_voitures_autopartage' (ne pas utiliser pour l'instant)
# * 6.6.3. Coût de l'autopartage (€/h) - 'autopartage_cout_rapport_revenu'
# * 6.6.4. Part des routes accessibles aux véhicules en autopartage 'part_routes_accessibles_vl'
# * 6.6.5. Part de surface couverte par l'autopartage 'part_surface_geofencing_citiz'
# * Sous-indicateur de mobilité :
#     * 6.6.6. Part des emplois accessibles en moins de T minutes - 'autopartage_ratio_emplois'
#     * 6.6.7. Part des services accessibles en moins de T minutes. Moyenne sur S types et/ou catégories de services - 'autopartage_ratio_services_dom_A' à 'autopartage_ratio_services_dom_F'

# #### 6.6.1. Nombre de stations d'autopartage de véhicules
# ---

# In[57]:


def calculer_nombre_stations_autopartage():
    # 1. Chargement des données
    carreaux = charger_fichier_parquet("maille_200m_avec_donnees", crs=3857).copy()
    citiz_stations = charger_fichier_parquet("citiz_stations", crs=3857).copy()

    # 2. Supprime la colonne 'index_right' si elle existe
    citiz_stations = nettoyer_colonnes(citiz_stations)

    # 3. Supprime l’ancienne colonne si elle existe pour éviter conflits de fusion
    carreaux = nettoyer_colonnes(carreaux, ['nb_stations_autopartage'])

    # 4. Filtrer uniquement les points valides
    citiz_stations = citiz_stations[
        citiz_stations.geometry.notnull() & (citiz_stations.geometry.type == "Point")
    ].copy()

    # 5. Jointure spatiale : associer chaque station à un carreau
    stations_avec_carreaux = gpd.sjoin(
        citiz_stations, 
        carreaux[["idINSPIRE", "geometry"]], 
        how="inner", 
        predicate="within"
    )

    # 6. Agrégation : nombre de stations par carreau
    nb_stations_par_carreau = stations_avec_carreaux.groupby("idINSPIRE").size().reset_index(name="nb_stations_autopartage")

    # 7. Fusion avec les carreaux
    carreaux = carreaux.merge(nb_stations_par_carreau, on="idINSPIRE", how="left")
    carreaux["nb_stations_autopartage"] = carreaux["nb_stations_autopartage"].fillna(0).astype(int)

    # 8. Export
    exporter_parquet(carreaux, "maille_200m_avec_donnees")
    exporter_gpkg(carreaux, "maille_200m_avec_donnees")
    print("Colonne 'nb_stations_autopartage' ajoutée avec succès.")

# Exécution
calculer_nombre_stations_autopartage()


# In[415]:


def afficher_nombre_stations_autopartage(export = False):
    # 1. Chargement des données
    limites_epci = charger_fichier_parquet("limites_epci", crs=3857)
    carreaux = charger_fichier_parquet("maille_200m_avec_donnees", crs=3857)

    # 2. Affichage
    fig, ax = plt.subplots(figsize=(10, 10))
    limites_epci.plot(ax=ax, alpha=1, edgecolor='black', facecolor='none')

    carreaux.plot(column="nb_stations_autopartage",
             cmap="YlGn",
             legend=True,
             #legend_kwds={'label': "Nombre de stations d'autopartage Citiz par maille"},
             ax=ax,
             linewidth=0.1,
             edgecolor="grey")

    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)

    ax.set_title("Nombre de stations d'autopartage Citiz par maille", fontsize = 18)
    ax.axis('off')
    plt.tight_layout()

    # (Optionnel) Export PNG
    if export :
        images_export_dir = os.path.join(images_dir, "nombre_stations_autopartage.png")
        plt.savefig(images_export_dir, dpi=300, bbox_inches='tight')
        print(f"Carte exportée vers : {images_export_dir}")

    plt.show()
    plt.close()

# Exécution
afficher_nombre_stations_autopartage(export = True)


# In[131]:


# Version normaliée, buffer de 400 m
def calculer_part_surface_couverte_autopartage():
    # 1. Chargement des données
    carreaux = charger_fichier_parquet("maille_200m_avec_donnees", crs=2154).copy()
    stations_autopartage = charger_fichier_parquet("citiz_stations", crs=2154)

    # 2. Nettoyage des colonnes existantes
    carreaux = nettoyer_colonnes(carreaux, ["part_surface_couverte_autopartage", "part_normalise_couverte_autopartage"])

    # 3. Nettoyage géométrie des stations
    stations_autopartage = stations_autopartage[
        stations_autopartage.geometry.notnull() & (stations_autopartage.geometry.type == "Point")
    ].copy()

    # 4. Création des buffers 400 m
    stations_autopartage["buffer_400m"] = stations_autopartage.geometry.buffer(400)

    # 5. Union des buffers
    zone_couverte = stations_autopartage["buffer_400m"].geometry.union_all()

    # 6. Intersection avec les carreaux
    intersections = carreaux.geometry.intersection(zone_couverte)

    # 7. Calcul de la part de surface couverte (max 100%)
    surfaces_totales = carreaux.geometry.area
    surfaces_intersectees = intersections.area
    part_surface = (surfaces_intersectees / surfaces_totales).clip(upper=1.0) * 100

    # 8. Ajout à la table
    carreaux["part_normalise_couverte_autopartage"] = part_surface.round(2).fillna(0)

    # 9. Export
    exporter_parquet(carreaux, "maille_200m_avec_donnees")
    exporter_gpkg(carreaux, "maille_200m_avec_donnees")

    # 10. Résumé
    print("Colonne 'part_normalise_couverte_autopartage' ajoutée avec succès.")
    print(f"Valeur moyenne : {carreaux['part_normalise_couverte_autopartage'].mean():.2f}%")

# Exécution
calculer_part_surface_couverte_autopartage()


# In[417]:


def afficher_part_surface_couverte_autopartage(export = False):
    # 1. Chargement des données
    limites_epci = charger_fichier_parquet("limites_epci", crs=3857)
    carreaux = charger_fichier_parquet("maille_200m_avec_donnees", crs=3857)

    # 2. Affichage
    fig, ax = plt.subplots(figsize=(10, 10))
    limites_epci.plot(ax=ax, alpha=1, edgecolor='black', facecolor='none')

    carreaux.plot(column="part_normalise_couverte_autopartage",
             cmap="YlGn",
             legend=True,
             legend_kwds={'label': "% de surface des carreaux présent à moins de 400 mètres d'une station d'autopartage"},
             ax=ax,
             linewidth=0.1,
             edgecolor="grey")

    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)

    ax.set_title("Part de surface présente à moins de 400 mètres \nd'une station d'autopartage", fontsize=18)
    ax.axis('off')
    plt.tight_layout()

    # (Optionnel) Export PNG
    if export :
        images_export_dir = os.path.join(images_dir, "indicateur_part_normalise_couverte_autopartage.png")
        plt.savefig(images_export_dir, dpi=300, bbox_inches='tight')
        print(f"Carte exportée vers : {images_export_dir}")

    plt.show()
    plt.close()

# Exécution
afficher_part_surface_couverte_autopartage(export = True)


# #### 6.6.2. Nombre de voitures en autopartage
# ---
# Plutôt que de se baser sur le nombre de véhicules en temps réel (qui peut varier), le nombre de voitures est déduit d'après le nombre de places de parkings par stations (données plus stables)
# 
# NOTE : ne pas utiliser cet indicateur pour le calcul de l'indicateur composé de l'autopartage ? Donnée trop peu spatialisée

# In[309]:


def calculer_nombre_voitures_autopartage():
    # 1. Chargement des données
    carreaux = charger_fichier_parquet("maille_200m_avec_donnees", crs=3857).copy()
    citiz_stations = charger_fichier_parquet("citiz_stations", crs=3857).copy()

    # 2. Nettoyage des colonnes déjà calculées et de jointure (si elles existent)
    citiz_stations = nettoyer_colonnes(citiz_stations)
    carreaux = nettoyer_colonnes(carreaux, ['nb_voitures_autopartage'])

    # 3. Filtrer uniquement les géométries valides
    citiz_stations = citiz_stations[
        citiz_stations.geometry.notnull() & (citiz_stations.geometry.type == "Point")
    ].copy()

    # 4. Jointure spatiale : associer chaque station à un carreau
    stations_avec_carreaux = gpd.sjoin(
        citiz_stations,
        carreaux[["idINSPIRE", "geometry"]],
        how="inner",
        predicate="within"
    )

    # 5. Agrégation : somme des capacités par carreau
    capacite_par_carreau = stations_avec_carreaux.groupby("idINSPIRE")["capacity"].sum().reset_index()
    capacite_par_carreau.rename(columns={"capacity": "nb_voitures_autopartage"}, inplace=True)

    # 6. Fusion avec les carreaux
    carreaux = carreaux.merge(capacite_par_carreau, on="idINSPIRE", how="left")
    carreaux["nb_voitures_autopartage"] = carreaux["nb_voitures_autopartage"].fillna(0).astype(int)

    # 7. Export
    exporter_parquet(carreaux, "maille_200m_avec_donnees")
    exporter_gpkg(carreaux, "maille_200m_avec_donnees")
    print("Colonne 'nb_voitures_autopartage' ajoutée avec succès.")

# Exécution
calculer_nombre_voitures_autopartage()


# In[310]:


def afficher_nombre_voitures_autopartage(export = False):
    # 1. Chargement des données
    limites_epci = charger_fichier_parquet("limites_epci", crs=3857)
    carreaux = charger_fichier_parquet("maille_200m_avec_donnees", crs=3857)

    # 2. Affichage
    fig, ax = plt.subplots(figsize=(10, 10))
    limites_epci.plot(ax=ax, alpha=1, edgecolor='black', facecolor='none')

    carreaux.plot(column="nb_voitures_autopartage",
             cmap="YlGn",
             legend=True,
             legend_kwds={'label': "Nombre de voitures en autopartage par maille"},
             ax=ax,
             linewidth=0.1,
             edgecolor="grey")

    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)

    ax.set_title("Nombre de voitures en autopartage par maille", fontsize = 18)
    ax.axis('off')
    plt.tight_layout()

    # (Optionnel) Export PNG
    if export :
        images_export_dir = os.path.join(images_dir, "nombre_voitures_autopartage.png")
        plt.savefig(images_export_dir, dpi=300, bbox_inches='tight')
        print(f"Carte exportée vers : {images_export_dir}")

    plt.show()
    plt.close()

# Exécution
afficher_nombre_voitures_autopartage(export = True)


# #### 6.6.3. Coût de l'autopartage (€/h)
# ---
# Dans l'EMS, Citiz gère les véhicules en autopartage. Les véhicules peuvent circuler librement pendant le laps où ils sont loués (plusieurs heures / jours).
# 
# Dans ces conditions, le coût de l'autopartage est calculé par heure et est considéré uniforme dans l'ensemble de l'EMS. On récupère le tarif horaire pour une location sans abonnement de la plus petite catégorie de véhicules (type S). L'assurance, le carburant et l'entretient sont inclus.
# 
# Il existe 2 types de locations : 
# 1. Celles où le dépôt des véhicules doit se faire dans la station de départ.
# 2. Dans l'EMS, les voitures peuvent être partagées en dépose-libre. Elles peuvent être prises et déposées au sein des périmètres dédiés, définis dans les zones de geofencing. Leur tarif est celui des catégories S facturés au quart-d'heure, auquel se rajoute 1,5€ de prise en charge (non pris en compte ici)
# 
# Il y a 2 versions du script :
# 1. Le premier applique le coût moyen de l'autopartage aux mailles possédant au moins une station Citiz
# 2. Le second exporte le coût moyen valable pour tout l'EMS
# 
# Documentation : 
# * https://citiz.coop/foire-aux-questions
# * https://grand-est.citiz.coop/particuliers/tarifs

# In[316]:


def calculer_cout_autopartage():
    # 1. Charger les données
    carreaux = charger_fichier_parquet("maille_200m_avec_donnees", crs=3857).copy()
    citiz_stations = charger_fichier_parquet("citiz_stations", crs=3857).copy()

    # Nettoyage des colonnes parasites
    citiz_stations = nettoyer_colonnes(citiz_stations)
    carreaux = nettoyer_colonnes(carreaux, ["cout_moyen_autopartage"])

    # 2. Téléchargement et extraction des données tarifaires GBFS
    nom_source = "citiz_gbfs"
    url = trouver_source_url(nom_source)
    json_path = os.path.join(dir, f"{nom_source}.json")
    telecharger_fichier(url, json_path)

    with open(json_path, "r", encoding="utf-8") as f:
        gbfs_root = json.load(f)
    flux = gbfs_root["data"]["feeds"]
    flux_urls = {feed["name"]: feed["url"] for feed in flux}
    pricing_url = flux_urls.get("system_pricing_plans")

    pricing_data = requests.get(pricing_url).json()
    plans = pricing_data["data"]["plans"]
    plan_s = next(
        (p for p in plans if any(n.get("text", "") == "Sans Abonnement - S" for n in p["name"])),
        None
    )
    if not plan_s:
        raise ValueError("Plan 'Sans Abonnement - S' introuvable.")

    tarifs_15min = [e for e in plan_s.get("per_min_pricing", []) if e.get("interval") == 15]
    couts_h = [e["rate"] * 4 for e in tarifs_15min]
    cout_horaire_moyen = round(sum(couts_h) / len(couts_h), 2) if couts_h else 0.0

    # 3. Jointure spatiale : identifier les carreaux contenant une station
    citiz_stations = citiz_stations[
        citiz_stations.geometry.notnull() & (citiz_stations.geometry.type == "Point")
    ].copy()

    stations_avec_carreaux = gpd.sjoin(
        citiz_stations,
        carreaux[["idINSPIRE", "geometry"]],
        how="inner",
        predicate="within"
    )

    # 4. Identifier les ID des carreaux contenant au moins une station
    ids_carreaux_avec_station = stations_avec_carreaux["idINSPIRE"].unique()

    # 5. Appliquer le tarif uniquement à ces carreaux
    carreaux["cout_moyen_autopartage"] = 0.0
    carreaux.loc[carreaux["idINSPIRE"].isin(ids_carreaux_avec_station), "cout_moyen_autopartage"] = cout_horaire_moyen

    # 6. Export
    exporter_parquet(carreaux, "maille_200m_avec_donnees")
    exporter_gpkg(carreaux, "maille_200m_avec_donnees")
    print(f"Tarif horaire Citiz ({cout_horaire_moyen} €/h) appliqué à {len(ids_carreaux_avec_station)} carreaux.")

# Exécution
calculer_cout_autopartage()


# In[317]:


def afficher_cout_autopartage(export = False):
    # 1. Chargement des données
    limites_epci = charger_fichier_parquet("limites_epci", crs=3857)
    carreaux = charger_fichier_parquet("maille_200m_avec_donnees", crs=3857)

    # 2. Affichage
    fig, ax = plt.subplots(figsize=(10, 10))
    limites_epci.plot(ax=ax, alpha=1, edgecolor='black', facecolor='none')

    carreaux.plot(column="cout_moyen_autopartage",
             cmap="YlGn",
             legend=True,
             legend_kwds={'label': "Coût de location pour les véhicules en autopartage (Citiz)"},
             ax=ax,
             linewidth=0.1,
             edgecolor="grey")

    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)

    ax.set_title("Coût de location pour les véhicules en autopartage (Citiz)")
    ax.axis('off')
    plt.tight_layout()

    # (Optionnel) Export PNG
    if export :
        images_export_dir = os.path.join(images_dir, "indicateur_cout_autopartage.png")
        plt.savefig(images_export_dir, dpi=300, bbox_inches='tight')
        print(f"Carte exportée vers : {images_export_dir}")

    plt.show()
    plt.close()

# Exécution
afficher_cout_autopartage(export = True)


# In[537]:


# Version exportant le coût unique d'1h de location sans faire de jointures
def calculer_cout_autopartage_moyen_non_geo():
    # 1. Télécharger le JSON GBFS principal
    nom_source = "citiz_gbfs"
    url = trouver_source_url(nom_source)
    json_path = os.path.join(dir, f"{nom_source}.json")
    telecharger_fichier(url, json_path)

    # 2. Lire le JSON et extraire l'URL du plan tarifaire
    with open(json_path, "r", encoding="utf-8") as f:
        gbfs_root = json.load(f)
    flux = gbfs_root["data"]["feeds"]
    flux_urls = {feed["name"]: feed["url"] for feed in flux}
    pricing_url = flux_urls.get("system_pricing_plans")

    # 3. Télécharger les données tarifaires
    pricing_data = requests.get(pricing_url).json()
    plans = pricing_data["data"]["plans"]

    # 4. Sélection du plan "Sans Abonnement - S"
    plan_s = next(
        (p for p in plans if any(n.get("text", "") == "Sans Abonnement - S" for n in p["name"])),
        None
    )
    if not plan_s:
        raise ValueError("Plan 'Sans Abonnement - S' introuvable.")

    # 5. Moyenne du coût horaire à partir des tarifs par 15 minutes
    tarifs_15min = [
        e for e in plan_s.get("per_min_pricing", []) if e.get("interval") == 15
    ]
    couts_h = [e["rate"] * 4 for e in tarifs_15min]
    cout_moyen_horaire = round(sum(couts_h) / len(couts_h), 2) if couts_h else 0.0

    # 6. Export sous forme de DataFrame
    df_resultat = pd.DataFrame([{
        "nom": "cout_moyen_autopartage",
        "valeur": cout_moyen_horaire
    }])
    df_resultat.to_parquet(os.path.join(exports_dir, "cout_moyen_autopartage.parquet"), index=False)
    print(df_resultat)

# Exécution
calculer_cout_autopartage_moyen_non_geo()


# In[116]:


# Version normalisée
def calculer_cout_autopartage_par_revenu():
    # 1. Chargement des données
    carreaux = charger_fichier_parquet("maille_200m_avec_donnees", crs=2154).copy()
    df_cout = pd.read_parquet(os.path.join(exports_dir, "cout_moyen_autopartage.parquet"))
    cout_horaire = df_cout[df_cout["nom"] == "cout_moyen_autopartage"]["valeur"].iloc[0]

    # 2. Suppression de la colonne existante si besoin
    carreaux = nettoyer_colonnes(carreaux, ["autopartage_cout_rapport_revenu"])

    # 3. Calcul du ratio coût horaire / revenu disponible médian
    def calcul_ratio(cout, revenu):
        if pd.isna(revenu) or revenu <= 0:
            return -1
        return round(cout / revenu, 4)

    carreaux["autopartage_cout_rapport_revenu"] = carreaux["DISP_MED21"].apply(
        lambda x: calcul_ratio(cout_horaire, x)
    )

    # 4. Export
    exporter_parquet(carreaux, "maille_200m_avec_donnees")
    exporter_gpkg(carreaux, "maille_200m_avec_donnees")

    # 5. Statistiques
    valides = carreaux[carreaux["autopartage_cout_rapport_revenu"] >= 0]
    print(f"""
Indicateur 'autopartage_cout_rapport_revenu' ajouté :
- Carreaux traités : {len(carreaux)}
- Carreaux valides : {len(valides)}
- Moyenne : {valides['autopartage_cout_rapport_revenu'].mean():.4f}
- Coût horaire pris en compte : {cout_horaire:.2f} €
""")

# Exécution
calculer_cout_autopartage_par_revenu()


# In[422]:


def afficher_cout_autopartage_par_revenu(export = False):
    # 1. Chargement des données
    limites_epci = charger_fichier_parquet("limites_epci", crs=3857)
    carreaux = charger_fichier_parquet("maille_200m_avec_donnees", crs=3857)
    carreaux = carreaux[carreaux["autopartage_cout_rapport_revenu"] != -1]

    # 2. Affichage
    fig, ax = plt.subplots(figsize=(10, 10))
    limites_epci.plot(ax=ax, alpha=1, edgecolor='black', facecolor='none')

    carreaux.plot(column="autopartage_cout_rapport_revenu",
             cmap="YlOrRd",
             legend=True,
             #legend_kwds={'label': "Cout horaire d'utilisation d'un véhicule en autopartage rapporté au revenu médian de la population (2021)"},
             ax=ax,
             linewidth=0.1,
             edgecolor="grey")

    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)

    ax.set_title("Cout horaire d'utilisation d'un véhicule en autopartage \nrapporté au revenu médian de la population (2021)", fontsize=18)
    ax.axis('off')
    plt.tight_layout()

    # (Optionnel) Export PNG
    if export :
        images_export_dir = os.path.join(images_dir, "indicateur_cout_autopartage.png")
        plt.savefig(images_export_dir, dpi=300, bbox_inches='tight')
        print(f"Carte exportée vers : {images_export_dir}")

    plt.show()
    plt.close()

# Exécution
afficher_cout_autopartage_par_revenu(export = True)


# #### 6.6.4. Part des routes accessibles aux véhicules en autopartage
# ---
# Cet indicateur est l'inverse de celui calculé en 6.1.5.

# In[82]:


def calculer_part_routes_accessibles_autopartage():
    # 1. Chargement des données
    carreaux = charger_fichier_parquet("maille_200m_avec_donnees", crs=2154)
    routes = charger_fichier_parquet("bd_topo_routes_epci", crs=2154)

    # 2. Filtrage des routes en service et non privées
    routes = routes[
        (routes.geometry.notnull()) &
        (routes["ETAT"] == "En service") &
        (routes["PRIVE"].str.lower() == "non")
    ].copy()

    # 3. Filtrage géométrique
    routes = routes[routes.geometry.type.isin(["LineString", "MultiLineString"])]

    # 4. Longueur totale des tronçons
    routes["longueur"] = routes.geometry.length

    # 5. Détermination des tronçons accessibles aux VL
    routes["accessible_vl"] = routes["ACCES_VL"].str.lower().isin(["libre", "a péage"])

    # 6. Intersection avec les carreaux
    routes_par_carreau = gpd.overlay(routes, carreaux[["idINSPIRE", "geometry"]], how="intersection")
    routes_par_carreau["longueur"] = routes_par_carreau.geometry.length
    routes_par_carreau["accessible_vl"] = routes_par_carreau["ACCES_VL"].str.lower().isin(["libre", "a péage"])

    # 7. Agrégation par carreau
    stats = routes_par_carreau.groupby("idINSPIRE").agg(
        longueur_totale=("longueur", "sum"),
        longueur_accessible=("longueur", lambda x: x[routes_par_carreau.loc[x.index, "accessible_vl"]].sum())
    ).reset_index()

    # 8. Fusion dans les carreaux
    carreaux = carreaux.merge(stats, on="idINSPIRE", how="left")

    # 9. Calcul de la part accessible (en %)
    carreaux["part_routes_accessibles_vl"] = (
        (carreaux["longueur_accessible"] / carreaux["longueur_totale"]) * 100
    ).round(2)
    carreaux["part_routes_accessibles_vl"] = carreaux["part_routes_accessibles_vl"].fillna(-1).clip(lower=-1, upper=100)

    # 10. Nettoyage
    carreaux.drop(columns=["longueur_totale", "longueur_accessible"], inplace=True)

    # 11. Export
    exporter_parquet(carreaux, "maille_200m_avec_donnees")
    exporter_gpkg(carreaux, "maille_200m_avec_donnees")

    # 12. Statistiques
    valides = carreaux[carreaux["part_routes_accessibles_vl"] >= 0]
    print(f"""
    Part des rues accessibles aux véhicules légers :
    - Carreaux avec données : {len(valides)}/{len(carreaux)}
    - Moyenne : {valides["part_routes_accessibles_vl"].mean():.2f}%
    - Médiane : {valides["part_routes_accessibles_vl"].median():.2f}%
    - Max : {valides["part_routes_accessibles_vl"].max():.2f}%
    """)

# Exécution
calculer_part_rues_accessibles_autopartage()


# In[420]:


def afficher_part_rues_accessibles_autopartage(export = False):
    limites_epci = charger_fichier_parquet("limites_epci", crs=3857)
    carreaux = charger_fichier_parquet("maille_200m_avec_donnees", crs=3857)
    carreaux = carreaux[carreaux["part_routes_inaccessibles_vl"] != -1]

    # Affichage
    fig, ax = plt.subplots(figsize=(10, 10))
    limites_epci.plot(ax=ax, alpha=1, edgecolor='black', facecolor='none')

    carreaux.plot(column="part_routes_inaccessibles_vl",
             cmap="YlGn",
             legend=True,
             #legend_kwds={'label': "% de routes inaccessibles aux voitures"},
             ax=ax,
             linewidth=0.1,
             edgecolor="grey")

    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)

    ax.set_title("Part des routes accessibles en voiture", fontsize=18)
    ax.axis('off')
    plt.tight_layout()

    # (Optionnel) Export PNG
    if export :
        images_export_dir = os.path.join(images_dir, "indicateur_part_rues_accessibles_autopartage.png")
        plt.savefig(images_export_dir, dpi=300, bbox_inches='tight')
        print(f"Carte exportée vers : {images_export_dir}")

    plt.show()
    plt.close()

# Exécution
afficher_part_rues_accessibles_autopartage(export = True)


# #### 6.4.5. Part de surface couverte par l'autopartage
# ---
# On calcule ici la part de surface du maillage couverte par la surface où la dépose libre des véhicules est autorisée

# In[86]:


def calculer_part_surface_geofencing_citiz():
    # 1. Chargement des données
    carreaux = charger_fichier_parquet("maille_200m_avec_donnees", crs=2154)
    geofencing = charger_fichier_parquet("citiz_geofencing", crs=2154)

    # Nettoyage éventuel
    carreaux = nettoyer_colonnes(carreaux, ["part_surface_geofencing_citiz"])

    # 2. Calcul de surface totale pour chaque carreau
    carreaux["surface_totale_m2"] = carreaux.geometry.area

    # 3. Intersection avec les zones de géofencing
    print("Intersection entre zones geofencing Citiz et carreaux...")
    intersections = gpd.overlay(
        geofencing[["geometry"]],
        carreaux[["idINSPIRE", "geometry"]],
        how="intersection"
    )

    # 4. Calcul des surfaces de recouvrement
    intersections["surface_geofencing_m2"] = intersections.geometry.area

    # 5. Agrégation des surfaces par carreau
    surfaces = (
        intersections
        .groupby("idINSPIRE")["surface_geofencing_m2"]
        .sum()
        .reset_index()
    )

    # 6. Fusion et calcul de la part (%) de surface couverte
    carreaux = carreaux.merge(surfaces, on="idINSPIRE", how="left")
    carreaux["surface_geofencing_m2"] = carreaux["surface_geofencing_m2"].fillna(0)

    carreaux["part_surface_geofencing_citiz"] = (
        (carreaux["surface_geofencing_m2"] / carreaux["surface_totale_m2"]) * 100
    ).round(2).clip(upper=100)

    # 7. Export
    exporter_parquet(carreaux, "maille_200m_avec_donnees")
    exporter_gpkg(carreaux, "maille_200m_avec_donnees")

    print("Part de surface couverte par les zones de dépôt libre Citiz ajoutée avec succès.")

# Exécution
calculer_part_surface_geofencing_citiz()


# In[421]:


def afficher_part_surface_geofencing_citiz(export = False):
    limites_epci = charger_fichier_parquet("limites_epci", crs=3857)
    carreaux = charger_fichier_parquet("maille_200m_avec_donnees", crs=3857)

    # Affichage
    fig, ax = plt.subplots(figsize=(10, 10))
    limites_epci.plot(ax=ax, alpha=1, edgecolor='black', facecolor='none')

    carreaux.plot(column="part_surface_geofencing_citiz",
             cmap="YlGn",
             legend=True,
             #legend_kwds={'label': "Part de surface couverte par des zones de dépose-libre Citiz"},
             ax=ax,
             linewidth=0.1,
             edgecolor="grey")

    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)

    ax.set_title("Part de surface couverte par des zones de dépose-libre Citiz", fontsize=18)
    ax.axis('off')
    plt.tight_layout()

    # (Optionnel) Export PNG
    if export :
        images_export_dir = os.path.join(images_dir, "indicateur_part_surface_geofencing_citiz.png")
        plt.savefig(images_export_dir, dpi=300, bbox_inches='tight')
        print(f"Carte exportée vers : {images_export_dir}")

    plt.show()
    plt.close()

# Exécution
afficher_part_surface_geofencing_citiz(export = True)


# #### 6.6.6. Part des emplois accessibles en moins de T minutes
# ---
# Utilise le même graphe que pour les véhicules légers, mais en partant des carreaux possédant des places réservées à l'autopartage. Techniquement, les véhicules peuvent être loués plusieurs jours (donc partir / arriver depuis n'importe quel lieu), mais on considère ici des déplacements locaux en partant des places réservés aux voitures Citiz.

# In[ ]:


calculer_ratio_emplois_services_proches(nom_graphe="autopartage", secondes=900, n_jobs=12)


# In[218]:


afficher_ratio_emplois_proches(nom_graphe = "autopartage", export = True)


# #### 6.6.7. Part des services accessibles en moins de T minutes. Moyenne sur S types et/ou catégories de services.
# ---

# In[220]:


calculer_moyenne_ratio_services_proches(transport="autopartage")


# In[221]:


afficher_ratio_services_proches(nom_graphe = "autopartage", export = True)


# ### 6.7. Indicateurs non-calculables par transport
# ---
# Indicateur de diversité fonctionnelle à calculer à partir des indicateurs suivants :
# * 6.7.1. HSMI (Housing - Service Mix Index) / Indice de mixité logement - services
# * 6.7.2. HEMI (Housing - Employment Mix Index) / Indice de mixité logement - emplois

# #### 6.7.1. HSMI (Housing - Service Mix Index) / Indice de mixité logement - services
# ---
# Cet indicateur est calculé via les étapes suivantes :
# 1. La proportion totale de la population habitant dans chaque carreau (données issues de Filosofi de l'INSEE): proportion_pop_carreau
# 2. La proportion totale de chaque type service dans chaque carreau (données issues de BPE de l'INSEE) : proportion_type_service_carreau
# 3. Pour chaque type de service, on calcule les différences absolues de ces proportions puis on les divise par leur somme (calcul fait uniquement si |proportion_pop_carreau - proportion_type_service_carreau| > 0):
# HSMI par carreau et par service = 1 - (|proportion_pop_carreau - proportion_type_service_carreau|) / (|proportion_pop_carreau - proportion_type_service_carreau|)
# 4. Le HSMI global du carreau fait la moyenne pour tout les HSMI par type de service 
# 
# Types de services listés dans la BPE :
# - A : Services pour les particuliers
# - B : Commerces
# - C : Enseignement
# - D : Santé et action sociale
# - E : Transports et déplacements
# - F : Sports, loisirs et culture
# - G : Tourisme. On ne conserve pas les données de ce dernier champ, car on considère qu'il ne s'agit pas d'un service à la population locale.
# 
# Le HSMI peut aller de 0 à 1. On suppose que plus le HSMI est élevé, plus les mobilités locales sont encourages, car une grande partie de la population a accès à une grande partie des services. Pour référence, les 10 carreaux avec les HSMI les plus elevés de l'EMS se trouvent tous dans l'hypercentre, 9 d'entre eux entre la place Kléber et la gare centrale.
# 
# Les carreaux sans population (estimée) n'ont pas de HSMI, même si ils possèdent des services.

# In[58]:


def calculer_hsmi():
    # 1. Charger les données nécessaires
    carreaux = charger_fichier_parquet("maille_200m_avec_donnees", crs=2154)
    bpe = charger_fichier_parquet("insee_bpe_2023_epci", crs=2154)

    # Supression des colonnes de jointures et de celles déjà calculées (si présentes) 
    colonnes_a_supprimer = ['HMSI_A', 'HMSI_B', 'HMSI_C', 'HMSI_D', 'HMSI_E', 'HMSI_F',
                           'HMSI_global', 'HSMI_A', 'HSMI_B', 'HSMI_C', 'HSMI_D', 'HSMI_E', 'HSMI_F',
                           'HSMI_global']
    carreaux = nettoyer_colonnes(carreaux, colonnes_a_supprimer)

    # 2. Préparation des données BPE - Extraction du type de service
    bpe['TYPE_SERVICE'] = bpe['TYPEQU'].str[0]  # Extrait A, B, C, etc.

    # 3. Jointure spatiale pour compter les services par carreau
    bpe_avec_carreaux = gpd.sjoin(
        bpe,
        carreaux[['idINSPIRE', 'geometry']],
        how='left',
        predicate='within'
    )

    # 4. Comptage des services par type et par carreau
    services_par_carreau = bpe_avec_carreaux.groupby(
        ['idINSPIRE', 'TYPE_SERVICE']
    ).size().unstack(fill_value=0)

    # 5. Calcul des proportions de services
    total_services_par_carreau = services_par_carreau.sum(axis=1)
    proportions_services = services_par_carreau.div(total_services_par_carreau.replace(0, 1), axis=0)  # Évite division par zéro
    proportions_services = proportions_services.astype(np.float64)  # Conversion explicite en float

    # 6. Calcul des proportions de population
    total_population = carreaux['population_estimee'].sum()
    carreaux['proportion_pop'] = (carreaux['population_estimee'] / total_population).astype(np.float64)

    # 7. Jointure des proportions
    hsmi_data = carreaux[['idINSPIRE', 'proportion_pop']].merge(
        proportions_services,
        on='idINSPIRE',
        how='left'
    ).fillna(0)

    # 8. Initialisation des colonnes HSMI à NaN
    types_service = ['A', 'B', 'C', 'D', 'E', 'F']
    for service in types_service:
        if service in hsmi_data.columns:
            hsmi_data[f'HSMI_{service}'] = np.nan

    # 9. Calcul du HSMI par type de service
    for service in types_service:
        if service in hsmi_data.columns:
            diff = abs(hsmi_data['proportion_pop'] - hsmi_data[service])
            mask = diff > 0
            hsmi_values = 1 - (diff[mask] / (hsmi_data.loc[mask, 'proportion_pop'] + hsmi_data.loc[mask, service]))
            hsmi_data.loc[mask, f'HSMI_{service}'] = hsmi_values.astype(np.float64)

    # 10. Calcul du HSMI global
    colonnes_hsmi = [f'HSMI_{s}' for s in types_service if f'HSMI_{s}' in hsmi_data.columns]
    hsmi_data['HSMI_global'] = hsmi_data[colonnes_hsmi].mean(axis=1)

    # 11. Fusion avec les données existantes
    carreaux_final = carreaux.merge(
        hsmi_data[['idINSPIRE'] + colonnes_hsmi + ['HSMI_global']],
        on='idINSPIRE',
        how='left'
    )

    print("\n--- Statistiques (carreaux avec un HSMI calculable uniquement) ---")
    print(f"- HSMI global moyen (%) : {carreaux_final['HSMI_global'].mean()*100:.3f}")
    for service in types_service:
        col = f'HSMI_{service}'
        if col in carreaux_final.columns:
            print(f"- {col} moyen (%): {carreaux_final[col].mean()*100:.3f}")

    # Remplissage explicite des valeurs manquantes avec -1 (HSMI non calculable)
    for col in colonnes_hsmi + ['HSMI_global']:
        carreaux_final[col] = carreaux_final[col].fillna(-1).astype(np.float64)

    # 12. Export
    exporter_parquet(carreaux_final, "maille_200m_avec_donnees")
    exporter_gpkg(carreaux_final, "maille_200m_avec_donnees")

    print("\nExport terminé avec succès.")

# Exécution
calculer_hsmi()


# In[423]:


def afficher_hsmi(export = False):
    # 1. Chargement des données
    limites_epci = charger_fichier_parquet("limites_epci", crs=3857)
    carreaux = charger_fichier_parquet("maille_200m_avec_donnees", crs=3857)
    carreaux = carreaux[carreaux["HSMI_global"] != -1]

    # 2. Affichage
    fig, ax = plt.subplots(figsize=(10, 10))
    limites_epci.plot(ax=ax, alpha=1, edgecolor='black', facecolor='none')

    carreaux.plot(column="HSMI_global",
             cmap="YlGn",
             legend=True,
             #legend_kwds={'label': "HSMI moyen par maille"},
             ax=ax,
             linewidth=0.1,
             edgecolor="grey")

    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)

    ax.set_title("HSMI moyen par maille", fontsize=18)
    ax.axis('off')
    plt.tight_layout()

    # (Optionnel) Export PNG
    if export :
        images_export_dir = os.path.join(images_dir, "visualisation_hsmi.png")
        plt.savefig(images_export_dir, dpi=300, bbox_inches='tight')
        print(f"Carte exportée vers : {images_export_dir}")

    plt.show()
    plt.close()

# Exécution
afficher_hsmi(export = True)


# #### 6.7.2. HEMI (Housing - Employment Mix Index) / Indice de mixité logement - emplois
# ---
# Cet indicateur est calculé via les étapes suivantes :
# 1. Pour chaque carreau de 200m, on estime le nombre d'emplois en faisant en additionnant le nombre estimé d'emplois dans les entreprises (données SIRENE) qui s'y trouvent : nb_emplois_carreaux
# 2. Pour chaque carreau de 200m, on récupère la population active occupée (15-64 ans) : nb_pop_active_occ
# 3. Les différences absolues de la population employée et des emplois sont calculées chaque maille et divisée par leur somme : HEMI carrau = 1 - (|nb_emplois_carreaux-nb_pop_active_occ| / (|nb_emplois_carreaux+nb_pop_active_occ|)). Pas de calcul si nb_emplois_carreaux+nb_pop_active_occ = 0.
# 
# Documentation :
# * La population active occupée (ou population active ayant un emploi) comprend, au sens du recensement de la population, les personnes qui déclarent être dans l'une des situations suivantes :
#     * exercer une profession (salariée ou non), même à temps partiel
#     * aider une personne dans son travail (même sans rémunération)
#     * être apprenti, stagiaire rémunéré
#     * être chômeur tout en exerçant une activité réduite
#     * être étudiant ou retraité mais occupant un emploi - https://www.insee.fr/fr/metadonnees/definition/c1737
# * Les salariés sont les personnes qui travaillent, aux termes d’un contrat, pour une autre entité résidente en échange d’un salaire ou d’une rétribution équivalente, avec un lien de subordination. - https://www.insee.fr/fr/metadonnees/definition/c1965
# * Les non-salariés sont les personnes qui travaillent mais sont rémunérées sous une autre forme qu’un salaire. En pratique, ils désignent l’ensemble des personnes affiliées à un régime de protection sociale des travailleurs non salariés : Sécurité sociale des indépendants (SSI) ou Mutualité sociale agricole (MSA). Sont concernés les micro-entrepreneurs et les non-salariés classiques ; ces derniers sont pour l’essentiel des entrepreneurs individuels classiques (hors micro-entrepreneurs) ou des gérants majoritaires de sociétés à responsabilité limitée (SARL, SELARL, EARL, etc.).  Toutes les personnes exerçant une activité non salariée sont prises en compte, qu’il s’agisse de leur activité principale ou d’une activité secondaire, complémentaire à une activité salariée. Cependant, les conjoints collaborateurs et les aides familiaux, non répertoriés dans les sources administratives utilisées, ainsi que les cotisants solidaires de la MSA, dont l’importance de l’activité agricole est inférieure à l’activité minimale d’assujettissement, ne sont pas comptés parmi les non-salariés.
#  Une partie des personnes affiliées à un régime de protection sociale des travailleurs non salariés ne sont pas indépendantes économiquement ou au sens du droit du travail ; c’est le cas des entrepreneurs économiquement dépendants (d’un client, d’une organisation en amont ou d’un intermédiaire comme une plateforme numérique). Ces derniers ne sont cependant pas identifiables au sein des non-salariés à partir des données administratives, mais peuvent être appréhendés par des données d’enquête. - https://www.insee.fr/fr/metadonnees/definition/c2301
# 
# 
# Le HEMI peut aller de 0 à 1. Il permet de différencier les zones avec autant d'emplois que de population active, favorisant théoriquement les mobilités locales (1) des zones purement résidentielles ou spécialisées dans l'emploi (0), décourageant les mobilités locales.

# In[60]:


def calculer_hemi():
    # 1. Charger les données nécessaires
    carreaux_donnees = charger_fichier_parquet("maille_200m_avec_donnees", crs=2154)
    etablissements = charger_fichier_parquet("etablissements_emplois_epci", crs=2154)

    # Supression des colonnes de jointures et de celles déjà calculées (si présentes) 
    colonnes_a_supprimer = ['nb_emplois_carreau', 'HEMI']
    carreaux_donnees = nettoyer_colonnes(carreaux_donnees, colonnes_a_supprimer)

    # 2. Préparation des données d'emplois
    # Jointure spatiale pour compter les emplois par carreau
    etabs_avec_carreaux = gpd.sjoin(
        etablissements,
        carreaux_donnees[['idINSPIRE', 'geometry']],
        how='left',
        predicate='within'
    )

    # Somme des emplois par carreau
    emplois_par_carreau = etabs_avec_carreaux.groupby('idINSPIRE')['emplois'].sum().reset_index()
    emplois_par_carreau.columns = ['idINSPIRE', 'nb_emplois_carreau']

    # 3. Jointure avec les données de population active
    hemi_data = carreaux_donnees.merge(
        emplois_par_carreau,
        on='idINSPIRE',
        how='left'
    ).fillna(0)  # Carreaux sans emplois

    # 4. Calcul du HEMI
    # Sélection des colonnes avec conversion explicite en float
    hemi_data['nb_emplois_carreau'] = hemi_data['nb_emplois_carreau'].astype(float)
    hemi_data['C21_ACTOCC15P_estime'] = hemi_data['C21_ACTOCC15P_estime'].astype(float)

    # Calcul du dénominateur (somme emplois + population active)
    denominateur = hemi_data['nb_emplois_carreau'] + hemi_data['C21_ACTOCC15P_estime']

    # Calcul du HEMI seulement pour les carreaux avec denominateur > 0
    mask = denominateur > 0
    hemi_data.loc[mask, 'HEMI'] = 1 - (
        abs(hemi_data.loc[mask, 'nb_emplois_carreau'] - hemi_data.loc[mask, 'C21_ACTOCC15P_estime']) / 
        denominateur[mask]
    )

    # 5. Statistiques descriptives
    print("\n--- Statistiques (carreaux avec un HEMI calculable uniquement) ---")
    print(f"Carreaux avec calcul HEMI : {mask.sum()}/{len(hemi_data)}")
    print(f"HEMI moyen : {hemi_data['HEMI'].mean():.3f}")
    print(f"Distribution des valeurs :")
    print(hemi_data['HEMI'].describe())

    # 6. Formatage et arrondi des valeurs HEMI
    hemi_data['HEMI'] = hemi_data['HEMI'].round(2)  # Arrondi à 2 décimales
    hemi_data['HEMI'] = hemi_data['HEMI'].fillna(-1)  # Valeurs manquantes à -1

    # 7. Export
    exporter_parquet(hemi_data, "maille_200m_avec_donnees")
    exporter_gpkg(hemi_data, "maille_200m_avec_donnees")

    print("\nExport terminé avec succès.")

# Exécution
calculer_hemi()


# In[424]:


def afficher_hemi(export = False):
    # 1. Chargement des données
    limites_epci = charger_fichier_parquet("limites_epci", crs=3857)
    carreaux = charger_fichier_parquet("carreaux_200m_avec_hemi", crs=3857)
    carreaux = carreaux[carreaux["HEMI"] != -1]

    # 2. Affichage
    fig, ax = plt.subplots(figsize=(10, 10))
    limites_epci.plot(ax=ax, alpha=1, edgecolor='black', facecolor='none')

    carreaux.plot(column="HEMI",
             cmap="YlGn",
             legend=True,
             #legend_kwds={'label': "HEMI moyen par maille"},
             ax=ax,
             linewidth=0.1,
             edgecolor="grey")

    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)

    ax.set_title("HEMI moyen par maille", fontsize=18)
    ax.axis('off')
    plt.tight_layout()

    # (Optionnel) Export PNG
    if export :
        images_export_dir = os.path.join(images_dir, "visualisation_hemi.png")
        plt.savefig(images_export_dir, dpi=300, bbox_inches='tight')
        print(f"Carte exportée vers : {images_export_dir}")

    plt.show()
    plt.close()

# Exécution
afficher_hemi(export = True)


# #### 6.7.3. Part de surface des mailles où le bruit routier moyen journalier est une nuisance
# ---
# Reprends les données de la partie 3.11 pour calculer le bruit routier moyen sur 24h (2022) par maille
# On considère un niveau ≥ à 53 dB comme une nuisance d'après : "WHO recommends a threshold of 53dB during the day-evening-night period for road traffic noise pollution levels and 45dB during the night." (voir p. 33 - https://www.eea.europa.eu/en/analysis/publications/transport-and-environment-report-2022/th-al-22-015-en-n_4-term-2022-final-26-04-2023.pdf/@@download/file)

# In[173]:


# Cette fonction n'est pas utilisée pour le calcul d'indicateurs, mais les résultats obtenus peuvent être intéressants
def calculer_db_routiers_moyens_journalier():
    # 1. Chargement des données
    carreaux = charger_fichier_parquet("maille_200m_avec_donnees", crs=3857)
    bruits = charger_fichier_parquet("bruits_2022_ems", crs=3857)

    # Supression des colonnes de jointures et de celles déjà calculées (si présentes) 
    colonnes_a_supprimer = ['bruit_m_route_db', 'bruit_moy_route_db']
    carreaux = nettoyer_colonnes(carreaux, colonnes_a_supprimer)

    # 2. Conversion des classes de bruit en valeur numérique moyenne (version corrigée)
    def extraire_valeur_moyenne(plage):
        if plage.startswith('>='):
            # Pour '>=75', on prend 80 comme valeur moyenne (peut être ajusté si besoin)
            # return int(plage[2:]) # Retourne 75
            return 80
        elif '-' in plage:
            # Calcul de la moyenne de la plage A-B (B exclu)
            min_val, max_val = map(int, plage.split('-'))
            return (min_val + (max_val - 1)) / 2  # Moyenne de la plage
        else:
            return int(plage)

    bruits["db_moyen"] = bruits["valeur"].apply(extraire_valeur_moyenne)

    # 3. Intersection spatiale entre polygones de bruit et carreaux
    print("Intersection entre polygones de bruit et carreaux...")
    bruits_intersect = gpd.overlay(
        bruits[["geometry", "db_moyen"]], 
        carreaux[["idINSPIRE", "geometry"]],
        how="intersection"
    )

    # 4. Surface d'intersection
    bruits_intersect["surface"] = bruits_intersect.geometry.area

    # 5. Moyenne pondérée corrigée
    print("Calcul du bruit moyen pondéré par surface...")

    # Solution optimisée sans warning
    bruits_intersect["temp_id"] = bruits_intersect["idINSPIRE"]
    grouped = (
        bruits_intersect
        .groupby("temp_id", group_keys=False)
        .apply(
            lambda g: pd.Series({
                "bruit_moy_route_db": np.average(g["db_moyen"], weights=g["surface"])
            }),
            include_groups=False
        )
        .reset_index()
        .rename(columns={"temp_id": "idINSPIRE"})
    )

    # 6. Fusion et arrondi à l'entier
    carreaux = carreaux.merge(grouped, on="idINSPIRE", how="left")
    carreaux["bruit_moy_route_"] = carreaux["bruit_moy_route_db"].round().fillna(0).astype(int)

    # 7. Export
    exporter_parquet(carreaux, "maille_200m_avec_donnees")
    exporter_gpkg(carreaux, "maille_200m_avec_donnees")

    print("Bruit routier moyen journalier estimé ajouté avec succès.")

# Exécution
calculer_db_routiers_moyens_journalier()


# In[331]:


def afficher_db_routiers_moyens_journalier(export = False):
    # 1. Chargement des données
    limites_epci = charger_fichier_parquet("limites_epci", crs=3857)
    carreaux_bruit = charger_fichier_parquet("maille_200m_avec_donnees", crs=3857)

    # 2. Définition des classes et couleurs (version corrigée)
    classes_bruit = [
        (0, 50, '#2b83ba', '0-49 dB (Très faible)'), 
        (50, 55, '#abdda4', '50-54 dB (Faible)'),
        (55, 60, '#66c2a5', '55-59 dB (Moyen-faible)'),
        (60, 65, '#ffffbf', '60-64 dB (Modéré)'),
        (65, 70, '#fee08b', '65-69 dB (Élevé)'),
        (70, 75, '#fdae61', '70-74 dB (Très élevé)'),
        (75, 100, '#d7191c', '≥75 dB (Le plus élevé)')
    ]

    # 3. Attribution des couleurs
    carreaux_bruit['couleur'] = '#cccccc'  # Valeur par défaut (gris)

    for min_db, max_db, couleur, _ in classes_bruit:
        if max_db == 100:  # Cas spécial pour ≥75 dB
            mask = carreaux_bruit['bruit_moy_route_db'] >= min_db
        else:
            # between() avec min inclus et max exclus pour correspondre aux plages EMS
            mask = carreaux_bruit['bruit_moy_route_db'].between(min_db, max_db, inclusive='left')
        carreaux_bruit.loc[mask, 'couleur'] = couleur

    # 4. Affichage
    fig, ax = plt.subplots(figsize=(10, 10))
    carreaux_bruit.plot(ax=ax, color=carreaux_bruit['couleur'], edgecolor='face', linewidth=0.05)
    limites_epci.plot(ax=ax, edgecolor='black', facecolor='none')

    legend_elements = [
        mpatches.Patch(color=couleur, label=label) 
        for _, _, couleur, label in classes_bruit
    ]

    ax.legend(handles=legend_elements, loc='upper left', title_fontsize=12, fontsize=10)
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
    ax.set_title("Bruit routier moyen par jour et par carreau de 200m (estimé, 2022)",fontsize=14)
    ax.set_axis_off()
    plt.tight_layout()

    # (Optionnel) Export PNG
    if export :
        images_export_dir = os.path.join(images_dir, "visualisation_bruit_routier_maille.png")
        plt.savefig(images_export_dir, dpi=300, bbox_inches='tight')
        print(f"Carte exportée vers : {images_export_dir}")

    plt.show()
    plt.close()

# Exécution
afficher_db_routiers_moyens_journalier(export = True)


# In[230]:


# Calcule la part de surface des mailles 
def calculer_part_surface_bruit_route_nuisance(seuil_db=53):
    # 1. Chargement des données
    carreaux = charger_fichier_parquet("maille_200m_avec_donnees", crs=2154)
    bruits = charger_fichier_parquet("bruits_2022_ems", crs=2154)

    # Suppression éventuelle de l'ancienne colonne
    carreaux = nettoyer_colonnes(carreaux, ["part_surface_bruit_nuisance", f"part_surface_bruit_{seuil_db}db", "surface_nuisance_m2", "surface_totale_m2"])

    # 2. Conversion des classes de bruit en valeur moyenne numérique
    def extraire_valeur_moyenne(plage):
        if plage.startswith('>='):
            return 80
        elif '-' in plage:
            min_val, max_val = map(int, plage.split('-'))
            return (min_val + (max_val - 1)) / 2
        else:
            return int(plage)

    bruits["db_moyen"] = bruits["valeur"].apply(extraire_valeur_moyenne)

    # 3. Filtrer les polygones de bruit dépassant le seuil
    bruits_nuisance = bruits[bruits["db_moyen"] >= seuil_db].copy()

    # 4. Intersection avec les carreaux
    print(f"Intersection des zones de bruit ≥ {seuil_db} dB avec les carreaux...")
    inter = gpd.overlay(
        bruits_nuisance[["geometry"]],
        carreaux[["idINSPIRE", "geometry"]],
        how="intersection"
    )

    # 5. Calcul de la surface intersectée
    inter["surface_nuisance_m2"] = inter.geometry.area

    # 6. Agrégation par carreau
    surfaces_nuisance = (
        inter.groupby("idINSPIRE")["surface_nuisance_m2"]
        .sum()
        .reset_index()
    )

    # 7. Fusion et calcul du pourcentage de nuisance
    carreaux["surface_totale_m2"] = carreaux.geometry.area
    carreaux = carreaux.merge(surfaces_nuisance, on="idINSPIRE", how="left")
    carreaux["surface_nuisance_m2"] = carreaux["surface_nuisance_m2"].fillna(0)

    carreaux[f"part_surface_bruit_{seuil_db}db"] = (
        (carreaux["surface_nuisance_m2"] / carreaux["surface_totale_m2"]) * 100
    ).round(2).clip(upper=100)

    # 8. Export
    exporter_parquet(carreaux, "maille_200m_avec_donnees")
    exporter_gpkg(carreaux, "maille_200m_avec_donnees")

    print(f"Part de surface exposée à un bruit routier ≥ {seuil_db} dB ajoutée avec succès.")

# Exécution
calculer_part_surface_bruit_route_nuisance(seuil_db=70)
calculer_part_surface_bruit_route_nuisance(seuil_db=53)


# In[439]:


def afficher_part_surface_bruit_route_nuisance(export = False):
    # 1. Chargement des données
    limites_epci = charger_fichier_parquet("limites_epci", crs=3857)
    carreaux = charger_fichier_parquet("maille_200m_avec_donnees", crs=3857)

    # 2. Affichage
    fig, ax = plt.subplots(figsize=(10, 10))
    limites_epci.plot(ax=ax, alpha=1, edgecolor='black', facecolor='none')

    carreaux.plot(column="part_surface_bruit_53db",
             cmap="YlOrRd",
             legend=True,
             #legend_kwds={'label': "Part de surface exposée à un bruit routier ≥ 53 dB (moyenne sur 24 h)"},
             ax=ax,
             linewidth=0.1,
             edgecolor="grey")

    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)

    ax.set_title("Part de surface exposée à un bruit routier ≥ 53 dB (sur 24 h)", fontsize=18)
    ax.axis('off')
    plt.tight_layout()

    # (Optionnel) Export PNG
    if export :
        images_export_dir = os.path.join(images_dir, "indicateur_bruit_routier_53_dB.png")
        plt.savefig(images_export_dir, dpi=300, bbox_inches='tight')
        print(f"Carte exportée vers : {images_export_dir}")

    plt.show()
    plt.close()

# Exécution
afficher_part_surface_bruit_route_nuisance(export = True)


# #### 6.7.4. Part d'occupation du sol par les infrastructures de transport
# ---
# Par infrastructure de transports, on comprends :
# * Les routes. Issues de la BD Topo
# * Les parkings. Déjà des données surfaciques dans la BD Topo.
# * Les lignes de tram. D'après l'imagerie satellite, leur largeur à Strabourg semble être de 7,5 mètres (pour les lignes à double sens). Plus d'infos ici : https://www.ecocitestrasbourg.org/IMG/pdf/S-S-_JNK.pdf
# * Les gares (pas calculé ici, impossible de sélectionner uniquement le bâti lié au transport dans la BD Topo)
# * Les arrêts de bus / tram (pas calculé ici, les seules données disponibles sont ponctuelles)
# 
# 
# La fonction créé des zones tampon autours des routes et lignes de tram selon leur largeur, puis calcule la part de la surface occupée par ces zones tampon (transports) sur le reste de la maille. Les zones tampon sont rectangulaires et n'étendent pas artificiellement la longueur des infrastructures.

# In[334]:


def calculer_part_surface_transport():
    # 1. Chargement des données
    carreaux = charger_fichier_parquet("maille_200m_avec_donnees", crs=2154)
    routes = charger_fichier_parquet("bd_topo_routes_epci", crs=2154)
    tram = charger_fichier_parquet("lignes_tram", crs=2154)
    limites_epci = charger_fichier_parquet("limites_epci", crs=2154)

    # 2. Chargement des équipements de transport
    equip_transport_path = trouver_fichier("EQUIPEMENT_DE_TRANSPORT.dbf", extraction_dir)
    equipements = gpd.read_file(equip_transport_path)
    equipements = equipements.to_crs(2154)

    """
    Les ports de commerces ne sont pas conservés, leurs polygones sont beaucoup trop larges
    et couvrent des surfaces non liées aux transports. Les parkings souterrains ne sont pas
    conservés, car ne se trouvent pas en surface.
    """
    # 3. Filtrage des objets : on conserve tout sauf certains cas
    equipements = equipements[~equipements["NAT_DETAIL"].isin(["Port de commerce", "Parking souterrain"])]

    # 4. Intersection avec la zone d’étude
    equipements = equipements[equipements.intersects(limites_epci.geometry.union_all())]

    # 5. Largeur des lignes de transport
    largeur_tram = 7.5
    routes["LARGEUR"] = pd.to_numeric(routes.get("LARGEUR", None), errors="coerce")
    routes["largeur_calculee"] = routes["LARGEUR"].where(routes["LARGEUR"] > 0, 2.0)
    routes["largeur_calculee"] = routes["largeur_calculee"].fillna(2.0)

    def line_to_rectangle(line, largeur):
        return line.buffer(largeur / 2, cap_style=2, join_style=2)

    # 6. Buffers pour routes et tram
    routes["poly"] = routes.geometry.apply(lambda geom: line_to_rectangle(geom, largeur=routes["largeur_calculee"].median()))
    tram["poly"] = tram.geometry.apply(lambda geom: line_to_rectangle(geom, largeur=largeur_tram))

    # 7. Fusion des surfaces transport
    surfaces_transport = pd.concat([
        routes["poly"],
        tram["poly"],
        equipements.geometry  # on ajoute les surfaces directement
    ])
    union_transport = gpd.GeoSeries(surfaces_transport, crs=2154).geometry.union_all()

    # 8. Calcul par carreau
    carreaux["surface_totale_m2"] = carreaux.geometry.area
    surface_transport = []

    for geom in carreaux.geometry:
        inter = geom.intersection(union_transport)
        surface_transport.append(inter.area if not inter.is_empty else 0)

    carreaux["surface_transport_m2"] = surface_transport
    carreaux["part_surface_transport"] = (carreaux["surface_transport_m2"] / carreaux["surface_totale_m2"]) * 100
    carreaux["part_surface_transport"] = carreaux["part_surface_transport"].clip(upper=100).round(2)

    # 9. Export
    exporter_gpkg(carreaux, "maille_200m_avec_donnees")
    exporter_parquet(carreaux, "maille_200m_avec_donnees")

    # 10. Résumé
    moyenne = (carreaux["surface_transport_m2"].sum() / carreaux["surface_totale_m2"].sum()) * 100
    print(f"Part moyenne réelle de surface occupée par les infrastructures de transport : {moyenne:.2f} %")

# Exécution
calculer_part_surface_transport()


# In[440]:


# Note : la taille de la gare de Triage d'Hausbergen est surestimée, le polygone est légèrement trop large,
# mais serait pire de ne pas la prendre en compte
def afficher_part_surface_transport(export = False):
    # 1. Charger les données
    limites_epci = charger_fichier_parquet("limites_epci", crs=3857)
    carreaux = charger_fichier_parquet("maille_200m_avec_donnees", crs=3857)

    # 2. Affichage
    fig, ax = plt.subplots(figsize=(10, 10))
    limites_epci.plot(ax=ax, alpha=1, edgecolor='black', facecolor='none')

    carreaux.plot(column="part_surface_transport",
             cmap="YlOrRd",
             legend=True,
             #legend_kwds={'label': "% de surface occupée par des infrastructures de transport"},
             ax=ax,
             linewidth=0.1,
             edgecolor="grey")

    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)

    ax.set_title("Part de surface occupée par des infrastructures de transport", fontsize=18)
    ax.axis('off')
    plt.tight_layout()

    # (Optionnel) Export PNG
    if export :
        images_export_dir = os.path.join(images_dir, "indicateur_part_surface_transport.png")
        plt.savefig(images_export_dir, dpi=300, bbox_inches='tight')
        print(f"Carte exportée vers : {images_export_dir}")

    plt.show()
    plt.close()

# Exécution
afficher_part_surface_transport(export = True)


# #### 6.7.5. Nombre d'accidents liés aux transports
# ---

# In[226]:


calculer_nb_accidents(transport="total")


# In[442]:


afficher_nb_accidents(transport="total", export = True)


# ### 6.8. Indicateurs de schémas de mobilité
# ---
# Indicateur de schémas de mobilité :
# * 6.8.1. Part des navetteurs utilisant principalement la marche - 'part_marche'
# * 6.8.2. Part des navetteurs utilisant principalement le vélo - 'part_velos'
# * 6.8.3. Part des navetteurs utilisant principalement les transports en commun - 'part_tcom'
# * 6.8.4. Part des navetteurs utilisant principalement des véhicules légers - 'part_vl'
# 
# marche

# #### 6.8.1. Part des navetteurs utilisant principalement la marche
# ---

# In[72]:


calculer_part_navetteurs(nom_transport="marche")


# In[431]:


afficher_part_navetteurs(nom_transport="marche", export = True)


# #### 6.8.2. Part des navetteurs utilisant principalement le vélo
# ---

# In[346]:


calculer_part_navetteurs(nom_transport="velos")


# In[435]:


afficher_part_navetteurs(nom_transport="velos", export = True)


# #### 6.8.3. Part des navetteurs utilisant principalement les transports en commun
# ---

# In[145]:


calculer_part_navetteurs(nom_transport="tcom")


# In[433]:


afficher_part_navetteurs(nom_transport="tcom", export = True)


# #### 6.8.4. Part des navetteurs utilisant principalement des véhicules légers
# ---

# In[76]:


calculer_part_navetteurs(nom_transport="vl")


# In[436]:


afficher_part_navetteurs(nom_transport="vl", export = True)


# ### 6.9. Export des données brutes et normalisées
# ---

# #### 6.9.1. Exports des données brutes
# ---

# In[176]:


def afficher_colonnes_presentes():
    # Chargement des données
    carreaux = charger_fichier_parquet("maille_200m_avec_donnees", crs=3857)

    # Affichage des colonnes triées
    colonnes_triees = sorted(carreaux.columns.to_list())
    for colonne in colonnes_triees:
        print(f"'{colonne}',")

# Exécution
afficher_colonnes_presentes()


# In[174]:


"""
import os
import geopandas as gpd
import pandas as pd

# Chemins
dir = "data"
exports_dir = os.path.join(dir, "exports")

# Fichiers
fichier_source_gpkg = os.path.join(dir, "maille_200m_avec_donnees.gpkg")
fichier_cible_parquet = os.path.join(exports_dir, "maille_200m_avec_donnees.parquet")

# Colonnes à récupérer
colonnes_a_transferer = [
    "idINSPIRE",
    "autopartage_ratio_emplois",
    "autopartage_ratio_services_dom_A",
    "autopartage_ratio_services_dom_B",
    "autopartage_ratio_services_dom_C",
    "autopartage_ratio_services_dom_D",
    "autopartage_ratio_services_dom_E",
    "autopartage_ratio_services_dom_F"
]

# Chargement
print("Chargement des fichiers...")
gdf_source = gpd.read_file(fichier_source_gpkg)[colonnes_a_transferer].copy()
df_cible = pd.read_parquet(fichier_cible_parquet).copy()

# Suppression des colonnes à réécrire si elles existent
colonnes_sans_id = [col for col in colonnes_a_transferer if col != "idINSPIRE"]
colonnes_a_supprimer = [col for col in colonnes_sans_id if col in df_cible.columns]
if colonnes_a_supprimer:
    print(f"[INFO] Colonnes existantes supprimées : {colonnes_a_supprimer}")
    df_cible.drop(columns=colonnes_a_supprimer, inplace=True)

# Fusion
df_fusion = df_cible.merge(gdf_source, on="idINSPIRE", how="left")

# Export
df_fusion.to_parquet(fichier_cible_parquet, index=False)
print(f"✅ Données fusionnées et exportées : {fichier_cible_parquet}")
"""


# In[443]:


def exporter_donnees_brutes_tableau_de_bord():
    # 1. Chargement des données
    carreaux = charger_fichier_parquet("maille_200m_avec_donnees", crs=4326)
    limites_epci = charger_fichier_parquet("limites_epci", crs=4326)

    # 2. Liste des colonnes à conserver pour les carreaux
    colonnes_a_conserver_carreaux = [
        'idINSPIRE', # Ne pas crééer de cartes pour cette donnée sur le tableau de bord
        'geometry', # Ne pas crééer de cartes pour cette donnée sur le tableau de bord
        'part_surface_pietonne', # Marchabilité
        'part_vegetation',
        'ratio_hl_bati_rues',
        'part_routes_lentes',
        'part_routes_inaccessibles_vl',
        'part_parcs',
        'marche_ratio_emplois',
        'marche_ratio_services_dom_moyenne',
        # 'marche_ratio_services_dom_A', 
        #'marche_ratio_services_dom_B', 
        #'marche_ratio_services_dom_C', 
        #'marche_ratio_services_dom_D', 
        #'marche_ratio_services_dom_E',
        #'marche_ratio_services_dom_F',
        'part_surface_cyclable', # Cyclabilité
        'part_normalise_surface_couverte_stations_velos',
        # 'nb_velos_libre_service' (difficile à normaliser, non utilisé pour l'instant)
        'part_routes_interdites_pl',
        'part_routes_pavees',
        'pente_moyenne_absolue',
        'part_normalise_couverte_stations_rep_velos',
        'velos_ratio_emplois',
        'velos_ratio_services_dom_moyenne',
        #'velos_ratio_services_dom_A', 
        #'velos_ratio_services_dom_B', 
        #'velos_ratio_services_dom_C', 
        #'velos_ratio_services_dom_D', 
        #'velos_ratio_services_dom_E',
        #'velos_ratio_services_dom_F',
        'att_moy_semaine_bus', # Qualité de service des bus
        'cout_horaire_bus',
        'ratio_arrets_pr_bus',
        'ratio_arrets_autopartage_bus',
        'part_routes_bus',
        'part_batie_autour_arrets_bus',
        'marche_bus_ratio_emplois',
        'marche_bus_ratio_services_dom_moyenne',
        #'marche_bus_ratio_services_dom_A', 
        #'marche_bus_ratio_services_dom_B', 
        #'marche_bus_ratio_services_dom_C', 
        #'marche_bus_ratio_services_dom_D', 
        #'marche_bus_ratio_services_dom_E',
        #'marche_bus_ratio_services_dom_F',
        'att_moy_semaine_tram', # Qualité de service des tram
        'cout_horaire_tram',
        'ratio_arrets_pr_tram',
        'ratio_arrets_autopartage_tram',
        'part_routes_tram',
        'part_batie_autour_arrets_tram',
        'marche_tram_ratio_emplois',
        'marche_tram_ratio_services_dom_moyenne',
        #'marche_tram_ratio_services_dom_A', 
        #'marche_tram_ratio_services_dom_B', 
        #'marche_tram_ratio_services_dom_C', 
        #'marche_tram_ratio_services_dom_D', 
        #'marche_tram_ratio_services_dom_E',
        #'marche_tram_ratio_services_dom_F',
        'part_sens_unique', # Qualité de service pour les VL
        'part_zfe',
        'nb_feux_circulation',
        # 'nb_stations_service', (difficile à normaliser, non utilisé pour l'instant)
        'places_stationnement_par_habitant',
        'cout_moyen_stationnement',
        # 'duree_moyenne_stationnement' (difficile à normaliser, non utilisé pour l'instant)
        'bornes_ve_par_habitant',
        'vl_ratio_emplois',
        'vl_ratio_services_dom_moyenne',
        #'vl_ratio_services_dom_A', 
        #'vl_ratio_services_dom_B', 
        #'vl_ratio_services_dom_C', 
        #'vl_ratio_services_dom_D', 
        #'vl_ratio_services_dom_E',
        #'vl_ratio_services_dom_F',
        'part_normalise_couverte_autopartage', # Qualité de service de l'autopartage
        # 'nb_voitures_autopartage' (difficile à normaliser, non utilisé pour l'instant)
        'autopartage_cout_rapport_revenu',
        'part_routes_accessibles_vl',
        'part_surface_geofencing_citiz',
        'autopartage_ratio_emplois',
        'autopartage_ratio_services_dom_moyenne',
        #'autopartage_ratio_services_dom_A', 
        #'autopartage_ratio_services_dom_B', 
        #'autopartage_ratio_services_dom_C', 
        #'autopartage_ratio_services_dom_D', 
        #'autopartage_ratio_services_dom_E',
        #'autopartage_ratio_services_dom_F',
        'HSMI_global', # Diversité fonctionnelle
        'HEMI',
        'part_marche', # Indicateurs de schéma de mobilité
        'part_velos',
        'part_communs',
        'part_vl', 
        'part_surface_bruit_53db', # Autres : indicateurs de mobilité durable
        'part_surface_transport',
        'nb_accidents_circulation'
    ]

    # 3. Liste des colonnes à conserver pour l'EPCI
    colonnes_a_conserver_epci = ['geometry', 'NOM', 'CODE_SIREN']

    # 4. Vérification et filtrage
    colonnes_presentes = [col for col in colonnes_a_conserver_carreaux if col in carreaux.columns]
    colonnes_manquantes = sorted(set(colonnes_a_conserver_carreaux) - set(carreaux.columns))

    # 5. Affichage  des colonnes absentes
    if colonnes_manquantes:
        print("[INFO] Colonnes manquantes dans le fichier des carreaux :")
        for col in colonnes_manquantes:
            print(f" - {col}")

    carreaux = carreaux[colonnes_presentes]
    limites_epci = limites_epci[[col for col in colonnes_a_conserver_epci if col in limites_epci.columns]]

    # 6. Export
    exporter_geojson(carreaux, "maille_200m_avec_donnees", tableau_bord_dir)
    exporter_geojson(limites_epci, "limites_epci", tableau_bord_dir)

    print(f"\nExport terminé")
    print(f"- Colonnes conservées (carreaux) : {len(colonnes_presentes)} / {len(colonnes_a_conserver_carreaux)}")
    if colonnes_manquantes:
        print(f"- Colonnes ignorées car manquantes : {len(colonnes_manquantes)}")

# Exécution
exporter_donnees_brutes_tableau_de_bord()


# #### 6.9.2. Export des données normalisées
# ---

# Normalise les données de 0 à 1 pour toutes les colonnes, où :
# * 0 : Très mauvaise durabilité
# * 1 : Très bonne durabilité
# 
# La configuration se fait sur chaque variable, et détermine d'après les données présentes dans la colonne avant modification "l'orientation' de la normalisation. Par exemple, la colonne 'bruit_moy_route_db' indique le bruit moyen routier journalier : l'indicateur prendra comme 1 la valeur la plus faible trouvée dans cette colonne, et 0 comme la plus élevée. A l'inverse, la colonne 'pourcentage_vegetation' indique le % de la surface occupée par la végétation : la valeur 1 sera la valeur la plus élevée de cette colonne.
# * +1 : Plus la valeur originale est élevée, plus c'est durable
# * -1 : Plus la valeur originale est faible, plus c'est durable
# 
# Une fois l'orientation attribuée, tout les indicateurs sont normalisés de 0 à 1, où 1 signifie toujours une très bonne durabilité.

# In[460]:


def exporter_donnees_normalisees_tableau_de_bord():
    # 1. Chargement
    gdf = charger_fichier_geojson("maille_200m_avec_donnees", crs=4326, dossier=tableau_bord_dir)

    # 2. Configuration : 1 si "+ c’est proche de 1, + c'est durable", -1 si "- c'est proche de 1, + c'est durable"
    # Ne pas normaliser 'geometry' ou 'idINSPIRE', ni les colonnes allant déjà de 0 à 1
    config_norm = {
        #'idINSPIRE',
        #'geometry',
        'part_surface_pietonne':+1, # Marchabilité
        'part_vegetation':+1,
        'ratio_hl_bati_rues':-1,
        'part_routes_lentes':+1,
        'part_routes_inaccessibles_vl':+1,
        'part_parcs':+1,
        'marche_ratio_emplois':+1,
        'marche_ratio_services_dom_moyenne':+1,
        # 'marche_ratio_services_dom_A', 
        #'marche_ratio_services_dom_B', 
        #'marche_ratio_services_dom_C', 
        #'marche_ratio_services_dom_D', 
        #'marche_ratio_services_dom_E',
        #'marche_ratio_services_dom_F',
        'part_surface_cyclable':+1, # Cyclabilité
        'part_normalise_surface_couverte_stations_velos':+1,
        # 'nb_velos_libre_service' (difficile à normaliser, non utilisé pour l'instant)
        'part_routes_interdites_pl':+1,
        'part_routes_pavees':-1,
        'pente_moyenne_absolue':-1,
        'part_normalise_couverte_stations_rep_velos':+1,
        'velos_ratio_emplois':+1,
        'velos_ratio_services_dom_moyenne':+1,
        #'velos_ratio_services_dom_A', 
        #'velos_ratio_services_dom_B', 
        #'velos_ratio_services_dom_C', 
        #'velos_ratio_services_dom_D', 
        #'velos_ratio_services_dom_E',
        #'velos_ratio_services_dom_F',
        'att_moy_semaine_bus':-1, # Qualité de service des bus
        'cout_horaire_bus':-1,
        'ratio_arrets_pr_bus':+1,
        'ratio_arrets_autopartage_bus':+1,
        'part_routes_bus':+1,
        'part_batie_autour_arrets_bus':+1,
        'marche_bus_ratio_emplois':+1,
        'marche_bus_ratio_services_dom_moyenne':+1,
        #'marche_bus_ratio_services_dom_A', 
        #'marche_bus_ratio_services_dom_B', 
        #'marche_bus_ratio_services_dom_C', 
        #'marche_bus_ratio_services_dom_D', 
        #'marche_bus_ratio_services_dom_E',
        #'marche_bus_ratio_services_dom_F',
        'att_moy_semaine_tram':-1, # Qualité de service des tram
        'cout_horaire_tram':-1,
        'ratio_arrets_pr_tram':+1,
        'ratio_arrets_autopartage_tram':+1,
        'part_routes_tram':+1,
        'part_batie_autour_arrets_tram':+1,
        'marche_tram_ratio_emplois':+1,
        'marche_tram_ratio_services_dom_moyenne':+1,
        #'marche_tram_ratio_services_dom_A', 
        #'marche_tram_ratio_services_dom_B', 
        #'marche_tram_ratio_services_dom_C', 
        #'marche_tram_ratio_services_dom_D', 
        #'marche_tram_ratio_services_dom_E',
        #'marche_tram_ratio_services_dom_F',
        'part_sens_unique':-1, # Qualité de service pour les VL
        'part_zfe':-1,
        'nb_feux_circulation':-1,
        # 'nb_stations_service', (difficile à normaliser, non utilisé pour l'instant)
        'places_stationnement_par_habitant':+1,
        'cout_moyen_stationnement':-1,
        # 'duree_moyenne_stationnement' (difficile à normaliser, non utilisé pour l'instant)
        'bornes_ve_par_habitant':+1,
        'vl_ratio_emplois':+1,
        'vl_ratio_services_dom_moyenne':+1,
        #'vl_ratio_services_dom_A', 
        #'vl_ratio_services_dom_B', 
        #'vl_ratio_services_dom_C', 
        #'vl_ratio_services_dom_D', 
        #'vl_ratio_services_dom_E',
        #'vl_ratio_services_dom_F',
        'part_normalise_couverte_autopartage':+1, # Qualité de service de l'autopartage
        # 'nb_voitures_autopartage' (difficile à normaliser, non utilisé pour l'instant)
        'autopartage_cout_rapport_revenu':-1,
        'part_routes_accessibles_vl':+1,
        'part_surface_geofencing_citiz':+1,
        'autopartage_ratio_emplois':+1,
        'autopartage_ratio_services_dom_moyenne':+1,
        #'autopartage_ratio_services_dom_A', 
        #'autopartage_ratio_services_dom_B', 
        #'autopartage_ratio_services_dom_C', 
        #'autopartage_ratio_services_dom_D', 
        #'autopartage_ratio_services_dom_E',
        #'autopartage_ratio_services_dom_F',
        'HSMI_global':+1, # Diversité fonctionnelle
        'HEMI':+1,
        'part_marche':+1, # Indicateurs de schéma de mobilité
        'part_velos':+1,
        'part_communs':+1,
        'part_vl':+1, 
        'part_surface_bruit_53db':-1, # Autres : indicateurs de mobilité durable
        'part_surface_transport':-1,
        'nb_accidents_circulation':-1
    }

    # 3. Normalisation
    for col, orientation in config_norm.items():
        if col not in gdf.columns:
            print(f"Colonne absente : {col}")
            continue

        serie = gdf[col]
        valides = serie[serie != -1]

        if len(valides) == 0:
            gdf[col] = -1
            continue

        vmin = valides.min()
        vmax = valides.max()

        if vmax == vmin:
            gdf[col] = np.where(serie == -1, -1, 1)
        else:
            if orientation == +1:
                gdf[col] = np.where(
                    serie == -1,
                    -1,
                    ((serie - vmin) / (vmax - vmin)).clip(0, 1).round(2)
                )
            elif orientation == -1:
                gdf[col] = np.where(
                    serie == -1,
                    -1,
                    ((vmax - serie) / (vmax - vmin)).clip(0, 1).round(2)
                )
            else:
                raise ValueError(f"Orientation invalide : {orientation} pour {col}")

    # 4. Export
    exporter_geojson(gdf, "maille_200m_avec_donnees_normalise", tableau_bord_dir)
    print("Fichier normalisé exporté avec succès.")

# Exécution
exporter_donnees_normalisees_tableau_de_bord()


# ### 7. Calcul des indicateurs composés
# ----

# #### 7.1. Marche / Indicateur de marchabilité
# ---
# Indicateur de marchabilité 'ind_compose_marche' à calculer à partir des indicateurs suivants :
# * 6.1.1. Part de la surface accessible par le réseau pédestre - 'part_surface_pietonne'
# * 6.1.2. Part de la surface occupée par la végétation (NDVI > 0,7) - 'part_vegatation'
# * 6.1.3. Rapport moyen entre la hauteur des bâtiments et la largeur des rues - 'ratio_hl_bati_rues'
# * 6.1.4. Part des rues dont la vitesse moyenne est de moins de 30 km/h - 'part_routes_lentes'
# * 6.1.5. Part des rues inaccessibles aux VL - 'part_routes_inaccessibles_vl"
# * 6.1.6. Part des parcs dans l'aire urbaine - 'part_parcs'
# * Sous-indicateur de mobilité :
#     * 6.1.7. Part des emplois accessibles en moins de T minutes - 'marche_ratio_emploi'
#     * 6.1.8. Part des services accessibles en moins de T minutes. Moyenne sur S types et/ou catégories de services - 'marche_ratio_services_dom_A' à 'marche_ratio_services_dom_F'

# In[466]:


def calcul_indicateurs_composes_marche():
    # 1. Chargement des données normalisées
    donnees_normalisees = charger_fichier_geojson(
        "maille_200m_avec_donnees_normalise", crs=2154, dossier=tableau_bord_dir
    ).copy()

    # 2. Colonnes nécessaires
    colonnes = [
        'part_surface_pietonne',
        'part_vegetation',
        'ratio_hl_bati_rues',
        'part_routes_lentes',
        'part_routes_inaccessibles_vl',
        'part_parcs',
        'marche_ratio_emplois',
        'marche_ratio_services_dom_moyenne',
    ]

    # 3. Vérifie les colonnes présentes
    colonnes_presentes = [col for col in colonnes if col in donnees_normalisees.columns]
    colonnes_absentes = sorted(set(colonnes) - set(colonnes_presentes))
    if colonnes_absentes:
        print("Colonnes manquantes (elles seront ignorées) :")
        for col in colonnes_absentes:
            print(f"  - {col}")

    # 4. Calcul du sous-indicateur d'accès (si possible)
    if all(col in donnees_normalisees.columns for col in ['marche_ratio_emplois', 'marche_ratio_services_dom_moyenne']):
        donnees_normalisees["sous_indic_access"] = (
            (donnees_normalisees["marche_ratio_emplois"].replace(-1, np.nan) +
             donnees_normalisees["marche_ratio_services_dom_moyenne"]) / 2
        )
    elif "marche_ratio_emplois" in donnees_normalisees.columns:
        donnees_normalisees["sous_indic_access"] = donnees_normalisees["marche_ratio_emplois"].replace(-1, np.nan)
    elif "marche_ratio_services_dom_moyenne" in donnees_normalisees.columns:
        donnees_normalisees["sous_indic_access"] = donnees_normalisees["marche_ratio_services_dom_moyenne"].replace(-1, np.nan)
    else:
        donnees_normalisees["sous_indic_access"] = np.nan
        print("[INFO] Aucun des indicateurs d’accessibilité disponibles.")

    # 5. Liste finale des colonnes disponibles pour l'indicateur
    composantes_finales = [
        col for col in [
            'part_surface_pietonne',
            'part_vegetation',
            'ratio_hl_bati_rues',
            'part_routes_lentes',
            'part_routes_inaccessibles_vl',
            'part_parcs',
            "sous_indic_access"
        ] if col in donnees_normalisees.columns
    ]

    # 6. Calcul de l'indicateur final
    donnees_normalisees["ind_compose_marche"] = (
        donnees_normalisees[composantes_finales]
        .replace(-1, np.nan)
        .mean(axis=1)
        .round(3)
    ).fillna(-1)

    # 7. Export
    exporter_geojson(donnees_normalisees, "maille_200m_avec_donnees_normalise", dossier=tableau_bord_dir)
    exporter_parquet(donnees_normalisees, "maille_200m_avec_donnees_normalise")

    # 8. Statistiques
    valides = donnees_normalisees[donnees_normalisees["ind_compose_marche"] >= 0]
    print(f"""
Indicateur composé de marchabilité calculé avec succès :
- Nombre de carreaux : {len(donnees_normalisees)}
- Carreaux avec données valides : {len(valides)}
- Moyenne (valeurs valides) : {valides["ind_compose_marche"].mean():.3f}
""")

# Exécution
calcul_indicateurs_composes_marche()


# In[467]:


def afficher_indicateur_compose_marchabilite(export = False):
    # 1. Chargement des données
    limites_epci = charger_fichier_parquet("limites_epci", crs=3857)
    carreaux = charger_fichier_parquet("maille_200m_avec_donnees_normalise", crs=3857)

    # 2. Affichage
    fig, ax = plt.subplots(figsize=(10, 10))
    limites_epci.plot(ax=ax, alpha=1, edgecolor='black', facecolor='none')

    carreaux.plot(column="ind_compose_marche",
             cmap="YlGn",
             legend=True,
             #legend_kwds={'label': "Indicateur de marchabilité par carreau"},
             ax=ax,
             linewidth=0.1,
             edgecolor="grey",
             vmin=0,
             vmax=1)

    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)

    ax.set_title("Indicateur de marchabilité",fontsize=18)
    ax.axis('off')
    plt.tight_layout()

    # (Optionnel) Export PNG
    if export :
        images_export_dir = os.path.join(images_dir, "indicateur_compose_marchabilite.png")
        plt.savefig(images_export_dir, dpi=300, bbox_inches='tight')
        print(f"Carte exportée vers : {images_export_dir}")

    plt.show()
    plt.close()

# Exécution
afficher_indicateur_compose_marchabilite(export = True)


# #### 7.2. Vélos / Indicateur de cyclabilité
# ---
# Indicateur de cyclabilité 'ind_compose_velo' à calculer à partir des indicateurs suivants :
# * 6.1.4. Part des rues dont la vitesse moyenne est de moins de 30 km/h - 'part_routes_lentes'
# * 6.1.5. Part des rues inaccessibles aux VL - 'part_inaccessibles_vl'
# * 6.1.6. Part des parcs dans l'aire urbaine - 'part_parcs'
# * 6.2.1. Part de la surface accessible par le réseau cyclable - 'part_surface_cyclable'
# * 6.2.2. Nombre de stations de vélos en libre-partage (Vélhop) - 'nb_stations_velos_libre_service' / 'part_normalise_surface_couverte_stations_velos'
# * 6.2.3. Nombre de vélos en libre-service - 'nb_velos_libre_service' (difficile à normaliser, non utilisé pour l'instant)
# * 6.2.4. Part des rues interdites aux poids lourds - 'part_routes_inaccessibles_pl'
# * 6.2.5. Part de rues pavées - 'part_routes_pavees'
# * 6.2.6.8. Pente moyenne des rues - 'pente_moyenne'
# * 6.2.9. Nombre de stations de réparation de vélos - 'nb_stations_reparation_velo' / 'part_normalise_couverte_stations_rep_velo'
# * Sous-indicateur de mobilité :
#     * 6.2.8. Part des emplois accessibles en moins de T minutes - 'velos_ratio_emploi'
#     * 6.2.9. Part des services accessibles en moins de T minutes. Moyenne sur S types et/ou catégories de services - 'velos_ratio_services_dom_A' à 'velos_ratio_services_dom_F'

# In[462]:


def calcul_indicateurs_composes_velos():
    # 1. Chargement des données normalisées
    donnees_normalisees = charger_fichier_geojson(
        "maille_200m_avec_donnees_normalise", crs=2154, dossier=tableau_bord_dir
    ).copy()

    # 2. Colonnes nécessaires
    colonnes = [
        'part_normalise_surface_couverte_stations_velos',
        'part_routes_interdites_pl',
        'part_routes_pavees',
        'pente_moyenne_absolue',
        'part_normalise_couverte_stations_rep_velos',
        'velos_ratio_emplois',
        'velos_ratio_services_dom_moyenne',
    ]

    # 3. Vérifie les colonnes présentes
    colonnes_presentes = [col for col in colonnes if col in donnees_normalisees.columns]
    colonnes_absentes = sorted(set(colonnes) - set(colonnes_presentes))
    if colonnes_absentes:
        print("Colonnes manquantes (elles seront ignorées) :")
        for col in colonnes_absentes:
            print(f"  - {col}")

    # 4. Calcul du sous-indicateur d'accès (si possible)
    if all(col in donnees_normalisees.columns for col in ['velos_ratio_emplois', 'velos_ratio_services_dom_moyenne']):
        donnees_normalisees["sous_indic_access"] = (
            (donnees_normalisees["velos_ratio_emplois"].replace(-1, np.nan) +
             donnees_normalisees["velos_ratio_services_dom_moyenne"].replace(-1, np.nan)) / 2
        )
    elif "velos_ratio_emplois" in donnees_normalisees.columns:
        donnees_normalisees["sous_indic_access"] = donnees_normalisees["velos_ratio_emplois"].replace(-1, np.nan)
    elif "velos_ratio_services_dom_moyenne" in donnees_normalisees.columns:
        donnees_normalisees["sous_indic_access"] = donnees_normalisees["velos_ratio_services_dom_moyenne"].replace(-1, np.nan)
    else:
        donnees_normalisees["sous_indic_access"] = np.nan
        print("[INFO] Aucun des indicateurs d’accessibilité disponibles.")

    # 5. Définir les colonnes finales à intégrer
    composantes_finales = [
        col for col in [
            'part_normalise_surface_couverte_stations_velos',
            'part_routes_interdites_pl',
            'part_routes_pavees',
            'pente_moyenne_absolue',
            'part_normalise_couverte_stations_rep_velos',
            "sous_indic_access"
        ] if col in donnees_normalisees.columns
    ]

    # 6. Calcul explicite de la moyenne ligne par ligne en excluant NaN et -1
    def moyenne_valeurs_valides(row):
        valeurs = [row[col] for col in composantes_finales if pd.notnull(row[col]) and row[col] != -1]
        return round(np.mean(valeurs), 3) if valeurs else -1

    donnees_normalisees["ind_compose_velos"] = donnees_normalisees.apply(moyenne_valeurs_valides, axis=1)

    # 7. Export
    exporter_geojson(donnees_normalisees, "maille_200m_avec_donnees_normalise", dossier=tableau_bord_dir)
    exporter_parquet(donnees_normalisees, "maille_200m_avec_donnees_normalise")

    # 8. Statistiques
    valides = donnees_normalisees[donnees_normalisees["ind_compose_velos"] >= 0]
    print(f"""
Indicateur composé de cyclabilité calculé avec succès :
- Nombre de carreaux : {len(donnees_normalisees)}
- Carreaux avec données valides : {len(valides)}
- Moyenne (valeurs valides) : {valides["ind_compose_velos"].mean():.3f}
""")

# Exécution
calcul_indicateurs_composes_velos()


# In[464]:


def afficher_indicateur_compose_cyclabilite(export = False):
    # 1. Chargement des données
    limites_epci = charger_fichier_parquet("limites_epci", crs=3857)
    carreaux = charger_fichier_parquet("maille_200m_avec_donnees_normalise", crs=3857)
    carreaux = carreaux[carreaux["ind_compose_velos"] >= 0]

    # 2. Affichage
    fig, ax = plt.subplots(figsize=(10, 10))
    limites_epci.plot(ax=ax, alpha=1, edgecolor='black', facecolor='none')

    carreaux.plot(column="ind_compose_velos",
             cmap="YlGn",
             legend=True,
             # legend_kwds={'label': "Indicateur de marchabilité par carreau"},
             ax=ax,
             linewidth=0.1,
             edgecolor="grey",
             vmin=0,
             vmax=1)

    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)

    ax.set_title("Indicateur de cyclabilité", fontsize=18)
    ax.axis('off')
    plt.tight_layout()

    # (Optionnel) Export PNG
    if export :
        images_export_dir = os.path.join(images_dir, "indicateur_compose_cyclabilite.png")
        plt.savefig(images_export_dir, dpi=300, bbox_inches='tight')
        print(f"Carte exportée vers : {images_export_dir}")

    plt.show()
    plt.close()

# Exécution
afficher_indicateur_compose_cyclabilite(export = True)


# #### 7.3. Bus / Indicateur de service des bus
# ---
# Indicateur de qualité de service des bus 'ind_compose_bus' à calculer à partir des indicateurs suivants :
# * 6.3.1. Temps d'accès effectif à un véhicule - 'att_moy_semaine_bus'
# * 6.3.2. Coût d'un trajet (€/h) - 'cout_horaire_bus' 
# * 6.3.3. Part des arrêts de bus disposant de parking relais - 'ratio_arrets_pr_bus'
# * 6.3.4. Part des arrêts de bus disposant de stations d'autopartage - 'ratio_arrets_autopartage_bus'
# * 6.3.5. Part des routes réservées aux bus - 'part_routes_bus'
# * 6.3.6. Part de surface bâtie à moins de 400m d'une station - 'part_bati_stations'
# * Sous-indicateur de mobilité :
#     * 6.3.7. Part des emplois accessibles en moins de T minutes - 'marche_bus_ratio_emploi'
#     * 6.3.8. Part des services accessibles en moins de T minutes. Moyenne sur S types et/ou catégories de services - 'marche_bus_ratio_services_dom_A' à 'marche_bus_ratio_services_dom_F'

# In[468]:


def calcul_indicateurs_composes_bus():
    # 1. Chargement des données normalisées
    donnees_normalisees = charger_fichier_geojson(
        "maille_200m_avec_donnees_normalise", crs=2154, dossier=tableau_bord_dir
    ).copy()

    # 2. Colonnes nécessaires
    colonnes = [
        'att_moy_semaine_bus',
        'cout_horaire_bus',
        'ratio_arrets_pr_bus',
        'ratio_arrets_autopartage_bus',
        'part_routes_bus',
        'part_batie_autour_arrets_bus',
        'marche_bus_ratio_emplois',
        'marche_bus_ratio_services_dom_moyenne',
    ]

    # 3. Vérifie les colonnes présentes
    colonnes_presentes = [col for col in colonnes if col in donnees_normalisees.columns]
    colonnes_absentes = sorted(set(colonnes) - set(colonnes_presentes))
    if colonnes_absentes:
        print("Colonnes manquantes (elles seront ignorées) :")
        for col in colonnes_absentes:
            print(f"  - {col}")

    # 4. Calcul du sous-indicateur d'accès
    if all(col in donnees_normalisees.columns for col in ['marche_bus_ratio_emplois', 'marche_bus_ratio_services_dom_moyenne']):
        donnees_normalisees["sous_indic_access"] = (
            (donnees_normalisees["marche_bus_ratio_emplois"].replace(-1, np.nan) +
             donnees_normalisees["marche_bus_ratio_services_dom_moyenne"].replace(-1, np.nan)) / 2
        )
    elif "marche_bus_ratio_emplois" in donnees_normalisees.columns:
        donnees_normalisees["sous_indic_access"] = donnees_normalisees["marche_bus_ratio_emplois"].replace(-1, np.nan)
    elif "marche_bus_ratio_services_dom_moyenne" in donnees_normalisees.columns:
        donnees_normalisees["sous_indic_access"] = donnees_normalisees["marche_bus_ratio_services_dom_moyenne"].replace(-1, np.nan)
    else:
        donnees_normalisees["sous_indic_access"] = np.nan
        print("[INFO] Aucun des indicateurs d’accessibilité disponibles.")

    # 5. Colonnes finales à intégrer
    composantes_finales = [
        col for col in [
            'att_moy_semaine_bus',
            'cout_horaire_bus',
            'ratio_arrets_pr_bus',
            'ratio_arrets_autopartage_bus',
            'part_routes_bus',
            'part_batie_autour_arrets_bus',
            "sous_indic_access"
        ] if col in donnees_normalisees.columns
    ]

    # 6. Moyenne ligne par ligne uniquement sur les valeurs valides
    def moyenne_valeurs_valides(row):
        valeurs = [row[col] for col in composantes_finales if pd.notnull(row[col]) and row[col] != -1]
        return round(np.mean(valeurs), 3) if valeurs else -1

    donnees_normalisees["ind_compose_bus"] = donnees_normalisees.apply(moyenne_valeurs_valides, axis=1)

    # 7. Export
    exporter_geojson(donnees_normalisees, "maille_200m_avec_donnees_normalise", dossier=tableau_bord_dir)
    exporter_parquet(donnees_normalisees, "maille_200m_avec_donnees_normalise")

    # 8. Statistiques
    valides = donnees_normalisees[donnees_normalisees["ind_compose_bus"] >= 0]
    print(f"""
Indicateur composé de qualité de service bus calculé avec succès :
- Nombre de carreaux : {len(donnees_normalisees)}
- Carreaux avec données valides : {len(valides)}
- Moyenne (valeurs valides) : {valides["ind_compose_bus"].mean():.3f}
""")

# Exécution
calcul_indicateurs_composes_bus()


# In[469]:


def afficher_indicateur_compose_qualite_bus(export = False):
    # 1. Chargement des données
    limites_epci = charger_fichier_parquet("limites_epci", crs=3857)
    carreaux = charger_fichier_parquet("maille_200m_avec_donnees_normalise", crs=3857)

    # 2. Affichage
    fig, ax = plt.subplots(figsize=(10, 10))
    limites_epci.plot(ax=ax, alpha=1, edgecolor='black', facecolor='none')

    carreaux.plot(column="ind_compose_bus",
             cmap="YlGn",
             legend=True,
             #legend_kwds={'label': "Indicateur de qualité de services pour les bus par carreau"},
             ax=ax,
             linewidth=0.1,
             edgecolor="grey",
             vmin=0,
             vmax=1)

    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)

    ax.set_title("Indicateur de qualité de services pour les bus", fontsize=18)
    ax.axis('off')
    plt.tight_layout()

    # (Optionnel) Export PNG
    if export :
        images_export_dir = os.path.join(images_dir, "indicateur_compose_bus.png")
        plt.savefig(images_export_dir, dpi=300, bbox_inches='tight')
        print(f"Carte exportée vers : {images_export_dir}")

    plt.show()
    plt.close()

# Exécution
afficher_indicateur_compose_qualite_bus(export = True)


# #### 7.4. Tram / Indicateur de service des trams
# ---

# In[470]:


def calcul_indicateurs_composes_tram():
    # 1. Chargement des données normalisées
    donnees_normalisees = charger_fichier_geojson(
        "maille_200m_avec_donnees_normalise", crs=2154, dossier=tableau_bord_dir
    ).copy()

    # 2. Colonnes nécessaires
    colonnes = [
        'att_moy_semaine_tram',
        'cout_horaire_tram',
        'ratio_arrets_pr_tram',
        'ratio_arrets_autopartage_tram',
        'part_routes_tram',
        'part_batie_autour_arrets_tram',
        'marche_tram_ratio_emplois',
        'marche_tram_ratio_services_dom_moyenne',
    ]

    # 3. Vérifie les colonnes présentes
    colonnes_presentes = [col for col in colonnes if col in donnees_normalisees.columns]
    colonnes_absentes = sorted(set(colonnes) - set(colonnes_presentes))
    if colonnes_absentes:
        print("Colonnes manquantes (elles seront ignorées) :")
        for col in colonnes_absentes:
            print(f"  - {col}")

    # 4. Calcul du sous-indicateur d'accès
    if all(col in donnees_normalisees.columns for col in ['marche_tram_ratio_emplois', 'marche_tram_ratio_services_dom_moyenne']):
        donnees_normalisees["sous_indic_access"] = (
            (donnees_normalisees["marche_tram_ratio_emplois"].replace(-1, np.nan) +
             donnees_normalisees["marche_tram_ratio_services_dom_moyenne"].replace(-1, np.nan)) / 2
        )
    elif "marche_tram_ratio_emplois" in donnees_normalisees.columns:
        donnees_normalisees["sous_indic_access"] = donnees_normalisees["marche_tram_ratio_emplois"].replace(-1, np.nan)
    elif "marche_tram_ratio_services_dom_moyenne" in donnees_normalisees.columns:
        donnees_normalisees["sous_indic_access"] = donnees_normalisees["marche_tram_ratio_services_dom_moyenne"].replace(-1, np.nan)
    else:
        donnees_normalisees["sous_indic_access"] = np.nan
        print("[INFO] Aucun des indicateurs d’accessibilité disponibles.")

    # 5. Colonnes finales
    composantes_finales = [
        col for col in [
            'att_moy_semaine_tram',
            'cout_horaire_tram',
            'ratio_arrets_pr_tram',
            'ratio_arrets_autopartage_tram',
            'part_routes_tram',
            'part_batie_autour_arrets_tram',
            "sous_indic_access"
        ] if col in donnees_normalisees.columns
    ]

    # 6. Moyenne ligne par ligne avec filtrage -1 / NaN
    def moyenne_valeurs_valides(row):
        valeurs = [row[col] for col in composantes_finales if pd.notnull(row[col]) and row[col] != -1]
        return round(np.mean(valeurs), 3) if valeurs else -1

    donnees_normalisees["ind_compose_tram"] = donnees_normalisees.apply(moyenne_valeurs_valides, axis=1)

    # 7. Export
    exporter_geojson(donnees_normalisees, "maille_200m_avec_donnees_normalise", dossier=tableau_bord_dir)
    exporter_parquet(donnees_normalisees, "maille_200m_avec_donnees_normalise")

    # 8. Statistiques
    valides = donnees_normalisees[donnees_normalisees["ind_compose_tram"] >= 0]
    print(f"""
Indicateur composé de qualité de service tram calculé avec succès :
- Nombre de carreaux : {len(donnees_normalisees)}
- Carreaux avec données valides : {len(valides)}
- Moyenne (valeurs valides) : {valides["ind_compose_tram"].mean():.3f}
""")

# Exécution
calcul_indicateurs_composes_tram()


# In[266]:


# ANCIENNE VERSION, laissée commme référence si besoin de changer
def calcul_indicateurs_composes_tram():
    # 1. Chargement des données normalisées
    donnees_normalisees = charger_fichier_geojson(
        "maille_200m_avec_donnees_normalise", crs=2154, dossier=tableau_bord_dir
    ).copy()

    # 2. Colonnes nécessaires
    colonnes = [
        'att_moy_semaine_tram', # Qualité de service des tram
        'cout_horaire_tram',
        'ratio_arrets_pr_tram',
        'ratio_arrets_autopartage_tram',
        'part_routes_tram',
        'part_batie_autour_arrets_tram',
        'marche_tram_ratio_emplois',
        'marche_tram_ratio_services_dom_moyenne',
    ]

    # 3. Vérifie les colonnes présentes
    colonnes_presentes = [col for col in colonnes if col in donnees_normalisees.columns]
    colonnes_absentes = sorted(set(colonnes) - set(colonnes_presentes))
    if colonnes_absentes:
        print("Colonnes manquantes (elles seront ignorées) :")
        for col in colonnes_absentes:
            print(f"  - {col}")

    # 4. Calcul du sous-indicateur d'accès (si possible)
    if all(col in donnees_normalisees.columns for col in ['marche_tram_ratio_emplois', 'marche_tram_ratio_services_dom_moyenne']):
        donnees_normalisees["sous_indic_access"] = (
            (donnees_normalisees["marche_tram_ratio_emplois"].replace(-1, np.nan) +
             donnees_normalisees["marche_tram_ratio_services_dom_moyenne"]) / 2
        )
    elif "marche_tram_ratio_emplois" in donnees_normalisees.columns:
        donnees_normalisees["sous_indic_access"] = donnees_normalisees["marche_tram_ratio_emplois"].replace(-1, np.nan)
    elif "marche_tram_ratio_services_dom_moyenne" in donnees_normalisees.columns:
        donnees_normalisees["sous_indic_access"] = donnees_normalisees["marche_tram_ratio_services_dom_moyenne"].replace(-1, np.nan)
    else:
        donnees_normalisees["sous_indic_access"] = np.nan
        print("[INFO] Aucun des indicateurs d’accessibilité disponibles.")

    # 5. Liste finale des colonnes disponibles pour l'indicateur
    composantes_finales = [
        col for col in [
            'part_surface_pietonne',
            'part_vegetation',
            'ratio_hl_bati_rues',
            'part_routes_lentes',
            'part_routes_inaccessibles_vl',
            'part_parcs',
            "sous_indic_access"
        ] if col in donnees_normalisees.columns
    ]

    # 6. Calcul de l'indicateur final
    donnees_normalisees["ind_compose_tram"] = (
        donnees_normalisees[composantes_finales]
        .replace(-1, np.nan)
        .mean(axis=1)
        .round(3)
    ).fillna(-1)

    # 7. Export
    exporter_geojson(donnees_normalisees, "maille_200m_avec_donnees_normalise", dossier=tableau_bord_dir)
    exporter_parquet(donnees_normalisees, "maille_200m_avec_donnees_normalise")

    # 8. Statistiques
    valides = donnees_normalisees[donnees_normalisees["ind_compose_tram"] >= 0]
    print(f"""
Indicateur composé de marchabilité calculé avec succès :
- Nombre de carreaux : {len(donnees_normalisees)}
- Carreaux avec données valides : {len(valides)}
- Moyenne (valeurs valides) : {valides["ind_compose_tram"].mean():.3f}
""")

# Exécution
calcul_indicateurs_composes_tram()


# In[474]:


def afficher_indicateur_compose_qualite_tram(export = False):
    # 1. Chargement des données
    limites_epci = charger_fichier_parquet("limites_epci", crs=3857)
    carreaux = charger_fichier_parquet("maille_200m_avec_donnees_normalise", crs=3857)

    # 2. Affichage
    fig, ax = plt.subplots(figsize=(10, 10))
    limites_epci.plot(ax=ax, alpha=1, edgecolor='black', facecolor='none')

    carreaux.plot(column="ind_compose_tram",
             cmap="YlGn",
             legend=True,
             #legend_kwds={'label': "Indicateur de qualité de services pour les tram par carreau"},
             ax=ax,
             linewidth=0.1,
             edgecolor="grey",
             vmin=0,
             vmax=1)

    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)

    ax.set_title("Indicateur de qualité de services pour les tram", fontsize=18)
    ax.axis('off')
    plt.tight_layout()

    # (Optionnel) Export PNG
    if export :
        images_export_dir = os.path.join(images_dir, "indicateur_compose_tram.png")
        plt.savefig(images_export_dir, dpi=300, bbox_inches='tight')
        print(f"Carte exportée vers : {images_export_dir}")

    plt.show()
    plt.close()

# Exécution
afficher_indicateur_compose_qualite_tram(export = True)


# #### 7.5. Véhicules légers / Indicateur de service des VL
# ---

# In[472]:


def calcul_indicateurs_composes_vl():
    # 1. Chargement des données normalisées
    donnees_normalisees = charger_fichier_geojson(
        "maille_200m_avec_donnees_normalise", crs=2154, dossier=tableau_bord_dir
    ).copy()

    # 2. Colonnes nécessaires
    colonnes = [
        'part_sens_unique',
        'part_zfe',
        'nb_feux_circulation',
        'places_stationnement_par_habitant',
        'cout_moyen_stationnement',
        'bornes_ve_par_habitant',
        'vl_ratio_emplois',
        'vl_ratio_services_dom_moyenne',
    ]

    # 3. Vérification des colonnes présentes
    colonnes_presentes = [col for col in colonnes if col in donnees_normalisees.columns]
    colonnes_absentes = sorted(set(colonnes) - set(colonnes_presentes))
    if colonnes_absentes:
        print("Colonnes manquantes (elles seront ignorées) :")
        for col in colonnes_absentes:
            print(f"  - {col}")

    # 4. Calcul du sous-indicateur d'accès
    if all(c in donnees_normalisees.columns for c in ['vl_ratio_emplois', 'vl_ratio_services_dom_moyenne']):
        donnees_normalisees["sous_indic_access"] = (
            (donnees_normalisees["vl_ratio_emplois"].replace(-1, np.nan) +
             donnees_normalisees["vl_ratio_services_dom_moyenne"].replace(-1, np.nan)) / 2
        )
    elif "vl_ratio_emplois" in donnees_normalisees.columns:
        donnees_normalisees["sous_indic_access"] = donnees_normalisees["vl_ratio_emplois"].replace(-1, np.nan)
    elif "vl_ratio_services_dom_moyenne" in donnees_normalisees.columns:
        donnees_normalisees["sous_indic_access"] = donnees_normalisees["vl_ratio_services_dom_moyenne"].replace(-1, np.nan)
    else:
        donnees_normalisees["sous_indic_access"] = np.nan
        print("[INFO] Aucun des indicateurs d’accessibilité disponibles.")

    # 5. Liste des colonnes utilisées
    composantes_finales = [
        col for col in [
            'part_sens_unique',
            'part_zfe',
            'nb_feux_circulation',
            'places_stationnement_par_habitant',
            'cout_moyen_stationnement',
            'bornes_ve_par_habitant',
            'sous_indic_access'
        ] if col in donnees_normalisees.columns
    ]

    # 6. Moyenne ligne par ligne
    def moyenne_valeurs_valides(row):
        valeurs = [row[c] for c in composantes_finales if pd.notnull(row[c]) and row[c] != -1]
        return round(np.mean(valeurs), 3) if valeurs else -1

    donnees_normalisees["ind_compose_vl"] = donnees_normalisees.apply(moyenne_valeurs_valides, axis=1)

    # 7. Export
    exporter_geojson(donnees_normalisees, "maille_200m_avec_donnees_normalise", dossier=tableau_bord_dir)
    exporter_parquet(donnees_normalisees, "maille_200m_avec_donnees_normalise")

    # 8. Statistiques
    valides = donnees_normalisees[donnees_normalisees["ind_compose_vl"] >= 0]
    print(f"""
Indicateur composé VL calculé avec succès :
- Nombre de carreaux : {len(donnees_normalisees)}
- Carreaux avec données valides : {len(valides)}
- Moyenne (valeurs valides) : {valides["ind_compose_vl"].mean():.3f}
""")

# Exécution
calcul_indicateurs_composes_vl()


# In[273]:


def calcul_indicateurs_composes_vl():
    # 1. Chargement des données normalisées
    donnees_normalisees = charger_fichier_geojson(
        "maille_200m_avec_donnees_normalise", crs=2154, dossier=tableau_bord_dir
    ).copy()

    # 2. Colonnes nécessaires
    colonnes = [
        'part_sens_unique', # Qualité de service pour les VL
        'part_zfe',
        'nb_feux_circulation',
        # 'nb_stations_service', (difficile à normaliser, non utilisé pour l'instant)
        'places_stationnement_par_habitant',
        'cout_moyen_stationnement',
        # 'duree_moyenne_stationnement' (difficile à normaliser, non utilisé pour l'instant)
        'bornes_ve_par_habitant',
        'vl_ratio_emplois',
        'vl_ratio_services_dom_moyenne',
    ]

    # 3. Vérifie les colonnes présentes
    colonnes_presentes = [col for col in colonnes if col in donnees_normalisees.columns]
    colonnes_absentes = sorted(set(colonnes) - set(colonnes_presentes))
    if colonnes_absentes:
        print("Colonnes manquantes (elles seront ignorées) :")
        for col in colonnes_absentes:
            print(f"  - {col}")

    # 4. Calcul du sous-indicateur d'accès (si possible)
    if all(col in donnees_normalisees.columns for col in ['vl_ratio_emplois', 'vl_ratio_services_dom_moyenne']):
        donnees_normalisees["sous_indic_access"] = (
            (donnees_normalisees["vl_ratio_emplois"].replace(-1, np.nan) +
             donnees_normalisees["vl_ratio_services_dom_moyenne"]) / 2
        )
    elif "vl_ratio_emplois" in donnees_normalisees.columns:
        donnees_normalisees["sous_indic_access"] = donnees_normalisees["vl_ratio_emplois"].replace(-1, np.nan)
    elif "vl_ratio_services_dom_moyenne" in donnees_normalisees.columns:
        donnees_normalisees["sous_indic_access"] = donnees_normalisees["vl_ratio_services_dom_moyenne"].replace(-1, np.nan)
    else:
        donnees_normalisees["sous_indic_access"] = np.nan
        print("[INFO] Aucun des indicateurs d’accessibilité disponibles.")

    # 5. Liste finale des colonnes disponibles pour l'indicateur
    composantes_finales = [
        col for col in [
            'part_sens_unique', # Qualité de service pour les VL
            'part_zfe',
            'nb_feux_circulation',
            # 'nb_stations_service', (difficile à normaliser, non utilisé pour l'instant)
            'places_stationnement_par_habitant',
            'cout_moyen_stationnement',
            # 'duree_moyenne_stationnement' (difficile à normaliser, non utilisé pour l'instant)
            'bornes_ve_par_habitant',
            "sous_indic_access"
        ] if col in donnees_normalisees.columns
    ]

    # 6. Calcul de l'indicateur final
    donnees_normalisees["ind_compose_vl"] = (
        donnees_normalisees[composantes_finales]
        .replace(-1, np.nan)
        .mean(axis=1)
        .round(3)
    ).fillna(-1)

    # 7. Export
    exporter_geojson(donnees_normalisees, "maille_200m_avec_donnees_normalise", dossier=tableau_bord_dir)
    exporter_parquet(donnees_normalisees, "maille_200m_avec_donnees_normalise")

    # 8. Statistiques
    valides = donnees_normalisees[donnees_normalisees["ind_compose_vl"] >= 0]
    print(f"""
Indicateur composé de marchabilité calculé avec succès :
- Nombre de carreaux : {len(donnees_normalisees)}
- Carreaux avec données valides : {len(valides)}
- Moyenne (valeurs valides) : {valides["ind_compose_vl"].mean():.3f}
""")

# Exécution
calcul_indicateurs_composes_vl()


# In[473]:


def afficher_indicateur_compose_qualite_vl(export = False):
    # 1. Chargement des données
    limites_epci = charger_fichier_parquet("limites_epci", crs=3857)
    carreaux = charger_fichier_parquet("maille_200m_avec_donnees_normalise", crs=3857)

    # 2. Affichage
    fig, ax = plt.subplots(figsize=(10, 10))
    limites_epci.plot(ax=ax, alpha=1, edgecolor='black', facecolor='none')

    carreaux.plot(column="ind_compose_vl",
             cmap="YlGn",
             legend=True,
             legend_kwds={'label': "Indicateur de qualité de services pour les vl par carreau"},
             ax=ax,
             linewidth=0.1,
             edgecolor="grey",
             vmin=0,
             vmax=1)

    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)

    ax.set_title("Indicateur de qualité de services pour les vl par carreau")
    ax.axis('off')
    plt.tight_layout()

    # (Optionnel) Export PNG
    if export :
        images_export_dir = os.path.join(images_dir, "indicateur_compose_vl.png")
        plt.savefig(images_export_dir, dpi=300, bbox_inches='tight')
        print(f"Carte exportée vers : {images_export_dir}")

    plt.show()
    plt.close()

# Exécution
afficher_indicateur_compose_qualite_vl(export = True)


# #### 7.6. Autopartage / Indicateur de service des véhicules en autopartage
# ---

# In[475]:


def calcul_indicateurs_composes_autopartage():
    # 1. Chargement des données normalisées
    donnees_normalisees = charger_fichier_geojson(
        "maille_200m_avec_donnees_normalise", crs=2154, dossier=tableau_bord_dir
    ).copy()

    # 2. Colonnes nécessaires
    colonnes = [
        'part_normalise_couverte_autopartage',
        'autopartage_cout_rapport_revenu',
        'part_routes_accessibles_vl',
        'part_surface_geofencing_citiz',
        'autopartage_ratio_emplois',
        'autopartage_ratio_services_dom_moyenne',
    ]

    # 3. Vérification des colonnes disponibles
    colonnes_presentes = [col for col in colonnes if col in donnees_normalisees.columns]
    colonnes_absentes = sorted(set(colonnes) - set(colonnes_presentes))
    if colonnes_absentes:
        print("Colonnes manquantes (elles seront ignorées) :")
        for col in colonnes_absentes:
            print(f"  - {col}")

    # 4. Calcul du sous-indicateur d'accès
    if all(c in donnees_normalisees.columns for c in ['autopartage_ratio_emplois', 'autopartage_ratio_services_dom_moyenne']):
        donnees_normalisees["sous_indic_access"] = (
            (donnees_normalisees["autopartage_ratio_emplois"].replace(-1, np.nan) +
             donnees_normalisees["autopartage_ratio_services_dom_moyenne"].replace(-1, np.nan)) / 2
        )
    elif "autopartage_ratio_emplois" in donnees_normalisees.columns:
        donnees_normalisees["sous_indic_access"] = donnees_normalisees["autopartage_ratio_emplois"].replace(-1, np.nan)
    elif "autopartage_ratio_services_dom_moyenne" in donnees_normalisees.columns:
        donnees_normalisees["sous_indic_access"] = donnees_normalisees["autopartage_ratio_services_dom_moyenne"].replace(-1, np.nan)
    else:
        donnees_normalisees["sous_indic_access"] = np.nan
        print("[INFO] Aucun des indicateurs d’accessibilité disponibles.")

    # 5. Liste finale des composantes disponibles
    composantes_finales = [
        col for col in [
            'part_normalise_couverte_autopartage',
            'autopartage_cout_rapport_revenu',
            'part_routes_accessibles_vl',
            'part_surface_geofencing_citiz',
            'sous_indic_access'
        ] if col in donnees_normalisees.columns
    ]

    # 6. Calcul ligne par ligne de la moyenne
    def moyenne_valeurs_valides(row):
        valeurs = [row[c] for c in composantes_finales if pd.notnull(row[c]) and row[c] != -1]
        return round(np.mean(valeurs), 3) if valeurs else -1

    donnees_normalisees["ind_compose_autopartage"] = donnees_normalisees.apply(moyenne_valeurs_valides, axis=1)

    # 7. Export
    exporter_geojson(donnees_normalisees, "maille_200m_avec_donnees_normalise", dossier=tableau_bord_dir)
    exporter_parquet(donnees_normalisees, "maille_200m_avec_donnees_normalise")

    # 8. Statistiques
    valides = donnees_normalisees[donnees_normalisees["ind_compose_autopartage"] >= 0]
    print(f"""
Indicateur composé autopartage calculé avec succès :
- Nombre de carreaux : {len(donnees_normalisees)}
- Carreaux avec données valides : {len(valides)}
- Moyenne (valeurs valides) : {valides["ind_compose_autopartage"].mean():.3f}
""")

# Exécution
calcul_indicateurs_composes_autopartage()


# In[275]:


def calcul_indicateurs_composes_autopartage():
    # 1. Chargement des données normalisées
    donnees_normalisees = charger_fichier_geojson(
        "maille_200m_avec_donnees_normalise", crs=2154, dossier=tableau_bord_dir
    ).copy()

    # 2. Colonnes nécessaires
    colonnes = [
        'part_normalise_couverte_autopartage', # Qualité de service de l'autopartage
        # 'nb_voitures_autopartage' (difficile à normaliser, non utilisé pour l'instant)
        'autopartage_cout_rapport_revenu',
        'part_routes_accessibles_vl',
        'part_surface_geofencing_citiz',
        'autopartage_ratio_emplois',
        'autopartage_ratio_services_dom_moyenne',
    ]

    # 3. Vérifie les colonnes présentes
    colonnes_presentes = [col for col in colonnes if col in donnees_normalisees.columns]
    colonnes_absentes = sorted(set(colonnes) - set(colonnes_presentes))
    if colonnes_absentes:
        print("Colonnes manquantes (elles seront ignorées) :")
        for col in colonnes_absentes:
            print(f"  - {col}")

    # 4. Calcul du sous-indicateur d'accès (si possible)
    if all(col in donnees_normalisees.columns for col in ['autopartage_ratio_emplois', 'autopartage_ratio_services_dom_moyenne']):
        donnees_normalisees["sous_indic_access"] = (
            (donnees_normalisees["autopartage_ratio_emplois"].replace(-1, np.nan) +
             donnees_normalisees["autopartage_ratio_services_dom_moyenne"]) / 2
        )
    elif "autopartage_ratio_emplois" in donnees_normalisees.columns:
        donnees_normalisees["sous_indic_access"] = donnees_normalisees["autopartage_ratio_emplois"].replace(-1, np.nan)
    elif "autopartage_ratio_services_dom_moyenne" in donnees_normalisees.columns:
        donnees_normalisees["sous_indic_access"] = donnees_normalisees["autopartage_ratio_services_dom_moyenne"].replace(-1, np.nan)
    else:
        donnees_normalisees["sous_indic_access"] = np.nan
        print("[INFO] Aucun des indicateurs d’accessibilité disponibles.")

    # 5. Liste finale des colonnes disponibles pour l'indicateur
    composantes_finales = [
        col for col in [
            'part_normalise_couverte_autopartage', # Qualité de service de l'autopartage
            # 'nb_voitures_autopartage' (difficile à normaliser, non utilisé pour l'instant)
            'autopartage_cout_rapport_revenu',
            'part_routes_accessibles_vl',
            'part_surface_geofencing_citiz',
            "sous_indic_access"
        ] if col in donnees_normalisees.columns
    ]

    # 6. Calcul de l'indicateur final
    donnees_normalisees["ind_compose_autopartage"] = (
        donnees_normalisees[composantes_finales]
        .replace(-1, np.nan)
        .mean(axis=1)
        .round(3)
    ).fillna(-1)

    # 7. Export
    exporter_geojson(donnees_normalisees, "maille_200m_avec_donnees_normalise", dossier=tableau_bord_dir)
    exporter_parquet(donnees_normalisees, "maille_200m_avec_donnees_normalise")

    # 8. Statistiques
    valides = donnees_normalisees[donnees_normalisees["ind_compose_autopartage"] >= 0]
    print(f"""
Indicateur composé de marchabilité calculé avec succès :
- Nombre de carreaux : {len(donnees_normalisees)}
- Carreaux avec données valides : {len(valides)}
- Moyenne (valeurs valides) : {valides["ind_compose_autopartage"].mean():.3f}
""")

# Exécution
calcul_indicateurs_composes_autopartage()


# In[477]:


def afficher_indicateur_compose_qualite_autopartage(export = False):
    # 1. Chargement des données
    limites_epci = charger_fichier_parquet("limites_epci", crs=3857)
    carreaux = charger_fichier_parquet("maille_200m_avec_donnees_normalise", crs=3857)

    # 2. Affichage
    fig, ax = plt.subplots(figsize=(10, 10))
    limites_epci.plot(ax=ax, alpha=1, edgecolor='black', facecolor='none')

    carreaux.plot(column="ind_compose_autopartage",
             cmap="YlGn",
             legend=True,
             #legend_kwds={'label': "Indicateur de qualité de services pour les véhicules en autopartage par carreau"},
             ax=ax,
             linewidth=0.1,
             edgecolor="grey",
             vmin=0,
             vmax=1)

    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)

    ax.set_title("Indicateur de qualité de service pour \nles véhicules en autopartage", fontsize=18)
    ax.axis('off')
    plt.tight_layout()

    # (Optionnel) Export PNG
    if export :
        images_export_dir = os.path.join(images_dir, "indicateur_compose_autopartage.png")
        plt.savefig(images_export_dir, dpi=300, bbox_inches='tight')
        print(f"Carte exportée vers : {images_export_dir}")

    plt.show()
    plt.close()

# Exécution
afficher_indicateur_compose_qualite_autopartage(export = True)


# #### 7.7. Diversité fonctionnelle
# ---
# Indicateur de diversité fonctionnelle : 
# * 6.7.1. 6.7.1. HSMI (Housing - Service Mix Index) / Indice de mixité logement - services - 'HSMI_global'
# * 6.7.2. HEMI (Housing - Employment Mix Index) / Indice de mixité logement - emplois - 'HEMI' 

# In[479]:


def calcul_indicateur_diversite_fonctionnelle():
    # 1. Chargement des données
    donnees = charger_fichier_geojson(
        "maille_200m_avec_donnees_normalise", crs=2154, dossier=tableau_bord_dir
    ).copy()

    # 2. Colonnes attendues
    colonnes_diversite = ["HSMI_global", "HEMI"]

    # 3. Vérification de la présence des colonnes
    colonnes_absentes = [col for col in colonnes_diversite if col not in donnees.columns]
    colonnes_presentes = [col for col in colonnes_diversite if col in donnees.columns]

    if colonnes_absentes:
        print(f"[INFO] Colonnes manquantes (elles seront ignorées) : {colonnes_absentes}")

    # 4. Calcul de l'indicateur de diversité fonctionnelle
    donnees["ind_diversite_fonctionnelle"] = (
        donnees[colonnes_presentes]
        .replace(-1, np.nan)
        .mean(axis=1)
        .round(3)
    )
    donnees["ind_diversite_fonctionnelle"] = donnees["ind_diversite_fonctionnelle"].fillna(-1)

    # 5. Export
    exporter_geojson(donnees, "maille_200m_avec_donnees_normalise", dossier=tableau_bord_dir)
    exporter_parquet(donnees, "maille_200m_avec_donnees_normalise")

    # 6. Statistiques
    valides = donnees[donnees["ind_diversite_fonctionnelle"] >= 0]
    print(f"""
Indicateur de diversité fonctionnelle calculé :
- Nombre de carreaux : {len(donnees)}
- Carreaux valides : {len(valides)}
- Moyenne (valeurs valides) : {valides["ind_diversite_fonctionnelle"].mean():.3f}
""")

# Exécution
calcul_indicateur_diversite_fonctionnelle()


# In[481]:


def afficher_indicateur_diversite_fonctionnelle(export = False):
    # 1. Chargement des données
    limites_epci = charger_fichier_parquet("limites_epci", crs=3857)
    carreaux = charger_fichier_parquet("maille_200m_avec_donnees_normalise", crs=3857)

    # 2. Affichage
    fig, ax = plt.subplots(figsize=(10, 10))
    limites_epci.plot(ax=ax, alpha=1, edgecolor='black', facecolor='none')

    carreaux.plot(column="ind_diversite_fonctionnelle",
             cmap="YlGn",
             legend=True,
             #legend_kwds={'label': "Indicateur de diversité fonctionnelle (HSMI + HEMI) / 2 par carreau"},
             ax=ax,
             linewidth=0.1,
             edgecolor="grey",
             vmin=0,
             vmax=1)

    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)

    ax.set_title("Indicateur de diversité fonctionnelle par carreau", fontsize = 18)
    ax.axis('off')
    plt.tight_layout()

    # (Optionnel) Export PNG
    if export :
        images_export_dir = os.path.join(images_dir, "indicateur_compose_diversite_fonctionnelle.png")
        plt.savefig(images_export_dir, dpi=300, bbox_inches='tight')
        print(f"Carte exportée vers : {images_export_dir}")

    plt.show()
    plt.close()

# Exécution
afficher_indicateur_diversite_fonctionnelle(export = True)


# #### 7.8. Schémas de mobilité
# ---
# Indicateur des schémas de mobilité :
# * 6.8.1. Part des navetteurs utilisant principalement la marche - 'part_marche'
# * 6.8.2. Part des navetteurs utilisant principalement le vélo - 'part_velos'
# * 6.8.3. Part des navetteurs utilisant principalement les transports en commun - 'part_tcom'
# * 6.8.4. Part des navetteurs utilisant principalement des véhicules légers - 'part_vl'
# 
# A la différence des autres, cet indicateur est égal à 1 si aucun trajet n'est réalisé en VL dans la maille. Il est à 0 si tout les trajets des navetteurs se font en VL.

# In[483]:


def calcul_indicateur_schemas_mobilite():
    # 1. Chargement des données
    donnees = charger_fichier_geojson(
        "maille_200m_avec_donnees_normalise", crs=2154, dossier=tableau_bord_dir
    ).copy()

    # 2. Vérification de la colonne 'part_vl'
    if "part_vl" not in donnees.columns:
        print("[ERREUR] La colonne 'part_vl' est absente du fichier.")
        return

    # 3. Calcul de l'indicateur (1 - part_vl), avec gestion des valeurs -1
    donnees["ind_schema_mobilite"] = (
        1 - donnees["part_vl"].replace(-1, np.nan)
    ).round(3)
    donnees["ind_schema_mobilite"] = donnees["ind_schema_mobilite"].fillna(-1)

    # 4. Export
    exporter_geojson(donnees, "maille_200m_avec_donnees_normalise", dossier=tableau_bord_dir)
    exporter_parquet(donnees, "maille_200m_avec_donnees_normalise")

    # 5. Statistiques
    valides = donnees[donnees["ind_schema_mobilite"] >= 0]
    print(f"""
Indicateur des schémas de mobilité calculé :
- Nombre de carreaux : {len(donnees)}
- Carreaux valides : {len(valides)}
- Moyenne (valeurs valides) : {valides["ind_schema_mobilite"].mean():.3f}
""")

# Exécution
calcul_indicateur_schemas_mobilite()


# In[484]:


def afficher_indicateur_schemas_mobilite(export = False):
    # 1. Chargement des données
    limites_epci = charger_fichier_parquet("limites_epci", crs=3857)
    carreaux = charger_fichier_parquet("maille_200m_avec_donnees_normalise", crs=3857)

    # 2. Affichage
    fig, ax = plt.subplots(figsize=(10, 10))
    limites_epci.plot(ax=ax, alpha=1, edgecolor='black', facecolor='none')

    carreaux.plot(column="ind_schema_mobilite",
             cmap="YlGn",
             legend=True,
             #legend_kwds={'label': "Indicateur des schémas de mobilité (% des trajets réalisés\nà pied, en vélos ou via les transports en commun)"},
             ax=ax,
             linewidth=0.1,
             edgecolor="grey",
             vmin=0,
             vmax=1)

    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)

    ax.set_title("Indicateur des schémas de mobilité (% des trajets réalisés\nà pied, en vélos ou via les transports en commun)", fontsize=18)
    ax.axis('off')
    plt.tight_layout()

    # (Optionnel) Export PNG
    if export :
        images_export_dir = os.path.join(images_dir, "indicateur_compose_schema_mobilite.png")
        plt.savefig(images_export_dir, dpi=300, bbox_inches='tight')
        print(f"Carte exportée vers : {images_export_dir}")

    plt.show()
    plt.close()

# Exécution
afficher_indicateur_schemas_mobilite(export = True)


# In[ ]:




