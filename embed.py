import os, uuid, zipfile, pathlib, shutil
from lxml import etree
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding
from tqdm.auto import tqdm
import torch

# ==========================================
# 1. KONFIGURACJA (POBIERANA Z GITHUB SECRETS)
# ==========================================
QDRANT_URL     = os.environ.get("QDRANT_URL")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
COLLECTION     = "gesetze"

ZIP_FOLDER = pathlib.Path("zips") 
ZIP_FILES = list(ZIP_FOLDER.glob("*.zip")))

if not QDRANT_URL or not QDRANT_API_KEY:
    print("❌ BŁĄD: Brak zmiennych środowiskowych QDRANT_URL lub QDRANT_API_KEY!")
    exit(1)

# ==========================================
# 2. LOGIKA DZIAŁANIA
# ==========================================

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# Detekcja urządzenia (GitHub Actions zawsze wybierze 'cpu')
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️ Urządzenie obliczeniowe: {device.upper()}")

# Modele (Identyczne jak w Colab, aby wektory pasowały)
print("⏳ Ładowanie modeli (Dense i Sparse)...")
dense_model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1", device=device)
sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")

# Przetwarzanie każdego znalezionego ZIPa
for zip_path in ZIP_FILES:
    ZIP_NAME = zip_path.name
    extract_path = f"./temp_{ZIP_NAME}"
    
    if os.path.exists(extract_path):
        shutil.rmtree(extract_path)
    
    with zipfile.ZipFile(ZIP_NAME, 'r') as z:
        z.extractall(extract_path)
    print(f"📦 Rozpakowano {ZIP_NAME}")

    all_files = list(pathlib.Path(extract_path).rglob("*.xml"))
    print(f"🚀 Startujemy z wektoryzacją {len(all_files)} plików z {ZIP_NAME}...")

    for file_path in tqdm(all_files, desc=f"Przetwarzanie {ZIP_NAME}"):
        try:
            tree = etree.parse(str(file_path), etree.XMLParser(recover=True))
            jurabk = (tree.xpath('//jurabk/text()') or [file_path.name])[0]
            
            norms = []
            for norm in tree.xpath('//norm'):
                p_text = ' '.join(norm.xpath('.//textdaten//Content//P//text()')).strip()
                if len(p_text) >= 5:
                    norms.append({
                        "text": p_text[:2500], 
                        "unit": (norm.xpath('.//enbez/text()') or [''])[0]
                    })
            
            if norms:
                texts = [n["text"] for n in norms]
                # Dense (na CPU będzie wolniej, ale matematycznie tak samo)
                dv = dense_model.encode(texts, show_progress_bar=False)
                # Sparse (zawsze na CPU)
                sv = list(sparse_model.embed(texts))
                
                points = []
                for i, n in enumerate(norms):
                    point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{jurabk}_{i}_{file_path.name}"))
                    points.append(models.PointStruct(
                        id=point_id,
                        vector={
                            "dense": dv[i].tolist(),
                            "sparse": models.SparseVector(
                                indices=sv[i].indices.tolist(), 
                                values=sv[i].values.tolist()
                            )
                        },
                        payload={"text": n["text"], "jurabk": jurabk, "unit": n["unit"]}
                    ))
                
                client.upsert(collection_name=COLLECTION, points=points, wait=True)
                
        except Exception as e:
            print(f"⚠️ Błąd w pliku {file_path.name}: {e}")
    
    # Sprzątanie po przetworzeniu ZIPa
    shutil.rmtree(extract_path)
    print(f"✅ Ukończono paczkę: {ZIP_NAME}")

print("\n🏁 KONIEC! Wszystkie ZIPy przetworzone.")
