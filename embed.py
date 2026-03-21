import os
import json
import uuid
from lxml import etree
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding, SparseTextEmbedding

# ==========================================
# KONFIGURACJA
# ==========================================
SOURCE_DIR     = "./xml/"   # <-- zmienisz jak chcesz
COLLECTION     = "gesetze"
INDEX_REGISTRY = "./indexed_files.json"

DENSE_MODEL    = "mixedbread-ai/mxbai-embed-large-v1"
SPARSE_MODEL   = "Qdrant/bm25"
CACHE_DIR      = "./models_cache"

DENSE_DIM      = 1024
BATCH_SIZE     = 10
# ==========================================

# Ustaw cache dla fastembed
os.environ["FASTEMBED_CACHE_PATH"] = CACHE_DIR

# Połącz z Qdrant (TY dopiszesz URL i KEY)
client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

# ------------------------------------------
# Rejestr plików
# ------------------------------------------
def load_registry():
    if os.path.exists(INDEX_REGISTRY):
        try:
            with open(INDEX_REGISTRY, 'r') as f:
                return json.load(f)
        except:
            return []
    return []

def save_to_registry(file_name):
    indexed = load_registry()
    if file_name not in indexed:
        indexed.append(file_name)
        with open(INDEX_REGISTRY, 'w') as f:
            json.dump(indexed, f, indent=4)

# ------------------------------------------
# Tworzenie kolekcji (jeśli nie istnieje)
# ------------------------------------------
if not client.collection_exists(COLLECTION):
    print(f"🆕 Tworzę kolekcję '{COLLECTION}'...")
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config={
            "dense": models.VectorParams(
                size=DENSE_DIM,
                distance=models.Distance.COSINE
            )
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams()
        }
    )

# ------------------------------------------
# Ładowanie modeli
# ------------------------------------------
print("🔄 Ładuję modele...")
dense_model = TextEmbedding(model_name=DENSE_MODEL, cache_dir=CACHE_DIR)
sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL, cache_dir=CACHE_DIR)

# ------------------------------------------
# Lista plików
# ------------------------------------------
indexed_files = load_registry()
all_files = sorted([f for f in os.listdir(SOURCE_DIR) if f.endswith('.xml')])
total_files = len(all_files)

print(f"📂 Znaleziono {total_files} plików XML.")

# ------------------------------------------
# Główna pętla
# ------------------------------------------
for idx, file_name in enumerate(all_files):

    if file_name in indexed_files:
        continue

    file_path = os.path.join(SOURCE_DIR, file_name)

    try:
        parser = etree.XMLParser(recover=True)
        tree = etree.parse(file_path, parser)

        # Metadane
        jurabk = tree.xpath('//jurabk/text()')
        jurabk = jurabk[0] if jurabk else file_name

        title = tree.xpath('//titel/text()')
        title = title[0] if title else "Gesetz"

        norms = []

        for norm in tree.xpath('//norm'):
            unit_str = (norm.xpath('.//enbez/text()') or [""])[0].strip()
            titel_str = (norm.xpath('.//titel/text()') or [""])[0].strip()

            paragraphs = norm.xpath('.//textdaten//Content//P//text()')
            content_text = " ".join(paragraphs).strip()

            if len(content_text) < 5 and not unit_str:
                continue

            doknr = norm.get('doknr') or "nodok"

            norms.append({
                "text": f"{unit_str} {titel_str}: {content_text}".strip(),
                "unit": unit_str,
                "doknr": doknr
            })

        if not norms:
            save_to_registry(file_name)
            continue

        # Batch embedding + upsert
        for start in range(0, len(norms), BATCH_SIZE):
            batch = norms[start:start + BATCH_SIZE]

            texts = [n["text"] for n in batch]
            dense_vecs  = list(dense_model.embed(texts))
            sparse_vecs = list(sparse_model.embed(texts))

            points = []

            for i, norm in enumerate(batch):
                string_id = f"{jurabk}_{norm['unit']}_{norm['doknr']}"
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, string_id))

                sv = sparse_vecs[i]

                points.append(models.PointStruct(
                    id=point_id,
                    vector={
                        "dense": dense_vecs[i].tolist(),
                        "sparse": models.SparseVector(
                            indices=sv.indices.tolist(),
                            values=sv.values.tolist()
                        )
                    },
                    payload={
                        "jurabk": jurabk,
                        "title": title,
                        "unit": norm["unit"],
                        "text": norm["text"],
                        "doknr": norm["doknr"],
                        "source_file": file_name
                    }
                ))

            client.upsert(collection_name=COLLECTION, points=points)

        save_to_registry(file_name)
        print(f"[{idx+1}/{total_files}] ✅ {file_name} ({len(norms)} fragmentów)")

    except Exception as e:
        print(f"[{idx+1}/{total_files}] ❌ Błąd w {file_name}: {e}")

print("\n🏁 Zakończono.")
