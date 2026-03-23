import os, uuid, zipfile, pathlib
from lxml import etree
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding, SparseTextEmbedding

COLLECTION   = "gesetze"
DENSE_MODEL  = "mixedbread-ai/mxbai-embed-large-v1"
SPARSE_MODEL = "Qdrant/bm25"
CACHE_DIR    = "./models_cache"
DENSE_DIM    = 1024
BATCH_SIZE   = 10

os.environ["FASTEMBED_CACHE_PATH"] = CACHE_DIR

workspace = pathlib.Path(os.getenv("GITHUB_WORKSPACE", "."))
zip_name  = os.getenv("BATCH_FILE", "paczka_1.zip")

with zipfile.ZipFile(workspace / "zips" / zip_name) as z:
    z.extractall(workspace / "data")

all_files = [f for f in (workspace / "data").rglob("*.xml") if "models_cache" not in str(f)]
print(f"📂 Znaleziono {len(all_files)} plików XML")

client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))

if not client.collection_exists(COLLECTION):
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config={"dense": models.VectorParams(size=DENSE_DIM, distance=models.Distance.COSINE)},
        sparse_vectors_config={"sparse": models.SparseVectorParams()}
    )

dense_model  = TextEmbedding(model_name=DENSE_MODEL, cache_dir=CACHE_DIR)
sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL, cache_dir=CACHE_DIR)

for idx, file_path in enumerate(all_files):
    try:
        tree   = etree.parse(file_path, etree.XMLParser(recover=True))
        jurabk = (tree.xpath('//jurabk/text()') or [file_path.name])[0]
        title  = (tree.xpath('//titel/text()') or ["Gesetz"])[0]

        norms = [
            {
                "text": f"{(norm.xpath('.//enbez/text()') or [''])[0].strip()} {(norm.xpath('.//titel/text()') or [''])[0].strip()}: {' '.join(norm.xpath('.//textdaten//Content//P//text()')).strip()}".strip(),
                "unit": (norm.xpath('.//enbez/text()') or [''])[0].strip(),
                "doknr": norm.get('doknr') or "nodok"
            }
            for norm in tree.xpath('//norm')
            if len(' '.join(norm.xpath('.//textdaten//Content//P//text()')).strip()) >= 5
        ]

        for start in range(0, len(norms), BATCH_SIZE):
            batch       = norms[start:start + BATCH_SIZE]
            texts       = [n["text"] for n in batch]
            dense_vecs  = list(dense_model.embed(texts))
            sparse_vecs = list(sparse_model.embed(texts))
            client.upsert(
                collection_name=COLLECTION,
                points=[
                    models.PointStruct(
                        id=str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{jurabk}_{n['unit']}_{n['doknr']}")),
                        vector={
                            "dense": dense_vecs[i].tolist(),
                            "sparse": models.SparseVector(
                                indices=sparse_vecs[i].indices.tolist(),
                                values=sparse_vecs[i].values.tolist()
                            )
                        },
                        payload={**n, "jurabk": jurabk, "title": title, "source_file": file_path.name}
                    )
                    for i, n in enumerate(batch)
                ]
            )
        print(f"[{idx+1}/{len(all_files)}] ✅ {file_path.name}")

    except Exception as e:
        print(f"❌ {file_path.name}: {e}")

print("🏁 Zakończono.")