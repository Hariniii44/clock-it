"""Build Qdrant collections directly from downloaded datasets — no FAISS step needed"""
import sys
sys.path.append('.')

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from datasets import load_from_disk
from tqdm import tqdm

from config import DATASETS_DIR, DATASETS_CONFIG, EMBEDDING_MODEL, QDRANT_URL, QDRANT_API_KEY

BATCH_SIZE = 64  # documents per upsert batch (smaller for cloud upload)


def _extract_texts_and_metadata(data, dataset_name: str):
    """
    Extract (text, metadata) pairs from a HuggingFace dataset.
    Mirrors the logic in FAISSIndexBuilder.prepare_texts_from_dataset.
    """
    texts, meta = [], []

    for idx, item in enumerate(tqdm(data, desc=f"  Preparing {dataset_name}", leave=False)):
        title, content = '', ''

        for f in ['title', 'headline', 'subject', 'summary']:
            if f in item and item[f]:
                title = str(item[f])
                break

        for f in ['content', 'body', 'text', 'chunk_text', 'description']:
            if f in item and item[f]:
                content = str(item[f])
                break

        if not title and content:
            title = content.split('\n')[0][:100]

        if title and content:
            search_text = f"{title}. {content}"
        elif content:
            search_text = content
        elif title:
            search_text = title
        else:
            search_text = '. '.join(
                str(v).strip() for v in item.values()
                if isinstance(v, str) and len(str(v).strip()) > 10
            )

        search_text = search_text[:2000]
        if len(search_text.strip()) < 20:
            continue

        texts.append(search_text)
        meta.append({
            'title': title,
            'date':  item.get('date', item.get('published_date', item.get('timestamp', ''))),
            'url':   item.get('url', item.get('link', '')),
        })

    return texts, meta


class QdrantIndexBuilder:
    """
    Reads datasets directly from data/datasets/ and upserts them into Qdrant.
    One collection per dataset. No FAISS dependency.
    """

    def __init__(self, model_name: str = None, qdrant_url: str = None):
        if model_name is None:
            model_name = EMBEDDING_MODEL
        if qdrant_url is None:
            qdrant_url = QDRANT_URL

        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"Embedding dimension: {self.dimension}")

        self.client = QdrantClient(url=qdrant_url, api_key=QDRANT_API_KEY, timeout=120)
        print(f"Qdrant connected at {qdrant_url}")

    def build_collection(self, dataset_name: str, force_rebuild: bool = False) -> bool:
        """
        Build a Qdrant collection for one dataset.
        Reads directly from data/datasets/<dataset_name>/.
        """
        dataset_path = DATASETS_DIR / dataset_name

        if not dataset_path.exists():
            print(f"  [SKIP] Dataset not downloaded: {dataset_path}")
            print(f"         Run: python scripts/download_datasets.py")
            return False

        # Check for existing collection
        existing = {c.name for c in self.client.get_collections().collections}
        if dataset_name in existing:
            if not force_rebuild:
                count = self.client.get_collection(dataset_name).points_count
                print(f"  [SKIP] '{dataset_name}' already exists ({count:,} points). Use --force to rebuild.")
                return True
            self.client.delete_collection(dataset_name)
            print(f"  Deleted existing collection '{dataset_name}'")

        # Load dataset from disk
        print(f"  Loading dataset from {dataset_path} ...")
        try:
            dataset = load_from_disk(str(dataset_path))
            if hasattr(dataset, 'keys') and 'train' in dataset:
                data = dataset['train']
            elif hasattr(dataset, 'keys'):
                data = dataset[list(dataset.keys())[0]]
            else:
                data = dataset
            print(f"  {len(data):,} documents")
        except Exception as e:
            print(f"  ERROR loading dataset: {e}")
            return False

        # Extract texts and metadata
        texts, doc_meta = _extract_texts_and_metadata(data, dataset_name)
        if not texts:
            print(f"  ERROR: No valid texts found in dataset")
            return False
        print(f"  {len(texts):,} valid texts extracted")

        # Create Qdrant collection
        self.client.create_collection(
            collection_name=dataset_name,
            vectors_config=VectorParams(size=self.dimension, distance=Distance.COSINE),
        )
        print(f"  Created collection '{dataset_name}'")

        # Embed and upsert in batches
        for start in tqdm(range(0, len(texts), BATCH_SIZE), desc=f"  Uploading {dataset_name}"):
            batch_texts = texts[start: start + BATCH_SIZE]
            batch_meta  = doc_meta[start: start + BATCH_SIZE]

            embeddings = self.model.encode(
                batch_texts,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=64,
            )

            points = [
                PointStruct(
                    id=start + i,
                    vector=emb.tolist(),
                    payload={
                        'text':    text,
                        'title':   m.get('title', ''),
                        'url':     m.get('url', ''),
                        'date':    m.get('date', ''),
                        'dataset': dataset_name,
                    },
                )
                for i, (emb, m, text) in enumerate(zip(embeddings, batch_meta, batch_texts))
            ]

            self.client.upsert(collection_name=dataset_name, points=points, wait=False)

        final_count = self.client.get_collection(dataset_name).points_count
        print(f"  Done: {final_count:,} points in '{dataset_name}'")
        return True

    def build_all_collections(self, dataset_names: list = None, force_rebuild: bool = False) -> dict:
        """Build Qdrant collections for all downloaded datasets."""
        if dataset_names is None:
            dataset_names = sorted(
                d.name for d in DATASETS_DIR.iterdir()
                if d.is_dir() and d.name in DATASETS_CONFIG
            )

        print(f"\nBuilding {len(dataset_names)} Qdrant collection(s) from {DATASETS_DIR}")

        successful, failed = [], []

        for name in dataset_names:
            print(f"\n[{name}]")
            try:
                ok = self.build_collection(name, force_rebuild=force_rebuild)
                (successful if ok else failed).append(name)
            except Exception as e:
                print(f"  ERROR: {e}")
                failed.append(name)

        print(f"\n{'='*50}")
        print(f"Done: {len(successful)} succeeded, {len(failed)} failed")
        if failed:
            print(f"Failed: {failed}")

        return {'successful': successful, 'failed': failed}

    def list_collections(self):
        """Print all Qdrant collections with point counts."""
        collections = self.client.get_collections().collections
        if not collections:
            print("No collections found in Qdrant.")
            return
        print(f"\n{'Collection':<40} {'Points':>10}")
        print("-" * 52)
        for c in sorted(collections, key=lambda x: x.name):
            info = self.client.get_collection(c.name)
            print(f"{c.name:<40} {info.points_count:>10,}")
