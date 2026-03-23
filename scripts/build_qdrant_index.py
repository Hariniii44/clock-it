"""
Build Qdrant collections directly from downloaded datasets.

Usage:
    python scripts/build_qdrant_index.py                  # build all missing collections
    python scripts/build_qdrant_index.py --force          # rebuild all from scratch
    python scripts/build_qdrant_index.py --list           # list current Qdrant collections
    python scripts/build_qdrant_index.py --dataset hansard_2020s         # single dataset
    python scripts/build_qdrant_index.py --dataset hansard_2020s --force # force rebuild one

Prerequisites:
    1. Qdrant running:   docker run -p 6333:6333 -p 6334:6334 -v "$(pwd)/qdrant_storage:/qdrant/storage:z" qdrant/qdrant
    2. Datasets present: data/datasets/<name>/  (download with scripts/download_datasets.py)
"""
import sys
import argparse
sys.path.append('.')
sys.path.append('src')

from src.retrieval.qdrant_indexer import QdrantIndexBuilder
from config import DATASETS_DIR, DATASETS_CONFIG


def main():
    parser = argparse.ArgumentParser(description='Build Qdrant collections from downloaded datasets')
    parser.add_argument('--force',   action='store_true', help='Rebuild collections even if they already exist')
    parser.add_argument('--list',    action='store_true', help='List existing Qdrant collections and exit')
    parser.add_argument('--dataset', type=str,            help='Build a single dataset by name')
    args = parser.parse_args()

    print("QDRANT INDEX BUILDER")
    print("=" * 50)

    builder = QdrantIndexBuilder()

    if args.list:
        builder.list_collections()
        return

    if args.dataset:
        dataset_names = [args.dataset]
    else:
        # Auto-discover all downloaded datasets that are in DATASETS_CONFIG
        dataset_names = sorted(
            d.name for d in DATASETS_DIR.iterdir()
            if d.is_dir() and d.name in DATASETS_CONFIG
        )

    if not dataset_names:
        print("ERROR: No downloaded datasets found in data/datasets/")
        print("Run 'python scripts/download_datasets.py' first.")
        sys.exit(1)

    print(f"\nDatasets to index: {dataset_names}")
    results = builder.build_all_collections(dataset_names=dataset_names, force_rebuild=args.force)

    if results['successful']:
        print(f"\nSUCCESS: {len(results['successful'])} collection(s) ready in Qdrant.")
        print("You can inspect them at http://localhost:6333/dashboard")
        print("\nNext step: run your pipeline normally — it will now use Qdrant for retrieval.")
    else:
        print("\nERROR: No collections were built. Check errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
