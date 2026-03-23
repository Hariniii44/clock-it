"""
Download all configured Sri Lankan government datasets from HuggingFace.

Usage:
    python scripts/download_datasets.py           # download all missing datasets
    python scripts/download_datasets.py --force   # re-download everything
    python scripts/download_datasets.py --status  # check what's already downloaded
"""
import sys
import argparse
sys.path.append('src')
sys.path.append('.')

from src.retrieval.dataset_loader import DatasetLoader
from config import DATASETS_CONFIG, print_config


def main():
    parser = argparse.ArgumentParser(description='Download Sri Lankan government datasets')
    parser.add_argument('--force',  action='store_true', help='Re-download already existing datasets')
    parser.add_argument('--status', action='store_true', help='Show download status and exit')
    args = parser.parse_args()

    print_config()

    loader = DatasetLoader()

    if args.status:
        status = loader.get_dataset_status()
        print("\nDATASET STATUS:")
        print("=" * 50)
        ready, missing = [], []
        for name, info in status.items():
            if info['exists'] and info.get('documents', 0) > 0:
                print(f"  {name}: {info['documents']:,} docs")
                ready.append(name)
            else:
                print(f"  {name}: NOT downloaded")
                missing.append(name)
        print(f"\n{len(ready)}/{len(status)} datasets ready")
        if missing:
            print(f"Missing: {missing}")
        return

    print(f"\nDownloading {len(DATASETS_CONFIG)} datasets...")
    results = loader.download_all_datasets(force_reload=args.force, skip_on_error=True)

    total_ready = len(results['successful']) + len(results['already_existed'])
    print(f"\n{total_ready}/{len(DATASETS_CONFIG)} datasets ready.")
    if total_ready > 0:
        print("Next step: python scripts/build_qdrant_index.py")


if __name__ == "__main__":
    main()
