"""
Download and manage Nuuwan's Sri Lankan government datasets
"""
from datasets import load_dataset, load_from_disk
from tqdm import tqdm
import pandas as pd
import sys
sys.path.append('.')
from config import DATASETS_CONFIG, DATASETS_DIR
import os
from pathlib import Path

class DatasetLoader:
    """Download and manage datasets from HuggingFace"""
    
    def __init__(self):
        """Initialize dataset loader"""
        self.datasets_dir = DATASETS_DIR
        self.datasets_config = DATASETS_CONFIG
        
        print(f"📁 Dataset storage: {self.datasets_dir}")

    def _load_and_validate_dataset(self, save_path):
        """Load dataset from disk and validate it properly"""
        try:
            # Use load_from_disk instead of load_dataset for saved datasets
            dataset = load_from_disk(str(save_path))
            
            # Handle different dataset structures
            if hasattr(dataset, 'keys') and callable(getattr(dataset, 'keys')):
                # Multi-split dataset (has 'train', 'test', etc.)
                available_splits = list(dataset.keys())
                print(f"   📊 Available splits: {available_splits}")
                
                if 'train' in dataset:
                    data = dataset['train']
                elif len(available_splits) == 1:
                    # Single split with custom name
                    split_name = available_splits[0]
                    data = dataset[split_name]
                    print(f"   📊 Using '{split_name}' split")
                else:
                    # Multiple splits - use largest
                    largest_split = max(available_splits, key=lambda k: len(dataset[k]))
                    data = dataset[largest_split]
                    print(f"   📊 Using '{largest_split}' split (largest)")
            else:
                # Single dataset without splits
                data = dataset
            
            doc_count = len(data)
            
            # Validate we have reasonable data
            if doc_count == 0:
                return None, "Empty dataset"
            
            if doc_count < 10:
                # Check if it's actually loaded properly
                print(f"   ⚠️ Small dataset detected ({doc_count} docs) - validating...")
                sample = data[0] if doc_count > 0 else {}
                sample_str = str(sample)
                if len(sample_str) < 100:
                    print(f"   ❌ Sample data seems too small: {sample_str[:200]}...")
                    return None, "Dataset appears corrupted (samples too small)"
            
            return data, doc_count
        
        except Exception as e:
            return None, f"Load error: {str(e)}"
        
    def download_dataset(self, dataset_name: str, force_reload: bool = False):
        """
        Download a single dataset
        
        Args:
            dataset_name: Name of dataset from config
            force_reload: If True, re-download even if exists
            
        Returns:
            tuple: (success: bool, dataset_info: dict)
        """
        if dataset_name not in self.datasets_config:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        config = self.datasets_config[dataset_name]
        hf_name = config['hf_name']
        save_path = self.datasets_dir / dataset_name
        
        print(f"\n{'='*60}")
        print(f"📥 DOWNLOADING: {dataset_name}")
        print(f"{'='*60}")
        print(f"HuggingFace ID: {hf_name}")
        print(f"Save location: {save_path}")
        print(f"Description: {config['description']}")
        print(f"Authority weight: {config['authority']}")
        print(f"Use for: {config['use_for']}")
        
        # Check if already exists
        if save_path.exists() and not force_reload:
            print(f"⚠️  Dataset already exists. Use force_reload=True to re-download.")
            try:
                # Load existing to get info using proper method
                data, result = self._load_and_validate_dataset(save_path)
                
                if data is None:
                    print(f"❌ Existing dataset appears corrupted: {result}")
                    print("🔄 Will re-download...")
                else:
                    print(f"📊 Existing dataset: {result:,} documents")
                    
                    return True, {
                        'name': dataset_name,
                        'documents': result,
                        'status': 'already_exists'
                    }
                
            except Exception as e:
                print(f"❌ Error reading existing dataset: {e}")
                print("🔄 Will re-download...")
        
        try:
            print(f"🚀 Starting download...")
            
            # Download from HuggingFace
            print(f"📡 Downloading from HuggingFace: {hf_name}")
            dataset = load_dataset(hf_name)
            
            # Get document count
            if 'train' in dataset:
                doc_count = len(dataset['train'])
                print(f"📊 Downloaded: {doc_count:,} documents")
            else:
                doc_count = len(dataset)
                print(f"📊 Downloaded: {doc_count:,} documents")
            
            # Save to disk
            print(f"💾 Saving to: {save_path}")
            dataset.save_to_disk(str(save_path))
            
            print(f"✅ SUCCESS: {dataset_name} downloaded successfully!")
            
            return True, {
                'name': dataset_name,
                'documents': doc_count,
                'hf_name': hf_name,
                'save_path': str(save_path),
                'status': 'downloaded'
            }
            
        except Exception as e:
            print(f"❌ ERROR downloading {dataset_name}: {str(e)}")
            return False, {
                'name': dataset_name,
                'error': str(e),
                'status': 'failed'
            }
    
    def download_all_datasets(self, force_reload: bool = False, skip_on_error: bool = True):
        """
        Download all configured datasets
        
        Args:
            force_reload: Re-download existing datasets
            skip_on_error: Continue downloading other datasets if one fails
            
        Returns:
            dict: Summary of download results
        """
        print("🚀 STARTING BULK DATASET DOWNLOAD")
        print("="*60)
        print(f"Total datasets to download: {len(self.datasets_config)}")
        print(f"Force reload: {force_reload}")
        print(f"Skip on error: {skip_on_error}")
        
        results = {
            'successful': [],
            'failed': [],
            'already_existed': [],
            'total_documents': 0
        }
        
        # Download each dataset
        for i, dataset_name in enumerate(self.datasets_config.keys(), 1):
            print(f"\n📋 DATASET {i}/{len(self.datasets_config)}: {dataset_name}")
            
            try:
                success, info = self.download_dataset(dataset_name, force_reload)
                
                if success:
                    if info['status'] == 'already_exists':
                        results['already_existed'].append(info)
                    else:
                        results['successful'].append(info)
                    
                    results['total_documents'] += info.get('documents', 0)
                else:
                    results['failed'].append(info)
                    
                    if not skip_on_error:
                        print(f"❌ Stopping due to error in {dataset_name}")
                        break
                
            except Exception as e:
                error_info = {
                    'name': dataset_name,
                    'error': str(e),
                    'status': 'failed'
                }
                results['failed'].append(error_info)
                
                if not skip_on_error:
                    print(f"❌ Stopping due to error in {dataset_name}: {e}")
                    break
        
        # Print summary
        self._print_download_summary(results)
        
        return results
    
    def _print_download_summary(self, results):
        """Print download summary"""
        print(f"\n{'='*60}")
        print("📊 DOWNLOAD SUMMARY")
        print(f"{'='*60}")
        
        successful = len(results['successful'])
        already_existed = len(results['already_existed'])
        failed = len(results['failed'])
        total = successful + already_existed + failed
        
        print(f"✅ Successfully downloaded: {successful}")
        print(f"📁 Already existed: {already_existed}")
        print(f"❌ Failed: {failed}")
        print(f"📋 Total processed: {total}")
        print(f"📊 Total documents: {results['total_documents']:,}")
        
        if results['successful']:
            print(f"\n📥 NEWLY DOWNLOADED:")
            for info in results['successful']:
                print(f"  ✅ {info['name']}: {info['documents']:,} docs")
        
        if results['already_existed']:
            print(f"\n📁 ALREADY EXISTED:")
            for info in results['already_existed']:
                print(f"  📋 {info['name']}: {info['documents']:,} docs")
        
        if results['failed']:
            print(f"\n❌ FAILED DOWNLOADS:")
            for info in results['failed']:
                print(f"  💥 {info['name']}: {info.get('error', 'Unknown error')}")
        
        print(f"{'='*60}")
        
        # Success rate
        success_rate = ((successful + already_existed) / total * 100) if total > 0 else 0
        print(f"🎯 Success rate: {success_rate:.1f}%")
        
        if success_rate == 100:
            print("🎉 ALL DATASETS READY!")
        elif success_rate >= 80:
            print("✅ Most datasets ready - you can proceed to next step")
        else:
            print("⚠️ Many downloads failed - check errors before proceeding")

    def get_dataset_status(self):
        """Get status of all configured datasets"""
        status = {}
        
        for dataset_name in self.datasets_config.keys():
            save_path = self.datasets_dir / dataset_name
            
            if save_path.exists():
                try:
                    data, doc_count = self._load_and_validate_dataset(save_path)
                    
                    if data is None:
                        status[dataset_name] = {
                            'exists': True,
                            'error': f'Dataset corrupted: {doc_count}',
                            'documents': 0
                        }
                    else:
                        status[dataset_name] = {
                            'exists': True,
                            'documents': doc_count,
                            'path': str(save_path)
                        }
                except Exception as e:
                    status[dataset_name] = {
                        'exists': True,
                        'error': f'Could not load dataset: {str(e)}',
                        'documents': 0
                    }
            else:
                status[dataset_name] = {
                    'exists': False,
                    'error': 'Not downloaded'
                }
        
        return status
    
    # def get_dataset_status(self):
    #     """Get status of all configured datasets"""
    #     status = {}
        
    #     for dataset_name in self.datasets_config.keys():
    #         save_path = self.datasets_dir / dataset_name
            
    #         if save_path.exists():
    #             try:
    #                 dataset = load_dataset(str(save_path))
    #                 if 'train' in dataset:
    #                     doc_count = len(dataset['train'])
    #                 else:
    #                     doc_count = len(dataset)
                    
    #                 status[dataset_name] = {
    #                     'exists': True,
    #                     'documents': doc_count,
    #                     'path': str(save_path)
    #                 }
    #             except:
    #                 status[dataset_name] = {
    #                     'exists': False,
    #                     'error': 'Could not load dataset'
    #                 }
    #         else:
    #             status[dataset_name] = {
    #                 'exists': False,
    #                 'error': 'Not downloaded'
    #             }
        
    #     return status

def main():
    """Test dataset loader"""
    loader = DatasetLoader()
    
    # Show current status
    print("📋 CURRENT DATASET STATUS:")
    status = loader.get_dataset_status()
    for name, info in status.items():
        if info['exists']:
            print(f"  ✅ {name}: {info['documents']:,} documents")
        else:
            print(f"  ❌ {name}: {info.get('error', 'Not found')}")
    
    # Ask user what to do
    print(f"\n🤔 What would you like to do?")
    print("1. Download all datasets")
    print("2. Download specific dataset")
    print("3. Show status only")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == "1":
        force = input("Force re-download existing? (y/N): ").strip().lower() == 'y'
        loader.download_all_datasets(force_reload=force)
    elif choice == "2":
        print(f"Available datasets: {list(DATASETS_CONFIG.keys())}")
        dataset_name = input("Enter dataset name: ").strip()
        if dataset_name in DATASETS_CONFIG:
            force = input("Force re-download if exists? (y/N): ").strip().lower() == 'y'
            loader.download_dataset(dataset_name, force_reload=force)
        else:
            print(f"❌ Unknown dataset: {dataset_name}")
    else:
        print("📊 Status shown above.")

if __name__ == "__main__":
    main()