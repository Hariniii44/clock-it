"""Build FAISS indices for fast semantic search"""
from sentence_transformers import SentenceTransformer
from datasets import load_from_disk
import faiss
import numpy as np
from tqdm import tqdm
import pickle
from pathlib import Path
import sys
sys.path.append('.')
from config import DATASETS_DIR, INDICES_DIR, EMBEDDING_MODEL

class FAISSIndexBuilder:
    """Build and save FAISS indices for datasets"""
    
    def __init__(self, model_name=None):
        """Initialize with embedding model"""
        if model_name is None:
            model_name = EMBEDDING_MODEL
            
        self.model_name = model_name

        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"Embedding dimension: {self.dimension}")
    
    def prepare_texts_from_dataset(self, dataset, dataset_name):
        """Extract searchable text from dataset"""
        texts = []
        metadata = []
        
        print(f"Analyzing dataset structure...")
        
        # Get a sample to understand structure
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"   Sample fields: {list(sample.keys())}")
            
            # Show sample content
            for key, value in sample.items():
                if isinstance(value, str) and len(value) > 20:
                    print(f"   {key}: {str(value)[:100]}...")
        
        # Extract text based on dataset structure
        for idx, item in enumerate(tqdm(dataset, desc="Preparing texts")):
            # Try different field combinations for text content
            title = ''
            content = ''
            
            # Common title fields
            for title_field in ['title', 'headline', 'subject', 'summary']:
                if title_field in item and item[title_field]:
                    title = str(item[title_field])
                    break
            
            # Common content fields  
            for content_field in ['content', 'body', 'text', 'chunk_text', 'description']:
                if content_field in item and item[content_field]:
                    content = str(item[content_field])
                    break
            
            # If no separate title, try to extract from content
            if not title and content:
                # Use first sentence/line as title
                lines = content.split('\n')
                if lines:
                    title = lines[0][:100]  # First 100 chars of first line
            
            # Create searchable text (title + content)
            if title and content:
                search_text = f"{title}. {content}"
            elif content:
                search_text = content
            elif title:
                search_text = title
            else:
                # Fallback: combine all string fields
                all_text = []
                for key, value in item.items():
                    if isinstance(value, str) and len(value.strip()) > 10:
                        all_text.append(value.strip())
                search_text = '. '.join(all_text)
            
            # Limit text length for embeddings
            if len(search_text) > 2000:
                search_text = search_text[:2000]
            
            # Skip if text too short
            if len(search_text.strip()) < 20:
                continue
                
            texts.append(search_text)
            
            # Store metadata
            metadata.append({
                'idx': idx,
                'title': title,
                'date': item.get('date', item.get('published_date', item.get('timestamp', ''))),
                'url': item.get('url', item.get('link', '')),
                'source': item.get('source', item.get('newspaper_id', item.get('dataset', dataset_name))),
                'raw_item': item  # Keep full item for reference
            })
        
        print(f"   Prepared {len(texts)} searchable texts from {len(dataset)} items")
        
        return texts, metadata
    
    def build_index_for_dataset(self, dataset_path, dataset_name):
        """Build FAISS index for a single dataset"""
        print(f"\n{'='*60}")
        print(f"BUILDING INDEX: {dataset_name}")
        print(f"{'='*60}")
        
        # Load dataset
        print("Loading dataset...")
        try:
            dataset = load_from_disk(str(dataset_path))
            
            # Handle dataset structure
            if hasattr(dataset, 'keys') and 'train' in dataset:
                data = dataset['train']
                print(f"   Using 'train' split")
            elif hasattr(dataset, 'keys'):
                splits = list(dataset.keys())
                data = dataset[splits[0]]
                print(f"   Using '{splits[0]}' split")
            else:
                data = dataset
            
            print(f"   Dataset size: {len(data):,} documents")
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None
        
        # Prepare texts for embedding
        texts, metadata = self.prepare_texts_from_dataset(data, dataset_name)
        
        if len(texts) == 0:
            print(f"No valid texts found in dataset")
            return None
        
        # Generate embeddings in batches
        print(f"\nGenerating embeddings for {len(texts):,} texts...")
        batch_size = 32
        embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
            batch = texts[i:i+batch_size]
            batch_embeddings = self.model.encode(
                batch,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True  # Important for cosine similarity
            )
            embeddings.append(batch_embeddings)
        
        embeddings = np.vstack(embeddings).astype('float32')
        print(f"   Generated {len(embeddings):,} embeddings")
        
        # Build FAISS index
        print(f"\nBuilding FAISS index...")
        
        # Use Inner Product (cosine similarity with normalized vectors)
        index = faiss.IndexFlatIP(self.dimension)
        
        # Add embeddings to index
        index.add(embeddings)
        print(f"   Index built with {index.ntotal:,} vectors")
        
        # Save index and metadata
        index_path = INDICES_DIR / f"{dataset_name}.index"
        metadata_path = INDICES_DIR / f"{dataset_name}.metadata.pkl"
        
        print(f"\nSaving index...")
        print(f"   Index: {index_path}")
        print(f"   Metadata: {metadata_path}")
        
        faiss.write_index(index, str(index_path))
        
        with open(metadata_path, 'wb') as f:
            pickle.dump({
                'metadata': metadata,
                'texts': texts,
                'dataset_name': dataset_name,
                'embedding_model': self.model_name,
                'total_documents': len(texts)
            }, f)
        
        print(f"   Index saved successfully")
        print(f"   Summary: {len(texts):,} documents indexed")
        
        return {
            'dataset_name': dataset_name,
            'total_documents': len(texts),
            'index_path': str(index_path),
            'metadata_path': str(metadata_path)
        }
    
    def build_all_indices(self, dataset_names=None):
        """Build indices for all available datasets"""
        print("STARTING FAISS INDEX BUILDING")
        print("="*60)
        
        # Get available datasets
        available_datasets = []
        for dataset_dir in DATASETS_DIR.iterdir():
            if dataset_dir.is_dir():
                available_datasets.append(dataset_dir.name)
        
        if dataset_names:
            # Filter to requested datasets
            available_datasets = [name for name in available_datasets if name in dataset_names]
        
        print(f"Will build indices for {len(available_datasets)} datasets:")
        for name in available_datasets:
            print(f"   - {name}")
        
        results = {
            'successful': [],
            'failed': [],
            'total_documents': 0
        }
        
        # Build each index
        for i, dataset_name in enumerate(available_datasets, 1):
            dataset_path = DATASETS_DIR / dataset_name
            
            if not dataset_path.exists():
                print(f"Skipping {dataset_name}: Dataset directory not found")
                continue
            
            try:
                print(f"\nINDEX {i}/{len(available_datasets)}: {dataset_name}")
                result = self.build_index_for_dataset(dataset_path, dataset_name)
                
                if result:
                    results['successful'].append(result)
                    results['total_documents'] += result['total_documents']
                else:
                    results['failed'].append({
                        'dataset_name': dataset_name,
                        'error': 'Index building returned None'
                    })
                    
            except Exception as e:
                print(f"Error building index for {dataset_name}: {e}")
                results['failed'].append({
                    'dataset_name': dataset_name,
                    'error': str(e)
                })
        
        # Print summary
        self._print_build_summary(results)
        
        return results
    
    def _print_build_summary(self, results):
        """Print index building summary"""
        print(f"\n{'='*60}")
        print("INDEX BUILDING SUMMARY")
        print(f"{'='*60}")
        
        successful = len(results['successful'])
        failed = len(results['failed'])
        total = successful + failed
        
        print(f"Successfully built: {successful}")
        print(f"Failed: {failed}")
        print(f"Total documents indexed: {results['total_documents']:,}")
        
        if results['successful']:
            print(f"\nSUCCESSFUL INDICES:")
            for result in results['successful']:
                print(f"   {result['dataset_name']}: {result['total_documents']:,} docs")
        
        if results['failed']:
            print(f"\nFAILED INDICES:")
            for result in results['failed']:
                print(f"   {result['dataset_name']}: {result['error']}")
        
        print(f"{'='*60}")
        
        success_rate = (successful / total * 100) if total > 0 else 0
        print(f"Success rate: {success_rate:.1f}%")
        
        if success_rate == 100 and successful > 0:
            print("ALL INDICES BUILT SUCCESSFULLY!")
            print("Ready for Step 4: Testing retrieval")
        elif success_rate >= 75:
            print("Most indices built - you can proceed to testing")
        else:
            print("Many indices failed - check errors before proceeding")

def main():
    """Test index builder"""
    builder = FAISSIndexBuilder()
    
    # Show available datasets
    available = []
    for dataset_dir in DATASETS_DIR.iterdir():
        if dataset_dir.is_dir():
            available.append(dataset_dir.name)
    
    print(f"Available datasets: {available}")
    
    if not available:
        print("No datasets found. Run dataset download first.")
        return
    
    # Ask user what to do
    print(f"\nWhat would you like to do?")
    print("1. Build all indices")
    print("2. Build specific dataset index")
    print("3. Test with one dataset first")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == "1":
        builder.build_all_indices()
    elif choice == "2":
        print(f"Available: {available}")
        dataset_name = input("Enter dataset name: ").strip()
        if dataset_name in available:
            dataset_path = DATASETS_DIR / dataset_name
            result = builder.build_index_for_dataset(dataset_path, dataset_name)
            if result:
                print(f"Index built for {dataset_name}")
        else:
            print(f"Dataset not found: {dataset_name}")
    elif choice == "3":
        # Test with first available dataset
        if available:
            test_dataset = available[0] 
            print(f"Testing with: {test_dataset}")
            dataset_path = DATASETS_DIR / test_dataset
            result = builder.build_index_for_dataset(dataset_path, test_dataset)
            if result:
                print(f"Test successful! Ready to build all indices.")

if __name__ == "__main__":
    main()