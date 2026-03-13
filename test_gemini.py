import google.genai as genai
import os
from dotenv import load_dotenv
load_dotenv()

def test_gemini_connection():
    """Test Gemini connection and list available models"""
    
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("ERROR: GEMINI_API_KEY not found in environment variables")
        return
    
    try:
        # Initialize client
        client = genai.Client(api_key=api_key)
        print("Client initialized successfully")
        
        # List available models
        print("\nListing available models...")
        models = client.models.list()
        
        print("Available models:")
        for model in models:
            print(f"- {model.name}")
            if hasattr(model, 'supported_generation_methods'):
                print(f"  Supported methods: {model.supported_generation_methods}")
        
        # Try a simple generation with the first available model
        if models:
            first_model = models[0].name
            print(f"\nTesting simple generation with: {first_model}")
            
            try:
                response = client.models.generate_content(
                    model=first_model,
                    contents="Hello, can you respond with 'Hello World'?"
                )
                print(f"Response: {response.text}")
            except Exception as e:
                print(f"Error with generate_content: {e}")
                
                # Try alternative API structure
                try:
                    response = client.models.generate_content(
                        model=first_model,
                        contents=[{"role": "user", "parts": [{"text": "Hello, can you respond with 'Hello World'?"}]}]
                    )
                    print(f"Response (alternative structure): {response.text}")
                except Exception as e2:
                    print(f"Error with alternative structure: {e2}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_gemini_connection()