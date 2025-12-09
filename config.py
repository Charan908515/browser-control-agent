import os
import requests
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
import dotenv
import sys

dotenv.load_dotenv()

class LLMConfig:
    _instance = None
    _ollama_available = False
    _checked_ollama = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LLMConfig, cls).__new__(cls)
        return cls._instance

    @classmethod
    def check_ollama_availability(cls):
        """Checks if Ollama is running locally."""
        if cls._checked_ollama:
            return cls._ollama_available
        
        try:
            response = requests.get("http://localhost:11434", timeout=2)
            if response.status_code == 200:
                print(">>> Ollama is detected and running.")
                cls._ollama_available = True
            else:
                print(">>> Ollama detected but returned unexpected status.")
                cls._ollama_available = False
        except requests.exceptions.ConnectionError:
            print(">>> Ollama not detected (ConnectionError).")
            cls._ollama_available = False
        except Exception as e:
            print(f">>> Error checking Ollama: {e}")
            cls._ollama_available = False
        
        cls._checked_ollama = True
        return cls._ollama_available

    @classmethod
    def get_main_llm(cls):
        """Returns the main Gemini LLM for text/reasoning tasks."""
        
        api_key = os.getenv("GOOGLE_API_KEY")
        
        if not api_key:
            raise ValueError("CRITICAL: No Google API Key found (checked 'GOOGLE_API_KEY'). Please set it in .env")

        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key,
            temperature=0.1
        )

    @classmethod
    def get_vision_llm(cls):
        """
        Returns the LLM to use for Vision tasks.
        Priority: Ollama (if available) -> Groq (Fallback)
        """ 
        if cls.check_ollama_availability():
            print(">>> Using Local Ollama for Vision.")
            # Instruct user to download model if likely missing (simple heuristic or just always warn)
            print(">>> Ensure you have 'llama3.2-vision:11b' installed: `ollama run llama3.2-vision:11b`")
            return ChatOllama(model="llama3.2-vision:11b", temperature=0.1)
        else:
            print(">>> Ollama not available. Falling back to Groq for Vision.")
            
            api_key=os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("CRITICAL: No Groq API Key found (checked 'GROQ_API_KEY'). Please set it in .env")
            
            return ChatGroq(
                api_key=api_key, 
                model="llama-3.2-90b-vision-preview" # Using a vision-capable model on Groq
            )

# Global validator to run on import/startup
def validate_config():
    print("--- Validating LLM Configuration ---")
    try:
        LLMConfig.get_main_llm()
        print("✅ Main LLM (Gemini) Configured")
    except Exception as e:
        print(f"❌ Main LLM Config Error: {e}")

    # Check Vision
    is_ollama = LLMConfig.check_ollama_availability()
    if not is_ollama:
        print("⚠️  Ollama not available. Checking Groq fallback...")
        try:
            LLMConfig.get_vision_llm()
            print("✅ Vision Fallback (Groq) Configured")
        except Exception as e:
             print(f"❌ Vision Config Error: {e}")
    else:
        print("✅ Vision LLM (Ollama) Configured")
    print("------------------------------------")

if __name__ == "__main__":
    validate_config()
