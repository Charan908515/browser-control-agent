import os
import requests
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_sambanova import ChatSambaNova
from langchain_ollama import ChatOllama
import dotenv
import sys

dotenv.load_dotenv()

class LLMConfig:
    _instance = None
    _ollama_available = False
    _checked_ollama = False
    _total_gemini_keys = 0
    _total_groq_keys = 0
    _total_sambanova_keys = 0
    

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
    def _build_gemini_rotation(cls):
        """Build rotation list for Gemini API keys (gemini_llm1 to gemini_llmn)."""
        rotation = []
        for i in range(1, cls._total_gemini_keys + 1):
            key = os.getenv(f"GOOGLE_API_KEY{i}")
            if key:
                rotation.append((f"gemini_llm{i}", key))
        return rotation

    @classmethod
    def _build_groq_rotation(cls):
        """Build rotation list for Groq API keys (groq_llm1 to groq_llm5)."""
        rotation = []
        for i in range(1, cls._total_groq_keys + 1):
            key = os.getenv(f"GROQ_API_KEY{i}")
            if key:
                rotation.append((f"groq_llm{i}", key))
        return rotation

    @classmethod
    def _build_sambanova_rotation(cls):
        """Build rotation list for SambaNova API keys (sambanova1 to sambanova3)."""
        rotation = []
        for i in range(1, cls._total_sambanova_keys + 1):
            key = os.getenv(f"SAMBANOVA_API_KEY{i}")
            if key:
                rotation.append((f"sambanova{i}", key))
        return rotation

    @classmethod
    def get_main_llm(cls):
        """Returns the main Gemini LLM for text/reasoning tasks (legacy, no rotation)."""
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("CRITICAL: No Google API Key found (checked 'GOOGLE_API_KEY'). Please set it in .env")
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key,
            temperature=0.1
        )

    @classmethod
    def get_main_llm_with_rotation(cls, start_index=0, provider=None):
        """
        Returns a list of (model_name, llm_instance) tuples for rotation.
        Use this in agents/tools to handle rate limits.
        
        Args:
            start_index: Starting index for rotation (from state)
            provider: Optional provider filter ("gemini", "groq", "sambanova", "ollama", or None for all)
            
        Returns:
            List of (name, llm) tuples to try in order
        """
        rotation_list = []
        
        # Add Gemini keys (if no provider specified or provider="gemini")
        if provider is None or provider == "gemini":
            gemini_keys = cls._build_gemini_rotation()
            for name, key in gemini_keys:
                try:
                    llm = ChatGoogleGenerativeAI(
                        model="gemini-2.5-flash",
                        google_api_key=key,
                        temperature=0.1
                    )
                    rotation_list.append((name, llm))
                except Exception as e:
                    print(f"Failed to initialize {name}: {e}")
        
        # Add Groq keys (if no provider specified or provider="groq")
        if provider is None or provider == "groq":
            groq_keys = cls._build_groq_rotation()
            for name, key in groq_keys:
                try:
                    llm = ChatGroq(
                        api_key=key,
                        model="llama-3.3-70b-versatile",
                        temperature=0.1
                    )
                    rotation_list.append((name, llm))
                except Exception as e:
                    print(f"Failed to initialize {name}: {e}")
        
        # Add SambaNova keys (if provider="sambanova")
        if provider == "sambanova":
            sambanova_keys = cls._build_sambanova_rotation()
            for name, key in sambanova_keys:
                try:
                    
                    llm = ChatSambaNova(
                        api_key=key,
                        model="gpt-oss-120b",
                        temperature=0.1
                    )
                    rotation_list.append((name, llm))
                except Exception as e:
                    print(f"Failed to initialize {name}: {e}")
        
        # Add Ollama (if provider="ollama")
        if provider == "ollama":
            try:
                llm = ChatOllama(model="llama3.2", temperature=0.1)
                rotation_list.append(("ollama_text", llm))
            except Exception as e:
                print(f"Failed to initialize Ollama text: {e}")
        
        if not rotation_list:
            provider_msg = f" for provider '{provider}'" if provider else ""
            raise ValueError(f"No valid API keys found for main LLM rotation{provider_msg}")
        
        # Rotate based on start_index
        if start_index > 0:
            rotation_list = rotation_list[start_index:] + rotation_list[:start_index]
        
        return rotation_list

    @classmethod
    def get_vision_llm(cls):
        """Returns the LLM to use for Vision tasks (legacy, no rotation)."""
        if cls.check_ollama_availability():
            print(">>> Using Local Ollama for Vision.")
            print(">>> Ensure you have 'llama3.2-vision:11b' installed: `ollama run llama3.2-vision:11b`")
            return ChatOllama(model="llama3.2-vision:11b", temperature=0.1)
        else:
            print(">>> Ollama not available. Falling back to Groq for Vision.")
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("CRITICAL: No Groq API Key found (checked 'GROQ_API_KEY'). Please set it in .env")
            return ChatGroq(
                api_key=api_key, 
                model="llama-3.2-90b-vision-preview"
            )

    @classmethod
    def get_vision_llm_with_rotation(cls, start_index=0):
        """
        Returns a list of (model_name, llm_instance) tuples for vision tasks with rotation.
        Priority: Ollama (if available) -> Groq rotation
        
        Args:
            start_index: Starting index for Groq rotation (from state)
            
        Returns:
            List of (name, llm) tuples to try in order
        """
        rotation_list = []
        
        # Check Ollama first (local, no rate limits)
        if cls.check_ollama_availability():
            try:
                llm = ChatOllama(model="llama3.2-vision:11b", temperature=0.1)
                rotation_list.append(("ollama_vision", llm))
                print(">>> Added Ollama to vision rotation (priority)")
            except Exception as e:
                print(f"Failed to initialize Ollama: {e}")
        
        # Add Groq vision keys as fallback
        groq_keys = cls._build_groq_rotation()
        for name, key in groq_keys:
            try:
                llm = ChatGroq(
                    api_key=key,
                    model="llama-3.2-90b-vision-preview",
                    temperature=0.1
                )
                rotation_list.append((f"{name}_vision", llm))
            except Exception as e:
                print(f"Failed to initialize {name} vision: {e}")
        
        if not rotation_list:
            raise ValueError("No valid vision LLM available (Ollama or Groq)")
        
        # Rotate based on start_index (only affects Groq keys if Ollama failed)
        if start_index > 0 and len(rotation_list) > 1:
            rotation_list = rotation_list[start_index:] + rotation_list[:start_index]
        
        return rotation_list

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
