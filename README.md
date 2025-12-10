# Autonomous Browser Agent

An advanced AI-powered agent capable of navigating the web, extracting data, and performing complex tasks autonomously. It leverages **Google Gemini** for reasoning and planning, and **Computer Vision (Ollama/Groq)** for understanding web interfaces.

## 🚀 Features

- **Autonomous Orchestration**: Uses [LangGraph](https://langchain-ai.github.io/langgraph/) to manage complex stateful workflows (Planning → Execution → Verification).
- **Vision-First Interaction**: Analyzes webpages using multi-modal LLMs (Llama 3.2 Vision) to detect elements, verify states, and handle dynamic content.
- **Smart Fallbacks**: 
  - Vision: Prioritizes local **Ollama** for privacy and cost, falls back to **Groq** for speed.
  - LLM: Centralized configuration using **Gemini 2.5 Flash**.
- **Self-Healing**: Detects execution errors and autonomously re-plans to recover.

## 🛠️ Architecture

- **`orchestation.py`**: The core brain. Defines the LangGraph state machine and agents (Planner, Executor, Redirector).
- **`config.py`**: Centralized configuration management for LLMs and API keys.
- **`analyze_tools.py`**: Vision analysis tools that capture scrolling screenshots for full-page context.
- **`browser_manager.py`**: Manages the Playwright browser instance.

## 📦 Setup

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com/) (Recommended for local vision)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository_url>
   cd <repository_name>
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   playwright install
   ```

3. **Install Local Vision Model (Optional but Recommended)**
   If you have Ollama installed:
   ```bash
   ollama pull llama3.2-vision:11b
   ```

4. **Configure Environment**
   Create a `.env` file in the root directory with your API keys.

   ### For Free Tier Users (Recommended: API Key Rotation)
   
   Free tier APIs have limited requests per minute (RPM). To handle rate limits automatically, configure multiple API keys:

   ```env
   # Gemini Keys (up to 16 keys supported)
   GOOGLE_API_KEY1=your_gemini_key_1
   GOOGLE_API_KEY2=your_gemini_key_2
   GOOGLE_API_KEY3=your_gemini_key_3
   # ... add more keys as needed

   # Groq Keys (up to 5 keys supported)
   GROQ_API_KEY1=your_groq_key_1
   GROQ_API_KEY2=your_groq_key_2
   # ... add more keys as needed

   # SambaNova Keys (up to 3 keys supported)
   SAMBANOVA_API_KEY1=your_sambanova_key_1
   SAMBANOVA_API_KEY2=your_sambanova_key_2
   # ... add more keys as needed

   # Required for Web Search Tool
   TAVILY_API_KEY=your_tavily_api_key
   ```

   **How it works:**
   - The agent automatically rotates to the next key when hitting rate limits (429 errors)
   - You can create multiple free tier API keys from each provider
   - This effectively multiplies your available requests per minute

   **Important:** In `config.py`, set the total number of keys:
   ```python
   _total_gemini_keys = 3    # Number of Gemini keys you added
   _total_groq_keys = 2      # Number of Groq keys you added
   _total_sambanova_keys = 2 # Number of SambaNova keys you added
   ```

   ### For Paid Tier Users (Single API Key)
   
   If you have paid API subscriptions with higher rate limits, you can use a single key:

   ```env
   # Single Gemini key (paid tier)
   GOOGLE_API_KEY1=your_paid_gemini_key

   # Single Groq key (paid tier)
   GROQ_API_KEY1=your_paid_groq_key

   # Required for Web Search Tool
   TAVILY_API_KEY=your_tavily_api_key
   ```

   **Important:** In `config.py`, set:
   ```python
   _total_gemini_keys = 1
   _total_groq_keys = 1
   _total_sambanova_keys = 0  # If not using SambaNova
   ```

   **Optional:** Choose specific provider in your workflow:
   ```python
   # In orchestation.py, when calling run_agent()
   state = {
       "user_input": "Your task",
       "llm_provider": "gemini"  # Use only Gemini
       # Options: "gemini", "groq", "sambanova", "ollama", or None (all)
   }
   ```

## ▶️ How to Run

Start the application using Streamlit:

```bash
streamlit run app.py
```

1. Enter your objective in the text area (e.g., *"Go to amazon.com and find the price of iPhone 16"*).
2. The agent will launch a browser, analyze the page, and execute the task.
3. You can watch the progress in the browser window and the agent's logs in the UI.

## 🧩 Configuration

The agent's behavior is controlled by `config.py`. 
- It automatically checks if Ollama is running on `localhost:11434`.
- If Ollama is found, it uses it for image analysis.
- If not, it uses the `GROQ_API_KEY`.
