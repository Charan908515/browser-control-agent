import streamlit as st
import json
from datetime import datetime
from orchestation import run_agent
from browser_manager import browser_manager

# Set page config
st.set_page_config(
    page_title="Browser Automation Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] button {
        font-size: 1.1rem;
        padding: 0.5rem 1.5rem;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d1e7dd;
        border: 1px solid #badbcc;
        color: #0f5132;
    }
    .error-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #842029;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #cfe2ff;
        border: 1px solid #b6d4fe;
        color: #084298;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("🤖 Autonomous Browser Agent")
st.markdown("Automated web scraping and interaction powered by LLMs")

# Sidebar configuration
st.sidebar.header("⚙️ Configuration")

with st.sidebar:
    st.subheader("Agent Settings")
    
    # Headless mode option
    headless_mode = st.checkbox(
        "Headless Mode",
        value=True,
        help="Run browser in headless mode (no UI)"
    )

# Main content area
tab1, tab3, tab4 = st.tabs(["🚀 Execute", "📊 History", "ℹ️ About"])

with tab1:
    st.subheader("Task Execution")
    
    # Input area for natural language prompt
    agent_input = st.text_area(
        "Agent Instructions",
        placeholder="e.g., Open youtube.com and search for 'Python tutorials', then extracting the titles of the first 5 videos.",
        height=150,
        help="Describe what you want the agent to do in natural language."
    )
    
    st.divider()
    
    # Execute button
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        execute_button = st.button(
            "▶️ Execute Task",
            type="primary",
            use_container_width=True
        )
    
    with col2:
        reset_button = st.button(
            "🔄 Reset",
            use_container_width=True
        )
    
    if reset_button:
        st.rerun()
    
    # Execution logic
    if execute_button:
        if not agent_input.strip():
            st.error("❌ Please enter instructions for the agent.")
        else:
            with st.spinner("🔄 Executing task... This may take a moment"):
                try:
                    # Configure headless mode
                    browser_manager.set_headless_mode(headless_mode)
                    
                    # Display task being executed
                    st.info(f"📝 Task: {agent_input}")
                    st.divider()
                    
                    # Create placeholder for progress updates
                    progress_placeholder = st.empty()
                    
                    with progress_placeholder:
                        st.info("🚀 Starting autonomous browser agent...")
                    
                    # Execute the agent
                    result = run_agent(agent_input)
                    
                    # Clear progress and show results
                    progress_placeholder.empty()
                    
                    # Check if execution was successful
                    if "error" in result:
                        st.error(f"❌ Error: {result['error']}")
                    else:
                        st.success("✅ Task completed successfully!")
                        
                        # Display results
                        st.markdown("#### 📊 Results")
                        
                        output_content = result.get("output", "")
                        
                        try:
                            if isinstance(output_content, str):
                                # Try to extract JSON from the output
                                import re
                                json_match = re.search(r'\{.*\}', output_content, re.DOTALL)
                                if json_match:
                                    parsed_output = json.loads(json_match.group())
                                    st.json(parsed_output)
                                else:
                                    st.markdown(f"```\n{output_content}\n```")
                            else:
                                st.json(output_content)
                        except (json.JSONDecodeError, AttributeError):
                            st.markdown(f"```\n{output_content}\n```")
                        
                        # Store in session for history
                        if "execution_history" not in st.session_state:
                            st.session_state.execution_history = []
                        
                        st.session_state.execution_history.append({
                            "timestamp": datetime.now().isoformat(),
                            "task_input": agent_input,
                            "status": "✅ Success",
                            "output": str(output_content)[:200] + "..." if len(str(output_content)) > 200 else str(output_content)
                        })
                    
                except Exception as e:
                    st.error(f"❌ Error executing task: {str(e)}")
                    st.info("💡 Tips: Ensure all required environment variables are set (API keys, etc.)")
                    
                    # Show error details in expandable section
                    with st.expander("🔧 Error Details"):
                        st.code(str(e))

with tab3:
    st.markdown("#### 📊 Execution History")
    
    # Initialize session state for history if not present
    if "execution_history" not in st.session_state:
        st.session_state.execution_history = []
    
    # Display history
    if st.session_state.execution_history:
        # Convert history to display format
        history_display = []
        for item in st.session_state.execution_history:
            history_display.append({
                "Timestamp": item["timestamp"],
                "Task": item["task_input"],
                "Status": item["status"],
                "Output": item["output"]
            })
        
        st.dataframe(
            history_display,
            use_container_width=True,
            hide_index=True
        )
        
        # Clear history button
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.execution_history = []
            st.rerun()
    else:
        st.info("📭 No execution history yet. Run a task to see results here.")
    
    st.markdown("---")
    st.info("💡 History is stored in your session and cleared when you refresh or close the app.")

with tab4:
    st.markdown("""
    ### ℹ️ About This Application
    
    **Autonomous Browser Agent** is an AI-powered web automation tool.
    
    #### Simply describe what you want to do:
    - "Go to google.com and search for 'latest AI news'"
    - "Open amazon.com and find the price of iPhone 15"
    - "Login to my account at example.com... (careful with credentials!)"
    
    The agent uses:
    - 🧠 **Large Language Models** for reasoning
    - 🎭 **Page Analysis** to understand web pages
    - 🎪 **Playwright** for browser control
    """)

# Footer
st.divider()
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 0.9rem; padding: 1rem;'>
        Built with ❤️ using Streamlit | Powered by Advanced LLMs
    </div>
    """,
    unsafe_allow_html=True
)
