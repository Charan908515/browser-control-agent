import os
from prompts import get_central_agent_prompt2,get_autonomous_browser_prompt4,get_central_agent_prompt5
from langchain_core.messages import ChatMessage
# Removed direct LLM imports
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from browser_manager import browser_manager
from config import LLMConfig
from browser_tools import (
        enable_vision_overlay, 
        find_element_ids,      
        click_id,              
        fill_id,               
        scroll_one_screen,
        press_key,
        get_page_text,
        hover_element,
        get_visible_input_fields,
        extract_text_from_selector,
        extract_attribute_from_selector,
        select_dropdown_option,
        open_dropdown_and_select    
)
from analyze_tools import (
     open_browser, close_browser,
    extract_and_analyze_selectors, analyze_using_vision,scrape_data_using_text
)
from schemas import Step,SupervisorOutput
import json
import time
import re
from urllib.parse import urlparse
from pydantic import BaseModel,Field
from typing import Optional,Literal,List,Annotated,Sequence
from langchain_community.tools.tavily_search import TavilySearchResults
import dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain.schema import Document
from langchain_core.messages import ChatMessage ,HumanMessage,BaseMessage,SystemMessage,AIMessage
import operator
from langgraph.graph import StateGraph,START,END,MessagesState
from dataclasses import field
from langgraph.types import Command
import nest_asyncio
import sys
import asyncio

# FIX: Enforce ProactorEventLoop on Windows for Playwright
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

nest_asyncio.apply()
dotenv.load_dotenv()


def extract_json_from_markdown(text: str) -> str:
    """Extract JSON from markdown code blocks like ```json ... ```"""
    # Try to find JSON in markdown code blocks
    pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    match = re.search(pattern, text, re.DOTALL)
    
    if match:
        return match.group(1).strip()
    
    # If no markdown blocks found, return the original text
    return text.strip()


def get_current_browser_info():
    """
    Directly inspects the open browser to get the current URL and Site Name.
    Source of Truth for where an error actually happened.
    """
    page = browser_manager.get_page()
    
    # Check if browser is actually open and valid
    if page and not page.is_closed():
        try:
            current_url = page.url
            # Extract domain (e.g., "https://www.google.com/..." -> "google.com")
            parsed = urlparse(current_url)
            site_name = parsed.netloc.replace("www.", "")
            
            # If local file or empty
            if not site_name: 
                site_name = "local_or_unknown"
                
            return current_url, site_name
        except Exception as e:
            print(f"Error reading browser URL: {e}")
            
    return "unknown_url", "unknown_site"

def central_agent1(state):
    user_input = state["user_input"]
    # Sanitize input
    user_input = user_input.replace("{", "{{").replace("}", "}}")
    
    site_names = state.get("site_names", [])
    urls = state.get("urls", [])
    
    # current_plan holds the state from the PREVIOUS run (or empty if fresh)
    current_plan = state.get("plan", [])
    
    # step_index is crucial: 
    # - If Redirector sends 0 and we have data, it's a "New Phase".
    # - If step_index > 0, it might be an error or continuation.
    current_index = state.get("step_index", 0)
    
    last_error = state.get("last_error", None)
    
    print(f">>> PLANNING AGENT: Step Index: {current_index}")
    
    # --- 1. MEMORY & CONTEXT PREPARATION ---
    
    # Retrieve general RAG errors (historical knowledge)
    if not site_names:
        historical_errors = "No previous errors"
    else:
        historical_errors = retrieve_errors(state) if vector_db else "No previous errors"
    historical_errors = historical_errors.replace("{", "{{").replace("}", "}}")

    # Get Browser State (Grounding)
    print(">>> Planner is scanning the page...")
    if browser_manager.get_page():
        try:
            # First 1000 chars to identify Page State
            raw_text = browser_manager.get_page().evaluate("document.body.innerText")[:1000]
            current_page_state = f"Page Title: {browser_manager.get_page().title()}\nVisible Text Snippet: {raw_text}..."
        except:
            current_page_state = "Browser is open but page content is unreadable."
    else:
        current_page_state = "Browser is NOT open. First step must be 'Open Browser'."
    current_page_state = current_page_state.replace("{", "{{").replace("}", "}}")

    # Initialize Context Variables
    completed_steps = []          # The steps we will KEEP in the final plan
    completed_context_str = ""    # The text description of history for the LLM
    extracted_data_context = ""   # Data found so far
    immediate_error_context = ""
    
    parser = JsonOutputParser(pydantic_object=SupervisorOutput)
    instructions = parser.get_format_instructions().replace("{", "{{").replace("}", "}}")

    # --- 2. DETERMINE PLANNING MODE ---

    # MODE A: PHASE 2 / RE-PLANNING (Data exists + Redirector reset index to 0)
    # This happens when the Redirector sees "PLANNER" step or finishes a loop and sends us back.
    if state.get("output_content") and current_index == 0:
        print(">>> PLANNER MODE: PHASE 2 (Data Driven Re-planning)")
        
        # 1. Build History String from the OLD plan (Phase 1)
        if current_plan:
            completed_context_str = "\n### COMPLETED HISTORY (PHASE 1):\n"
            for step in current_plan:
                completed_context_str += f"✓ Step {step['step_number']}: {step['query']} (Agent: {step['agent']})\n"
        completed_context_str = completed_context_str.replace("{", "{{").replace("}", "}}")
        # 2. Prepare Data Context
        recent_data = state["output_content"][-2:] # Last 2 items to save tokens
        extracted_data_context = f"\n### DATA EXTRACTED SO FAR (Use this to plan Phase 2):\n{str(recent_data)}\n"
        extracted_data_context = extracted_data_context.replace("{", "{{").replace("}", "}}")
        
        # 3. Reset Steps (We want a FRESH plan for Phase 2, not to append to Phase 1)
        completed_steps = [] 

        system_message = f"""
                    You are the **Browser Automation Architect**.
                    We have just completed Phase 1 of the user's request.

                    USER REQUEST: "{user_input}"

                    {completed_context_str}

                    {extracted_data_context}

                    THE CURRENT PAGE STATE IS:
                    {current_page_state}

                    ### INSTRUCTIONS FOR PHASE 2:
                    1. Review the 'COMPLETED HISTORY' and 'DATA EXTRACTED'.
                    2. Plan the **NEXT** logical actions based on the extracted data.
                    - Example: If we extracted "Gladiator 2" from IMDB, your next steps should be "Go to YouTube" and "Search for Gladiator 2".
                    3. **DO NOT** repeat the completed steps.
                    4. If this next phase also requires a pause for data extraction, the FINAL step of this plan should be agent='PLANNER'.
                    5. If the task is fully complete, the final step should be agent='end'.

                    {instructions}
                    """
        

    # MODE B: ERROR RECOVERY (Something failed mid-execution)
    elif last_error:
        print(f">>> PLANNER MODE: ERROR RECOVERY (Step {current_index} Failed)")
        
        # 1. Keep successful steps
        if current_plan:
            completed_steps = current_plan[:current_index]
            print(f">>> RETAINING {len(completed_steps)} SUCCESSFUL STEPS.")
        
        # 2. Build Context
        completed_context_str = "\nTHE FOLLOWING STEPS ARE ALREADY COMPLETED. DO NOT RE-PLAN THEM:\n"
        for step in completed_steps:
            completed_context_str += f"- Step {step['step_number']}: {step['query']}\n"
            
        immediate_error_context = (
            f"\n\nCRITICAL FAILURE IN PREVIOUS ATTEMPT:\n"
            f"The execution failed at Step {current_index + 1} with error: {last_error}\n"
            f"YOU MUST GENERATE A NEW PLAN STARTING FROM STEP {current_index + 1} THAT FIXES THIS ERROR."
        )

        completed_context_str = completed_context_str.replace("{", "{{").replace("}", "}}")
        immediate_error_context = immediate_error_context.replace("{", "{{").replace("}", "}}")
        
        system_message = f"""
You are the **Browser Automation Architect**.
We are in the middle of an execution that encountered an error.

USER REQUEST: "{user_input}"

{completed_context_str}

{immediate_error_context}

THE CURRENT PAGE STATE IS:
{current_page_state}

### INSTRUCTIONS FOR REFINEMENT:
1. **IGNORE** the completed steps in your output JSON (they are kept automatically).
2. **GENERATE ONLY** the remaining steps needed to fix the error and finish.
3. The first step of your new plan must address the error described above.

{instructions}
"""

    # MODE C: FRESH START (First run)
    else:
        print(">>> PLANNER MODE: FRESH START")
        completed_steps = []
        system_message = get_central_agent_prompt5(user_input, historical_errors, instructions)


    # --- 3. LLM EXECUTION ---
    
    # Configure LLMs (Primary & Fallback)
    llm = LLMConfig.get_main_llm()
    
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("human", "Please generate the plan."),
        MessagesPlaceholder(variable_name="chat_history"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])

    tavily = TavilySearchResults(tavily_api_key="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")
    tools = [tavily, close_browser]

    agent = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=30,
        handle_parsing_errors=True,
        return_intermediate_steps=True
    )

    # Sanitize History for LLM
    sanitized_history = []
    for msg in state["messages"]:
        sanitized_content = msg.content.replace("{", "{{").replace("}", "}}")
        if isinstance(msg, (HumanMessage, AIMessage)):
            sanitized_history.append(type(msg)(content=sanitized_content))
        else:
            sanitized_history.append(HumanMessage(content=sanitized_content))

    try:
        response = executor.invoke({"chat_history": sanitized_history})
        clean_json = extract_json_from_markdown(response["output"])
        result = json.loads(clean_json)
        
        detected_urls = result.get("target_urls", [])
        detected_sites = result.get("site_names", [])
        new_steps = result.get("steps", result) if isinstance(result, dict) else result
        
        if not isinstance(new_steps, list):
            new_steps = []

        # RAG: Save Error Resolution if applicable
        if last_error and vector_db and len(new_steps) > 0:
            try:
                url, site_name = get_current_browser_info()
                fix_action = new_steps[0].get('query', 'Unknown Action')
                rag_content = f"Error Encountered: {last_error}\nSuccessful Fix/Next Step: {fix_action}"
                doc = Document(
                    page_content=rag_content,
                    metadata={
                        "url": url, "site_name": site_name,
                        "type": "error_resolution", "related_step_index": current_index
                    }
                )
                vector_db.add_documents([doc])
                print(f">>> SAVED ERROR & SOLUTION TO RAG")
            except Exception as e:
                print(f"RAG Save Failed: {e}")

        # --- 4. MERGE PLANS ---
        # If Phase 2: completed_steps is [], so final_plan = new_steps. (Correct: We overwrite old plan)
        # If Error: completed_steps is [0..X], so final_plan = [0..X] + new_steps. (Correct: We append)
        final_plan = completed_steps + new_steps
        
        for i, step in enumerate(final_plan):
            step['step_number'] = i + 1

        print(f">>> PLAN GENERATED. New Steps: {len(new_steps)}, Total Steps: {len(final_plan)}")

    except Exception as e:
        print(f"Planning Logic Error: {e}")
        final_plan = current_plan # Fallback to existing plan on crash

    return {
        "plan": final_plan,
        "step_index": len(completed_steps), # Start at 0 for Phase 2, or at failure point for Error
        "last_error": None, 
        "urls": detected_urls,
        "site_names": detected_sites,
        "messages": [HumanMessage(content=f"[Planner]: Plan updated. Phase steps: {len(new_steps)}.")]
    }



def redirector(state):
    #print("wait for 30 seconds")
    #time.sleep(30)
    print(">>>> Redirector WORKING")
    plan=state["plan"]
    index=state["step_index"]
    steps=plan
    if index >= len(plan):
        print("Plan execution completed.")
        return Command(goto=END)

    step = plan[index]
    print(f"Executing step {index + 1}/{len(steps)} → {step['agent']}: {step['query']}")
    next_step_update = {"step_index": index + 1}
    if step["agent"] == "PLANNER":
        print(">>> Step is 'PLANNER'. Resetting plan and sending back to Architect.")
        
        # 1. We keep 'output_content' and 'messages' so the Planner knows history.
        # 2. We RESET 'plan' to empty so the Planner can write a NEW list.
        # 3. We RESET 'step_index' to 0 for the new plan.
        return Command(
            update={
                "step_index": 0,    
                # Optional: Add a message to help the planner focus
                "messages": [HumanMessage(content="Phase 1 complete. Data extracted. Please plan Phase 2.")]
            },
            goto="planner"
        )
    elif step["agent"]=="RAG":
        # Use rag_message if available, otherwise fall back to query
        message_content = step["rag_message"] or step["query"]
        new_msg = HumanMessage(content=message_content)
        
        return Command(goto="rag_agent", update={**next_step_update, "rag_messages": [new_msg]})
    elif step["agent"]=="EXECUTION":
        new_msg = HumanMessage(content=step["query"])
        return Command(goto="executor",update={**next_step_update,"execution_messages":[new_msg]})
    elif step["agent"]=="OUTPUT_FORMATTING":
        new_msg = HumanMessage( content=step["query"])
        content_to_append = step.get("content", "")
        new_content_list = state["output_content"] + [content_to_append] if content_to_append else state["output_content"]
        
        return Command(goto="output_agent",update={**next_step_update,"output_agent_messages":[new_msg],"output_content":new_content_list})
    elif step["agent"]=="end":
        return Command(goto=END,update={"step_index":0,"plan":[]})
    else:
        error_msg = f"Error at step {index}: Unknown agent {step['agent']}"
        return Command(
            update={"messages": [ChatMessage(role="redirector", content=error_msg)]},
            goto="planner"
        )


def execution_agent(state):
    task_msg=state["execution_messages"][-1]
    task = task_msg.content if hasattr(task_msg, 'content') else str(task_msg)
    task=task.replace("{", "{{").replace("}", "}}")
    sanitized_history = []
    for msg in state["execution_messages"][:-1]: # Exclude the current task
        if isinstance(msg, HumanMessage) or isinstance(msg, AIMessage):
            sanitized_history.append(msg)
        else:
            # Convert generic BaseMessage or others to HumanMessage
            sanitized_history.append(HumanMessage(content=str(msg.content)))
            
    # USE CONFIG
    llm = LLMConfig.get_main_llm()
    
    tavily = TavilySearchResults(tavily_api_key="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")
    tools = [
        tavily,
        enable_vision_overlay, # The "Scanner"
        find_element_ids,      # The "Search Engine"
        click_id,              # The "Finger"
        fill_id,               # The "Keyboard"
        scroll_one_screen,
        press_key,
        get_page_text,
        open_browser,
        scrape_data_using_text,
        analyze_using_vision ,
        extract_and_analyze_selectors,
         hover_element,
        get_visible_input_fields,
        extract_text_from_selector,
        extract_attribute_from_selector,
        select_dropdown_option,
        open_dropdown_and_select
    ]
    
    # Use Gemini for execution (it supports tool calling)
    agent = create_tool_calling_agent(llm, tools, get_autonomous_browser_prompt4())
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=30,
        handle_parsing_errors=True,
        return_intermediate_steps=True
    )

    try:
        print(f">>> Starting  Execution Agent Task {state['step_index']}...")
        result = agent_executor.invoke({"input": task,"chat_history":[]})
        output_text = result["output"]
        print("\n>>> FINAL OUTPUT:")
        output_lower = output_text.lower()
        trigger_word = next((w for w in ["unable", "error", "couldn't", "failed","execution failed"] if w in output_lower), None)
        
        if trigger_word:
            if "no error" not in output_lower:
                print(f"\n>>> Execution Agent Error: Detected potential failure keyword '{trigger_word}'")
                return Command(
                    update={"last_error": output_text},
                    goto="planner"
                )
        new_msg = ChatMessage(role="execution_agent", content=output_text)
        
        update_dict = {"execution_messages": [new_msg],"messages": [new_msg]}
        extracted_data = []
        
        # 1. Check Intermediate Steps for raw tool outputs (The Fix)
        if "intermediate_steps" in result:
            for action, observation in result["intermediate_steps"]:
                # Capture data from specific extraction tools
                if action.tool in ["scrape_data_using_text", "analyze_using_vision", "extract_and_analyze_selectors"]:
                    # Ensure observation is a string for the message history
                    content = json.dumps(observation) if isinstance(observation, (dict, list)) else str(observation)
                    extracted_data.append(content)

        # 2. Fallback: If tools didn't return data, check the final text
        if not extracted_data and len(output_text) > 20 and ("{" in output_text or "[" in output_text):
            extracted_data = [output_text]

        if extracted_data:
            update_dict["output_content"] = extracted_data

    

        return Command(update=update_dict, goto="redirector")
        
    except Exception as e:
        error_str = str(e).lower()
        
        # --- NEW LOGIC: CRITICAL EXIT ON RATE LIMIT ---
        if "429" in error_str or "rate limit" in error_str or "quota" in error_str:
            print(f"\n>>> 🛑 CRITICAL TERMINATION: Rate Limit/Quota Exceeded. Stopping Agent.\nError: {str(e)}")
            
            # Send a final message explaining why it stopped
            final_msg = ChatMessage(
                role="execution_agent", 
                content=f"Workflow Terminated: API Rate Limit (429) or Quota Exceeded."
            )
            
            return Command(
                update={
                    "messages": [final_msg],
                    "last_error": f"CRITICAL 429: {str(e)}"
                },
                goto=END  # <--- EXITS THE GRAPH IMMEDIATELY
            )
        # ----------------------------------------------

        # Standard error handling (send back to Planner for retry)
        error_msg = f"AGENT CRASHED: {str(e)}"
        print(f"\n>>> {error_msg}")
        new_msg = ChatMessage(role="execution_agent", content=error_msg)
        
        return Command(
            update={
                "execution_messages": [new_msg], 
                "messages": [new_msg],
                "step_index": state["step_index"], 
                "last_error": error_msg 
            },
            goto="planner"
        )




def output_formatting_agent(state):
    print(">>> OUTPUT FORMATTING AGENT")
    # Define LLM inside the function since it wasn't defined globally or passed in
    llm = LLMConfig.get_main_llm()
    input_message=state["output_agent_messages"][-1]
    if hasattr(input_message, 'content'):
        input_message=input_message.content
    else:
        input_message=str(input_message)
    content_to_format = state["output_content"] if state["output_content"] else "No content to format."
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a data extraction specialist. \nINSTRUCTIONS:\n{instructions}"),
        ("human", "RAW DATA:\n{data}")
    ])
    
    chain = prompt |llm
    result = chain.invoke({"instructions": input_message, "data": content_to_format})
    formatted_output = result.content

    print(f">>> Formatted Output: {formatted_output[:100]}...")
    
    return Command(
        update={"Output": formatted_output}, 
        goto="redirector" # Go back to redirector to finish or do next step
    )


try:
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = Chroma(
        persist_directory="./rag_data", 
        embedding_function=embeddings,
        collection_name="agent_memories"
    )
except Exception as e:
    print(f"Error loading vector database: {str(e)}")
    vector_db = None


def rag(state):
    print(">>> RAG WORKING")
    rag_content = state["rag_messages"][-1]
    if hasattr(rag_content, 'content'): 
        rag_content = rag_content.content
    else:
        rag_content=str(rag_content)
    url,site_name=get_current_browser_info()
    
    plan = state.get("plan", [])
    index = state.get("step_index", 0)
    
    if index < len(plan):
        task = plan[index]['query']
        agent = plan[index]['agent']
    else:
        task = "Unknown"
        agent = "Unknown"

    doc = Document(
        page_content=str(rag_content),  
        metadata={
            "url": url,
            "site_name": site_name,
            "task": task,
            "agent": agent
        }
    )
    vector_db.add_documents([doc])
    
    # Return Command to control flow and avoid parallel execution
    return Command(
        update={"messages": [ChatMessage(role="RAG Agent", content=f"Memory Saved: '{rag_content[:50]}...'")]},
        goto="redirector"
    )

def retrieve_errors(state):
    # --- NEW LOGIC ---
    current_sites = state.get("site_names", [])
    if not current_sites:
        return "No specific sites identified yet."

    combined_errors = "PAST ERRORS/LESSONS:\n"
    found_any = False

    for site in current_sites:
        try:
            results = vector_db.similarity_search(
                query="error failure issue fix", 
                k=3, 
                filter={"site_name": site} 
            )
            if results:
                found_any = True
                combined_errors += f"\n--- For {site} ---\n"
                for i, doc in enumerate(results):
                    prev_task = doc.metadata.get('task', 'General Task')
                    combined_errors += f"{i+1}. [Task: {prev_task}]: {doc.page_content}\n"
        except Exception as e:
            print(f"Error retrieving for {site}: {e}")

    if not found_any:
        return "No previous errors found for these sites."
            
    return combined_errors




class AgentState(MessagesState):
    user_input: str
    urls: Annotated[List[str], operator.add]
    site_names: Annotated[List[str], operator.add]   
    step_index: int
    plan: List[dict]
    execution_messages: Annotated[List[BaseMessage], operator.add]
    rag_messages: Annotated[List[BaseMessage], operator.add]
    output_agent_messages: Annotated[List[BaseMessage], operator.add]
    output_content: Annotated[List[str], operator.add]
    Output: str
    last_error:Optional[str]=None
def create_agent():
    workflow=StateGraph(AgentState)
    workflow.add_node("planner",central_agent1)
    workflow.add_node("executor",execution_agent)
    workflow.add_node("output_agent",output_formatting_agent)
    workflow.add_node("redirector",redirector)
    workflow.add_node("rag_agent",rag)
    workflow.add_edge(START,"planner")
    workflow.add_edge("planner","redirector")
    # Removed static edges to allow Command to control flow
    # workflow.add_edge("redirector","executor")
    # workflow.add_edge("redirector","output_agent")
    # workflow.add_edge("redirector","rag_agent")
    # workflow.add_edge("executor","planner")
    # workflow.add_edge("rag_agent","planner")
    # workflow.add_edge("redirector",END)
    # workflow.add_edge("output_agent",END)

    app=workflow.compile()
    return app

def run_agent(input_str:str):
    state = {
    "user_input": input_str,
    "site_names": [],
    "urls": [],
    "plan": [],
    "messages": [],
    "step_index": 0,
    "execution_messages": [],
    "rag_messages": [],
    "output_agent_messages": [],
    "output_content": [],
    "Output": ""
    }
    try:
        app=create_agent()
        response = app.invoke(state)
        for key, value in response.items():
            print(">>>>",key)
            print(value)
            print("#"*100)
        return {"output":response["Output"]}
    except Exception as e:
        print(f"An error occurred during execution: {e}")
        return {"error":"Error during agent running: {e}"}
    finally:
        print(">>> CLEANUP: Closing Browser...")
        
        if browser_manager.is_browser_open():
            browser_manager.close_browser()
    
# Wrap the execution in a try/finally block
if __name__ == "__main__":
    a="Open naukri.com, login with ncharankumareddy@gmail.com/charan#30, search for AI Engineer, extract results."
    output=run_agent(a)