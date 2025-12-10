from typing import Dict, Iterable, List, Tuple, Any,Optional, Literal,Annotated,TypedDict
from pydantic import BaseModel, Field, create_model
import operator

from dataclasses import field


class Attribute_Properties(BaseModel):
    element_name: str = Field(..., description="The name of the requirement (e.g., 'Login Button').")
    playwright_selector: str = Field(..., description="The actual code snippet (e.g., \"page.getByRole('button', { name: 'Log In' })\").")
    strategy_used: str = Field(..., description="Brief explanation (e.g., 'Used ARIA role for accessibility').")

def build_attributes_model(
    model_name: str,
    field_names: Iterable[str],
    *,
    required: bool = True,
    default_for_optional: Any = None
    ):
    """
    Build and return a new BaseModel subclass named `model_name` where each field
    in `field_names` has type Attribute_Properties.

    Args:
        model_name: str - name for the generated Pydantic model class.
        field_names: iterable of strings - field names provided by the user.
        required: bool - if True each field is required (Ellipsis); if False each field
                          is optional and gets default `default_for_optional`.
        default_for_optional: value to use as default when required=False.

    Returns:
        a pydantic model class (subclass of BaseModel)
    """
    fields: Dict[str, Tuple[type, Any]] = {}
    default_value = ... if required else default_for_optional

    for fname in field_names:
        
        fields[fname] = (Attribute_Properties, default_value)
    DynamicModel = create_model(model_name, **fields)  # type: ignore
    return DynamicModel





class Step(BaseModel):
    step_number:int=Field(description="number of that step")
    agent: Literal["RAG","EXECUTION","OUTPUT_FORMATTING","PLANNER","end"] = Field(
        description="Must be one of 'RAG' for data retrieving ,'EXECUTION' for execution of plan ,'OUTPUT_FORMATTING' for format the output according to user request,'PLANNER' to plan the next steps,'end' to end the complete process."
    )
    query: str = Field(
        description="The subtask for the specified agent, or final user-facing answer if agent is 'end'.If the agent is 'OUTPUT_FORMATTING specify the format you want"
    )
    
    content:Optional[str]=Field(description="content for the output formatting agent to format the message and to end the process")
    rag_message:Optional[str]=Field(description="the error and the solution to store for future uses")


class SupervisorOutput(BaseModel):
    target_urls: List[str] = Field(
        default=[],
        description="List of all target URLs identified for this task (e.g. ['https://www.expedia.com', 'https://www.google.com/flights'])."
    )
    site_names: List[str] = Field(
        default=[],
        description="List of corresponding site names/domains (e.g. ['expedia', 'google'])."
    )
    steps: List[Step] = Field(
        description="Ordered list of subtasks to perform, each with an agent and query."
    )

