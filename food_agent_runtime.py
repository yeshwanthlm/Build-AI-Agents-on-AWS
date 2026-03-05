"""
Food Recommendation Agent for AgentCore Runtime

This agent uses AgentCore Memory to remember user food preferences across sessions.
Deploy this to AgentCore Runtime with the following environment variables:
  - MEMORY_ID: Your pre-created memory resource ID (e.g., FoodAgentMemory-xyz)
  - MODEL_ID: Bedrock model ID (e.g., us.anthropic.claude-3-5-haiku-20241022-v1:0)
  - AWS_REGION: AWS region (e.g., us-east-1)
"""

import os
import logging
from datetime import datetime

from strands import Agent, tool
from strands.models import BedrockModel
from strands.hooks import (
    AgentInitializedEvent, 
    AfterInvocationEvent,
    HookProvider, 
    HookRegistry
)
from bedrock_agentcore.memory import MemoryClient
from bedrock_agentcore.runtime import BedrockAgentCoreApp

from ddgs import DDGS
from ddgs.exceptions import DDGSException, RatelimitException

# ==========================================
# Configuration & Setup
# ==========================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("food_agent_runtime")

app = BedrockAgentCoreApp()

REGION = os.getenv('AWS_REGION', 'us-east-1')
MODEL_ID = os.getenv('MODEL_ID', 'us.anthropic.claude-3-5-haiku-20241022-v1:0')
MEMORY_ID = os.getenv('MEMORY_ID', 'FoodAgentMemory-2SXptmCV1E') # Needs to be provided in environment for prod

# Global agent instance
agent = None

# ==========================================
# Memory Hook Provider
# ==========================================
class FoodMemoryHookProvider(HookProvider):
    """Automatic memory management for food agent"""
    
    def __init__(self, region_name: str):
        logger.info(f"Initializing FoodMemoryHookProvider with region {region_name}")
        self.memory_client = MemoryClient(region_name=region_name)
    
    def on_agent_initialized(self, event: AgentInitializedEvent):
        """Load food preferences when agent starts"""
        logger.info("Agent initialization hook triggered (FoodMemoryHookProvider)")

        memory_id = event.agent.state.get("memory_id")
        actor_id = event.agent.state.get("actor_id")

        if not memory_id or not actor_id:
            logger.warning(
                f"Missing required state - memory_id: {memory_id}, actor_id: {actor_id}"
            )
            return

        try:
            namespace = f"user/{actor_id}/food_preferences"
            preferences = self.memory_client.retrieve_memories(
                memory_id=memory_id,
                namespace=namespace,
                query="food preferences cuisines dietary restrictions favorites",
                top_k=5
            )
            
            if preferences:
                pref_texts = []
                for pref in preferences:
                    if isinstance(pref, dict):
                        content = pref.get('content', {})
                        if isinstance(content, dict):
                            text = content.get('text', '').strip()
                            if text:
                                pref_texts.append(f"- {text}")
                
                if pref_texts:
                    context = "\\n".join(pref_texts)
                    event.agent.system_prompt += f"\\n\\n## User's Food Preferences:\\n{context}"
                    logger.info(f"✅ Loaded {len(pref_texts)} food preferences for user: {actor_id}")
            else:
                 logger.info("No previous food preferences found - starting fresh!")
                 
        except Exception as e:
            logger.error(f"Error loading preferences: {e}", exc_info=True)
    
    def on_after_invocation(self, event: AfterInvocationEvent):
        """Save conversation after each interaction"""
        logger.info("After invocation hook triggered (FoodMemoryHookProvider)")
        
        memory_id = event.agent.state.get("memory_id")
        actor_id = event.agent.state.get("actor_id")
        session_id = event.agent.state.get("session_id")

        if not memory_id or not actor_id or not session_id:
            logger.warning(
                f"Missing required state for saving - memory_id: {memory_id}, actor_id: {actor_id}, session_id: {session_id}"
            )
            return

        try:
            messages = event.agent.messages
            if len(messages) < 2:
                return
            
            user_msg = None
            assistant_msg = None
            
            for msg in reversed(messages):
                if msg["role"] == "assistant" and not assistant_msg:
                    content = msg.get("content", [])
                    if content and isinstance(content[0], dict) and "text" in content[0]:
                        assistant_msg = content[0]["text"]
                elif msg["role"] == "user" and not user_msg:
                    content = msg.get("content", [])
                    if content and isinstance(content[0], dict) and "text" in content[0]:
                        if "toolResult" not in content[0]:
                            user_msg = content[0]["text"]
                            break
            
            if user_msg and assistant_msg:
                self.memory_client.create_event(
                    memory_id=memory_id,
                    actor_id=actor_id,
                    session_id=session_id,
                    messages=[(user_msg, "USER"), (assistant_msg, "ASSISTANT")]
                )
                logger.info(f"💾 Saved conversation event to memory for session: {session_id}")
                
        except Exception as e:
            logger.error(f"Error saving conversation: {e}", exc_info=True)
    
    def register_hooks(self, registry: HookRegistry):
        logger.info("Registering food memory hooks")
        registry.add_callback(AgentInitializedEvent, self.on_agent_initialized)
        registry.add_callback(AfterInvocationEvent, self.on_after_invocation)


# ==========================================
# Agent Tools (Search & M2M Identity Mock)
# ==========================================
@tool
def search_food(query: str, max_results: int = 5) -> str:
    """Search for food information, recipes, cuisines, or restaurant recommendations.
    
    Args:
        query: Search query about food 
        max_results: Maximum number of results to return
    """
    try:
        results = DDGS().text(f"{query} food recipe restaurant", region="us-en", max_results=max_results)
        if not results:
            return "No results found."
        
        formatted = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "No title")
            body = r.get("body", "")
            formatted.append(f"{i}. {title}\\n   {body}")
        
        return "\\n\\n".join(formatted)
    except RatelimitException:
        return "Rate limit reached. Please try again later."
    except Exception as e:
        return f"Search error: {str(e)}"


def get_system_prompt() -> str:
    """Generate the system prompt for the food agent"""
    return f"""You are a concise food assistant. Help users discover new foods & remember their preferences. 
You can search the web for recipes using `search_food`. 
Date: {datetime.today().strftime("%Y-%m-%d")}"""

# ==========================================
# Core Agent Creator
# ==========================================
def initialize_agent(actor_id: str, session_id: str):
    """Initialize the food agent with memory hooks"""
    global agent

    logger.info(
        f"Initializing food agent for actor_id={actor_id}, session_id={session_id}"
    )

    logger.info(f"Creating BedrockModel with ID: {MODEL_ID}")
    model = BedrockModel(model_id=MODEL_ID)

    logger.info(f"Creating memory hook with region: {REGION}")
    memory_hook = FoodMemoryHookProvider(region_name=REGION)

    agent = Agent(
        model=model,
        hooks=[memory_hook],
        tools=[search_food],
        system_prompt=get_system_prompt(),
        state={
            "memory_id": MEMORY_ID, 
            "actor_id": actor_id, 
            "session_id": session_id
        },
    )

    logger.info(f"✅ Food agent initialized with state: {agent.state.get()}")

# ==========================================
# Bedrock AgentCore Runtime Entrypoint
# ==========================================
@app.entrypoint
def food_agent(payload: dict, context):
    """
    Main entry point for the food recommendation agent.

    Expected payload:
    {
        "prompt": "User's message",
        "actor_id": "unique_user_id"  # Optional, defaults to "default_user"
    }

    The session_id comes from context.session_id (managed by AgentCore Runtime)
    """
    global agent

    logger.info(f"Received payload: {payload}")
    logger.info(f"Context session_id: {context.session_id}")
    print(f"Session ID: {context.session_id}")

    # Extract values from payload
    user_input = payload.get("prompt")
    actor_id = payload.get("actor_id", "default_user")
    session_id = context.session_id

    print(f"Actor ID: {actor_id}")
    print(f"Memory ID: {MEMORY_ID}")

    # Validate required fields
    if not user_input:
        error_msg = "❌ ERROR: Missing 'prompt' field in payload"
        logger.error(error_msg)
        return error_msg

    if not MEMORY_ID:
        error_msg = "❌ ERROR: MEMORY_ID environment variable not set. Please deploy properly to Agent Core runtime."
        logger.error(error_msg)
        return error_msg

    # Initialize agent on first request or if session changed
    if agent is None:
        logger.info("First request - initializing agent")
        initialize_agent(actor_id, session_id)
    else:
        # Update state if actor_id or session_id changed
        current_session = agent.state.get("session_id")
        current_actor = agent.state.get("actor_id")

        if current_session != session_id or current_actor != actor_id:
            logger.info(f"Session or actor changed - reinitializing agent")
            initialize_agent(actor_id, session_id)

    # Invoke the agent
    logger.info(f"Invoking agent with input: {user_input}")
    response = agent(user_input)

    # Extract response text
    response_text = response.message["content"][0]["text"]
    logger.info(f"✅ Agent response: {response_text[:100]}...")

    return response_text


if __name__ == "__main__":
    logger.info("Starting Food Agent on Bedrock AgentCore Runtime")
    app.run()
