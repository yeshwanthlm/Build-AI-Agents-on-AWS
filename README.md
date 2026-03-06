# Food Recommendation Agent - AWS AgentCore Demo

This repository demonstrates how to build and deploy a sophisticated GenAI Food Recommendation Agent using the **Strands** framework and **AWS Bedrock AgentCore**.

The agent is capable of searching the web for food/recipes and utilizing **AgentCore Memory** to persistently remember a user's dietary preferences (like peanut allergies) or favorite cuisines across different sessions.

Presentation Deck: [Build & Deploy AI Agent.pdf](https://github.com/user-attachments/files/25800769/Build.Deploy.AI.Agent.pdf)

## 📁 Repository Structure

```text
Build-AI-Agent-on-AWS/
├── food_agent_runtime.py       # Core AgentCore Application (Production deployment ready)
├── food-agent.ipynb            # Jupyter Notebook demonstrating local execution, memory initialization, & Identity/Policy concepts
├── .env                        # Local environment variables
├── requirements.txt            # Python dependencies (strands-agents, bedrock-agentcore, ddgs, etc.)
```

---

## 🚀 Production Demo: AgentCore CLI Workflow

To deploy and test this agent using the **AWS AgentCore Runtime**, follow entirely serverless paradigm using the internal `agentcore` CLI tool.

### 1. Environment Setup

Ensure your virtual environment is activated and the necessary toolkits are installed:

```bash
# Create your python virtual environment
python3 -m venv venv
# Activate your python virtual environment
source .venv/bin/activate

# Install Strands, Bedrock AgentCore SDK, and the internal AgentCore CLI toolkits
pip install strands-agents
pip install bedrock-agentcore-starter-toolkit
pip install agentcore
```

### 2. Configure AgentCore

Configure the CLI to link your local workspace to your AWS environment (region, memory ID defaults, model defaults, etc.):

```bash
agentcore configure
```
*(Follow the interactive prompts to define the required `MEMORY_ID` and Foundation Model bindings like `us.anthropic.claude-3-5-haiku-20241022-v1:0`)*

### 3. Deploy the Runtime

Package and launch the `food_agent_runtime.py` file to the AWS Bedrock AgentCore infrastructure:

```bash
agentcore launch
```

### 4. Invoke the Deployed Agent

Test the deployed agent persistently. Notice how passing the `"actor_id": "food-lover-001"` allows it to pull from the `FoodAgentMemory` we configured!

```bash
agentcore invoke '{"prompt": "Recommend me some dishes based on my preferences", "actor_id":"food-lover-001"}'
```

_If your established memory remembered your peanut allergy and preference for Thai food, the deployed agent will immediately recognize those traits when generating the response!_

---

## 🔐 Advanced Topics Explored

- **Memory Persistence Strategy:** Using the `HookProvider` to automatically intercept `AgentInitializedEvent` (to load context) and `AfterInvocationEvent` (to save conversational turns natively into Bedrock Memory).
- **Least-Privilege DevOps:** Included `terraform/` configurations restrict the agent's IAM execution role explicitly just to the required Haiku models and the specific `/memory/*` actions.
- **AgentCore Identity & Policy:** Outlined concepts (via the local notebook) on how `@requires_iam_access_token` and the Agent Core Gateway dynamically evaluate and govern tool access using JWTs and Cedar policies. 
