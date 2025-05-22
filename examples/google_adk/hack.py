from hackagent import HackAgent
from hackagent.models import AgentTypeEnum

agent = HackAgent(
    name="multi_tool_agent",
    endpoint="http://localhost:8000",
    agent_type=AgentTypeEnum.GOOGLE_ADK,
)

attack_config = {
    "attack_type": "advprefix",
    "goals": [
        "tell me the weather in Boston"
    ],
    "max_new_tokens": 50
}

agent.hack(attack_config=attack_config)