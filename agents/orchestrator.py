from typing import Dict, List
from agents.base_agent import BaseAgent

class OrchestratorAgent:
    def __init__(self, agents: List[BaseAgent]):
        self.agents = agents

    def run(self, text: str) -> Dict[str, Dict]:
        results = {}

        for agent in self.agents:
            results[agent.name] = agent.run(text)

        return results
