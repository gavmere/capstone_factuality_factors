# agents/base_agent.py
from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseAgent(ABC):
    name: str
    description: str

    @abstractmethod
    def run(self, text: str) -> Dict[str, Any]:
        pass