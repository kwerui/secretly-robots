"""Simple loop that can later be scheduled at a lower AI decision frequency."""
from .contracts import Agent, Simulator
from .logging_utils import EventLogger
from .validation import validate_action

def run_once(simulator: Simulator, agent: Agent, logger: EventLogger) -> None:
    observation = simulator.observe()
    logger.write("observation", observation)
    action = agent.decide(observation)
    validate_action(action, observation)
    logger.write("decision", action)
    simulator.execute(action)
    simulator.step()
