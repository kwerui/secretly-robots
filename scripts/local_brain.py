"""Try a real local Gemma call (requires Ollama running and model downloaded)."""
from brain.engine import Brain
from brain.intent import OllamaIntentParser, IntentError
from brain.contracts import Observation, ObjectSighting

if __name__ == "__main__":
    brain = Brain(parser=OllamaIntentParser())
    try:
        print("Interpreted:", brain.start_task("Please fetch my water bottle"))
        print("Next action:", brain.next_action(Observation(0, "sofa", (ObjectSighting("water_bottle", "cupboard", "drawer"),))))
    except IntentError as e:
        print("Local Gemma is not ready:", e)
        print("Install and start Ollama; pull gemma4:e2b; or use the offline rule-based parser.")
