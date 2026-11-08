import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "assistive_g1"))

from relocation_benchmark import EPISODES, build_episode_command


def test_relocation_benchmark_uses_supported_sequence_and_shared_memory(tmp_path):
    assert [(episode.name, episode.bottle_target) for episode in EPISODES] == [
        ("episode-1-drawer", "kitchen_drawer"),
        ("episode-2-counter-relocated", "kitchen_counter"),
        ("episode-3-counter-reload", "kitchen_counter"),
    ]

    memory_path = tmp_path / "memory.json"
    command = build_episode_command(
        python="python", demo=Path("house_brain_demo.py"), episode=EPISODES[1], memory_path=memory_path,
        results_dir=tmp_path / "episode-2-counter-relocated", seed=7, max_turns=15,
    )

    assert "--live-memory" in command
    assert command[command.index("--bottle-target") + 1] == "kitchen_counter"
    assert command[command.index("--memory-path") + 1] == str(memory_path)
    assert command[command.index("--results-dir") + 1].endswith("episode-2-counter-relocated")
