import csv
import json

from assistive_g1.evaluation import (
    EpisodeMetrics,
    Outcome,
    TrialRecorder,
    TruthSnapshot,
    verify_truth,
    write_results,
)


def test_truth_verification_requires_physical_delivery_after_settling():
    verified = verify_truth(
        TruthSnapshot(
            item_on_tray=True,
            item_on_tray_after_settling=True,
            grasp_attached=False,
            drawer_closed=True,
            robot_fallen=False,
        )
    )

    assert verified.delivery is Outcome.SUCCESS
    assert verified.grasp_released is Outcome.SUCCESS
    assert verified.drawer_closed is Outcome.SUCCESS
    assert verified.task_outcome is Outcome.SUCCESS


def test_truth_verification_does_not_equate_skill_success_with_delivery():
    verified = verify_truth(
        TruthSnapshot(
            item_on_tray=False,
            item_on_tray_after_settling=False,
            grasp_attached=False,
            drawer_closed=True,
            robot_fallen=False,
        )
    )

    assert verified.delivery is Outcome.FAILURE
    assert verified.task_outcome is Outcome.FAILURE
    assert verified.failure_stage == "delivery"


def test_failed_search_keeps_delivery_not_attempted_and_records_navigation_target():
    recorder = TrialRecorder(EpisodeMetrics.baseline("first-baseline-shape", "water_bottle"))
    recorder.record_feedback("search", False, "searched every likely place")
    recorder.record_navigation_attempts([
        {"station": "kitchen_counter", "reported_success": True, "accepted_for_search": True, "error": ""},
        {"station": "kitchen_drawers", "reported_success": False, "accepted_for_search": False,
         "error": "navigation timeout"},
    ])

    metrics = recorder.finish(
        TruthSnapshot(
            item_on_tray=False,
            item_on_tray_after_settling=False,
            grasp_attached=False,
            drawer_closed=True,
            robot_fallen=False,
        )
    )

    assert metrics.delivery is Outcome.NOT_ATTEMPTED
    assert metrics.physical_delivery is Outcome.FAILURE
    assert metrics.first_failed_action == {
        "action": "search", "target": None, "reason": "searched every likely place"
    }
    assert metrics.first_failed_navigation_attempt["station"] == "kitchen_drawers"
    assert metrics.failure_stage == "search"


def test_search_failure_does_not_override_successful_navigation_attempts():
    recorder = TrialRecorder(EpisodeMetrics.baseline("search-vs-navigation", "water_bottle"))
    recorder.record_feedback("search", False, "drawer opening failed", target="water_bottle")
    recorder.record_navigation_attempts([
        {"station": "kitchen_counter", "reported_success": True, "accepted_for_search": True, "error": ""},
        {"station": "kitchen_drawers", "reported_success": True, "accepted_for_search": True, "error": ""},
        {"station": "kitchen_table", "reported_success": True, "accepted_for_search": True, "error": ""},
    ])
    recorder.record_drawer_opening_attempts([
        {"station": "kitchen_drawers", "result_ok": False, "result_error": "opening stalled",
         "before": {"drawer_open_m": 0.0}, "after": {"drawer_open_m": 0.078}},
    ])
    metrics = recorder.finish(TruthSnapshot(robot_fallen=False))

    assert metrics.navigation is Outcome.SUCCESS
    assert metrics.first_failed_navigation_attempt is None
    assert metrics.drawer_opening is Outcome.FAILURE
    assert metrics.first_failed_drawer_opening_attempt["after"]["drawer_open_m"] == 0.078
    assert metrics.failure_stage == "search"


def test_grasp_release_is_not_attempted_when_no_grasp_action_ran():
    recorder = TrialRecorder(EpisodeMetrics.baseline("no-grasp", "water_bottle"))
    metrics = recorder.finish(TruthSnapshot(grasp_attached=False, robot_fallen=False))

    assert metrics.grasp is Outcome.NOT_ATTEMPTED
    assert metrics.grasp_attachment_released is Outcome.NOT_ATTEMPTED


def test_attempted_delivery_and_physical_check_are_reported_separately():
    recorder = TrialRecorder(EpisodeMetrics.baseline("delivery-shape", "water_bottle"))
    recorder.record_feedback("deliver", True, "placed on user_tray", target="sofa")
    metrics = recorder.finish(
        TruthSnapshot(item_on_tray=False, item_on_tray_after_settling=False, grasp_attached=False,
                      drawer_closed=True, robot_fallen=False)
    )

    assert metrics.delivery is Outcome.SUCCESS
    assert metrics.physical_delivery is Outcome.FAILURE
    assert metrics.failure_stage == "physical_delivery"


def test_open_drawer_after_successful_delivery_is_drawer_closing_failure():
    recorder = TrialRecorder(EpisodeMetrics.baseline("open-drawer", "water_bottle"))
    recorder.record_feedback("grasp", True, "attached", target="water_bottle")
    recorder.record_feedback("deliver", True, "placed on user_tray", target="sofa")
    metrics = recorder.finish(
        TruthSnapshot(item_on_tray=True, item_on_tray_after_settling=True, grasp_attached=False,
                      drawer_closed=False, robot_fallen=False)
    )

    assert metrics.physical_delivery is Outcome.SUCCESS
    assert metrics.drawer_closed_in_sim is Outcome.FAILURE
    assert metrics.task_outcome is Outcome.FAILURE
    assert metrics.failure_stage == "drawer_closing"
    assert metrics.failure_reason == "drawer remained open"


def test_incomplete_episode_keeps_unattempted_stages_explicit():
    recorder = TrialRecorder(EpisodeMetrics.baseline("exp-1", "water_bottle"))
    recorder.record_decision("navigate", "deterministic_planner")
    recorder.record_feedback("navigate", True, "")
    metrics = recorder.finish(TruthSnapshot(robot_fallen=False))

    assert metrics.navigation is Outcome.SUCCESS
    assert metrics.drawer_opening is Outcome.NOT_ATTEMPTED
    assert metrics.delivery is Outcome.NOT_ATTEMPTED
    assert metrics.task_outcome is Outcome.UNKNOWN


def test_recorder_counts_grasp_retry_and_cloud_fallback_provenance():
    recorder = TrialRecorder(EpisodeMetrics.baseline("exp-2", "water_bottle"))
    recorder.record_decision("grasp", "gemini_argon")
    recorder.record_feedback("grasp", False, "missed")
    recorder.record_decision("grasp", "deterministic_planner", fallback=True)
    recorder.record_feedback("grasp", True, "", retry_count=1)
    metrics = recorder.finish(TruthSnapshot(robot_fallen=False))

    assert metrics.grasp is Outcome.SUCCESS
    assert metrics.grasp_retry_count == 1
    assert metrics.decision_provenance == ["gemini_argon", "fallback_after_cloud_rejection_or_failure"]


def test_recorder_keeps_memory_evidence_separate_from_physical_outcome():
    recorder = TrialRecorder(EpisodeMetrics.baseline("memory-evidence", "water_bottle"))
    recorder.record_memory_execution({
        "memory_beliefs_before": {"kitchen_counter": 0.5, "kitchen_drawers": 0.5},
        "memory_beliefs_after": {"kitchen_counter": 0.2, "kitchen_drawers": 0.8},
        "memory_updates": [{"item": "water_bottle", "station": "kitchen_drawers",
                            "evidence_source": "head_camera_sighting"}],
        "search_order": ["kitchen_counter", "kitchen_drawers"],
        "stations_visited": ["kitchen_counter", "kitchen_drawers"],
        "confirmed_negative_observations": ["kitchen_counter"],
        "inaccessible_or_unobserved_stations": ["kitchen_table"],
    })
    metrics = recorder.finish(TruthSnapshot(robot_fallen=False))

    assert metrics.memory_beliefs_after["kitchen_drawers"] == 0.8
    assert metrics.memory_update_events[0]["evidence_source"] == "head_camera_sighting"
    assert metrics.search_order == ["kitchen_counter", "kitchen_drawers"]
    assert metrics.persistent_memory_updated is False
    assert metrics.physical_delivery is Outcome.UNKNOWN


def test_recorder_reports_relocation_prediction_and_camera_observations():
    recorder = TrialRecorder(EpisodeMetrics.baseline("relocation-2", "water_bottle"))
    recorder.record_memory_execution({
        "predicted_top_ranked_station": "kitchen_drawers",
        "positive_camera_observations": [{"item": "water_bottle", "station": "kitchen_counter"}],
        "camera_diagnostics": [{"station": "kitchen_counter", "marker_ids": [2, 3],
                                "absence_confirmed": False}],
        "stations_visited": ["kitchen_drawers", "kitchen_counter"],
        "stations_searched_before_detection": 2,
    })
    metrics = recorder.finish(TruthSnapshot(robot_fallen=False))

    assert metrics.predicted_top_ranked_station == "kitchen_drawers"
    assert metrics.positive_camera_observations == [{"item": "water_bottle", "station": "kitchen_counter"}]
    assert metrics.camera_diagnostics[0]["marker_ids"] == [2, 3]
    assert metrics.stations_searched_before_detection == 2


def test_result_writers_emit_json_csv_and_human_summary(tmp_path):
    metrics = EpisodeMetrics.baseline("exp-3", "water_bottle")
    metrics.navigation = Outcome.SUCCESS
    metrics.task_outcome = Outcome.UNKNOWN
    output = write_results([metrics], tmp_path)

    result = json.loads((tmp_path / "exp-3.json").read_text())
    assert result["schema_version"] == 3
    assert result["navigation"] == "success"
    with (tmp_path / "results.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["experiment_id"] == "exp-3"
    assert "exp-3" in (tmp_path / "summary.md").read_text()
    assert output["csv"].name == "results.csv"
