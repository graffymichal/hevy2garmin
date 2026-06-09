"""Tests for merge mode: matching heuristic + FIT-replace flow."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hevy2garmin.merge import (
    MergeResult,
    attempt_merge,
    reset_circuit_breaker,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_garmin_activity(
    activity_id: int = 12345,
    start: str = "2026-03-15 18:02:00",
    duration_s: float = 43 * 60,
    type_key: str = "strength_training",
) -> dict:
    return {
        "activityId": activity_id,
        "startTimeGMT": start,
        "startTimeLocal": start,
        "duration": duration_s,
        "activityType": {"typeKey": type_key},
    }


HEVY_WORKOUT = {
    "id": "test-123",
    "title": "Push",
    "start_time": "2026-03-15T18:00:00+00:00",
    "end_time": "2026-03-15T18:45:00+00:00",
    "exercises": [
        {
            "title": "Bench Press (Barbell)",
            "sets": [
                {"type": "warmup", "weight_kg": 40, "reps": 12},
                {"type": "normal", "weight_kg": 60, "reps": 10},
                {"type": "normal", "weight_kg": 60, "reps": 8},
            ],
        },
        {
            "title": "Shoulder Press (Dumbbell)",
            "sets": [
                {"type": "normal", "weight_kg": 14, "reps": 12},
                {"type": "normal", "weight_kg": 14, "reps": 10},
            ],
        },
    ],
}


# ---------------------------------------------------------------------------
# Matching heuristic tests (unchanged from before — exercises find_matching_garmin_activity)
# ---------------------------------------------------------------------------

class TestFindMatchingActivity:

    def test_exact_overlap_matches(self):
        from hevy2garmin.garmin import find_matching_garmin_activity

        client = MagicMock()
        client.get_activities_by_date.return_value = [
            _make_garmin_activity(start="2026-03-15 18:02:00", duration_s=43 * 60),
        ]
        match = find_matching_garmin_activity(client, HEVY_WORKOUT)
        assert match is not None
        assert match["activityId"] == 12345

    def test_low_overlap_rejected(self):
        from hevy2garmin.garmin import find_matching_garmin_activity

        client = MagicMock()
        client.get_activities_by_date.return_value = [
            _make_garmin_activity(start="2026-03-15 18:22:00", duration_s=23 * 60),
        ]
        match = find_matching_garmin_activity(client, HEVY_WORKOUT)
        assert match is None

    def test_wrong_type_rejected(self):
        from hevy2garmin.garmin import find_matching_garmin_activity

        client = MagicMock()
        client.get_activities_by_date.return_value = [
            _make_garmin_activity(type_key="running"),
        ]
        match = find_matching_garmin_activity(client, HEVY_WORKOUT)
        assert match is None

    def test_incomplete_activity_rejected(self):
        """Activity still in progress (end time in future) → no match."""
        from datetime import datetime, timezone, timedelta
        from hevy2garmin.garmin import find_matching_garmin_activity

        now = datetime.now(timezone.utc)
        recent_start = now.strftime("%Y-%m-%d %H:%M:%S")
        hevy_now = {
            **HEVY_WORKOUT,
            "start_time": now.isoformat(),
            "end_time": (now + timedelta(minutes=45)).isoformat(),
        }
        client = MagicMock()
        client.get_activities_by_date.return_value = [
            _make_garmin_activity(start=recent_start, duration_s=999999),
        ]
        match = find_matching_garmin_activity(client, hevy_now)
        assert match is None

    def test_best_of_multiple_candidates(self):
        from hevy2garmin.garmin import find_matching_garmin_activity

        client = MagicMock()
        client.get_activities_by_date.return_value = [
            _make_garmin_activity(activity_id=1, start="2026-03-15 18:10:00", duration_s=35 * 60),
            _make_garmin_activity(activity_id=2, start="2026-03-15 18:01:00", duration_s=44 * 60),
        ]
        match = find_matching_garmin_activity(client, HEVY_WORKOUT)
        assert match is not None
        assert match["activityId"] == 2


# ---------------------------------------------------------------------------
# FIT-replace flow tests
# ---------------------------------------------------------------------------

class TestExtractHrSamples:
    """Round-trip: write a FIT with known HR, read it back via the new extractor."""

    def test_round_trip(self, sample_profile: dict, tmp_path: Path) -> None:
        from hevy2garmin.fit import generate_fit
        from hevy2garmin.garmin import extract_hr_samples

        workout = {
            "id": "hr-rt",
            "title": "HR Round Trip",
            "start_time": "2026-04-01T20:00:00+00:00",
            "end_time": "2026-04-01T20:30:00+00:00",
            "exercises": [{
                "index": 0, "title": "Bench Press (Barbell)", "exercise_template_id": "X",
                "sets": [{"type": "normal", "weight_kg": 60, "reps": 8}],
            }],
        }
        hr_in = [110, 112, 115, 118, 120, 119, 117, 115, 113, 110]
        fit_path = tmp_path / "rt.fit"
        generate_fit(workout, hr_samples=hr_in, output_path=str(fit_path), profile=sample_profile)

        hr_out = extract_hr_samples(fit_path.read_bytes())
        assert hr_out == hr_in, f"Round-trip mismatch: in={hr_in} out={hr_out}"

    def test_no_hr_records_returns_empty(self, sample_profile: dict, tmp_path: Path) -> None:
        """FIT generated with hr_samples=None has zero RecordMessage HR → extractor returns []."""
        from hevy2garmin.fit import generate_fit
        from hevy2garmin.garmin import extract_hr_samples

        workout = {
            "id": "no-hr",
            "title": "No HR",
            "start_time": "2026-04-01T20:00:00+00:00",
            "end_time": "2026-04-01T20:30:00+00:00",
            "exercises": [{
                "index": 0, "title": "Bench Press (Barbell)", "exercise_template_id": "X",
                "sets": [{"type": "normal", "weight_kg": 60, "reps": 8}],
            }],
        }
        fit_path = tmp_path / "no-hr.fit"
        generate_fit(workout, hr_samples=None, output_path=str(fit_path), profile=sample_profile)

        assert extract_hr_samples(fit_path.read_bytes()) == []


class TestFitReplaceMerge:
    """End-to-end attempt_merge with FIT-replace strategy (mocked I/O)."""

    def setup_method(self):
        reset_circuit_breaker()

    def _patches(self):
        """Common patch stack for attempt_merge tests."""
        return {
            "find": patch("hevy2garmin.merge.find_matching_garmin_activity"),
            "download": patch("hevy2garmin.merge.download_activity_fit"),
            "extract": patch("hevy2garmin.merge.extract_hr_samples"),
            "generate": patch("hevy2garmin.merge.generate_fit"),
            "upload": patch("hevy2garmin.merge.upload_fit"),
            "rename": patch("hevy2garmin.merge.rename_activity"),
            "set_desc": patch("hevy2garmin.merge.set_description"),
            "delete": patch("hevy2garmin.merge.delete_activity"),
            "load_cfg": patch("hevy2garmin.merge.load_config"),
        }

    def test_calls_in_order_with_delete_default(self):
        """Match → download → extract → generate → upload → rename → set_desc → delete."""
        ps = self._patches()
        mocks = {k: p.start() for k, p in ps.items()}
        try:
            mocks["find"].return_value = _make_garmin_activity()
            mocks["download"].return_value = b"fake-fit-bytes"
            mocks["extract"].return_value = [120, 121, 122]
            mocks["generate"].return_value = {"exercises": 2, "total_sets": 5, "calories": 200, "avg_hr": 121}
            mocks["upload"].return_value = {"activity_id": 99999}
            mocks["load_cfg"].return_value = {"merge_delete_original": True}

            result = attempt_merge(MagicMock(), HEVY_WORKOUT, MagicMock())

            assert result.merged is True
            assert result.activity_id == 99999
            mocks["download"].assert_called_once()
            mocks["extract"].assert_called_once_with(b"fake-fit-bytes")
            mocks["generate"].assert_called_once()
            # generate_fit was called with hr_samples = the extracted list
            assert mocks["generate"].call_args.kwargs.get("hr_samples") == [120, 121, 122]
            mocks["upload"].assert_called_once()
            mocks["rename"].assert_called_once()
            mocks["set_desc"].assert_called_once()
            mocks["delete"].assert_called_once_with(mocks["delete"].call_args.args[0], 12345)
        finally:
            for p in ps.values():
                p.stop()

    def test_keeps_original_when_flag_off(self):
        """merge_delete_original=False → delete_activity is NOT called."""
        ps = self._patches()
        mocks = {k: p.start() for k, p in ps.items()}
        try:
            mocks["find"].return_value = _make_garmin_activity()
            mocks["download"].return_value = b"fake"
            mocks["extract"].return_value = [120]
            mocks["generate"].return_value = {"exercises": 1, "total_sets": 1, "calories": 50, "avg_hr": 120}
            mocks["upload"].return_value = {"activity_id": 88888}
            mocks["load_cfg"].return_value = {"merge_delete_original": False}

            result = attempt_merge(MagicMock(), HEVY_WORKOUT, MagicMock())

            assert result.merged is True
            assert result.activity_id == 88888
            mocks["delete"].assert_not_called()
        finally:
            for p in ps.values():
                p.stop()

    def test_falls_back_to_no_hr_on_download_error(self):
        """download_activity_fit raises → generate_fit called with hr_samples=None."""
        ps = self._patches()
        mocks = {k: p.start() for k, p in ps.items()}
        try:
            mocks["find"].return_value = _make_garmin_activity()
            mocks["download"].side_effect = RuntimeError("network down")
            mocks["generate"].return_value = {"exercises": 1, "total_sets": 1, "calories": 50, "avg_hr": 90}
            mocks["upload"].return_value = {"activity_id": 77777}
            mocks["load_cfg"].return_value = {"merge_delete_original": True}

            result = attempt_merge(MagicMock(), HEVY_WORKOUT, MagicMock())

            assert result.merged is True
            mocks["extract"].assert_not_called()  # download blew up before extract
            assert mocks["generate"].call_args.kwargs.get("hr_samples") is None
            mocks["upload"].assert_called_once()
            mocks["delete"].assert_called_once()
        finally:
            for p in ps.values():
                p.stop()

    def test_empty_hr_samples_passed_as_none(self):
        """extract_hr_samples returns [] → generate_fit called with hr_samples=None."""
        ps = self._patches()
        mocks = {k: p.start() for k, p in ps.items()}
        try:
            mocks["find"].return_value = _make_garmin_activity()
            mocks["download"].return_value = b"fake"
            mocks["extract"].return_value = []
            mocks["generate"].return_value = {"exercises": 1, "total_sets": 1, "calories": 50, "avg_hr": 90}
            mocks["upload"].return_value = {"activity_id": 66666}
            mocks["load_cfg"].return_value = {"merge_delete_original": True}

            attempt_merge(MagicMock(), HEVY_WORKOUT, MagicMock())

            assert mocks["generate"].call_args.kwargs.get("hr_samples") is None
        finally:
            for p in ps.values():
                p.stop()

    def test_returns_merged_even_if_delete_fails(self):
        """delete_activity raises → still merged=True; both activities exist."""
        ps = self._patches()
        mocks = {k: p.start() for k, p in ps.items()}
        try:
            mocks["find"].return_value = _make_garmin_activity()
            mocks["download"].return_value = b"fake"
            mocks["extract"].return_value = [120]
            mocks["generate"].return_value = {"exercises": 1, "total_sets": 1, "calories": 50, "avg_hr": 120}
            mocks["upload"].return_value = {"activity_id": 55555}
            mocks["delete"].side_effect = RuntimeError("delete API down")
            mocks["load_cfg"].return_value = {"merge_delete_original": True}

            result = attempt_merge(MagicMock(), HEVY_WORKOUT, MagicMock())

            assert result.merged is True
            assert result.activity_id == 55555
        finally:
            for p in ps.values():
                p.stop()

    def test_no_match_returns_fallback(self):
        """find_matching_garmin_activity returns None → no upload/delete attempted."""
        ps = self._patches()
        mocks = {k: p.start() for k, p in ps.items()}
        try:
            mocks["find"].return_value = None

            result = attempt_merge(MagicMock(), HEVY_WORKOUT, MagicMock())

            assert result.merged is False
            assert result.fallback_reason is not None
            assert "No matching" in result.fallback_reason
            mocks["download"].assert_not_called()
            mocks["upload"].assert_not_called()
            mocks["delete"].assert_not_called()
        finally:
            for p in ps.values():
                p.stop()

    def test_circuit_breaker_trips_after_upload_failures(self):
        """3 consecutive upload failures → 4th call is short-circuited."""
        ps = self._patches()
        mocks = {k: p.start() for k, p in ps.items()}
        try:
            mocks["find"].return_value = _make_garmin_activity()
            mocks["download"].return_value = b"fake"
            mocks["extract"].return_value = [120]
            mocks["generate"].return_value = {"exercises": 1, "total_sets": 1, "calories": 50, "avg_hr": 120}
            mocks["upload"].side_effect = RuntimeError("upload failed")
            mocks["load_cfg"].return_value = {"merge_delete_original": True}

            for _ in range(3):
                attempt_merge(MagicMock(), HEVY_WORKOUT, MagicMock())

            result = attempt_merge(MagicMock(), HEVY_WORKOUT, MagicMock())
            assert result.merged is False
            assert result.fallback_reason is not None
            assert "Circuit breaker" in result.fallback_reason
        finally:
            for p in ps.values():
                p.stop()


# ---------------------------------------------------------------------------
# Sync integration tests
# ---------------------------------------------------------------------------

class TestSyncIntegration:
    """Test merge mode wired into sync.py."""

    WORKOUTS = [
        {
            "id": "w1", "title": "Push",
            "start_time": "2026-03-15T18:00:00+00:00", "end_time": "2026-03-15T18:45:00+00:00",
            "updated_at": "2026-03-15T18:45:00+00:00",
            "exercises": [{"title": "Bench Press (Barbell)", "sets": [{"type": "normal", "weight_kg": 60, "reps": 8}]}],
        },
        {
            "id": "w2", "title": "Pull",
            "start_time": "2026-03-16T18:00:00+00:00", "end_time": "2026-03-16T18:45:00+00:00",
            "updated_at": "2026-03-16T18:45:00+00:00",
            "exercises": [{"title": "Bent Over Row (Barbell)", "sets": [{"type": "normal", "weight_kg": 50, "reps": 10}]}],
        },
    ]

    def _mock_hevy(self):
        h = MagicMock()
        h.get_workout_count.return_value = 2
        h.get_workouts.return_value = {"workouts": self.WORKOUTS, "page_count": 1}
        return h

    @patch("hevy2garmin.sync.db")
    @patch("hevy2garmin.sync.get_client")
    @patch("hevy2garmin.sync.HevyClient")
    @patch("hevy2garmin.sync.attempt_merge")
    def test_merge_on_both_match(self, mock_merge, mock_hevy_cls, mock_gclient, mock_db):
        """merge ON, both match → both use merge path."""
        mock_hevy_cls.return_value = self._mock_hevy()
        mock_gclient.return_value = MagicMock()
        mock_db.is_synced.return_value = False
        mock_merge.return_value = MergeResult(merged=True, activity_id=12345)

        from hevy2garmin.sync import sync
        stats = sync(config={"hevy_api_key": "t", "merge_mode": True}, limit=2)

        assert stats["merged"] == 2
        assert stats["merge_fallback"] == 0
        assert mock_merge.call_count == 2
        calls = mock_db.mark_synced.call_args_list
        assert all(c.kwargs.get("sync_method") == "merge" for c in calls)

    @patch("hevy2garmin.sync.db")
    @patch("hevy2garmin.sync.get_client")
    @patch("hevy2garmin.sync.HevyClient")
    @patch("hevy2garmin.sync.attempt_merge")
    @patch("hevy2garmin.sync.generate_fit", return_value={"exercises": 1, "total_sets": 1, "calories": 100, "avg_hr": 90})
    @patch("hevy2garmin.sync.upload_fit", return_value={"activity_id": 222})
    @patch("hevy2garmin.sync.find_activity_by_start_time", return_value=None)
    @patch("hevy2garmin.sync.rename_activity")
    @patch("hevy2garmin.sync.set_description")
    @patch("hevy2garmin.sync.generate_description", return_value="test")
    def test_merge_on_second_falls_back(self, *mocks):
        """merge ON, first matches, second doesn't → fallback to upload."""
        (mock_desc, mock_setdesc, mock_rename, mock_find, mock_upload,
         mock_fit, mock_merge, mock_hevy_cls, mock_gclient, mock_db) = mocks

        mock_hevy_cls.return_value = self._mock_hevy()
        mock_gclient.return_value = MagicMock()
        mock_db.is_synced.return_value = False
        call_count = [0]
        def alt(c, w, d, **kwargs):
            call_count[0] += 1
            return MergeResult(merged=True, activity_id=111) if call_count[0] == 1 else MergeResult(merged=False, fallback_reason="No match")
        mock_merge.side_effect = alt

        from hevy2garmin.sync import sync
        stats = sync(config={"hevy_api_key": "t", "merge_mode": True}, limit=2)

        assert stats["merged"] == 1
        assert stats["merge_fallback"] == 1
        calls = mock_db.mark_synced.call_args_list
        assert calls[0].kwargs.get("sync_method") == "merge"
        assert calls[1].kwargs.get("sync_method") == "upload_fallback"

    @patch("hevy2garmin.sync.db")
    @patch("hevy2garmin.sync.get_client")
    @patch("hevy2garmin.sync.HevyClient")
    @patch("hevy2garmin.sync.attempt_merge")
    @patch("hevy2garmin.sync.generate_fit", return_value={"exercises": 1, "total_sets": 1, "calories": 100, "avg_hr": 90})
    @patch("hevy2garmin.sync.upload_fit", return_value={"activity_id": 333})
    @patch("hevy2garmin.sync.find_activity_by_start_time", return_value=None)
    @patch("hevy2garmin.sync.rename_activity")
    @patch("hevy2garmin.sync.set_description")
    @patch("hevy2garmin.sync.generate_description", return_value="test")
    def test_merge_off_normal_upload(self, *mocks):
        """merge OFF → normal upload, merge never attempted."""
        (mock_desc, mock_setdesc, mock_rename, mock_find, mock_upload,
         mock_fit, mock_merge, mock_hevy_cls, mock_gclient, mock_db) = mocks

        mock_hevy_cls.return_value = self._mock_hevy()
        mock_gclient.return_value = MagicMock()
        mock_db.is_synced.return_value = False

        from hevy2garmin.sync import sync
        stats = sync(config={"hevy_api_key": "t", "merge_mode": False}, limit=2)

        assert stats["merged"] == 0
        assert mock_merge.call_count == 0
        calls = mock_db.mark_synced.call_args_list
        assert all(c.kwargs.get("sync_method") == "upload" for c in calls)
