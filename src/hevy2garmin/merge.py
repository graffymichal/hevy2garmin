"""Merge mode: replace a watch-recorded Garmin activity with a Hevy-sourced
FIT that preserves the watch's per-second HR and adds correctly-named exercises.

Why this isn't a true API merge: Garmin's `/exerciseSets` PUT endpoint accepts
our payload but the resulting activities show every exercise as "Unknown"
(upstream issue drkostas/hevy2garmin#138). The FIT-upload path renders names
correctly because `fit_tool` writes the FIT exercise enums directly. So in
merge mode we:

  1. Find the watch activity that overlaps the Hevy workout.
  2. Download its original FIT and extract per-second HR samples.
  3. Generate a fresh FIT containing those HR samples + Hevy exercises.
  4. Upload the new FIT (rendering correct exercise names).
  5. Delete the watch activity (default; toggleable in /settings).

Public API:
    attempt_merge(client, hevy_workout, db) -> MergeResult
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path

from hevy2garmin.config import load_config
from hevy2garmin.fit import generate_fit
from hevy2garmin.garmin import (
    delete_activity,
    download_activity_fit,
    extract_hr_samples,
    find_matching_garmin_activity,
    generate_description,
    rename_activity,
    set_description,
    upload_fit,
)

logger = logging.getLogger("hevy2garmin")

# Circuit breaker: disable merge after N consecutive failures
_MAX_CONSECUTIVE_FAILURES = 3
_consecutive_failures = 0


@dataclass
class MergeResult:
    """Result of a merge attempt."""
    merged: bool
    activity_id: int | None = None
    fallback_reason: str | None = None


def reset_circuit_breaker() -> None:
    """Reset the failure counter (call at start of each sync run)."""
    global _consecutive_failures
    _consecutive_failures = 0


def _circuit_breaker_tripped() -> bool:
    return _consecutive_failures >= _MAX_CONSECUTIVE_FAILURES


def _fit_replace_merge(client, hevy_workout: dict, database, match: dict) -> MergeResult:
    """Replace the matched watch activity with a Hevy-sourced FIT carrying watch HR."""
    global _consecutive_failures

    original_id = match.get("activityId")
    if not original_id:
        return MergeResult(merged=False, fallback_reason="Matched activity missing activityId")

    # 1. Pull watch HR from its original FIT (best effort).
    watch_hr: list[int] | None
    try:
        fit_bytes = download_activity_fit(client, original_id)
        samples = extract_hr_samples(fit_bytes)
        watch_hr = samples if samples else None
        if watch_hr:
            logger.info("  Extracted %d HR samples from watch activity %s", len(watch_hr), original_id)
        else:
            logger.info("  Watch FIT for %s contained no HR records", original_id)
    except Exception as e:
        logger.warning("  Could not extract watch HR from %s: %s — uploading without watch HR", original_id, e)
        watch_hr = None

    # 2. Generate + upload the new FIT.
    wid = hevy_workout.get("id", "unknown")
    title = hevy_workout.get("title", "Workout")
    start_time = hevy_workout.get("start_time") or hevy_workout.get("startTime", "")

    try:
        with tempfile.TemporaryDirectory() as tmp:
            fit_path = str(Path(tmp) / f"{wid}.fit")
            result = generate_fit(hevy_workout, hr_samples=watch_hr, output_path=fit_path)
            logger.info(
                "  FIT: %d exercises, %d sets, %d cal",
                result["exercises"], result["total_sets"], result["calories"],
            )
            upload_result = upload_fit(client, fit_path, workout_start=start_time)
        _consecutive_failures = 0
    except Exception as e:
        _consecutive_failures += 1
        logger.error("  FIT upload failed for workout %s: %s", wid, e)
        return MergeResult(merged=False, fallback_reason=f"FIT upload failed: {e}")

    new_id = upload_result.get("activity_id")
    if not new_id:
        # Upload returned 200 but we couldn't resolve the new activity ID.
        # Don't delete the original — that would lose data.
        logger.warning("  Upload succeeded but new activity ID not found; leaving original %s in place", original_id)
        return MergeResult(merged=False, fallback_reason="Uploaded but new activity ID not resolved")

    # 3. Rename + describe the new activity.
    try:
        rename_activity(client, new_id, title)
        desc = generate_description(
            hevy_workout,
            calories=result.get("calories"),
            avg_hr=result.get("avg_hr"),
        )
        if not desc.endswith("— synced by hevy2garmin"):
            desc += "\n— synced by hevy2garmin"
        desc = "⚡ Replaced by hevy2garmin (watch HR preserved)\n\n" + desc
        set_description(client, new_id, desc)
    except Exception as e:
        logger.warning("  Rename/description failed for new activity %s: %s", new_id, e)
        # Non-fatal — the upload itself succeeded.

    # 4. Delete the original watch activity (configurable).
    cfg = load_config()
    if cfg.get("merge_delete_original", True):
        try:
            delete_activity(client, original_id)
        except Exception as e:
            logger.error(
                "  Uploaded new activity %s but failed to delete original %s: %s. "
                "You will have two strength activities at this timestamp.",
                new_id, original_id, e,
            )
    else:
        logger.info("  Kept original watch activity %s (merge_delete_original=False)", original_id)

    return MergeResult(merged=True, activity_id=new_id)


def attempt_merge(
    client,
    hevy_workout: dict,
    database,
    overlap_threshold: float = 0.70,
    max_drift_minutes: int = 20,
) -> MergeResult:
    """Try to merge a Hevy workout into a matching watch-recorded Garmin activity.

    Returns MergeResult with merged=True if the FIT-replace succeeded, or
    merged=False with a fallback_reason explaining why (no match, circuit
    breaker, upload failure).
    """
    if _circuit_breaker_tripped():
        return MergeResult(merged=False, fallback_reason="Circuit breaker: too many failures")

    match = find_matching_garmin_activity(
        client, hevy_workout,
        overlap_threshold=overlap_threshold,
        max_drift_minutes=max_drift_minutes,
    )
    if not match:
        return MergeResult(merged=False, fallback_reason="No matching Garmin activity found")

    return _fit_replace_merge(client, hevy_workout, database, match)
