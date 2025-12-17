from src.models.memory import compute_memory_score


def test_compute_memory_score_recency():
    meta_recent = {
        "type": "episode",
        "created_at": "2025-11-23T00:00:00",
        "last_accessed_at": "2025-11-23T00:00:00",
        "access_count": 1,
        "importance": 0.3,
    }
    meta_old = {
        "type": "episode",
        "created_at": "2025-01-01T00:00:00",
        "last_accessed_at": "2025-01-01T00:00:00",
        "access_count": 1,
        "importance": 0.3,
    }
    s_recent = compute_memory_score(meta_recent)
    s_old = compute_memory_score(meta_old)

    assert s_recent > s_old
