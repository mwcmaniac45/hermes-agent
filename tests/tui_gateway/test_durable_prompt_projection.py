"""PROVENANCE_UNVERIFIED — vertical durable v1 text projection tests."""
from __future__ import annotations

import threading

import pytest

from hermes_state import SessionDB
from tui_gateway import server
from tui_gateway.prompt_submission_contract import (
    DisplayKind,
    build_canonical_semantic_object,
    compute_semantic_fingerprint,
    compute_text_sha256,
)


def _db(tmp_path, *session_ids):
    db = SessionDB(db_path=tmp_path / "state.db")
    for session_id in session_ids:
        db.create_session(session_id, "tui")
    return db


def _work(db, session_id="session-a", submission_id="submission-a", text="private text"):
    return db.create_or_read_prompt_submission(
        session_id=session_id,
        submission_id=submission_id,
        contract_version="1",
        semantic_fingerprint="a" * 64,
        payload={"text": text},
    )


def _fingerprint(text, *, queued=False, interrupted=False):
    return compute_semantic_fingerprint(build_canonical_semantic_object(
        text_sha256=compute_text_sha256(text),
        display_kind=DisplayKind.NORMAL,
        queued=queued,
        interrupted=interrupted,
        surface="",
        truncation_target=None,
        truncation_consent=False,
        attachments=[],
        replay_controls={"attachments": "unsupported", "truncation": "unsupported"},
    ))


def _session(profile_home, session_id="session-a"):
    session = {
        "session_key": session_id,
        "profile_home": str(profile_home),
        "history": [],
        "history_lock": threading.RLock(),
        "running": False,
        "client_surface": "",
    }
    with server._sessions_lock:
        server._sessions[session_id] = session
    return session


@pytest.fixture(autouse=True)
def _clear_test_session_registry():
    """Keep each durable scheduler test on its own live session generation."""
    with server._sessions_lock:
        server._sessions.pop("session-a", None)
    yield
    with server._sessions_lock:
        server._sessions.pop("session-a", None)


def test_claim_returns_only_claimed_canonical_payload_and_session_identity(tmp_path):
    db = _db(tmp_path, "session-a", "session-b")
    try:
        work = _work(db)
        claim = db.claim_prompt_submission_work(
            work["work_id"], owner_token="owner-a", session_id="session-a"
        )
        assert {key: claim[key] for key in ("work_id", "session_id", "owner_generation", "payload")} == {
            "work_id": work["work_id"],
            "session_id": "session-a",
            "owner_generation": 1,
            "payload": {
                "payload_version": 1,
                "text": "private text",
                "display_kind": "normal",
                "queued": False,
                "interrupted": False,
                "truncation_target": None,
                "truncation_consent": False,
                "attachments": [],
            },
        }
    finally:
        db.close()


def test_fresh_session_scoped_recovery_marks_only_its_invocation_unknown(tmp_path):
    db = _db(tmp_path, "session-a", "session-b")
    try:
        first, second = _work(db, "session-a", "first"), _work(db, "session-b", "second")
        for work, owner, attempt in ((first, "owner-a", "attempt-a"), (second, "owner-b", "attempt-b")):
            db.claim_prompt_submission_work(work["work_id"], owner_token=owner, session_id=work is first and "session-a" or "session-b")
            db.mark_prompt_submission_invoking(work["work_id"], owner_token=owner, attempt_token=attempt)
        assert db.recover_prompt_submission_work(session_id="session-a", now=0) == []
        assert db.create_or_read_prompt_submission(
            session_id="session-a", submission_id="first", contract_version="1", semantic_fingerprint="a" * 64, payload={"text": "private text"}
        )["ack"]["invocation_status"] == "unknown_outcome"
        assert db.create_or_read_prompt_submission(
            session_id="session-b", submission_id="second", contract_version="1", semantic_fingerprint="a" * 64, payload={"text": "private text"}
        )["ack"]["invocation_status"] == "invoking"
    finally:
        db.close()


def test_owner_attempt_fencing_rejects_stale_renewal_and_settlement(tmp_path):
    db = _db(tmp_path, "session-a")
    try:
        work = _work(db)
        db.claim_prompt_submission_work(work["work_id"], owner_token="owner-a", session_id="session-a")
        db.mark_prompt_submission_invoking(work["work_id"], owner_token="owner-a", attempt_token="attempt-a")
        assert db.renew_prompt_submission_lease(
            work["work_id"], owner_token="owner-b", attempt_token="attempt-a", lease_seconds=60
        ) is False
        assert db.renew_prompt_submission_lease(
            work["work_id"], owner_token="owner-a", attempt_token="attempt-b", lease_seconds=60
        ) is False
        assert db.complete_prompt_submission_work(
            work["work_id"], owner_token="owner-b", attempt_token="attempt-a", state="COMPLETED"
        ) is False
        assert db.renew_prompt_submission_lease(
            work["work_id"], owner_token="owner-a", attempt_token="attempt-a", lease_seconds=60
        ) is True
    finally:
        db.close()


def test_dispatching_owner_can_release_only_its_uninvoked_claim(tmp_path):
    db = _db(tmp_path, "session-a")
    try:
        work = _work(db)
        db.claim_prompt_submission_work(work["work_id"], owner_token="owner-a", session_id="session-a")
        assert db.release_prompt_submission_dispatch(
            work["work_id"], owner_token="owner-b"
        ) is False
        assert db.release_prompt_submission_dispatch(
            work["work_id"], owner_token="owner-a"
        ) is True
        replay = db.create_or_read_prompt_submission(
            session_id="session-a", submission_id="submission-a", contract_version="1",
            semantic_fingerprint="a" * 64, payload={"text": "private text"},
        )
        assert replay["ack"]["invocation_status"] == "accepted"
    finally:
        db.close()


def test_scheduler_build_failure_releases_claim_and_preserves_other_inflight(monkeypatch, tmp_path):
    profile = tmp_path / "profile"
    profile.mkdir()
    db = SessionDB(db_path=profile / "state.db")
    db.create_session("session-a", "tui")
    work = _work(db)
    db.close()
    session = _session(profile)
    session["inflight_turn"] = {"submission_id": "another-turn", "status": "running"}

    class InlineThread:
        def __init__(self, *, target, daemon):
            self.target = target
        def start(self):
            self.target()

    provider_entries = []
    monkeypatch.setattr(server.threading, "Thread", InlineThread)
    monkeypatch.setattr(server, "_start_agent_build", lambda *_a: None)
    monkeypatch.setattr(server, "_wait_agent_for_prompt", lambda *_a: {"error": "build failed"})
    monkeypatch.setattr(server, "_run_prompt_submit", lambda *_a, **_k: provider_entries.append(True))

    server._schedule_durable_prompt_projection(session, work["work_id"])

    check = SessionDB(db_path=profile / "state.db")
    try:
        replay = check.create_or_read_prompt_submission(
            session_id="session-a", submission_id="submission-a", contract_version="1",
            semantic_fingerprint="a" * 64, payload={"text": "private text"},
        )
        assert replay["ack"]["invocation_status"] == "accepted"
    finally:
        check.close()
    assert session["running"] is False
    assert session["inflight_turn"]["submission_id"] == "another-turn"
    assert provider_entries == []


def test_scheduler_projects_real_accepted_work_to_completed_before_exact_replay(monkeypatch, tmp_path):
    profile = tmp_path / "profile"
    profile.mkdir()
    db = SessionDB(db_path=profile / "state.db")
    db.create_session("session-a", "tui")
    work = _work(db)
    db.close()
    session = _session(profile)
    ready = threading.Event()
    ready.set()
    session["agent_ready"] = ready
    switches, provider_entries, emitted, marker_starts = [], [], [], []

    class InlineThread:
        def __init__(self, *, target, daemon):
            self.target = target
        def start(self):
            self.target()

    class JoinedThread:
        def join(self):
            return None

    class Agent:
        session_id = "session-a"
        provider = "test"
        model = "before"
        interim_assistant_callback = None
        def run_conversation(self, message, **_kwargs):
            assert self.model == "after"
            provider_entries.append(message)
            return {"messages": [{"role": "user", "content": message}, {"role": "assistant", "content": "ok"}], "final_response": "ok"}

    session["agent"] = Agent()
    session["pending_model_switch"] = {"raw": "after", "confirm_expensive_model": False}
    def apply_switch(_sid, current, raw, **_kwargs):
        assert raw == "after"
        current["agent"].model = raw
        switches.append(raw)
        return {"value": raw, "warning": "", "confirm_required": False}

    monkeypatch.setattr(server.threading, "Thread", InlineThread)
    monkeypatch.setattr(server, "_apply_model_switch", apply_switch)
    monkeypatch.setattr(server, "_sync_agent_model_with_config", lambda *_a: None)
    monkeypatch.setattr(server, "_sync_bot_capabilities", lambda *_a: None)
    monkeypatch.setattr(server, "_start_usage_ticker", lambda *_a: (threading.Event(), JoinedThread()))
    monkeypatch.setattr(server, "_get_usage", lambda *_a: {})
    monkeypatch.setattr(server, "_emit", lambda *a: emitted.append(a))
    monkeypatch.setattr(server, "record_turn_start", lambda *a, **_k: marker_starts.append(a))

    server._schedule_durable_prompt_projection(session, work["work_id"])

    check = SessionDB(db_path=profile / "state.db")
    try:
        replay = check.create_or_read_prompt_submission(
            session_id="session-a", submission_id="submission-a", contract_version="1",
            semantic_fingerprint="a" * 64, payload={"text": "private text"},
        )
        assert replay["ack"]["invocation_status"] == "completed"
        assert check._conn is not None
        row = check._conn.execute(
            "SELECT state, owner_token, invocation_attempt_token FROM prompt_accepted_work WHERE work_id=?",
            (work["work_id"],),
        ).fetchone()
        assert tuple(row) == ("COMPLETED", None, row[2])
        assert row[2]
    finally:
        check.close()
    assert provider_entries == ["private text"]
    assert switches == ["after"]
    assert "pending_model_switch" not in session
    assert marker_starts == []
    assert not any(event[0] == "message.delta" for event in emitted)
    assert session["running"] is False


def test_scheduler_cancellation_during_real_wait_releases_before_provider_entry(monkeypatch, tmp_path):
    """A cancelled durable wait must release its claim without starting a turn."""
    profile = tmp_path / "profile"
    profile.mkdir()
    db = SessionDB(db_path=profile / "state.db")
    db.create_session("session-a", "tui")
    work = _work(db)
    db.close()
    session = _session(profile)
    wait_entered, provider_entered = threading.Event(), threading.Event()
    emitted, marker_starts, dispatch_threads = [], [], []

    class ObservedReady(threading.Event):
        def wait(self, timeout=None):
            wait_entered.set()
            return super().wait(timeout)

    class DispatchThread:
        def __init__(self, *, target, daemon):
            self.thread = server._RealThread(target=target, daemon=daemon)
            dispatch_threads.append(self.thread)

        def start(self):
            self.thread.start()

    class JoinedThread:
        def join(self):
            return None

    class Agent:
        session_id = "session-a"
        provider = "test"
        model = "test"
        interim_assistant_callback = None

        def run_conversation(self, *_args, **_kwargs):
            provider_entered.set()
            return {"final_response": "unexpected"}

    session["agent_ready"] = ObservedReady()
    session["agent"] = Agent()
    monkeypatch.setattr(server.threading, "Thread", DispatchThread)
    monkeypatch.setattr(server, "_start_agent_build", lambda *_args: None)
    monkeypatch.setattr(server, "_sync_agent_model_with_config", lambda *_args: None)
    monkeypatch.setattr(server, "_sync_bot_capabilities", lambda *_args: None)
    monkeypatch.setattr(server, "_start_usage_ticker", lambda *_args: (threading.Event(), JoinedThread()))
    monkeypatch.setattr(server, "_get_usage", lambda *_args: {})
    monkeypatch.setattr(server, "_emit", lambda *args: emitted.append(args))
    monkeypatch.setattr(server, "record_turn_start", lambda *args, **kwargs: marker_starts.append(args))

    server._schedule_durable_prompt_projection(session, work["work_id"])
    assert wait_entered.wait(timeout=1)
    with session["history_lock"]:
        session["_turn_cancel_requested"] = True
    session["agent_ready"].set()
    dispatch_threads[0].join(timeout=1)

    assert not provider_entered.wait(timeout=0.2)
    check = SessionDB(db_path=profile / "state.db")
    try:
        replay = check.create_or_read_prompt_submission(
            session_id="session-a", submission_id="submission-a", contract_version="1",
            semantic_fingerprint="a" * 64, payload={"text": "private text"},
        )
        assert replay["ack"]["invocation_status"] == "accepted"
    finally:
        check.close()
    assert session["running"] is False
    assert "_durable_projection_work_id" not in session
    assert not session.get("inflight_turn")
    assert not any(event[0] == "message.start" for event in emitted)
    assert marker_starts == []


def test_durable_runtime_entry_cancellation_releases_only_matching_projection(monkeypatch, tmp_path):
    """A cancellation observed at durable runtime entry must stay pre-invocation."""
    profile = tmp_path / "profile"
    profile.mkdir()
    db = SessionDB(db_path=profile / "state.db")
    db.create_session("session-a", "tui")
    work = _work(db)
    owner_token = "owner-a"
    claim = db.claim_prompt_submission_work(
        work["work_id"], owner_token=owner_token, session_id="session-a"
    )
    db.close()
    session = _session(profile)
    session.update({
        "running": True,
        "_durable_projection_work_id": work["work_id"],
        "_turn_cancel_requested": True,
        "inflight_turn": {"submission_id": "another-turn", "status": "running"},
    })
    provider_entries, emitted, marker_starts = [], [], []

    class Agent:
        def run_conversation(self, *_args, **_kwargs):
            provider_entries.append(True)

    session["agent"] = Agent()
    durable_context = {
        "work_id": work["work_id"],
        "owner_token": owner_token,
        "owner_generation": claim["owner_generation"],
        "attempt_token": None,
        "session_id": "session-a",
    }
    monkeypatch.setattr(server, "_emit", lambda *args: emitted.append(args))
    monkeypatch.setattr(
        server, "record_turn_start", lambda *args, **kwargs: marker_starts.append(args)
    )

    assert server._run_prompt_submit(
        "__durable__runtime-entry", "session-a", session, "private text",
        durable_context=durable_context,
    ) is False

    check = SessionDB(db_path=profile / "state.db")
    try:
        replay = check.create_or_read_prompt_submission(
            session_id="session-a", submission_id="submission-a", contract_version="1",
            semantic_fingerprint="a" * 64, payload={"text": "private text"},
        )
        assert replay["ack"]["invocation_status"] == "accepted"
    finally:
        check.close()
    assert provider_entries == []
    assert session["running"] is False
    assert "_durable_projection_work_id" not in session
    assert session["inflight_turn"] == {"submission_id": "another-turn", "status": "running"}
    assert not any(event[0] == "message.start" for event in emitted)
    assert marker_starts == []


def test_scheduler_registry_replacement_during_real_wait_releases_before_turn_start(monkeypatch, tmp_path):
    """A replaced durable session must release before it creates or starts a turn."""
    profile = tmp_path / "profile"
    profile.mkdir()
    db = SessionDB(db_path=profile / "state.db")
    db.create_session("session-a", "tui")
    work = _work(db)
    db.close()
    session = _session(profile)
    replacement = _session(profile)
    replacement_state = dict(replacement)
    replacement_state.pop("history_lock")
    wait_entered, provider_entered = threading.Event(), threading.Event()
    emitted, marker_starts, dispatch_threads = [], [], []

    class ObservedReady(threading.Event):
        def wait(self, timeout=None):
            wait_entered.set()
            return super().wait(timeout)

    class DispatchThread:
        def __init__(self, *, target, daemon):
            self.thread = server._RealThread(target=target, daemon=daemon)
            dispatch_threads.append(self.thread)

        def start(self):
            self.thread.start()

    class JoinedThread:
        def join(self):
            return None

    class Agent:
        session_id = "session-a"
        provider = "test"
        model = "test"
        interim_assistant_callback = None

        def run_conversation(self, *_args, **_kwargs):
            provider_entered.set()
            return {"final_response": "unexpected"}

    session["agent_ready"] = ObservedReady()
    session["agent"] = Agent()
    monkeypatch.setattr(server.threading, "Thread", DispatchThread)
    monkeypatch.setattr(server, "_start_agent_build", lambda *_args: None)
    monkeypatch.setattr(server, "_sync_agent_model_with_config", lambda *_args: None)
    monkeypatch.setattr(server, "_sync_bot_capabilities", lambda *_args: None)
    monkeypatch.setattr(server, "_start_usage_ticker", lambda *_args: (threading.Event(), JoinedThread()))
    monkeypatch.setattr(server, "_get_usage", lambda *_args: {})
    monkeypatch.setattr(server, "_emit", lambda *args: emitted.append(args))
    monkeypatch.setattr(server, "record_turn_start", lambda *args, **kwargs: marker_starts.append(args))
    with server._sessions_lock:
        server._sessions["session-a"] = session
    try:
        server._schedule_durable_prompt_projection(session, work["work_id"])
        assert wait_entered.wait(timeout=1)
        with server._sessions_lock:
            server._sessions["session-a"] = replacement
        session["agent_ready"].set()
        dispatch_threads[0].join(timeout=1)

        assert not provider_entered.wait(timeout=0.2)
        check = SessionDB(db_path=profile / "state.db")
        try:
            replay = check.create_or_read_prompt_submission(
                session_id="session-a", submission_id="submission-a", contract_version="1",
                semantic_fingerprint="a" * 64, payload={"text": "private text"},
            )
            assert replay["ack"]["invocation_status"] == "accepted"
        finally:
            check.close()
        assert session["running"] is False
        assert "_durable_projection_work_id" not in session
        assert not session.get("inflight_turn")
        assert not any(event[0] == "message.start" for event in emitted)
        assert marker_starts == []
        assert {key: value for key, value in replacement.items() if key != "history_lock"} == replacement_state
    finally:
        with server._sessions_lock:
            server._sessions.pop("session-a", None)


@pytest.mark.parametrize(
    ("transition", "raises"),
    (("invoking", False), ("invoking", True), ("running", False), ("running", True)),
)
def test_scheduler_transition_refusal_clears_only_matching_durable_projection_and_completes_ui(
    monkeypatch, tmp_path, transition, raises,
):
    profile = tmp_path / "profile"
    profile.mkdir()
    db = SessionDB(db_path=profile / "state.db")
    db.create_session("session-a", "tui")
    work = _work(db)
    db.close()
    session = _session(profile)
    provider_entries, emitted = [], []

    class InlineThread:
        def __init__(self, *, target, daemon): self.target = target
        def start(self): self.target()

    class Agent:
        session_id = "session-a"
        provider = "test"
        model = "test"
        interim_assistant_callback = None
        def run_conversation(self, *_a, **_k):
            provider_entries.append(True)
            return {"final_response": "unexpected"}

    session["agent"] = Agent()
    monkeypatch.setattr(server.threading, "Thread", InlineThread)
    monkeypatch.setattr(server, "_start_agent_build", lambda *_a: None)
    monkeypatch.setattr(server, "_wait_agent_for_prompt", lambda *_a: None)
    monkeypatch.setattr(server, "_sync_agent_model_with_config", lambda *_a: None)
    monkeypatch.setattr(server, "_sync_bot_capabilities", lambda *_a: None)
    monkeypatch.setattr(server, "_emit", lambda *args: emitted.append(args))
    if transition == "invoking":
        if raises:
            def refuse_invoking(*_a, **_k): raise RuntimeError("refused")
            monkeypatch.setattr(SessionDB, "mark_prompt_submission_invoking", refuse_invoking)
        else:
            monkeypatch.setattr(SessionDB, "mark_prompt_submission_invoking", lambda *_a, **_k: None)
    else:
        if raises:
            def refuse_running(*_a, **_k): raise RuntimeError("refused")
            monkeypatch.setattr(SessionDB, "mark_prompt_submission_running", refuse_running)
        else:
            monkeypatch.setattr(SessionDB, "mark_prompt_submission_running", lambda *_a, **_k: False)

    server._schedule_durable_prompt_projection(session, work["work_id"])

    check = SessionDB(db_path=profile / "state.db")
    try:
        replay = check.create_or_read_prompt_submission(
            session_id="session-a", submission_id="submission-a", contract_version="1",
            semantic_fingerprint="a" * 64, payload={"text": "private text"},
        )
        assert replay["ack"]["invocation_status"] == (
            "accepted" if transition == "invoking" else "terminal_error"
        )
    finally:
        check.close()
    assert provider_entries == []
    assert session["running"] is False
    assert not session.get("inflight_turn")
    # Both transition failures now occur before RUNNING permits UI publication.
    assert not any(event[0] in {"message.start", "message.complete"} for event in emitted)


@pytest.mark.parametrize("race", ("closing", "replacement"))
def test_scheduler_preprovider_entry_races_release_claim_and_close_started_ui(monkeypatch, tmp_path, race):
    profile = tmp_path / "profile"
    profile.mkdir()
    db = SessionDB(db_path=profile / "state.db")
    db.create_session("session-a", "tui")
    work = _work(db)
    db.close()
    session = _session(profile)
    provider_entries, emitted = [], []

    class InlineThread:
        def __init__(self, *, target, daemon): self.target = target
        def start(self): self.target()

    class Agent:
        session_id = "session-a"
        provider = "test"
        model = "test"
        interim_assistant_callback = None
        def run_conversation(self, *_a, **_k): provider_entries.append(True)

    session["agent"] = Agent()
    if race == "closing":
        session["_closing"] = True
    else:
        with server._sessions_lock:
            server._sessions["session-a"] = {"replacement": True}
    monkeypatch.setattr(server.threading, "Thread", InlineThread)
    monkeypatch.setattr(server, "_start_agent_build", lambda *_a: None)
    monkeypatch.setattr(server, "_wait_agent_for_prompt", lambda *_a: None)
    monkeypatch.setattr(server, "_emit", lambda *args: emitted.append(args))
    try:
        server._schedule_durable_prompt_projection(session, work["work_id"])
    finally:
        server._sessions.pop("session-a", None)

    check = SessionDB(db_path=profile / "state.db")
    try:
        replay = check.create_or_read_prompt_submission(
            session_id="session-a", submission_id="submission-a", contract_version="1",
            semantic_fingerprint="a" * 64, payload={"text": "private text"},
        )
        assert replay["ack"]["invocation_status"] == "accepted"
    finally:
        check.close()
    assert provider_entries == []
    assert session["running"] is False
    assert not session.get("inflight_turn")
    if race == "replacement":
        assert not any(event[0] == "message.start" for event in emitted)


def test_scheduler_invalid_payload_releases_without_agent_or_provider(monkeypatch, tmp_path):
    profile = tmp_path / "profile"
    profile.mkdir()
    db = SessionDB(db_path=profile / "state.db")
    db.create_session("session-a", "tui")
    work = db.create_or_read_prompt_submission(
        session_id="session-a", submission_id="invalid", contract_version="1",
        semantic_fingerprint="b" * 64,
        payload={"text": "private text", "queued": True},
    )
    db.close()
    session = _session(profile)

    class InlineThread:
        def __init__(self, *, target, daemon): self.target = target
        def start(self): self.target()

    calls = []
    monkeypatch.setattr(server.threading, "Thread", InlineThread)
    monkeypatch.setattr(server, "_start_agent_build", lambda *_a: calls.append("build"))
    monkeypatch.setattr(server, "_run_prompt_submit", lambda *_a, **_k: calls.append("provider"))
    server._schedule_durable_prompt_projection(session, work["work_id"])

    check = SessionDB(db_path=profile / "state.db")
    try:
        assert check.create_or_read_prompt_submission(
            session_id="session-a", submission_id="invalid", contract_version="1",
            semantic_fingerprint="b" * 64, payload={"text": "private text", "queued": True},
        )["ack"]["invocation_status"] == "accepted"
    finally:
        check.close()
    assert calls == [] and session["running"] is False


@pytest.mark.parametrize("outcome", ("returned_error", "raised_error"))
def test_durable_provider_failures_never_surface_hostile_data(monkeypatch, tmp_path, outcome, capsys):
    profile = tmp_path / "profile"
    profile.mkdir()
    db = SessionDB(db_path=profile / "state.db")
    db.create_session("session-a", "tui")
    work = _work(db)
    db.close()
    session = _session(profile)
    hostile = "token=secret /private/path prompt=never-render"
    emitted = []

    class InlineThread:
        def __init__(self, *, target, daemon): self.target = target
        def start(self): self.target()

    class JoinedThread:
        def join(self): return None

    class Agent:
        session_id = "session-a"
        provider = "test"
        model = "test"
        interim_assistant_callback = None
        def run_conversation(self, *_a, **_k):
            if outcome == "raised_error":
                raise RuntimeError(hostile)
            return {"error": hostile, "failed": True}

    session["agent"] = Agent()
    monkeypatch.setattr(server.threading, "Thread", InlineThread)
    monkeypatch.setattr(server, "_start_agent_build", lambda *_a: None)
    monkeypatch.setattr(server, "_wait_agent_for_prompt", lambda *_a: None)
    monkeypatch.setattr(server, "_apply_pending_model_switch", lambda *_a: None)
    monkeypatch.setattr(server, "_sync_agent_model_with_config", lambda *_a: None)
    monkeypatch.setattr(server, "_sync_bot_capabilities", lambda *_a: None)
    monkeypatch.setattr(server, "_start_usage_ticker", lambda *_a: (threading.Event(), JoinedThread()))
    monkeypatch.setattr(server, "_get_usage", lambda *_a: {})
    monkeypatch.setattr(server, "_emit", lambda *a: emitted.append(a))
    monkeypatch.setattr(server, "_CRASH_LOG", str(tmp_path / "crash.log"))

    server._schedule_durable_prompt_projection(session, work["work_id"])

    check = SessionDB(db_path=profile / "state.db")
    try:
        replay = check.create_or_read_prompt_submission(
            session_id="session-a", submission_id="submission-a", contract_version="1",
            semantic_fingerprint="a" * 64, payload={"text": "private text"},
        )
        assert replay["ack"]["invocation_status"] in {"terminal_error", "unknown_outcome"}
    finally:
        check.close()
    assert hostile not in repr(emitted)
    assert hostile not in capsys.readouterr().err
    assert not (tmp_path / "crash.log").exists()
    assert not session.get("inflight_turn")


@pytest.mark.parametrize("race", ("cancel", "close", "replace", "not_running"))
def test_durable_handoff_releases_post_runtime_gate_races_without_turn_side_effects(
    monkeypatch, tmp_path, race,
):
    """A live-state winner before handoff stays ACCEPTED and invisible."""
    profile = tmp_path / "profile"
    profile.mkdir()
    db = SessionDB(db_path=profile / "state.db")
    db.create_session("session-a", "tui")
    work = _work(db)
    db.close()
    session = _session(profile)
    replacement = _session(profile, "replacement")
    with server._sessions_lock:
        server._sessions.pop("replacement", None)
    session["inflight_turn"] = {"submission_id": "unrelated", "status": "running"}
    provider_entries, emitted, markers = [], [], []

    class InlineThread:
        def __init__(self, *, target, daemon): self.target = target
        def start(self): self.target()
        def join(self, *_a, **_k): return None

    class Agent:
        session_id = "session-a"
        provider = "test"
        model = "test"
        interim_assistant_callback = None
        def run_conversation(self, message, **_kwargs):
            provider_entries.append(message)
            return {"final_response": "unexpected"}

    session["agent"] = Agent()
    def checkpoint(_sid, current, _context):
        if race == "cancel":
            # Same state transition owned by session.interrupt before hard interrupt.
            with current["history_lock"]:
                current["_turn_cancel_requested"] = True
        elif race == "close":
            # Exercise the real close ownership detach, not a synthetic flag.
            assert server._pop_session_by_id("session-a") is current
        elif race == "replace":
            with server._sessions_lock:
                server._sessions["session-a"] = replacement
        else:
            with current["history_lock"]:
                current["running"] = False

    monkeypatch.setattr(server.threading, "Thread", InlineThread)
    monkeypatch.setattr(server, "_start_agent_build", lambda *_a: None)
    monkeypatch.setattr(server, "_wait_agent_for_prompt", lambda *_a: None)
    monkeypatch.setattr(server, "_sync_agent_model_with_config", lambda *_a: None)
    monkeypatch.setattr(server, "_sync_bot_capabilities", lambda *_a: None)
    monkeypatch.setattr(server, "_start_usage_ticker", lambda *_a: (threading.Event(), InlineThread(target=lambda: None, daemon=True)))
    monkeypatch.setattr(server, "_get_usage", lambda *_a: {})
    monkeypatch.setattr(server, "_emit", lambda *a: emitted.append(a))
    monkeypatch.setattr(server, "record_turn_start", lambda *a, **_k: markers.append(a))
    monkeypatch.setattr(server, "_durable_handoff_checkpoint", checkpoint, raising=False)

    server._schedule_durable_prompt_projection(session, work["work_id"])

    check = SessionDB(db_path=profile / "state.db")
    try:
        assert check.create_or_read_prompt_submission(
            session_id="session-a", submission_id="submission-a", contract_version="1",
            semantic_fingerprint="a" * 64, payload={"text": "private text"},
        )["ack"]["invocation_status"] == "accepted"
    finally:
        check.close()
    assert provider_entries == []
    assert not any(event[0] == "message.start" for event in emitted)
    assert markers == []
    assert session.get("inflight_turn") == {"submission_id": "unrelated", "status": "running"}
    assert "_durable_projection_work_id" not in session
    assert replacement.get("inflight_turn") is None


def test_durable_preflight_failure_releases_without_unknown_or_turn_side_effects(monkeypatch, tmp_path):
    """Explicit non-provider preflight failures are still pre-invocation."""
    profile = tmp_path / "profile"
    profile.mkdir()
    db = SessionDB(db_path=profile / "state.db")
    db.create_session("session-a", "tui")
    work = _work(db)
    db.close()
    session = _session(profile)
    session["inflight_turn"] = {"submission_id": "unrelated", "status": "running"}
    provider_entries, emitted, markers = [], [], []

    class InlineThread:
        def __init__(self, *, target, daemon): self.target = target
        def start(self): self.target()
        def join(self, *_a, **_k): return None
    class Agent:
        def run_conversation(self, *_a, **_k): provider_entries.append(True)
    session["agent"] = Agent()
    monkeypatch.setattr(server.threading, "Thread", InlineThread)
    monkeypatch.setattr(server, "_start_agent_build", lambda *_a: None)
    monkeypatch.setattr(server, "_wait_agent_for_prompt", lambda *_a: None)
    monkeypatch.setattr(server, "_emit", lambda *a: emitted.append(a))
    monkeypatch.setattr(server, "record_turn_start", lambda *a, **_k: markers.append(a))
    monkeypatch.setattr(server, "_durable_nonprovider_preflight", lambda *_a: (_ for _ in ()).throw(RuntimeError("preflight")), raising=False)

    server._schedule_durable_prompt_projection(session, work["work_id"])

    check = SessionDB(db_path=profile / "state.db")
    try:
        assert check.create_or_read_prompt_submission(
            session_id="session-a", submission_id="submission-a", contract_version="1",
            semantic_fingerprint="a" * 64, payload={"text": "private text"},
        )["ack"]["invocation_status"] == "accepted"
    finally:
        check.close()
    assert provider_entries == []
    assert not any(event[0] == "message.start" for event in emitted)
    assert markers == []
    assert session["inflight_turn"] == {"submission_id": "unrelated", "status": "running"}


def test_durable_handoff_wins_when_interrupt_is_waiting_on_history_lock(tmp_path):
    """An interrupt queued behind history_lock observes an already RUNNING turn."""
    profile = tmp_path / "profile"
    profile.mkdir()
    db = SessionDB(db_path=profile / "state.db")
    db.create_session("session-a", "tui")
    work = _work(db)
    claim = db.claim_prompt_submission_work(
        work["work_id"], owner_token="owner-a", session_id="session-a",
    )
    db.close()
    session = _session(profile)
    session.update({"running": True, "_durable_projection_work_id": work["work_id"]})
    context = {
        "work_id": work["work_id"], "owner_token": "owner-a",
        "owner_generation": claim["owner_generation"], "attempt_token": None,
        "session_id": "session-a",
    }
    waiting, interrupted = threading.Event(), threading.Event()

    def interrupt_equivalent():
        waiting.set()
        with session["history_lock"]:
            session["_turn_cancel_requested"] = True
            interrupted.set()

    # The outer acquisition mirrors a handoff already at the history boundary.
    # _begin_durable_invocation re-enters it, while the interrupt must wait.
    with session["history_lock"]:
        thread = threading.Thread(target=interrupt_equivalent)
        thread.start()
        assert waiting.wait(timeout=1)
        assert server._begin_durable_invocation("session-a", session, context, "private text") == "running"
        assert context["attempt_token"]
        assert not interrupted.is_set()
    thread.join(timeout=1)
    assert interrupted.is_set()
    check = SessionDB(db_path=profile / "state.db")
    try:
        assert check.create_or_read_prompt_submission(
            session_id="session-a", submission_id="submission-a", contract_version="1",
            semantic_fingerprint="a" * 64, payload={"text": "private text"},
        )["ack"]["invocation_status"] == "running"
    finally:
        check.close()


@pytest.mark.parametrize("control", ("running", "queued", "interrupted", "attachments", "hidden", "truncation"))
def test_v1_unsupported_or_busy_admission_has_no_receipt_or_runtime_mutation(monkeypatch, tmp_path, control):
    profile = tmp_path / "profile"
    profile.mkdir()
    db = SessionDB(db_path=profile / "state.db")
    db.create_session("session-a", "tui")
    db.close()
    session = _session(profile)
    if control == "running":
        session["running"] = True
    if control == "attachments":
        session["attached_images"] = ["/not-used"]
    server._sessions["session-a"] = session
    calls = []
    monkeypatch.setattr(server, "_ensure_session_db_row", lambda *_a, **_k: calls.append("ensure"))
    monkeypatch.setattr(server, "_schedule_durable_prompt_projection", lambda *_a, **_k: calls.append("project"))
    params = {
        "session_id": "session-a", "text": "private text", "submission_id": f"submission-{control}",
        "contract_version": "1", "semantic_fingerprint": _fingerprint("private text"),
    }
    if control in {"queued", "interrupted"}:
        params[control] = True
        params["semantic_fingerprint"] = _fingerprint("private text", **{control: True})
    elif control == "hidden":
        params["display_kind"] = "hidden"
    elif control == "truncation":
        params["confirm_truncate"] = True
    try:
        response = server.handle_request({"id": "r", "method": "prompt.submit", "params": params})
        assert response["error"]["code"] == 4092
        check = SessionDB(db_path=profile / "state.db")
        try:
            assert check.prompt_submission_counts("session-a") == (0, 0)
        finally:
            check.close()
        assert calls == []
    finally:
        server._sessions.pop("session-a", None)
