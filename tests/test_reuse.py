"""End-to-end tests for historical resolution reuse.

POST /merges/{merge_id}/reuse-resolutions matches the target merge's open
conflicts against a resolved source merge's creation-time conflict snapshot
by JSON Pointer; only conflicts whose base/left/right/missing are identical
after normalisation (missing key vs explicit null distinguished) are
reused. Matched items are applied atomically; the per-pointer applied and
skipped lists carry reasons; zero matches do not bump the revision.
"""

import json as _json
import os
import tempfile

os.environ["DATABASE_PATH"] = os.path.join(tempfile.mkdtemp(), "test_reuse.db")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app import db  # noqa: E402
from app.main import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    # Context-manager form runs the lifespan, which creates the tables.
    with TestClient(app) as c:
        yield c


def make_merge(client, base, left, right, schema=None):
    payload = {"base": base, "left": left, "right": right}
    if schema is not None:
        payload["result_schema"] = schema
    resp = client.post("/merges", json=payload)
    assert resp.status_code == 201, resp.text
    return resp.json()


def resolve(client, merge_id, revision, resolutions):
    resp = client.post(
        f"/merges/{merge_id}/resolutions",
        json={"revision": revision, "resolutions": resolutions},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def reuse(client, merge_id, revision, source_id):
    return client.post(
        f"/merges/{merge_id}/reuse-resolutions",
        json={"revision": revision, "source_merge_id": source_id},
    )


THREE_WAY = {
    "base": {"a": 1, "b": 1, "c": 1},
    "left": {"a": 2, "b": 2, "c": 2},
    "right": {"a": 3, "b": 3, "c": 3},
}


def test_reuse_applies_matching_resolutions_with_single_bump(client):
    # Source: /a takes an explicit value, /b takes left; the reused value
    # must come from the source's *final result*, not its candidate sides.
    source = make_merge(client, **THREE_WAY)
    sid = source["merge_id"]
    done = resolve(
        client,
        sid,
        1,
        [
            {"pointer": "/a", "value": 99},
            {"pointer": "/b", "choice": "left"},
            {"pointer": "/c", "choice": "right"},
        ],
    )
    assert done["status"] == "resolved"

    target = make_merge(client, **THREE_WAY)
    tid = target["merge_id"]
    resp = reuse(client, tid, 1, sid)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["source_merge_id"] == sid
    assert body["status"] == "resolved"
    assert body["revision"] == 2  # exactly one bump for the whole batch
    assert body["result"] == {"a": 99, "b": 2, "c": 3}
    assert body["applied"] == [
        {"pointer": "/a", "action": "set", "value": 99},
        {"pointer": "/b", "action": "set", "value": 2},
        {"pointer": "/c", "action": "set", "value": 3},
    ]
    assert body["skipped"] == []


def test_reuse_partial_match_leaves_other_conflicts_open(client):
    # Source only ever conflicted on /a; the target additionally conflicts
    # on /b and /c, which the source snapshot cannot cover.
    source = make_merge(
        client, base={"a": 1}, left={"a": 2}, right={"a": 3}
    )
    sid = source["merge_id"]
    resolve(client, sid, 1, [{"pointer": "/a", "choice": "right"}])

    target = make_merge(client, **THREE_WAY)
    tid = target["merge_id"]
    resp = reuse(client, tid, 1, sid)
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "pending"
    assert body["revision"] == 2
    assert body["applied"] == [{"pointer": "/a", "action": "set", "value": 3}]
    assert sorted(s["pointer"] for s in body["skipped"]) == ["/b", "/c"]
    assert sorted(c["pointer"] for c in body["conflicts"]) == ["/b", "/c"]

    # The remaining conflicts can then be resolved normally.
    done = resolve(
        client, tid, 2, [{"pointer": "/b", "value": 7}, {"pointer": "/c", "value": 8}]
    )
    assert done["status"] == "resolved"
    assert done["result"] == {"a": 3, "b": 7, "c": 8}


def test_reuse_shape_mismatch_is_skipped_with_differences(client):
    # Same pointer /a, but the right candidate differs: not reusable.
    source = make_merge(
        client, base={"a": 1}, left={"a": 2}, right={"a": 3}
    )
    sid = source["merge_id"]
    resolve(client, sid, 1, [{"pointer": "/a", "choice": "left"}])

    target = make_merge(
        client, base={"a": 1}, left={"a": 2}, right={"a": 30}
    )
    tid = target["merge_id"]
    resp = reuse(client, tid, 1, sid)
    assert resp.status_code == 200
    body = resp.json()
    assert body["applied"] == []
    assert body["skipped"] == [
        {"pointer": "/a", "reason": "shape_mismatch", "differences": ["right"]}
    ]
    # No matches -> nothing written and no revision bump.
    assert body["revision"] == 1
    assert body["status"] == "pending"


def test_reuse_missing_key_does_not_match_explicit_null(client):
    # Source /a: left deleted the key (missing left); target /a: left holds
    # an explicit JSON null. These must never be treated as equal.
    source = make_merge(
        client,
        base={"a": 1, "b": 1},
        left={"b": 2},
        right={"a": 3, "b": 3},
    )
    sid = source["merge_id"]
    src_conflict = next(c for c in source["conflicts"] if c["pointer"] == "/a")
    assert src_conflict["missing"] == ["left"]
    resolve(
        client,
        sid,
        1,
        [{"pointer": "/a", "choice": "left"}, {"pointer": "/b", "choice": "right"}],
    )

    target = make_merge(
        client,
        base={"a": 1, "b": 1},
        left={"a": None, "b": 2},
        right={"a": 3, "b": 3},
    )
    tid = target["merge_id"]
    tgt_conflict = next(c for c in target["conflicts"] if c["pointer"] == "/a")
    assert tgt_conflict["missing"] == [] and tgt_conflict["left"] is None

    resp = reuse(client, tid, 1, sid)
    assert resp.status_code == 200
    body = resp.json()
    skipped_a = next(s for s in body["skipped"] if s["pointer"] == "/a")
    assert skipped_a["reason"] == "shape_mismatch"
    assert sorted(skipped_a["differences"]) == ["left", "missing"]
    # /b is shape-identical and still applied atomically in the same call.
    assert body["applied"] == [{"pointer": "/b", "action": "set", "value": 3}]
    assert body["revision"] == 2
    fetched = client.get(f"/merges/{tid}").json()
    assert fetched["document"]["a"] == {"__merge_conflict__": "/a"}


def test_reuse_explicit_null_value_is_reused(client):
    source = make_merge(
        client, base={"a": 1}, left={"a": None}, right={"a": 2}
    )
    sid = source["merge_id"]
    resolve(client, sid, 1, [{"pointer": "/a", "choice": "left"}])

    target = make_merge(
        client, base={"a": 1}, left={"a": None}, right={"a": 2}
    )
    tid = target["merge_id"]
    resp = reuse(client, tid, 1, sid)
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "resolved"
    assert body["result"] == {"a": None}
    assert body["applied"] == [{"pointer": "/a", "action": "set", "value": None}]


def test_reuse_source_deletion_is_honoured_in_target(client):
    # Left deletes /a (delete-vs-modify conflict); the source resolution
    # accepts the deletion, so reuse removes /a from the target as well.
    docs = {
        "base": {"a": 1, "b": 1},
        "left": {"b": 2},
        "right": {"a": 3, "b": 3},
    }
    source = make_merge(client, **docs)
    sid = source["merge_id"]
    resolve(
        client,
        sid,
        1,
        [{"pointer": "/a", "choice": "left"}, {"pointer": "/b", "choice": "left"}],
    )
    assert client.get(f"/merges/{sid}").json()["result"] == {"b": 2}

    target = make_merge(client, **docs)
    tid = target["merge_id"]
    resp = reuse(client, tid, 1, sid)
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "resolved"
    assert body["result"] == {"b": 2}
    assert body["applied"] == [
        {"pointer": "/a", "action": "delete"},
        {"pointer": "/b", "action": "set", "value": 2},
    ]


def test_reuse_bool_is_not_equal_to_integer(client):
    # Python would equate True == 1; the JSON comparison must not.
    source = make_merge(
        client, base={"a": 0}, left={"a": True}, right={"a": 2}
    )
    sid = source["merge_id"]
    resolve(client, sid, 1, [{"pointer": "/a", "choice": "left"}])

    target = make_merge(
        client, base={"a": 0}, left={"a": 1}, right={"a": 2}
    )
    resp = reuse(client, target["merge_id"], 1, sid)
    body = resp.json()
    assert body["applied"] == []
    assert body["skipped"] == [
        {"pointer": "/a", "reason": "shape_mismatch", "differences": ["left"]}
    ]
    assert body["revision"] == 1


def test_reuse_nested_values_compared_structurally(client):
    source = make_merge(
        client,
        base={"a": {"x": 1}},
        left={"a": {"x": 2}},
        right={"a": {"x": 3}},
    )
    sid = source["merge_id"]
    assert [c["pointer"] for c in source["conflicts"]] == ["/a/x"]
    resolve(client, sid, 1, [{"pointer": "/a/x", "choice": "left"}])

    # Structurally identical nested candidate reuses the source value.
    same = make_merge(
        client,
        base={"a": {"x": 1}},
        left={"a": {"x": 2}},
        right={"a": {"x": 3}},
    )
    resp = reuse(client, same["merge_id"], 1, sid)
    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "resolved"
    assert resp.json()["result"] == {"a": {"x": 2}}

    # A nested candidate that differs inside the conflicting leaf changes
    # the shape: not reused (an added sibling key would auto-merge at its
    # own path and leave /a/x shape-identical, so the difference must sit
    # inside the compared subtree).
    different = make_merge(
        client,
        base={"a": {"x": 1}},
        left={"a": {"x": 2}},
        right={"a": {"x": [3, 4]}},
    )
    resp = reuse(client, different["merge_id"], 1, sid)
    body = resp.json()
    assert body["applied"] == []
    assert body["skipped"][0]["differences"] == ["right"]
    assert body["revision"] == 1


def test_reuse_snapshot_covers_conflicts_resolved_across_batches(client):
    # The snapshot is taken at creation time, so a source that resolved its
    # conflicts over several batches still offers every original conflict.
    source = make_merge(client, **THREE_WAY)
    sid = source["merge_id"]
    resolve(client, sid, 1, [{"pointer": "/a", "value": 9}])
    resolve(client, sid, 2, [{"pointer": "/b", "choice": "left"}])
    resolve(client, sid, 3, [{"pointer": "/c", "choice": "right"}])

    target = make_merge(client, **THREE_WAY)
    resp = reuse(client, target["merge_id"], 1, sid)
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "resolved"
    assert body["result"] == {"a": 9, "b": 2, "c": 3}

    # The frozen snapshot itself persists the complete original conflict
    # list even though the source's live conflict list shrank to empty.
    row = db.query_one(
        "SELECT conflicts, conflict_snapshot FROM merges WHERE id = ?", (sid,)
    )
    assert _json.loads(row["conflicts"]) == []
    snapshot = _json.loads(row["conflict_snapshot"])
    assert sorted(c["pointer"] for c in snapshot) == ["/a", "/b", "/c"]


def test_reuse_stale_revision_returns_409(client):
    source = make_merge(
        client, base={"a": 1}, left={"a": 2}, right={"a": 3}
    )
    sid = source["merge_id"]
    resolve(client, sid, 1, [{"pointer": "/a", "choice": "left"}])

    target = make_merge(client, **THREE_WAY)
    tid = target["merge_id"]
    resp = reuse(client, tid, 999, sid)
    assert resp.status_code == 409
    assert resp.json()["current_revision"] == 1
    # Nothing changed by the rejected request.
    assert client.get(f"/merges/{tid}").json()["revision"] == 1


def test_reuse_unknown_target_and_source_404(client):
    source = make_merge(
        client, base={"a": 1}, left={"a": 2}, right={"a": 3}
    )
    sid = source["merge_id"]
    resolve(client, sid, 1, [{"pointer": "/a", "choice": "left"}])

    assert reuse(client, "does-not-exist", 1, sid).status_code == 404

    target = make_merge(client, **THREE_WAY)
    resp = reuse(client, target["merge_id"], 1, "does-not-exist")
    assert resp.status_code == 404
    assert client.get(f"/merges/{target['merge_id']}").json()["revision"] == 1


def test_reuse_pending_source_or_self_returns_409(client):
    pending_source = make_merge(
        client, base={"a": 1}, left={"a": 2}, right={"a": 3}
    )
    target = make_merge(client, **THREE_WAY)
    tid = target["merge_id"]

    resp = reuse(client, tid, 1, pending_source["merge_id"])
    assert resp.status_code == 409
    assert resp.json()["source_merge_id"] == pending_source["merge_id"]
    assert resp.json()["source_status"] == "pending"

    # A merge cannot reuse from itself (it is not resolved).
    resp = reuse(client, tid, 1, tid)
    assert resp.status_code == 409
    assert resp.json()["source_merge_id"] == tid


def test_reuse_against_resolved_target_returns_409(client):
    source = make_merge(
        client, base={"a": 1}, left={"a": 2}, right={"a": 3}
    )
    sid = source["merge_id"]
    resolve(client, sid, 1, [{"pointer": "/a", "choice": "left"}])

    target = make_merge(client, **THREE_WAY)
    tid = target["merge_id"]
    resolve(
        client,
        tid,
        1,
        [
            {"pointer": "/a", "value": 1},
            {"pointer": "/b", "value": 1},
            {"pointer": "/c", "value": 1},
        ],
    )
    resp = reuse(client, tid, 2, sid)
    assert resp.status_code == 409


def test_reuse_result_schema_failure_writes_nothing(client):
    # Source resolves /a to a string; the target demands an integer final
    # result. The last conflict is being resolved, so the gate fails and
    # the whole atomic reuse batch is rejected with 422.
    source = make_merge(client, **THREE_WAY)
    sid = source["merge_id"]
    resolve(
        client,
        sid,
        1,
        [
            {"pointer": "/a", "value": "not-an-int"},
            {"pointer": "/b", "choice": "left"},
            {"pointer": "/c", "choice": "right"},
        ],
    )

    schema = {
        "type": "object",
        "properties": {"a": {"type": "integer"}},
    }
    target = make_merge(client, schema=schema, **THREE_WAY)
    tid = target["merge_id"]
    resp = reuse(client, tid, 1, sid)
    assert resp.status_code == 422
    errors = resp.json()["errors"]
    assert any(e["instance_path"] == "/a" for e in errors)

    # Rolled back entirely: revision, status and conflicts are untouched.
    fetched = client.get(f"/merges/{tid}").json()
    assert fetched["revision"] == 1
    assert fetched["status"] == "pending"
    assert sorted(c["pointer"] for c in fetched["conflicts"]) == [
        "/a",
        "/b",
        "/c",
    ]


def test_reuse_partial_match_does_not_trigger_schema_gate(client):
    # Only /a is reusable, so the target stays pending: the partial working
    # document (still carrying placeholders) is not schema-checked. A small
    # one-conflict source with shape-identical string candidates covers /a.
    small = make_merge(
        client,
        base={"a": 1},
        left={"a": "x"},
        right={"a": "y"},
    )
    resolve(client, small["merge_id"], 1, [{"pointer": "/a", "choice": "left"}])

    schema = {
        "type": "object",
        "properties": {"a": {"type": "integer"}},
    }
    target = make_merge(
        client,
        schema=schema,
        base={"a": 1, "b": 1, "c": 1},
        left={"a": "x", "b": 2, "c": 2},
        right={"a": "y", "b": 3, "c": 3},
    )
    resp = reuse(client, target["merge_id"], 1, small["merge_id"])
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "pending"
    assert body["revision"] == 2
    assert body["applied"] == [{"pointer": "/a", "action": "set", "value": "x"}]
    assert sorted(s["pointer"] for s in body["skipped"]) == ["/b", "/c"]


def test_reuse_matches_by_pointer_not_position(client):
    # Identical scalar shape at *different* pointers must not be reused.
    source = make_merge(
        client, base={"a": 1}, left={"a": 2}, right={"a": 3}
    )
    sid = source["merge_id"]
    resolve(client, sid, 1, [{"pointer": "/a", "choice": "left"}])

    target = make_merge(
        client, base={"z": 1}, left={"z": 2}, right={"z": 3}
    )
    resp = reuse(client, target["merge_id"], 1, sid)
    body = resp.json()
    assert body["applied"] == []
    assert body["skipped"] == [
        {"pointer": "/z", "reason": "no_source_conflict"}
    ]
    assert body["revision"] == 1


def test_resolved_source_without_conflicts_offers_nothing(client):
    # A conflict-free source is resolved immediately but has an empty
    # snapshot: reuse reports every target conflict as unmatched.
    source = make_merge(
        client, base={"a": 1}, left={"a": 2}, right={"a": 2}
    )
    assert source["status"] == "resolved"
    target = make_merge(
        client, base={"a": 1}, left={"a": 2}, right={"a": 3}
    )
    resp = reuse(client, target["merge_id"], 1, source["merge_id"])
    assert resp.status_code == 200
    body = resp.json()
    assert body["applied"] == []
    assert body["skipped"] == [
        {"pointer": "/a", "reason": "no_source_conflict"}
    ]
    assert body["revision"] == 1
