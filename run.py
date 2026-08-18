




import json as _json, os as _os, sys as _sys
_OPERON_TAPE = _json.load(open(_os.path.join(_os.path.dirname(__file__) if "__file__" in dir() else ".", "operon_tape.json")))


_OPERON_SKIPPABLE = frozenset({"list_artifacts", "artifacts", "current_model", "reasoning_model", "list_models"})
class _OperonTapeError(RuntimeError):



    pass





class ContactEmailUnavailable(RuntimeError):
    status = "unavailable"
    _DEFAULT_MESSAGE = (
        "No contact-email decision could be obtained. Omit the parameter "
        "— most services work without it."
    )
    def __init__(self, message=None):
        super().__init__(message or self._DEFAULT_MESSAGE)
class ContactEmailDeclined(ContactEmailUnavailable):
    status = "declined"
    _DEFAULT_MESSAGE = (
        "The user declined to share a contact email. Do not ask the "
        "user; omit the parameter — most services work without it."
    )
class CredentialUnavailable(RuntimeError):
    status = "unavailable"
    _DEFAULT_MESSAGE = (
        "No credential could be obtained for this provider. Skip the "
        "steps that need it; the user can add one under Customize "
        "→ Credentials."
    )
    def __init__(self, message=None, provider=None):
        super().__init__(message or self._DEFAULT_MESSAGE)
        self.provider = provider
class CredentialDeclined(CredentialUnavailable):
    status = "declined"
    _DEFAULT_MESSAGE = (
        "The user declined to provide this provider's credential. Do not "
        "ask again; skip the steps that need it and say why."
    )
class _OperonReplay:


    ContactEmailUnavailable = ContactEmailUnavailable
    ContactEmailDeclined = ContactEmailDeclined
    CredentialUnavailable = CredentialUnavailable
    CredentialDeclined = CredentialDeclined
    def __init__(self): self._i = 0
    def _next(self, m):
        if self._i >= len(_OPERON_TAPE):
            raise RuntimeError(f"host-SDK replay: tape exhausted at call {self._i} ({m!r}). The notebook made more host.*/operon.* calls than were recorded.")
        rec = _OPERON_TAPE[self._i]












        j = self._i
        while rec["method"] != m and rec["method"] in _OPERON_SKIPPABLE:
            j += 1
            if j >= len(_OPERON_TAPE):
                raise RuntimeError(f"host-SDK replay: tape exhausted at call {j} ({m!r})")
            rec = _OPERON_TAPE[j]
        if rec["method"] != m:
            raise RuntimeError(f"host-SDK replay drift at step {j}: recorded {rec['method']!r}, requested {m!r}")

        for k in range(self._i, j):
            print(f"host-SDK replay: skipping tape entry {k} "
                  f"({_OPERON_TAPE[k]['method']!r}) — not in extracted_code", file=_sys.stderr)
        self._i = j + 1
        if rec.get("error"): raise _OperonTapeError(rec["error"])
        return rec["data"]
    class _L:



        def __init__(s): s._c = {}
        def __getitem__(s, k):



            if not isinstance(k, str):
                raise TypeError(
                    "host.lineage[...]: version_id must be a str (UUID), "
                    f"got {type(k).__name__}"
                )
            if not k:
                raise ValueError(
                    "host.lineage[...]: version_id must be a non-empty str"
                )
            k = k.lower()
            if k in s._c: return s._c[k]
            r = _replay._next("get_lineage")




            if not (isinstance(r, dict) and r.get("extraction_pending")):
                s._c[k] = r
            return r
        def __contains__(s, k): return str(k).lower() in s._c
        def clear(s): s._c.clear()
        def graph(s, version_id, direction="up", max_depth=None, max_nodes=None):










            if not isinstance(version_id, str) or not version_id:
                raise TypeError(
                    "host.lineage.graph(...): version_id must be a non-empty "
                    f"str (UUID), got {type(version_id).__name__}"
                )
            if direction not in ("up", "down"):
                raise ValueError(
                    f"host.lineage.graph(...): direction must be 'up' or 'down', got {direction!r}"
                )
            if max_depth is not None: int(max_depth)
            if max_nodes is not None: int(max_nodes)
            return _replay._next("get_lineage_topology")
    lineage = _L()
    class _Ar:










        def search(s, query, k=8, from_idx=None, to_idx=None):
            if not isinstance(query, str) or not query.strip():
                raise TypeError(
                    f"host.archive.search: query must be a non-empty str, "
                    f"got {query!r}"
                )
            if isinstance(k, bool) or not isinstance(k, int) or k < 1:
                raise TypeError(
                    f"host.archive.search: k must be a positive int, got {k!r}"
                )
            for name, v in (("from_idx", from_idx), ("to_idx", to_idx)):
                if v is None:
                    continue
                if isinstance(v, bool) or not isinstance(v, int) or v < 0:
                    raise TypeError(
                        f"host.archive.search: {name} must be a non-negative "
                        f"int, got {v!r}"
                    )
            r = _replay._next("archive_search")
            if isinstance(r, dict) and "error" in r and len(r) == 1:
                raise RuntimeError(r["error"])
            return r
        def page(s, start=0, count=20, to_idx=None):
            if isinstance(start, bool) or not isinstance(start, int) or start < 0:
                raise TypeError(
                    f"host.archive.page: start must be a non-negative int, "
                    f"got {start!r}"
                )
            if isinstance(count, bool) or not isinstance(count, int) or count < 1:
                raise TypeError(
                    f"host.archive.page: count must be a positive int, "
                    f"got {count!r}"
                )
            if to_idx is not None and (
                isinstance(to_idx, bool)
                or not isinstance(to_idx, int)
                or to_idx < 0
            ):
                raise TypeError(
                    f"host.archive.page: to_idx must be a non-negative int, "
                    f"got {to_idx!r}"
                )
            r = _replay._next("archive_page")
            if isinstance(r, dict) and "error" in r and len(r) == 1:
                raise RuntimeError(r["error"])
            return r
        def __repr__(s): return "<host.archive (replay shim)>"
    archive = _Ar()
    class _C:









        def list(s):
            try: return _replay._next("credentials_list")
            except _OperonTapeError: return []
        def get(s, name):
            try: return _replay._next("credentials_get")
            except _OperonTapeError: return {}
        def request(s, provider):








            if not isinstance(provider, str) or not provider:
                raise TypeError(
                    "host.credentials.request: provider must be a non-empty str"
                )
            try:
                r = _replay._next("credentials_request")
            except _OperonTapeError as e:
                raise CredentialUnavailable(str(e), provider=provider) from None

            if isinstance(r, str):
                return r
            if isinstance(r, dict):
                status = r.get("status")
                value = r.get("value")
                if status == "available" and isinstance(value, str) and value:
                    return value
                message = r.get("message")
                if not (isinstance(message, str) and message):
                    message = None
                prov = r.get("provider") if isinstance(r.get("provider"), str) else provider
                if status == "declined":
                    raise CredentialDeclined(message, provider=prov)
                raise CredentialUnavailable(message, provider=prov)
            raise CredentialUnavailable(provider=provider)
        def __repr__(s):


            try: _replay._next("credentials_list")
            except Exception: pass
            return "<host.credentials (replay)>"
    credentials = _C()
    class _Q:




        def __call__(s, sql, params=None, limit=None, df=False, scope="project"):



            if not isinstance(sql, str):
                raise TypeError(f"host.query: sql must be a str, got {type(sql).__name__}")
            if not sql.strip():
                raise ValueError("host.query: sql must be a non-empty str")
            if params is not None and not isinstance(params, (list, tuple)):
                raise TypeError(
                    f"host.query: params must be a list or tuple, got "
                    f"{type(params).__name__}. (dicts iterate keys, strings "
                    f"iterate chars — both produce silent wrong results.)"
                )
            if limit is not None and (
                not isinstance(limit, int) or isinstance(limit, bool) or limit < 1
            ):
                raise TypeError(f"host.query: limit must be a positive int or None, got {limit!r}")
            if not isinstance(df, bool):
                raise TypeError(f"host.query: df must be a bool, got {type(df).__name__}")
            if scope not in ("project", "global"):
                raise ValueError(f"host.query: scope must be 'project' or 'global', got {scope!r}")
            r = _replay._next("query_db")
            if df:
                try:
                    import pandas as _pd
                    return _pd.DataFrame(r["rows"], columns=r["columns"])
                except ImportError: pass
            return r
        def schema(s):
            return _replay._next("query_schema")
    query = _Q()
    class _Rt:











        def _call(s, method):



            r = _replay._next("routine_" + method)
            if isinstance(r, dict) and "error" in r and len(r) == 1:
                raise RuntimeError(f"host.routine.{method}: {r['error']}")
            return r
        def configure(s, every_minutes, *, on_tick, label=None):
            if isinstance(every_minutes, bool) or not isinstance(every_minutes, int):
                raise TypeError(
                    "host.routine.configure: every_minutes must be an int "
                    f"(minutes, 5–1440), got {type(every_minutes).__name__}"
                )
            if not (5 <= every_minutes <= 1440):
                raise ValueError(
                    "host.routine.configure: every_minutes must be between "
                    f"5 and 1440 (inclusive), got {every_minutes!r}"
                )
            if not isinstance(on_tick, str) or not on_tick.strip():
                raise TypeError(
                    "host.routine.configure: on_tick must be a non-empty "
                    f"str, got {on_tick!r}"
                )
            if len(on_tick) > 2000:
                raise ValueError(
                    "host.routine.configure: on_tick must be ≤ 2000 chars, "
                    f"got {len(on_tick)}"
                )
            if label is not None:
                if not isinstance(label, str):
                    raise TypeError(
                        "host.routine.configure: label must be a str or "
                        f"None, got {type(label).__name__}"
                    )
                if len(label) > 120:
                    raise ValueError(
                        "host.routine.configure: label must be ≤ 120 "
                        f"chars, got {len(label)}"
                    )
            return s._call("configure")
        def status(s):
            return s._call("status")
        def done(s, had_work, summary=""):
            if not isinstance(had_work, bool):
                raise TypeError(
                    "host.routine.done: had_work must be a bool, got "
                    f"{type(had_work).__name__}"
                )
            if not isinstance(summary, str):
                raise TypeError(
                    "host.routine.done: summary must be a str, got "
                    f"{type(summary).__name__}"
                )
            if len(summary) > 500:
                raise ValueError(
                    "host.routine.done: summary must be ≤ 500 chars, got "
                    f"{len(summary)}"
                )
            return s._call("done")
        def __repr__(s): return "<host.routine (replay shim)>"
    routine = _Rt()
    class _F:









        def __call__(s, status="unresolved", frame=None, all_branches=False):
            if not isinstance(status, str) or status not in (
                    "unresolved", "open", "claimed", "unaddressed",
                    "resolved", "all"):
                raise ValueError(
                    f"host.findings: status={status!r} is not valid. "
                    f"Must be one of: ['unresolved', 'open', 'claimed', "
                    f"'unaddressed', 'resolved', 'all']"
                )
            if frame is not None and (not isinstance(frame, str) or not frame):
                raise TypeError(
                    f"host.findings: frame must be a non-empty str or None, "
                    f"got {frame!r}"
                )
            if not isinstance(all_branches, bool):
                raise TypeError(
                    f"host.findings: all_branches must be a bool, got "
                    f"{type(all_branches).__name__}"
                )
            return _replay._next("list_findings")
        def mark_addressed(s, ids, note):
            if isinstance(ids, str):
                ids = [ids]
            if (not isinstance(ids, (list, tuple)) or not ids
                    or any(not isinstance(i, str) or not i for i in ids)):
                raise TypeError(
                    "host.findings.mark_addressed: ids must be a non-empty "
                    "finding id or list of finding ids (exact ids from "
                    "host.findings())"
                )
            if len(ids) > 100:
                raise ValueError(
                    f"host.findings.mark_addressed: at most 100 ids per "
                    f"call (got {len(ids)})"
                )
            if not isinstance(note, str) or not note.strip():
                raise ValueError(
                    "host.findings.mark_addressed: note is required — a "
                    "non-empty description of what you did; the user sees "
                    'it as "addressed by agent: <note>"'
                )
            if len(note) > 2000:
                raise ValueError(
                    f"host.findings.mark_addressed: note exceeds "
                    f"2000 chars (got {len(note)})"
                )
            return _replay._next("mark_findings_addressed")
        def __repr__(s): return "<host.findings (replay shim)>"
    findings = _F()
    class _ME:






        def free_port(s):
            r = _replay._next("model_endpoints_free_port")
            return r["port"]
        def register(s, name, url, skill, start=None, stop=None, live=None,
                     credential="NVIDIA_API_KEY"):
            for label, v in (("name", name), ("url", url), ("skill", skill)):
                if not isinstance(v, str) or not v.strip():
                    raise TypeError(
                        f"host.model_endpoints.register: {label} must be a "
                        "non-empty str"
                    )
            hosted = isinstance(url, str) and url.strip().lower().startswith(
                "https://"
            )
            if hosted:
                for label, v in (("start", start), ("stop", stop), ("live", live)):
                    if v is not None:
                        raise TypeError(
                            "host.model_endpoints.register: remote endpoints "
                            f"(https url) take no `{label}` — the daemon owns "
                            "no lifecycle for them; omit start/stop/live"
                        )
            else:
                for label, v in (("start", start), ("stop", stop), ("live", live)):
                    if not isinstance(v, str) or not v.strip():
                        raise TypeError(
                            f"host.model_endpoints.register: {label} must be a "
                            "non-empty str for a local endpoint"
                        )
            if credential is not None and (
                not isinstance(credential, str) or not credential.strip()
            ):
                raise TypeError(
                    "host.model_endpoints.register: credential must be the NAME "
                    "of a saved credential (a str), never a key value"
                )
            return _replay._next("model_endpoints_register")
        def __repr__(s): return "<host.model_endpoints (replay shim)>"
    model_endpoints = _ME()
    class _At:







        _DELETE_BATCH_CAP = 200
        def __call__(s, *a, **k): return _replay._next("list_artifacts")
        def delete(s, artifact_ids, reason=None):
            if isinstance(artifact_ids, str):
                artifact_ids = [artifact_ids]
            if not isinstance(artifact_ids, (list, tuple)) or not artifact_ids:
                raise TypeError(
                    "host.artifacts.delete: artifact_ids must be an artifact-id "
                    "str or a non-empty list of them"
                )
            ids = list(artifact_ids)
            for x in ids:
                if not isinstance(x, str) or not x:
                    raise TypeError(
                        f"host.artifacts.delete: each artifact id must be a "
                        f"non-empty str, got {x!r}"
                    )
            if len(set(ids)) > s._DELETE_BATCH_CAP:
                raise ValueError(
                    f"host.artifacts.delete: batch of {len(set(ids))} exceeds the "
                    f"{s._DELETE_BATCH_CAP}-artifact cap — split into "
                    f"smaller batches"
                )
            if reason is not None and not isinstance(reason, str):
                raise TypeError(
                    f"host.artifacts.delete: reason must be a str or None, got "
                    f"{type(reason).__name__}"
                )
            return _replay._next("delete_artifacts")
        def rename(s, artifact_id, filename):
            if not isinstance(artifact_id, str) or not artifact_id:
                raise TypeError(
                    "host.artifacts.rename: artifact_id must be a non-empty str "
                    "(the 'id' field from host.artifacts())"
                )
            if not isinstance(filename, str) or not filename.strip():
                raise TypeError(
                    "host.artifacts.rename: filename must be a non-empty str"
                )
            return _replay._next("rename_artifact")
        def __repr__(s): return "<host.artifacts (replay shim)>"
    artifacts = _At()
    def _llm_offense(self, req):





        s = req.get("system")
        if s is not None and not isinstance(s, str):
            return "system"
        msgs = req.get("messages")
        if isinstance(msgs, (list, tuple)):
            for m in msgs:
                if not isinstance(m, dict) or not isinstance(
                        m.get("role"), str):
                    return "shape"
                if m.get("role") not in ("user", "assistant"):
                    return "role"
        return None
    def _llm_legacy_or_raise(self, wire, kind, label=""):



















        if wire is not None and self._i < len(_OPERON_TAPE):
            rec = _OPERON_TAPE[self._i]
            if rec.get("method") == wire and rec.get("legacy_system"):
                return self._next(wire)
        if kind == "shape":
            raise TypeError(
                f"host.llm: {label}messages entries must be "
                f"{{'role', 'content'}} dicts with role 'user' or "
                f"'assistant'"
            )
        if kind == "role":
            raise TypeError(
                f"host.llm: {label}messages role must be 'user' or "
                f"'assistant' — use the top-level `system` field instead"
            )
        raise TypeError(
            f"host.llm: {label}system must be a str or None"
        )
    def llm(self, request=None, system=None, model=None, max_tokens=None,
            max_concurrency=None, **k):













        if system is not None and not isinstance(system, str):
            return self._llm_legacy_or_raise(None, "system")
        if isinstance(request, (list, tuple)):
            if (system is not None or model is not None
                    or max_tokens is not None or k):
                raise TypeError(
                    "host.llm: when passing a list of requests, put "
                    "per-request options inside each request dict"
                )
            if max_concurrency is not None and (
                    not isinstance(max_concurrency, int)
                    or isinstance(max_concurrency, bool)
                    or max_concurrency < 1):
                raise TypeError(
                    f"host.llm: max_concurrency must be a positive int "
                    f"or None, got {max_concurrency!r}"
                )


            if not request:
                return []
            kind = None
            label = ""
            for i, r in enumerate(request):
                if isinstance(r, str):
                    continue
                if not isinstance(r, dict):
                    raise TypeError(
                        f"host.llm: requests[{i}] must be a dict or a "
                        f"prompt str, got {type(r).__name__}"
                    )
                if kind is None:
                    kind = self._llm_offense(r)
                    if kind:
                        label = f"requests[{i}]: "
            if kind:
                return self._llm_legacy_or_raise("llm_batch", kind, label)
            return self._next("llm_batch")
        if max_concurrency is not None:
            raise TypeError(
                "host.llm: max_concurrency only applies when passing a "
                "list of requests"
            )
        if isinstance(request, dict):
            if (system is not None or model is not None
                    or max_tokens is not None or k):
                raise TypeError(
                    "host.llm: pass options inside the request dict OR "
                    "as keyword arguments, not both"
                )
            _req = dict(request)
        elif isinstance(request, str) or request is None:
            _req = dict(k)
            if isinstance(request, str):
                if "prompt" in _req:
                    raise TypeError(
                        "host.llm: prompt given both positionally and "
                        "as a keyword"
                    )
                _req["prompt"] = request
            if system is not None:
                _req["system"] = system
            if model is not None:
                _req["model"] = model
            if max_tokens is not None:
                _req["max_tokens"] = max_tokens
        else:
            raise TypeError(
                f"host.llm: expected a prompt str, a request dict, or a "
                f"list of request dicts; got {type(request).__name__}"
            )
        extended = any(_req.get(x) is not None for x in
                       ("tools", "tool_choice", "images", "messages",
                        "temperature"))
        kind = self._llm_offense(_req)
        if kind:


            out = self._llm_legacy_or_raise(
                "llm_batch" if extended else "llm", kind)
            if not extended:
                return out
            r = out[0] if isinstance(out, list) and out else {}
            if isinstance(r, dict) and "error" in r and "text" not in r:
                raise RuntimeError(f"host.llm: {r['error']}")
            return r
        _req.pop("system", None)
        if extended:
            out = self._next("llm_batch")
            r = out[0] if isinstance(out, list) and out else {}
            if isinstance(r, dict) and "error" in r and "text" not in r:
                raise RuntimeError(f"host.llm: {r['error']}")
            return r
        return self._next("llm")
    def llm_batch(self, requests, max_concurrency=8):













        if not isinstance(requests, (list, tuple)):
            raise TypeError(
                f"host.llm_batch: requests must be a list, got "
                f"{type(requests).__name__}"
            )
        return self.llm(list(requests), max_concurrency=max_concurrency)
    def current_model(self):      return self._next("current_model")
    def reasoning_model(self):    return self._next("reasoning_model")
    def list_models(self):        return self._next("list_models")
    def artifact_path(self, version_id):



        if not isinstance(version_id, str) or not version_id:
            raise TypeError(
                "host.artifact_path: version_id must be a non-empty str, got "
                f"{type(version_id).__name__}"
            )
        return self._next("artifact_path")
    def view_image(self, source=None, *a, **k):





        import re as _re
        if isinstance(source, str) and _re.fullmatch(r"[0-9a-fA-F-]{36}", source):




            try: self._next("artifact_path")
            except _OperonTapeError: pass
        return {"source": "replay", "original_size": (0, 0), "crop": None,
                "output_size": (0, 0), "saved_to": "<replay>"}
    def artifact_marker(self, version_id):

        if not isinstance(version_id, str) or not version_id:
            raise TypeError(
                "host.artifact_marker: version_id must be a non-empty str, "
                f"got {type(version_id).__name__}"
            )
        return "{" + "{" + "artifact:" + version_id + "}" + "}"
    def capabilities(self):






        _m = ("query", "llm", "llm_batch", "current_model",
              "reasoning_model", "list_models",
              "frames", "children", "delegate", "delegation_stats",
              "mcp", "lineage", "artifacts", "credentials", "app",
              "findings", "submit_output")
        caps = {**{c: True for c in _m},
                **{c: False for c in ("agents", "skills", "compute",
                                      "exec_peek", "exec_interrupt")}}



        caps["goal"] = "goal" in type(self).__dict__
        return caps
    def frames(self, *a, **k):    return self._next("list_frames")
    def children(self, *a, **k):  return self._next("list_running_children")











    @staticmethod
    def _slots(val, label):


        def _one(x):
            if isinstance(x, str) and x:
                return ("ok", x)
            if isinstance(x, dict):
                fid = x.get("frame_id") or x.get("child_frame_id")
                if isinstance(fid, str) and fid:
                    return ("ok", fid)
                return ("bad", (
                    f"{label}: dict entry has no usable frame_id — pass the "
                    f"descriptor/result dict from host.delegate() (it "
                    f"carries frame_id), or the id string itself; got keys "
                    f"{sorted(x.keys())!r}"
                ))
            return ("bad", (
                f"{label}: each entry must be a frame-id str or a "
                f"descriptor/result dict carrying frame_id, got "
                f"{type(x).__name__}"
            ))
        if isinstance(val, (list, tuple)):
            if not val:
                raise TypeError(f"{label}: list must be non-empty")
            return [_one(x) for x in val], True
        return [_one(val)], False
    def collect(self, frame_ids, timeout=30.0):
        slots, _ = self._slots(frame_ids, "host.collect")
        if timeout is None:
            raise ValueError(
                "host.collect: timeout=None (unbounded) is not allowed — "
                "collect is deadline-bounded like wait_for_notification "
                "(default 30s, max 1800s). Loop short slices for long waits: "
                "still-running slots come back as {'status': 'running'} "
                "descriptors you can pass straight back in."
            )
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
            raise TypeError(
                f"host.collect: timeout must be a positive number of "
                f"seconds (default 30, max 1800), got {type(timeout).__name__}"
            )
        if timeout <= 0:
            raise ValueError(
                "host.collect: timeout must be a positive number of seconds "
                "(default 30, max 1800)"
            )
        if timeout > 1800:
            raise ValueError(
                "host.collect: timeout is capped at 1800s (matches the "
                "wait_for_notification guidance ceiling) — loop shorter "
                "slices instead of one long wait"
            )
        good = [fid for (tag, fid) in slots if tag == "ok"]
        if good:
            r = self._next("collect")
            if isinstance(r, dict) and "error" in r and len(r) == 1:
                raise RuntimeError(f"host.collect: {r['error']}")
        else:
            r = []
        it = iter(r)
        out = []
        for tag, v in slots:
            if tag == "ok":
                out.append(next(it))
            else:
                out.append({"status": "failed", "error": v})
        return out
    def send_message(self, target=None, message=None, kind="info", *,
                     child_frame_id=None):
        if target is None:
            target = child_frame_id
        if isinstance(target, (list, tuple)):
            raise TypeError(
                "host.send_message: one target per call — pass a single "
                "frame-id str or descriptor dict (loop for a wave)"
            )
        if not isinstance(message, str) or not message.strip():
            raise TypeError(
                "host.send_message: message must be a non-empty str"
            )
        if kind not in ("info", "question"):
            raise ValueError(
                "host.send_message: kind must be 'info' or 'question', "
                f"got {kind!r}"
            )
        if target == "parent":
            fid = "parent"
        elif target == "main":
            fid = "main"
        else:
            slots, _ = self._slots(target, "host.send_message")
            if slots[0][0] == "bad":
                return {"status": "failed", "error": slots[0][1]}
            fid = slots[0][1]
        r = self._next("send_message")
        if isinstance(r, dict) and "error" in r and len(r) == 1:
            return {"target": fid, "status": "failed", "error": r["error"]}
        return r
    def stop_child(self, child_frame_id, reason=None):
        if reason is not None and not isinstance(reason, str):
            raise TypeError(
                f"host.stop_child: reason must be a str or None, got "
                f"{type(reason).__name__}"
            )
        slots, was_list = self._slots(child_frame_id, "host.stop_child")
        good = [fid for (tag, fid) in slots if tag == "ok"]
        if not was_list:
            if not good:
                return {"status": "failed", "error": slots[0][1]}
            r = self._next("stop_child")
            if isinstance(r, dict) and "error" in r and len(r) == 1:
                return {
                    "child_frame_id": good[0],
                    "status": "failed",
                    "error": r["error"],
                }
            return r
        if good:
            r = self._next("stop_child")
            if isinstance(r, dict) and "error" in r and len(r) == 1:
                host_results = [
                    {"child_frame_id": fid, "status": "failed", "error": r["error"]}
                    for fid in good
                ]
            else:
                host_results = r.get("results", []) if isinstance(r, dict) else []
        else:
            host_results = []
        it = iter(host_results)
        merged = []
        for tag, v in slots:
            if tag == "ok":
                try:
                    merged.append(next(it))
                except StopIteration:
                    merged.append({"status": "failed", "error": "no result"})
            else:
                merged.append({"status": "failed", "error": v})
        return merged
    def submit_output(self, output, completion_bullets):
        if not isinstance(output, dict):
            raise TypeError(
                f"host.submit_output: output must be a dict matching your "
                f"task's OUTPUT SCHEMA, got {type(output).__name__}"
            )
        if isinstance(completion_bullets, str):
            completion_bullets = [completion_bullets]
        if (not isinstance(completion_bullets, (list, tuple))
                or not completion_bullets
                or not all(isinstance(b, str) and b.strip()
                           for b in completion_bullets)):
            raise TypeError(
                "host.submit_output: completion_bullets must be a non-empty "
                "list of short past-tense strings (e.g. ['Computed square "
                "root of 144'])"
            )
        return self._next("submit_output")
    def get_user_email(self):







        try:
            r = self._next("get_user_email")
        except _OperonTapeError as e:
            raise ContactEmailUnavailable(str(e)) from None
        if isinstance(r, str):
            return r
        if isinstance(r, dict):
            status = r.get("status")
            email = r.get("email")
            if status == "allowed" and isinstance(email, str) and email:
                return email
            message = r.get("message")
            if not (isinstance(message, str) and message):
                message = None
            if status == "declined":
                raise ContactEmailDeclined(message)
            raise ContactEmailUnavailable(message)
        raise ContactEmailUnavailable()
    def delegate(self, request=None, task=None, name=None,
                 context_summary=None, profile=None, output_schema=None,
                 model=None, max_concurrency=None):







        if isinstance(request, (list, tuple)):
            if not request:
                return []
            return self._next("delegate")
        out = self._next("delegate")
        return out[0] if isinstance(out, list) and out else {}
    class _M:










        def __call__(s, server, method, /, **kwargs):
            if not isinstance(server, str) or not server:
                raise TypeError(
                    f"host.mcp: server must be a non-empty str, got {server!r}"
                )
            if not isinstance(method, str) or not method:
                raise TypeError(
                    f"host.mcp: method must be a non-empty str, got {method!r}"
                )
            r = _replay._next("mcp")
            if isinstance(r, str) and r and r[0] in "{[":
                try: return _json.loads(r)
                except _json.JSONDecodeError: pass
            return r
        def __repr__(s): return "<host.mcp (replay shim)>"
    mcp = _M()
    def app(self, server):
        _n = self._next
        class _A:
            def tools(s): return _n("app_tools_list")
            def __getattr__(s, n):
                if n.startswith("_"): raise AttributeError(n)
                return lambda *a, **k: _n("app_tool")
        return _A()
    def __getattr__(self, name):







        if name.startswith("_"): raise AttributeError(name)
        return lambda *a, **k: self._next(name)



_replay = _OperonReplay()
host = _replay


operon = _replay






_sys.modules["host"] = _replay
_sys.modules["operon"] = _replay

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import json
import traceback
import warnings
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from sksurv.util import Surv
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw

warnings.filterwarnings("ignore")

DATA = os.environ.get("BC_BENCH", "/mnt/kedargouri/sachin/projects/paper2/harmonised")
SEED = 20260725
N_BOOT = 2000
PARTS = "parts"
os.makedirs(PARTS, exist_ok=True)

OS_COHORTS = ["TCGA", "METABRIC", "SCANB_GSE96058", "SCANB_GSE202203",
              "GSE20711", "GSE58812"]
SEC_COHORTS = ["GSE6532", "GSE11121", "GSE21653"]

with open(os.path.join(DATA, "gene_sets.json")) as fh:
    _RAW = json.load(fh)
GENE_SETS = {k: list(v["genes"]) if isinstance(v, dict) else list(v)
             for k, v in _RAW.items()}
GS_FAMILY = {k: (v.get("family", "") if isinstance(v, dict) else "")
             for k, v in _RAW.items()}
ALL_GENES = sorted({g for v in GENE_SETS.values() for g in v})
print({k: len(v) for k, v in GENE_SETS.items()}, flush=True)
print(f"union of gene-set genes: {len(ALL_GENES)}", flush=True)

SURV_KEEP = ["time_months", "event", "endpoint", "cohort", "platform"]


def load_cohort(c):
    surv = pd.read_parquet(os.path.join(DATA, f"{c}_surv.parquet"))
    if "sample" in surv.columns:
        surv = surv.set_index("sample")
    surv = surv[[k for k in SURV_KEEP if k in surv.columns]]
    import pyarrow.parquet as pq
    have = set(pq.ParquetFile(os.path.join(DATA, f"{c}_expr.parquet")).schema.names)
    cols = [g for g in ALL_GENES if g in have]
    expr = pd.read_parquet(os.path.join(DATA, f"{c}_expr.parquet"), columns=cols)
    common = surv.index.intersection(expr.index)
    surv, expr = surv.loc[common], expr.loc[common]
    ok = (surv["time_months"].astype(float) > 0) & surv["time_months"].notna() \
         & surv["event"].notna()
    return surv.loc[ok.values], expr.loc[ok.values].astype(np.float32)


COH = {}
for c in OS_COHORTS + SEC_COHORTS:
    COH[c] = load_cohort(c)
    s, e = COH[c]
    print(f"loaded {c}: n={len(s)} events={int(s['event'].sum())} "
          f"endpoint={s['endpoint'].iloc[0]} panel_genes={e.shape[1]}", flush=True)


def y_of(surv):
    return Surv.from_arrays(event=surv["event"].astype(float).astype(bool).values,
                            time=surv["time_months"].astype(float).values)


def make_models(n_feat):
    m = {}
    m["CoxPH_ridge"] = ([(f"alpha={a}", lambda a=a: CoxPHSurvivalAnalysis(alpha=a, n_iter=200))
                         for a in [0.01, 0.1, 1.0, 10.0, 100.0]], 5)
    m["Coxnet"] = ([(f"alpha={a}", lambda a=a: CoxnetSurvivalAnalysis(
                        l1_ratio=0.9, alphas=[a], fit_baseline_model=False,
                        max_iter=100000, tol=1e-6))
                    for a in [0.5, 0.2, 0.1, 0.05, 0.02, 0.01]], 5)
    m["RSF"] = ([("leaf=50,frac=0.5", lambda: RandomSurvivalForest(
                    n_estimators=100, min_samples_leaf=50, max_features="sqrt",
                    max_samples=0.5, n_jobs=1, random_state=SEED, low_memory=True)),
                 ("leaf=100,frac=0.4", lambda: RandomSurvivalForest(
                    n_estimators=100, min_samples_leaf=100, max_features="sqrt",
                    max_samples=0.4, n_jobs=1, random_state=SEED, low_memory=True))], 3)
    m["GBSA"] = ([(f"n={n},lr={lr},d={d}", lambda n=n, lr=lr, d=d:
                    GradientBoostingSurvivalAnalysis(
                        n_estimators=n, learning_rate=lr, max_depth=d,
                        subsample=0.8, random_state=SEED))
                  for (n, lr, d) in [(100, 0.1, 2), (150, 0.05, 2)]], 3)
    return m


def risk(est, X):
    return np.asarray(est.predict(X), dtype=float).ravel()


def comparable_matrix(t, e):
    t = np.asarray(t, float)
    e = np.asarray(e, bool)
    later = t[None, :] > t[:, None]
    same_cens = (t[None, :] == t[:, None]) & (~e[None, :])
    A = (e[:, None] & (later | same_cens))
    return A.astype(np.float32)


def conc_matrix(A, s):
    d = s[:, None] - s[None, :]
    W = np.where(d > 0, 1.0, np.where(d == 0, 0.5, 0.0)).astype(np.float32)
    return A * W


def boot_ci_fast(A, N, n_boot=N_BOOT, seed=SEED, block=200):
    n = A.shape[0]
    rng = np.random.default_rng(seed)
    vals = []
    for start in range(0, n_boot, block):
        b = min(block, n_boot - start)
        Wm = rng.multinomial(n, np.full(n, 1.0 / n), size=b).astype(np.float32)
        den = np.einsum("bi,bi->b", Wm @ A, Wm)
        num = np.einsum("bi,bi->b", Wm @ N, Wm)
        good = den > 0
        vals.append(num[good] / den[good])
    v = np.concatenate(vals)
    if v.size < 50:
        return np.nan, np.nan, int(v.size)
    lo, hi = np.percentile(v, [2.5, 97.5])
    return float(lo), float(hi), int(v.size)


def uno(y_train, surv_test, score):
    try:
        ev = surv_test["event"].astype(float).astype(bool).values
        tt = surv_test["time_months"].astype(float).values
        if ev.sum() < 2:
            return np.nan, "too few events"
        y_te = Surv.from_arrays(event=ev, time=tt)
        tau = min(float(y_train["time"].max()), float(np.quantile(tt[ev], 0.95)))
        keep = tt < tau
        if keep.sum() < 10 or ev[keep].sum() < 2:
            return np.nan, "too few events below tau"
        return float(concordance_index_ipcw(y_train, y_te[keep], score[keep],
                                            tau=tau)[0]), ""
    except Exception as e:
        return np.nan, f"uno:{type(e).__name__}: {e}"


def cohort_folds(cohort_labels, n_splits):
    uniq, counts = np.unique(cohort_labels, return_counts=True)
    k = min(n_splits, len(uniq))
    assign, load = {}, np.zeros(k)
    for i in np.argsort(-counts):
        j = int(np.argmin(load))
        assign[uniq[i]] = j
        load[j] += counts[i]
    fold_of = np.array([assign[c] for c in cohort_labels])
    folds = []
    for j in range(k):
        te = np.where(fold_of == j)[0]
        tr = np.where(fold_of != j)[0]
        if len(te) >= 10 and len(tr) >= 30:
            folds.append((tr, te))
    return folds


def run_cell(held_out, gs_name, train_cohorts, tag):
    rows, score_rows = [], []
    genes_req = GENE_SETS[gs_name]
    surv_te, expr_te = COH[held_out]

    avail = set(genes_req) & set(expr_te.columns)
    for c in train_cohorts:
        avail &= set(COH[c][1].columns)
    genes = [g for g in genes_req if g in avail]
    n_genes = len(genes)

    base = dict(held_out_cohort=held_out, gene_set=gs_name,
                gene_set_family=GS_FAMILY.get(gs_name, ""),
                n_genes_requested=len(genes_req), n_genes_used=n_genes,
                genes_used="|".join(genes),
                n_test=len(surv_te), events_test=int(surv_te["event"].sum()),
                test_endpoint=surv_te["endpoint"].iloc[0],
                train_cohorts="|".join(train_cohorts),
                train_endpoints="|".join(sorted({COH[c][0]["endpoint"].iloc[0]
                                                 for c in train_cohorts})),
                track=tag)

    if n_genes == 0:
        for mdl in ["CoxPH_ridge", "Coxnet", "RSF", "GBSA"]:
            rows.append({**base, "model": mdl, "n_train": 0, "events_train": 0,
                         "cindex": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
                         "uno_c": np.nan,
                         "error": "no gene of this set present in both the training pool and the held-out cohort"})
        return rows, score_rows

    Xtr = pd.concat([COH[c][1][genes] for c in train_cohorts], axis=0)
    Str = pd.concat([COH[c][0] for c in train_cohorts], axis=0)
    Xtr = Xtr.loc[Str.index].astype(np.float64).fillna(0.0)
    ytr = y_of(Str)
    Xte = expr_te[genes].astype(np.float64).fillna(0.0)
    base.update(n_train=len(Str), events_train=int(Str["event"].sum()))

    ev_te = surv_te["event"].astype(float).astype(bool).values
    tt_te = surv_te["time_months"].astype(float).values
    A = comparable_matrix(tt_te, ev_te)
    coh_lab = Str["cohort"].values
    Xtr_v, Xte_v = Xtr.values, Xte.values

    for mdl, (grid, nsplit) in make_models(n_genes).items():
        try:
            folds = cohort_folds(coh_lab, nsplit)
            best_lab, best_sc, cv_note = grid[0][0], np.nan, ""
            if len(grid) > 1 and folds:
                bs = -np.inf
                for lab, fac in grid:
                    sc = []
                    for tr_i, te_i in folds:
                        try:
                            est = fac().fit(Xtr_v[tr_i], ytr[tr_i])
                            sc.append(concordance_index_censored(
                                ytr["event"][te_i], ytr["time"][te_i],
                                risk(est, Xtr_v[te_i]))[0])
                        except Exception:
                            sc.append(np.nan)
                    ms = np.nanmean(sc) if not np.all(np.isnan(sc)) else -np.inf
                    if ms > bs:
                        bs, best_lab = ms, lab
                best_sc = bs
                cv_note = f"cohort_grouped_cv n_folds={len(folds)} inner_c={bs:.4f}"
            est = dict(grid)[best_lab]().fit(Xtr_v, ytr)
            s_te = risk(est, Xte_v)
            if np.unique(s_te).size < 2:
                raise ValueError("degenerate (constant) risk score on held-out cohort")

            c_h = float(concordance_index_censored(ev_te, tt_te, s_te)[0])
            N = conc_matrix(A, s_te)
            den = float(A.sum())
            c_fast = float(N.sum() / den) if den > 0 else np.nan
            lo, hi, nb = boot_ci_fast(A, N)
            u, u_err = uno(ytr, surv_te, s_te)
            rows.append({**base, "model": mdl, "best_param": best_lab,
                         "cindex": c_h, "ci_lo": lo, "ci_hi": hi, "uno_c": u,
                         "n_boot_ok": nb, "n_folds_inner": len(folds),
                         "inner_cv": cv_note,
                         "c_fast_minus_sksurv": c_fast - c_h,
                         "error": u_err})
            if gs_name == "Novel5":
                for smp, tm, evn, sv in zip(surv_te.index, tt_te, ev_te, s_te):
                    score_rows.append(dict(sample=smp, cohort=held_out, model=mdl,
                                           gene_set=gs_name, track=tag,
                                           time_months=float(tm), event=int(evn),
                                           risk_score=float(sv)))
        except Exception as e:
            rows.append({**base, "model": mdl, "best_param": "",
                         "cindex": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
                         "uno_c": np.nan, "n_boot_ok": 0,
                         "error": f"{type(e).__name__}: {e} | {traceback.format_exc(limit=1)}"})

    with open(os.path.join(PARTS, f"{tag}__{held_out}__{gs_name}.json"), "w") as fh:
        json.dump({"rows": rows, "scores": score_rows}, fh)
    return rows, score_rows


TASKS = []
for h in OS_COHORTS:
    for gs in GENE_SETS:
        TASKS.append((h, gs, [c for c in OS_COHORTS if c != h], "loco_os"))
for h in SEC_COHORTS:
    for gs in GENE_SETS:
        TASKS.append((h, gs, [c for c in SEC_COHORTS if c != h], "loco_secondary"))
for h in SEC_COHORTS:
    for gs in GENE_SETS:
        TASKS.append((h, gs, list(OS_COHORTS), "transfer"))
print(f"n_tasks={len(TASKS)}", flush=True)

res = Parallel(n_jobs=54, verbose=10, backend="loky")(
    delayed(run_cell)(h, gs, tc, tag) for (h, gs, tc, tag) in TASKS)

rows_all, score_all = [], []
for rows, sc in res:
    rows_all.extend(rows)
    score_all.extend(sc)

df = pd.DataFrame(rows_all)
COLS = ["held_out_cohort", "gene_set", "gene_set_family", "model", "n_test",
        "events_test", "n_genes_requested", "n_genes_used", "cindex", "ci_lo",
        "ci_hi", "uno_c", "n_train", "events_train", "best_param",
        "n_folds_inner", "inner_cv", "n_boot_ok", "c_fast_minus_sksurv",
        "test_endpoint", "train_cohorts", "train_endpoints", "genes_used", "error"]
df = df.reindex(columns=[c for c in COLS if c in df.columns]
                + [c for c in df.columns if c not in COLS])

for tag, fn in [("loco_os", "loco_os.csv"), ("loco_secondary", "loco_secondary.csv"),
                ("transfer", "cross_endpoint_transfer.csv")]:
    df[df.track == tag].drop(columns=["track"]).to_csv(fn, index=False)