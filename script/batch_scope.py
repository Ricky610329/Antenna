"""批次任務範圍：先過濾，再碰 claim、fail 或結果；舊 worker 只收無 scope 的工作。"""
import re


def validate_scope(scope):
    if scope is not None and not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}", scope):
        raise ValueError("scope 只接受 1–80 個英數、底線或連字號")
    return scope


def job_in_scope(job, scope):
    return job.get("scope") == validate_scope(scope)


def validate_worker_scope(args):
    scope = validate_scope(getattr(args, "scope", None))
    if scope and getattr(args, "selfgen", 0):
        raise ValueError("限定 scope 的 worker 必須明確使用 --selfgen 0")
    return scope
