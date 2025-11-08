# -*- coding: utf-8 -*-
from __future__ import absolute_import

# Minimal package initializer for jobs so "backend.jobs.tasks" is accessible to RQ.
try:
    # Expose the tasks module as an attribute of the package so RQ can import
    # backend.jobs.tasks.<function> (RQ uses getattr on the parent package).
    from . import tasks  # noqa: F401
except Exception:
    # If tasks cannot be imported here (e.g. missing heavy deps), keep package importable.
    tasks = None

__all__ = ["tasks", "worker"]