"""The scope-profiler tag of opencr's profiling regions.

The regions are opt-in: :mod:`opencr` registers the tag on import, so they
record nothing unless the application enables it, e.g. with
``scope_profiler.setup(enable_tags=["opencr"])``.
"""

#: Tags of every opencr region; the first is the opt-in tag.
PROFILE_TAGS = ("opencr",)
