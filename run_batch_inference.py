#!/usr/bin/env python3
"""Compatibility entrypoint for the packaged batch runner."""

from characonsist.runners.batch import *  # noqa: F401,F403


if __name__ == "__main__":
    raise SystemExit(main())
