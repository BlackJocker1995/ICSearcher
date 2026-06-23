"""ICSearcher pipeline entry points.

Each stage is a module with a ``main()`` function:
    collect, convert, train, fuzz, validate, range

Run via the console scripts (preferred, no ``python`` needed):
    uv run icsearcher-collect
    uv run icsearcher-train extract
    uv run icsearcher-validate validate --device 1

Or via the module dispatcher:
    uv run python -m pipelines collect
    uv run python -m pipelines train extract
"""
