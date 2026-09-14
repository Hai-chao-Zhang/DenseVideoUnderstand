"""Empty task package required by lmms-eval's external-plugin discovery.

The campaign reuses the repository's signed ``densevideo`` task definition;
this package exists solely so ``importlib.util.find_spec(<plugin>.tasks)`` is
well-defined during the real CLI startup path.
"""
