# Contributing

This project prioritizes claim traceability and leakage control.

1. Open an issue describing the research or implementation problem.
2. Do not commit raw IFND data, model weights, credentials, or personal data.
3. Add or update tests for changes to preprocessing, splitting, or metrics.
4. Run `python -m unittest discover -s tests -v`.
5. Run the synthetic smoke configuration before proposing a change.
6. Link every changed result or figure to a run ID and machine-readable artifact.

Bug reports should include the command, configuration, Python/platform details,
data checksum, traceback, and the smallest safe reproduction example.

