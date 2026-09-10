# Contributing

Use [GitHub issues](https://github.com/Jonsnow-willow/GPUMD-Wizard/issues) for bug reports, questions, and feature requests. For a reproducible report, include the GPUMD-Wizard revision, Python and dependency versions, a small input or script, the expected result, and the actual output or traceback. For a material-property question, also specify the calculator, units, reference state, boundary conditions, and relaxation settings.

Keep pull requests focused on one behavior or physical task. Explain the change and its validation, update affected examples, and add a small regression test when fixing a bug. Changes to physical quantities should state their units and assumptions explicitly. Avoid including large simulation outputs or private potential files.

From the repository root, in an isolated Python 3.10 or newer environment:

```bash
python -m pip install '.[test]'
python -m pytest -q
```

The public tests use small CPU examples and temporary directories. See [tests/README.md](tests/README.md) for their scope and independent reference calculations. Tutorials that require external solvers, particular potential files, or long simulations need separate manual validation; include the command, inputs, software versions, and result when such a workflow is affected.
