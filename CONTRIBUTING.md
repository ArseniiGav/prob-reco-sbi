# Contributing

Questions, bug reports and improvements are welcome, from the students of the course and from anyone else. This page
explains how to report a problem and how to propose a change.

## Asking a question or reporting a bug

Open an [issue](https://github.com/ArseniiGav/prob-reco-sbi/issues). Please search the existing issues first; a
question that is already answered there saves everybody time. A good bug report contains:

- what you ran: the exact command or a short piece of code that shows the problem;
- what happened: the complete error message, copied as text (not a screenshot);
- what you expected instead;
- your environment: the output of

  ```bash
  python -c "import platform, numpy, torch, h5py, prob_reco_sbi; print(platform.platform(), platform.python_version(), 'numpy', numpy.__version__, 'torch', torch.__version__, 'h5py', h5py.__version__, 'prob_reco_sbi', prob_reco_sbi.__version__, 'cuda', torch.cuda.is_available())"
  ```

  and, if a data file is involved, the output of `flashsim info FILE` (it shows how the file was made).

## Proposing a change

Changes go through pull requests from your own fork of the repository
([GitHub's guide](https://docs.github.com/en/get-started/exploring-projects-on-github/contributing-to-a-project)
explains every step).

1. Fork the repository on GitHub and clone your fork:

   ```bash
   git clone https://github.com/<your-user-name>/prob-reco-sbi.git
   cd prob-reco-sbi
   git remote add upstream https://github.com/ArseniiGav/prob-reco-sbi.git
   ```

2. Install the package with the development tools and the git hooks:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
   python -m pip install -e ".[analysis,dev]"
   pre-commit install
   ```

   The pip upgrade matters: the pip bundled with Python 3.10 cannot install the CPU build of PyTorch from the PyTorch
   index, and the following `pip install -e` would then silently fetch a CUDA build from PyPI.

3. Make a branch for your change, with a name that says what it does:

   ```bash
   git switch -c fix-units-in-data-format
   ```

4. Make the change, then run the checks that the continuous integration runs. `pre-commit` only sees the files that
   git knows, so add new files with `git add` first:

   ```bash
   pre-commit run --all-files   # ruff lint and format, whitespace, file checks, notebook outputs
   mypy                         # type checks of the package
   python -m pytest             # fast tests
   python -m pytest -m slow     # slow tests, if you changed the physics (most need course data and are skipped)
   ```

5. Commit with a short message in the imperative ("Fix the units of time_first in the data format"), push the branch to
   your fork with `git push -u origin fix-units-in-data-format` and open a pull request against `main` on GitHub.
   Describe what you changed and why, and link the issue it addresses. The continuous integration runs ruff, mypy and
   the pre-commit hooks, and the tests on Python 3.10 to 3.14; a pull request is merged when they pass and the change
   has been reviewed.

To bring your fork up to date with the main repository:

```bash
git switch main
git pull upstream main
git push origin main
```

## Style

- **Code:** formatted and linted by [ruff](https://github.com/astral-sh/ruff) (line length 120; the rules are in
  `pyproject.toml`). `pre-commit` runs it before every commit and fixes most problems itself.
- **Public functions** have type hints and a docstring in the
  [NumPy style](https://numpydoc.readthedocs.io/en/latest/format.html), with units for physical quantities.
- **Tests:** every new feature or bug fix comes with a test in `tests/`. Statistical tests compare with an expectation
  within a few standard deviations (see `tests/helpers.py`) and use a fixed seed. Tests that take more than a few
  seconds get the marker `@pytest.mark.slow`.
- **Notebooks** are committed without outputs; the `nbstripout` hook removes them.
- **Data** never go into git: `data/` and `*.h5` are ignored, and files larger than 3000 kB are refused by a hook.
  Describe how to generate a data set (the command and its seed) instead.
- **Changelog:** a change that users notice gets a line under "Unreleased" in [CHANGELOG.md](CHANGELOG.md).
- Code, comments and documentation are written in English.

## Your solutions to the tasks

Keep your answers to [TASKS.md](TASKS.md) on a branch of your fork or in a folder outside the repository, and bring
them to the session; they do not go into pull requests to this repository. Improvements to the tasks themselves, such
as an unclear question or a broken command, are welcome as issues.
