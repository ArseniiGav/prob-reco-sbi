"""``python -m prob_reco_sbi.flash``: the same as the ``flashsim`` command."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
