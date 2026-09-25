"""Pluggable ID extraction from prediction-record paths.

Different projects encode a case/exam ID differently in filenames — this is
a factory for the common "regex, else filename stem" pattern instead of
hardcoding one project's ID format into the analysis code.
"""

import re
from pathlib import Path
from typing import Callable


def regex_id_extractor(pattern: str) -> Callable[[str], str]:
    """Build an extractor: first regex match in the path, else the file stem.

    Example: ``regex_id_extractor(r"(EX\\d{4})")``.
    """
    compiled = re.compile(pattern)

    def extract(path: str) -> str:
        match = compiled.search(path)
        if match:
            return match.group(1)
        return Path(path).stem

    return extract
