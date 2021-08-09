import numpy as np
import json

def _find_insert_index(s: str, i: int) -> int:
    """
    returns the index after the first char from the left of `i`
    that is space or `[`

    This is where spaces can be inserted without disturbing the data.
    """
    while i >= 0 and s[i] not in (' ', '['):
        i -= 1
    return i + 1

def ndarray2str(a: np.ndarray) -> str:
    lines = json.dumps(a.tolist()).replace("], ", "],\n ").splitlines()
    decimal_counts = [s.count(".") for s in lines]
    if len(set(decimal_counts)) == 1:
        start = 0
        for _ in range(decimal_counts[0]):
            this_decimal = [s.find(".", start) for s in lines]
            target = max(this_decimal)
            for i in range(len(lines)):
                space_count = target - this_decimal[i]
                if space_count:
                    where = _find_insert_index(lines[i], this_decimal[i])
                    lines[i] = lines[i][:where] + (' ' * space_count) + lines[i][where:]
            start = target + 1
    return '\n'.join(lines)

def str2ndarray(s: str) -> np.ndarray:
    return np.array(json.loads(s))
