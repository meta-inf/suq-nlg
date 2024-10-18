import re
from typing import Dict, Any


def flatten_dict(d: Dict[str, Any], parent_key='', sep='_') -> Dict[str, Any]:
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def parse_float_between_0_and_1(s):
    # Regular expression pattern to match integers and floats (including optional signs)
    pattern = r'[+-]?\d*\.?\d+'
    # Find all matches in the string
    matches = re.finditer(pattern, s)
    for match in matches:
        num_str = match.group()
        try:
            num = float(num_str)
            # Check if the number is between 0 and 1 inclusive
            if 0 <= num <= 1:
                return num
        except ValueError:
            continue  # Skip if conversion to float fails
    return None  # Return None if no number is found
