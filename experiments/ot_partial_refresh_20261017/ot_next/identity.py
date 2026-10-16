"""Permit only order changes in Diffusers' set-derived default-key metadata."""


def compatible_scheduler_metadata(old, current):
    if set(old) != set(current):
        return False
    for key in old:
        if key == '_use_default_values':
            if sorted(old[key]) != sorted(current[key]):
                return False
        elif old[key] != current[key]:
            return False
    return True
