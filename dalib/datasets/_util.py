
def subset(s: dict, keys: list):
    return dict((key, value) for key, value in s.items() if key in keys)

