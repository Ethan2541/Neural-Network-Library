class DimensionMismatchError(Exception):
    def __init__(self, expected, got):
        self.expected = expected
        self.got = got

    def __str__(self):
        return f"Expected dimensions {self.expected}, got {self.got}."