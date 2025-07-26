"""Print a table row by row"""


class Table:
    def __init__(self, headers: list[str], formats: list[str]):
        self.headers = headers
        self.formats = ["{:" + f + "}" for f in formats]
        self.row = 0
        self.widths = [len(header) for header in self.headers]

    def print(self, *data):
        text = [self.formats[i].format(data[i]) for i in range(len(data))]
        if self.row == 0:
            for i, s in enumerate(text):
                self.widths[i] = max(self.widths[i], len(s))
                self.headers[i] = self.headers[i].center(self.widths[i])
            print(" | ".join(self.headers))

        text = [t.rjust(self.widths[i]) for i, t in enumerate(text)]
        print(" | ".join(text))
        self.row += 1


if __name__ == "__main__":
    T = Table(["P1", "P2", "Score"], ["2d", "2d", "4.2f"])

    T.print(1, 2, 0.5)
    T.print(1, 3, 0.25)
    T.print(2, 3, 1.0)
