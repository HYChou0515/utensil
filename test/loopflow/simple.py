from utensil.loopflow.loopflow import NodeTask


class Constant(NodeTask):

    def __init__(self, value):
        super().__init__()
        self.value = value

    def main(self):
        return self.value


class AddValue(NodeTask):

    def __init__(self, value):
        super().__init__()
        self.value = value

    def main(self, a):
        return a + self.value


class Add(NodeTask):

    def main(self, a, b):
        return a + b


class TimeValue(NodeTask):

    def __init__(self, value):
        super().__init__()
        self.value = value

    def main(self, a):
        return a * self.value


class ListAddSum(NodeTask):

    def main(self, add, *args):
        return sum([a + add for a in args])


class Sum(NodeTask):

    def main(self, l):
        return sum(l)


class Divide(NodeTask):

    def main(self, a, b):
        return a / b


class Pickle(NodeTask):

    def __init__(self, path):
        super().__init__()
        self.path = path

    def main(self, obj):
        import pickle
        with open(self.path, "wb") as f:
            pickle.dump(obj, f)
