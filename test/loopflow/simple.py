from utensil.loopflow.loopflow import NodeProcessFunction


class Constant(NodeProcessFunction):

    def __init__(self, value):
        super().__init__()
        self.value = value

    def main(self):
        return self.value


class AddValue(NodeProcessFunction):

    def __init__(self, value):
        super().__init__()
        self.value = value

    def main(self, a):
        return a + self.value


class Add(NodeProcessFunction):

    def main(self, a, b):
        return a + b


class TimeValue(NodeProcessFunction):

    def __init__(self, value):
        super().__init__()
        self.value = value

    def main(self, a):
        return a * self.value


class ListAddSum(NodeProcessFunction):

    def main(self, add, *args):
        return sum([a + add for a in args])


class Sum(NodeProcessFunction):

    def main(self, l):
        return sum(l)


class Divide(NodeProcessFunction):

    def main(self, a, b):
        return a / b


class Pickle(NodeProcessFunction):

    def __init__(self, path):
        super().__init__()
        self.path = path

    def main(self, obj):
        import pickle
        with open(self.path, "wb") as f:
            pickle.dump(obj, f)
