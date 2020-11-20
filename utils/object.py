class Person:
    def __init__(self):
        self.location=list()
        self.belongings=list()
        self.id=-1
        self.is_deleted = False

    def add_object(self, _object):
        self.belongings.append(_object)

class Object:
    def __init__(self, _label):
        self.location=list()
        self.label=_label
        self.id=-1
        self.is_abandoned = False
