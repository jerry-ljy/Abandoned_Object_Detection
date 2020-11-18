class Person:
    def __init__(self, _location, _confs):
        self.location=_location
        self.confs=_confs
        self.belongings=list()

    def add_object(self, _object):
        self.belongings.append(_object)

class Object:
    def __init__(self, _location, _confs, _label):
        self.location=_location
        self.confs = _confs
        self.label=_label
