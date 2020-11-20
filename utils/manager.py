import numpy as np

class Manager:
    def __init__(self):
        self.person_list = list()
        self.object_list = list()
        self.new_object_list = list()

    def add_person(self, _person):
        self.person_list.append(_person)

    def get_person(self, _index):
        for person in self.person_list:
            if person.id == _index:
                return person
        return None

    def get_object(self, _index):
        for obj in self.object_list:
            if obj.id == _index:
                return obj

        for new_obj in self.new_object_list:
            if new_obj.id == _index:
                return new_obj
        return None

    def add_object(self, _obj):
        self.new_object_list.append(_obj)

    def update(self):
        ## delete person
        for person in self.person_list:
            if person.is_deleted:
                for obj in person.belongings:
                    self.object_list.remove(obj)
                self.person_list.remove(person)

        ## match objects to person
        if self.new_object_list:
            for _obj in self.new_object_list:
                if self.person_list:
                    distance = float("inf")
                    target = None
                    for person in self.person_list:
                        temp = calculate_distance(_obj, person)
                        if temp < distance:
                            distance = temp
                            target = person
                    if target:
                        target.add_object(_obj)
                    else:
                        _obj.is_abandoned = True
                self.object_list.append(_obj)
        self.new_object_list=list()


    def get_ab_objects(self):
        ab_objects = [obj for obj in self.object_list if obj.is_abandoned]
        return ab_objects


def calculate_distance(_object, _person):
    obj_cen = np.array([(_object.location[0] + _object.location[2])/2, (_object.location[1] + _object.location[3])/2])
    per_cen = np.array([(_person.location[0] + _person.location[2])/2, (_person.location[1] + _person.location[3])/2])
    distance = np.sqrt(np.sum(np.square(obj_cen - per_cen)))
    return distance