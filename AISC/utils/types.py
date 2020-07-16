# Copyright 2020-present, Mayo Clinic Department of Neurology - Laboratory of Bioelectronics Neurophysiology and Engineering
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


def DicToObj(d):
    """
    Converts dictionary into object where, keys - converts keys into attributes
    """
    top = type('new', (object,), d)
    seqs = tuple, list, set, frozenset
    for i, j in d.items():
        if isinstance(j, dict):
            setattr(top, i, DicToObj(j))
        elif isinstance(j, seqs):
            setattr(top, i,
                    type(j)(DicToObj(sj) if isinstance(sj, dict) else sj for sj in j))
        else:
            setattr(top, i, j)
    return top


class ObjDict(dict):
    """
    Dictionary which you can access a) as a dict; b) as a struct with attributes. Can use both foo adding and deleting
    attributes resp items. Inherits from dict
    """
    def __init__(self, VT_={}):
        super().__init__(VT_)
        for key in VT_.keys():
            self.__setattr__(key, VT_[key])

    def __setitem__(self, key, value):
        if key in self:
            del self[key]

        super().__setitem__(key, value)
        if not key in self.__dir__():
            self.__setattr__(key, value)

    def __setattr__(self, key, value):
        if key in self.__dir__():
            self.__delattr__(key)

        super().__setattr__(key, value)
        if not key in self:
            self.__setitem__(key, value)

    def __delitem__(self, key):
        value = super().pop(key)
        super().pop(value, None)
        if key in dir(self):
            self.__delattr__(key)

    def __delattr__(self, key):
        super().__delattr__(key)
        if key in self:
            self.__delitem__(key)

    def __repr__(self):
        return f"{type(self).__name__}({super().__repr__()})"

    def __missing__(self, key):
        self[key] = ObjDict()
        return self[key]

    def __getattr__(self, item):
        if not item in self.__dir__():
            self.__missing__(item)
        return super().__getattribute__(item)

















