import logging
class Param:
    def __init__(self):
        self._para = {}
        pass
    @property
    def parameter(self):
        return self._para
    @parameter.setter
    def parameter(self,value):
        assert type(value) == dict
        if not value.keys():
            logging.warning("Empty parameters passed in")
        if value.keys() & self._para.keys() and not self._para.keys():
            logging.warning("New parameters would overide")
        try:
            self._para.update(value)
        except:
            logging.error("Error when updating params")
    def __getitem__(self, item:str):
        if item in self._para.keys():
            return self._para[item]
        else:
            logging.error("Parameter not registered")

    def __str__(self):
        rep_str = "Parameters in the model"
        for key,item in self._para:
            rep_str += "{}:{}\n".format(key,item)
        return rep_str

