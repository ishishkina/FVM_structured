def is_integer(line):
    try:
        int(line)
    except ValueError:
        return False
    else:
        return True

def is_float(line):
    try:
        float(line)
    except ValueError:
        return False
    else:
        return True

class Initializer:
    def __init__(self, filename):
        self.__storage = {}

        with open(filename, 'r') as initf:
            for line in initf:
                try:
                    key, value = line.strip().split('=', 1)
                except ValueError:
                    continue
                if not key.isidentifier():
                    raise ValueError(f"{key} is invalid name!")
                if is_integer(value):
                    value = int(value)
                elif is_float(value):
                    value = float(value)
                self.__storage[key] = value

        self.__check()

    def __check(self):
        if not 'mesh' in self.__storage:
            raise AttributeError('Mesh name has not been found')
        if not isinstance(self.__storage['mesh'], str):
            raise TypeError('Expected string, but has not been found!')

        if not 'iterations' in self.__storage:
            raise AttributeError('Number of iteration at Green-Gauss method has not been found')
        if not isinstance(self.__storage['iterations'], int):
            raise TypeError('Expected integer value, but it has not been found!')

        if not 'mod' in self.__storage:
            raise AttributeError('Scheme for convective terms has not been found')
        if not isinstance(self.__storage['mod'], int):
            raise TypeError('Expected integer value, but it has not been found!')

        if 'data' in self.__storage and not isinstance(self.__storage['data'], str):
            raise TypeError('Expected string, but has not been found!')

    @property
    def mesh(self):
        return self.__storage['mesh']

    @property
    def iterations(self):
        return self.__storage['iterations']

    @property
    def mod(self):
        return self.__storage['mod']

    @property
    def data(self):
        if 'data' in self.__storage:
            return self.__storage['data']
        return ''
