import os

is_camber = False

def get_file(filename, dir):
    path = os.path.join(dir, filename)
    if is_camber:
        path = get_file_from_camber(path)
        return path
    
    return os.path.join("../pascal", path) 

def get_file_from_camber(path):
    return os.path.join()
