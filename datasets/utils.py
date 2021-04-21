import os


def name_cond(name_filter: str):
    if name_filter is None:
        return lambda filename: True
    else:
        return lambda filename: name_filter in filename


def ext_cond(ext_filter: str):
    if ext_filter is None:
        return lambda filename: True
    else:
        return lambda filename: filename.endswith(ext_filter)


def get_files(dir_path, name_filter=None, extension_filter=None):
    if not os.path.isdir(dir_path):
        raise RuntimeError("\"{0}\" is not a directory.".format(dir_path))
    filtered_files = []
    for path, _, files in os.walk(dir_path):
        files.sort()
        for f in files:
            if name_cond(f) and ext_cond(f):
                full_path = os.path.join(path, f)
                filtered_files.append(full_path)
    return filtered_files