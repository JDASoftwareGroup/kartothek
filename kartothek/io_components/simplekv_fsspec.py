from fsspec.spec import AbstractFileSystem
import storefact


class SimplekvFsspecWrapper(AbstractFileSystem):
    def __init__(self, store):
        if type(store) is str:
            self.store = storefact.get_store_from_url(store)
        else:
            self.store = store

    def open(self, path, mode):
        return self.store.open(path)

    def isfile(self, path):
        return path in self.store

    def ls(self, path, detail=True):
        if path in self.store:
            return {"name": path, "size": None, "type": "file"}
    
    def info(self, path):
        return self.ls(path, detail=True)
