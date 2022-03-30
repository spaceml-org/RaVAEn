from pathlib import Path


class FilterBase:
    def __init__(self,
                 root_folder: str,
                 folder_path: str,
                 extension: str,
                 **kwargs):

        self.root_folder = root_folder
        self.folder_path = folder_path
        self.extension = extension

    def __call__(self):
        raise NotImplementedError


class NoFilter(FilterBase):
    def __call__(self):
        folder_path = Path(self.root_folder) / self.folder_path
        assert folder_path.exists(), f'Folder {folder_path} does not exist'

        files = sorted(list(folder_path.glob(self.extension)))
        return files


class SliceFilter(NoFilter):
    def __init__(self,
                 root_folder: str,
                 folder_path: str,
                 extension: str,
                 f_slice: str,
                 **kwargs):
        super().__init__(root_folder, folder_path, extension, **kwargs)

        self.f_slice = f_slice

    def __call__(self):
        files = super().__call__()
        return eval(f'files[{self.f_slice}]')
