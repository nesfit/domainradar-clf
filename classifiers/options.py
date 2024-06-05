import os


class PipelineOptions:
    def __init__(self, base_dir: str | None = None, models_dir: str | None = None, boundaries_dir: str | None = None):
        self.base_dir = base_dir if base_dir is not None else os.path.dirname(__file__)
        self.models_dir = models_dir if models_dir is not None else os.path.join(self.base_dir, "models")
        self.boundaries_dir = boundaries_dir if boundaries_dir is not None else (
            os.path.join(self.base_dir, "boundaries"))
