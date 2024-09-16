from dataclasses import dataclass
from pathlib import Path


@dataclass(kw_only=True, frozen=True)
class ConfigDict:
    def get(self, key: str, default=None):
        return getattr(self, key, default)

    def to_dict(self):
        """Converts the config to a dictionary.

        Helpful for saving to disk or logging.
        """
        d = {}
        for k, v in self.__dict__.items():
            if isinstance(v, ConfigDict) or hasattr(v, "to_dict"):
                d[k] = v.to_dict()
            elif isinstance(v, (tuple, list)):
                d[k] = tuple([x.to_dict() if isinstance(x, ConfigDict) or hasattr(v, "to_dict") else x for x in v])
            elif isinstance(v, Path):
                d[k] = v.as_posix()
            else:
                d[k] = v
        return d