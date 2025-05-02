from dataclasses import dataclass

@dataclass
class PrefixedName(str):
    prefix: str
    name: str

    def __hash__(self):
        return hash((self.prefix, self.name))