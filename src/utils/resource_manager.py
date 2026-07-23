import json
from pathlib import Path


class ResourceManager:

    _instance = None

    def __new__(cls):

        if cls._instance is None:

            cls._instance = super().__new__(cls)

            cls._instance._load()

        return cls._instance

    ############################################################

    def _load(self):

        root = Path(__file__).resolve().parents[2]

        self.resource_dir = root / "resources"

        self.resources = {}

        self.categories = {}

        self._load_files()

    ############################################################

    def _load_files(self):

        for file in self.resource_dir.glob("*.json"):

            name = file.stem

            with open(file, encoding="utf8") as f:

                data = json.load(f)

            self.resources[name] = data

            self.categories[name] = self._flatten(data)

    ############################################################

    def _flatten(self, obj):

        words = []

        if isinstance(obj, dict):

            for value in obj.values():

                words.extend(self._flatten(value))

        elif isinstance(obj, list):

            for item in obj:

                words.extend(self._flatten(item))

        elif isinstance(obj, str):

            words.append(obj.lower())

        return words

    ############################################################

    def get(self, name):

        return self.resources.get(name)

    ############################################################

    def words(self, name):

        return self.categories.get(name, [])

    ############################################################

    def has(self, name):

        return name in self.resources

    ############################################################

    def all_categories(self):

        return list(self.resources.keys())

    ############################################################

    def all_words(self):

        result = []

        for words in self.categories.values():

            result.extend(words)

        return list(set(result))

    ############################################################

    def reload(self):

        self.resources.clear()

        self.categories.clear()

        self._load_files()
