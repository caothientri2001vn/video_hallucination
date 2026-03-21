import os

class PromptTemplate:
    def __init__(self, path: str, name: str):
        self.path = path
        self.name = name
        with open(self.path) as fin:
            content = fin.read()
        self.content = content

PromptTemplateCollections = {
    "extract_fact": PromptTemplate("prompt_asset/extract_fact.txt", "extract_fact")
}
        