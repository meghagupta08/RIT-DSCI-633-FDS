from openkg.entity_extraction.entity_extraction import EntityExtraction
from openkg.entity_extraction.flair_ner import Flair
from openkg.entity_extraction.spacy_ner import Spacy
from openkg.entity_extraction.heuristics_ner import HeuristicsNER
from openkg.entity_extraction.set_expander import SetExpander


class EntityExtractionFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_entity_extraction_model(name: str, config: dict) -> EntityExtraction:
        name = name.lower()
        if name == 'flair':
            return Flair(config)
        elif name == 'spacy':
            return Spacy(config)
        elif name == 'heuristics':
            return HeuristicsNER(config)
        elif name == 'set_expander':
            return SetExpander(config)
        else:
            raise ValueError('Unknown entity extraction model')
