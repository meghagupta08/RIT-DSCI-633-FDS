class Entity:
    def __init__(self, start, end, text, entity_type):
        """
        :param start: start position of the entity in the source text
        :param end: end position of the entity in the source text
        :param text: entity text
        :param entity_type: type of the entity
        """
        self.start = start
        self.end = end
        self.text = text
        self.entity_type = entity_type

    def __str__(self, ):
        return 'text: {}, label: {}, start: {}, end: {}'.format(self.text, self.entity_type, self.start, self.end)
