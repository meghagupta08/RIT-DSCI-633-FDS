from openkg.entity_extraction.entity_extraction_factory import EntityExtractionFactory
import tarfile
import wget

def test_flair():
    config = {'model': 'ner'}
    entity_factory = EntityExtractionFactory()
    model = entity_factory.get_entity_extraction_model('flair', config)
    text = 'Apple is looking to buy USA startup for $1 billion'
    print('Flair prediction for the text: {}'.format(text))
    entities = model.get_entities(text)
    for x in entities:
        print(x)


def test_spacy():
    config = {'model': 'en_core_web_trf'}
    entity_factory = EntityExtractionFactory()
    model = entity_factory.get_entity_extraction_model('spacy', config)
    text = 'Apple is looking to buy USA startup for $1 billion'
    print('Spacy prediction for the text: {}'.format(text))
    entities = model.get_entities(text)
    for x in entities:
        print(x)


def test_heuristics():
    config = {}
    entity_factory = EntityExtractionFactory()
    model = entity_factory.get_entity_extraction_model('heuristics', config)
    text = 'Example email abcd@gmail.com. Example website www.google.com. A sample file name xy_2.exe'
    print('Heuristics prediction for the text: {}'.format(text))
    entities = model.get_entities(text)
    for x in entities:
        print(x)


# def test_set_expander_get_entities():
#     '''
#     This method test the get_entities method of the set_expander
#     '''
#
#     # If model file already exists skip downloading
#     if not os.path.isfile('enwiki-20171201_pretrained_set_expansion.txt'):
#         url = 'https://d2zs9tzlek599f.cloudfront.net/models/term_set/enwiki-20171201_pretrained_set_expansion.txt.tar.gz'
#         wget.download(url, 'enwiki-20171201_pretrained_set_expansion.txt.tar.gz')
#         tarf = tarfile.open('enwiki-20171201_pretrained_set_expansion.txt.tar.gz')
#         tarf.extractall()
#     config = {'model': 'enwiki-20171201_pretrained_set_expansion.txt',
#               'topn': 10,
#               'minimum_similarity_score' :0.83,
#               'num_iteration': 5
#               }
#
#     entity_factory = EntityExtractionFactory()
#     model = entity_factory.get_entity_extraction_model('set_expander', config)
#
#     seed_terms = ['apple', 'orange', 'bananas']
#     entities = model.get_entities(seed_terms)
#     # printing expanded terms
#     for x in entities:
#         print(x)
#
#     # removing the downloaded tar file if exists
#     if os.path.isfile('enwiki-20171201_pretrained_set_expansion.txt.tar.gz'):
#         os.remove('enwiki-20171201_pretrained_set_expansion.txt.tar.gz')
#
#
# def test_set_expander_train():
#     '''
#     This method test the train method of the set_expander
#     '''
#
#     if not os.path.isfile('enwiki-20171201_spacy_marked.txt'):
#         url = 'https://d2zs9tzlek599f.cloudfront.net/models/term_set/enwiki-20171201_spacy_marked.txt.tar.gz'
#         wget.download(url, 'enwiki-20171201_spacy_marked.txt.tar.gz')
#         tarf = tarfile.open('enwiki-20171201_spacy_marked.txt.tar.gz')
#         tarf.extractall()
#
#     config = {'size': 100,
#               'min_count': 10,
#               'window': 10,
#               'hs': 0,
#               'corpus': 'enwiki-20171201_spacy_marked.txt',
#               'np2vec_model_file': 'trained.model',
#               'corpus_format': 'txt',
#               'iter': 1
#               }
#
#     entity_factory = EntityExtractionFactory()
#     model = entity_factory.get_entity_extraction_model('set_expander', config)
#
#     model.train()
#
#     # removing the downloaded tar file if exists
#     if os.path.isfile('enwiki-20171201_spacy_marked.txt.tar.gz'):
#         os.remove('enwiki-20171201_spacy_marked.txt.tar.gz')

def test_ner():
    test_flair()
    test_spacy()
    test_heuristics()
    # test_set_expander_get_entities()
    # test_set_expander_train()


if __name__ == '__main__':
    test_ner()
