from pycorenlp import StanfordCoreNLP
from collections import defaultdict
# Provides functions to return the pos and dep dicts and length for a sentence.

# No longer used, as we have all the data stored.
def get_pos_dep_raw(sentence):
    nlp = StanfordCoreNLP('http://localhost:9000')
    try:
        output = nlp.annotate(sentence, properties={
          'annotators': 'tokenize,ssplit,pos,depparse',
          'outputFormat': 'json'
          })
    except UnicodeDecodeError:
        #sentence = unidecode(sentence)
        print('Unicode Fail')
        output = nlp.annotate(sentence, properties={
            'annotators': 'tokenize,ssplit,pos,depparse',
            'outputFormat': 'json'
        })

    tokens = output['sentences'][0]['tokens']
    dependencies = output['sentences'][0]['basicDependencies']

    return get_pos_dep(tokens, dependencies)


def get_pos_dep(tokens, dependencies):
    S = defaultdict(dict)

    for t in tokens:
        i = t['index'] - 1  # Shift the index by 1
        word = str(t['word']).lower()
        S[i]['word'] = word
        S[i]['pos'] = t['pos']
        S[i]['ner'] = t['ner']
        S[i]['deps'] = {}

    for dep in dependencies:
        g = dep['governor'] - 1
        if g < 0:  # Don't include the ROOT as a governor, for now.
            continue
        d = dep['dependent'] - 1
        S[g]['deps'][d] = dep['dep']

    return S


# Testing
#S = "Whenever I go home, I'm happy to see you"
#d = get_pos_dep(S)
#print d