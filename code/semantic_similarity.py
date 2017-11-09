from collections import defaultdict
import json
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
stemmer = PorterStemmer()

#S = 'Where is the puppy running to?'
#T = 'Where is the dog running to?'
# EQ = {} # This will be loaded as a json file, if we use it.
#STOP = set(stopwords.words('english')) # I don't think we should remove these.

# These should only load once.
wn_n = json.load(open('WordNet/wn_synonyms_n.json'))
wn_a = json.load(open('WordNet/wn_synonyms_a.json'))
wn_v = json.load(open('WordNet/wn_synonyms_v.json'))
wn_dicts = {'n': wn_n, 'a': wn_a, 'v': wn_v }
ppdb_dict = json.load(open('PPDB/ppdb_xl_paraphrases.json'))

def align(S,T):
    A = set()
    A_E = {'i': defaultdict(set), 'j': defaultdict(set)} # Already aligned indices
    word_sim = create_word_sim(S,T)
    A = A.union(cwDepAlign(S,T,A_E,word_sim))
    #info_A = [(S[i]['word'],T[j]['word']) for (i,j) in A]
    A = A.union(cwTextAlign(S,T,A_E,word_sim))
    #info_A = [(S[i]['word'],T[j]['word']) for (i,j) in A]
    return A

def cwDepAlign(S,T,A_E,word_sim):
    aligned_pairs_scores = set()
    pairs_context = dict()

    for i in range(len(S)):
        for j in range(len(T)):
            # Only when i and j have not yet been aligned
            if not A_E['i'][i] and not A_E['j'][j]:
                if word_sim[(i,j)] > 0:
                    context = depContext(S,T,i,j,word_sim)
                    contextSim = sum([word_sim[(k,l)] for (k,l) in context])
                    if contextSim > 0:
                        weighted_score = 0.75 * word_sim[(i,j)] + 0.25 * contextSim
                        aligned_pairs_scores.add(((i,j),weighted_score))
                        pairs_context[(i,j)] = context

    a = set()
    a_E = {'i': defaultdict(set), 'j': defaultdict(set)}  # Already aligned indices for a
    # Sort by decreasing score
    aligned_pairs_scores = sorted(aligned_pairs_scores,key=lambda x: x[1],reverse=True)
    # Populate aligned pairs
    for (i,j),score in aligned_pairs_scores:
        if not a_E['i'][i] and not a_E['j'][j]:
            a.add((i,j))
            a_E['i'][i].add(j)
            a_E['j'][j].add(i)
        for (k,l) in pairs_context[(i,j)]:
            # Check that neither k or l is matched in A or a
            if not a_E['i'][k] and not a_E['j'][l] and not A_E['i'][k] and not A_E['j'][l]:
                a.add((k,l))
                a_E['i'][i].add(l)
                a_E['j'][j].add(k)
    return a

def cwTextAlign(S,T,A_E,word_sim):
    aligned_pairs_scores = set()
    for i in range(len(S)):
        for j in range(len(T)):
            # Only when i and j have not yet been aligned
            if not A_E['i'][i] and not A_E['j'][j]:
                if word_sim[(i, j)] > 0:
                    context = textContext(S,T,i,j)
                    contextSim = sum([word_sim[(k,l)] for (k,l) in context])
                    if contextSim > 0: # We require there to be some contextual similarity.
                        weighted_score = 0.75 * word_sim[(i,j)] + 0.25 * contextSim
                        aligned_pairs_scores.add(((i,j), weighted_score))

    a = set()
    a_E = {'i': defaultdict(set), 'j': defaultdict(set)}  # Already aligned indices for a
    # Sort by decreasing score
    aligned_pairs_scores = sorted(aligned_pairs_scores, key=lambda x: x[1], reverse=True)
    # Populate aligned pairs
    for (i,j), score in aligned_pairs_scores:
        if not a_E['i'][i] and not a_E['j'][j]:
            a.add((i, j))
            a_E['i'][i].add(j)
            a_E['j'][j].add(i)
    return a

# Given two sentences S and T, s in S and t in T are a candidate aligned pair if
# (s,t) in R_sim
# (r_s in R and r_t in T) in R_sim

# For any pair, (a,b) in R_sim if...
# 1) a,b are identical
# 2) a,b are synonyms in WordNet
# 3) a,b are paraphrases in PPDB

def depContext(S,T,i,j,word_sim):
    context = set()
    for k in range(len(S)):
        for l in range(len(T)):
            if i != k and j != l:
                if word_sim[(k,l)] > 0:
                    S_dep_forward = S[i]['deps'].get(k,0)
                    S_dep_backward = S[k]['deps'].get(i,0)
                    T_dep_forward = T[j]['deps'].get(l,0)
                    T_dep_backward = T[l]['deps'].get(j,0)
                    # Only proceed if dependencies exist. Orientations may be used later.
                    if (S_dep_forward and T_dep_forward) or (S_dep_backward and T_dep_backward):
                        orientation = 'a'
                    elif (S_dep_forward and T_dep_backward) or (S_dep_backward and T_dep_forward):
                        orientation = 'c'
                    else:
                        continue
                    if S[i]['pos'] == T[j]['pos'] and S[k]['pos'] == T[l]['pos']:
                        S_dep = S_dep_forward if S_dep_forward else S_dep_backward
                        T_dep = T_dep_forward if T_dep_forward else T_dep_backward
                        if S_dep == T_dep: # Additional case of EQ goes here
                            context.add((k,l))
    return context


def textContext(S,T,i,j):
    left_i = i-3 if i-3 >= 0 else 0
    right_i = i+3 if i+3 < len(S) else len(S) - 1
    left_j = j-3 if j-3 >= 0 else 0
    right_j = j+3 if j+3 < len(T) else len(T) - 1
    C_i = [k for k in range(left_i,right_i + 1) if k != i]
    C_j = [l for l in range(left_j, right_j + 1) if l != j]
    return [(k,l) for l in C_j for k in C_i] # Cross product

# Index pairs that are similar. i,j are indices. S,T are tokenized sentence dictionaries.
def create_word_sim(S,T):
    return {(i,j): similarity(S,T,i,j) for j in range(len(T)) for i in range(len(S))}

def dotproduct_word_sim(S,T):
    return [similarity(S,T,i,i) for i in range(len(T))]

# Words can pass the similarity check in 3 ways.
def similarity(S,T,i,j):
    # Returns a value based on the type of match. Arbitrary for now.
    if identical_words(S,T,i,j):
        return 1
    elif wn_synonyms(S,T,i,j):
        return 0.9
    elif ppdb_paraphrases(S,T,i,j):
        return 0.7
    else:
        return 0

def identical_words(S,T,i,j):
    return S[i]['word'] == T[j]['word']

def wn_synonyms(S,T,i,j):
    # Load based on pos tag
    w_i = S[i]['word']
    w_j = T[j]['word']
    pos_i = infer_pos(S[i]['pos'])
    pos_j = infer_pos(T[j]['pos'])
    if pos_i and pos_j and (pos_i == pos_j):
        wn_dict = wn_dicts[pos_i]
        return w_j in wn_dict.get(w_i,[]) or w_i in wn_dict.get(w_j,[]) # Check both ways
    return False

# Helper for wn_synonyms.
def infer_pos(pos):
    if str(pos).startswith('N'):
        return 'n'
    if str(pos).startswith('J'):
        return 'a'
    if str(pos).startswith('V'):
        return 'v'
    else:
        return 0


def ppdb_paraphrases(S,T,i,j):
    w_i = S[i]['word']
    w_j = T[j]['word']
    return w_j in ppdb_dict.get(w_i,[]) or w_i in ppdb_dict.get(w_j,[])


#s_100 = [S for i in range(100)]
#print s_100
#start = timeit.timeit()
#A = align(S,T)
#end = timeit.timeit()
#print A
#print end - start

