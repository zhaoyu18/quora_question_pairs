from semantic_similarity import infer_pos, similarity

# Semantic similarity features.
def percentage_semantic_similarity_one(S, A):
    sim = float(len(A)) / len(S)
    return sim if sim <= 1 else 1


def percentage_semantic_similarity_both(S, T, A):
    sim = 2.0 * len(A) / (len(S) + len(T))
    return sim if sim <= 1 else 1


def percent_unmatched(S, T, A, pos, inferred_pos):
    S_num = num_pos(S, pos, inferred_pos)
    T_num = num_pos(T, pos, inferred_pos)
    if inferred_pos:
        S_unmatched, T_unmatched = number_unmatched(S, T, A, pos, True)
    else:
        S_unmatched, T_unmatched = number_unmatched(S, T, A, pos, False)
    S_p = float(S_unmatched) / S_num if S_num else 0
    T_p = float(T_unmatched) / T_num if T_num else 0
    return S_p, T_p


def number_unmatched(S, T, A, pos, inferred_pos):
    # First, filter out those matched indices.
    S, T = filter_aligned(S, T, A)
    if inferred_pos:
        S, T = filter_infer_pos(S, T, pos)
    else:
        S, T = filter_pos(S, T, pos)
    # print('S Nouns Unmatched:')
    # for i in S:
    #     print(i,S[i]['word'])
    # print('T Nouns Unmatched:')
    # for j in T:
    #     print(j,T[j]['word'])
    S_nouns = set(S.keys())
    T_nouns = set(T.keys())
    for i in S:
        for j in T:
            if similarity(S, T, i, j):
                S_nouns.discard(i)
                T_nouns.discard(j)
    # print('S Nouns Still Unmatched:')
    # for i in S_nouns:
    #     print(i, S[i]['word'])
    # print('T Nouns Still Unmatched:')
    # for j in T_nouns:
    #     print(j, T[j]['word'])
    return len(S_nouns), len(T_nouns)


# Helpers for 'unmatched' features.
def filter_aligned(S, T, A):
    A_i = {i for (i, j) in A}
    A_j = {j for (i, j) in A}
    S = {k:v for k,v in S.items() if k not in A_i}
    T = {k:v for k,v in T.items() if k not in A_j}
    return S, T


def filter_pos(S, T, pos):
    S = {k:v for k,v in S.items() if S[k]['pos'] == pos}
    T = {k:v for k,v in T.items() if T[k]['pos'] == pos}
    return S, T


def filter_infer_pos(S, T, pos):
    S = {k:v for k,v in S.items() if infer_pos(S[k]['pos']) == pos}
    T = {k:v for k,v in T.items() if infer_pos(T[k]['pos']) == pos}
    return S, T


def num_pos(S, pos, inferred_pos):
    if inferred_pos:
        return len([i for i in S if infer_pos(S[i]['pos']) == pos])
    else:
        return len([i for i in S if S[i]['pos'] == pos])

def len_difference_p(S, T):
    len_dif = len_difference(S, T)
    return 2 * float(len_dif) / (len(S) + len(T))

def len_difference(S, T):
    return abs(len(S) - len(T))

# Cutoff at 3
def ner_unmatched(S, T):
    S_ner = [S[i]['word'] for i in S if S[i]['ner'] != 'O']
    T_ner = [T[j]['word'] for j in T if T[j]['ner'] != 'O']
    S_words = [S[i]['word'] for i in S]
    T_words = [T[j]['word'] for j in T]
    S_unmatch_ner = len([w for w in S_ner if w not in T_words])
    S_unmatch_ner = S_unmatch_ner if S_unmatch_ner <= 3 else 3
    T_unmatch_ner = len([w for w in T_ner if w not in S_words])
    T_unmatch_ner = T_unmatch_ner if T_unmatch_ner <= 3 else 3
    return S_unmatch_ner, T_unmatch_ner

