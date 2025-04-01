import gensim
import numpy as np
import ot

def train_cbow(corpus, vector_size=100, window=5, min_count=1, epochs=10):
    model = gensim.models.Word2Vec(sentences=corpus, vector_size=vector_size, window=window, 
                                   min_count=min_count, sg=0, epochs=epochs)
    return model

def load_corpus(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        corpus = [line.strip().split() for line in f.readlines()]
    return corpus


def compute_distance_matrix(model):
    words = list(model.wv.index_to_key)
    vectors = np.array([model.wv[word] for word in words])
    distances = np.linalg.norm(vectors[:, None, :] - vectors[None, :, :], axis=-1)
    return distances

def compute_gw_distance(dist_matrix1, dist_matrix2):
    n, m = dist_matrix1.shape[0], dist_matrix2.shape[0]
    p, q = np.ones(n) / n, np.ones(m) / m 
    gw_distance = ot.gromov.gromov_wasserstein2(dist_matrix1, dist_matrix2, p, q, 'square_loss')
    return gw_distance

corpora_paths = {
    "human_russian": "human_russian.txt",
    "balaboba": "balaboba.txt",
    "chatgpt2": "chatgpt2.txt",
    "azerbaijani": "azerbaijani.txt"
}

models = {}
distance_matrices = {}
for name, path in corpora_paths.items():
    corpus = load_corpus(path)
    model = train_cbow(corpus)
    models[name] = model
    distance_matrices[name] = compute_distance_matrix(model)

gw_distances = {}
corpus_names = list(corpora_paths.keys())
for i in range(len(corpus_names)):
    for j in range(i+1, len(corpus_names)):
        gw_dist = compute_gw_distance(distance_matrices[corpus_names[i]], 
                                      distance_matrices[corpus_names[j]])
        gw_distances[(corpus_names[i], corpus_names[j])] = gw_dist

for (corpus1, corpus2), dist in gw_distances.items():
    print(f"GW distance between {corpus1} and {corpus2}: {dist:.4f}")