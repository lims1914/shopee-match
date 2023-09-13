import torch
import re
from clip.simple_tokenizer import SimpleTokenizer
from cuml.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.preprocessing import Normalizer
import numpy as np
import tqdm
import gc



_tokenizer = SimpleTokenizer()
# Copied from https://github.com/openai/CLIP/blob/beba48f35392a73c6c47ae67ddffced81ad1916d/clip/clip.py#L164
def tokenize(texts, context_length: int = 77) :
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        n = min(len(tokens), context_length)
        result[i, :n] = torch.tensor(tokens)[:n]
        if len(tokens) > context_length:
            result[i, -1] = tokens[-1]

    return result


# Remove EMOJI
RE_EMOJI = re.compile(r"\\x[A-Za-z0-9./]+", flags=re.UNICODE)


def strip_emoji(text):
    return RE_EMOJI.sub(r'', text)


def f1_score(y_true, y_pred):
    y_true = y_true.apply(lambda x: set(x.split()))
    y_pred = y_pred.apply(lambda x: set(x.split()))
    intersection = np.array([len(x[0] & x[1]) for x in zip(y_true, y_pred)])
    len_y_pred = y_pred.apply(lambda x: len(x)).values
    len_y_true = y_true.apply(lambda x: len(x)).values
    f1 = 2 * intersection / (len_y_pred + len_y_true)
    return f1




def get_image_predictions(df, embeddings, threshold=0.0):
    model = NearestNeighbors(n_neighbors=50, metric='cosine')
    model.fit(embeddings)
    distances, indices = model.kneighbors(embeddings)
    predictions = []
    for k in tqdm(range(embeddings.shape[0])):
        idx = np.where(distances[k,] < threshold)[0]
        ids = indices[k, idx]
        posting_ids = df['posting_id'].iloc[ids].values
        if len(posting_ids) >= 3:
            idx_s = np.where(distances[k,] < threshold - 0.09)[0]
            ids_s = indices[k, idx_s]
            posting_ids_b = df['posting_id'].iloc[ids_s].values
            if len(posting_ids_b) >= 2:
                predictions.append(posting_ids_b)
            else:
                predictions.append(posting_ids)
        else:
            idx = np.where(distances[k,] < threshold + 0.09)[0]
            ids = indices[k, idx]
            posting_ids = df['posting_id'].iloc[ids].values
            predictions.append(posting_ids[:2])

    del model, distances, indices
    gc.collect()
    df['img_pred'] = predictions
    df['img_pred'] = df.img_pred.apply(lambda x: ' '.join(x))
    print(df['img_pred'])
    print(df['target'])
    score = f1_score(df['target'], df['img_pred']).mean()
    print(f'Our f1 score for threshold {threshold} is {score}')
    return score


def extra_same_elem(list1, list2, list3):
    counter = {}
    res = []
    new_matrix = np.concatenate((np.concatenate((list1, list2), axis=0), list3), axis=0)
    for i in range(new_matrix.shape[0]):
        if counter.get(new_matrix[i]):
            counter[new_matrix[i]] += 1
            res.append(new_matrix[i])
        else:
            counter[new_matrix[i]] = 1
    return list(set(res))


def get_voted_predictions(df, embeddings1, embeddings2, embeddings3, threshold=0.0):
    KNN = 50
    model1 = NearestNeighbors(n_neighbors=KNN, metric='cosine')
    model1.fit(embeddings1)
    distances1, indices1 = model1.kneighbors(embeddings1)
    model2 = NearestNeighbors(n_neighbors=KNN, metric='cosine')
    model2.fit(embeddings2)
    distances2, indices2 = model2.kneighbors(embeddings2)
    model3 = NearestNeighbors(n_neighbors=KNN, metric='cosine')
    model3.fit(embeddings3)
    distances3, indices3 = model3.kneighbors(embeddings3)

    predictions = []
    for k in tqdm(range(embeddings1.shape[0])):
        idx1 = np.where(distances1[k,] < threshold)[0]
        ids1 = indices1[k, idx1]
        posting_ids1 = df['posting_id'].iloc[ids1].values
        idx2 = np.where(distances2[k,] < threshold)[0]
        ids2 = indices1[k, idx2]
        posting_ids2 = df['posting_id'].iloc[ids2].values
        idx3 = np.where(distances3[k,] < threshold)[0]
        ids3 = indices3[k, idx1]
        posting_ids3 = df['posting_id'].iloc[ids3].values
        res = extra_same_elem(posting_ids1, posting_ids2, posting_ids3)
        predictions.append(res)

    del model1, distances1, indices1, model2, distances2, indices2, model3, distances3, indices3
    gc.collect()
    df['img_pred'] = predictions
    df['img_pred'] = df.img_pred.apply(lambda x: ' '.join(x))
    # print(df['img_pred'])
    # print(df['target'])
    score = f1_score(df['target'], df['img_pred']).mean()
    print(f'Our f1 score for threshold {threshold} is {score}')
    return score


def pca(image_embedding):
    pca = PCA(n_components=512)
    new_image_imbedding = pca.fit_transform(image_embedding)
    norm2 = Normalizer(norm='l2')
    normalized_image_embed = norm2.fit_transform(new_image_imbedding)
    return normalized_image_embed