# Get embeddings for the kNN retriever in the many-shot ICL predictor

import os, sys, tqdm
import os.path as osp
from typing import List, Optional

import numpy as onp
import vertexai
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel

sys.path.append(os.getcwd())
from mt.utils import load_data


DATA_PATH = './data/flores/floresp-v2.0-rc.3'
DEST_PATH = './run/mt/emb-retrieval.npz'


def embed_text(
    texts: list = None,
    task: str = "RETRIEVAL_DOCUMENT",
    dimensionality: Optional[int] = 256,
) -> List[List[float]]:
    """Embeds texts with a pre-trained, foundational model.
    Args:
        texts (List[str]): A list of texts to be embedded.
        task (str): The task type for embedding. Check the available tasks in the model's documentation.
        dimensionality (Optional[int]): The dimensionality of the output embeddings.
    Returns:
        List[List[float]]: A list of lists containing the embedding vectors for each input text
    """
    model = TextEmbeddingModel.from_pretrained("text-embedding-004")
    inputs = [TextEmbeddingInput(text, task) for text in texts]
    kwargs = dict(output_dimensionality=dimensionality) if dimensionality else {}
    embeddings = model.get_embeddings(inputs, **kwargs)
    return [embedding.values for embedding in embeddings]


def embed_batched(sentences):
    embs = []
    for i in tqdm.trange(0, len(sentences), 100):
        embs.extend(embed_text(sentences[i: i+100]))
    return embs


if __name__ == '__main__':
    vertexai.init(location='us-central1')
    src_dev = load_data(DATA_PATH, 'dev', 'eng_Latn')
    src_devtest = load_data(DATA_PATH, 'devtest', 'eng_Latn')
    emb_dev = embed_batched(src_dev)
    emb_devtest = embed_batched(src_devtest)
    emb_dev, emb_devtest = map(onp.asarray, (emb_dev, emb_devtest))
    onp.savez(DEST_PATH, **{
        'emb_dev': emb_dev,
        'emb_devtest': emb_devtest,
    })
