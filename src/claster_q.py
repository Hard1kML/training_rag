from typing import List, Dict, Union
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
from rag_RS import encoder, RAGEval 

eval = RAGEval('data/documents.json', 'data/questions.json', retrieval_limit=5, similarity_threshold=0.6)
retrieval_score, useless_docs_idx, questions_idx = eval.evaluate()
questions = [eval.questions[i] for i in questions_idx]
enc=encoder()
class QCluster:

    def __init__(self, questions_idx: List[int], questions: List[str], encoder):

        if not questions_idx or not questions:
            raise ValueError("questions_idx и questions не могут быть пустыми")

        if len(questions_idx) != len(questions):
            raise ValueError("questions_idx и questions разной длины")

        self.questions_idx = questions_idx
        self.questions = questions
        self.encoder = encoder
        self.clusters = None
        self.kmeans = None

    def _to_numpy(self, embeddings):
        """
        Безопасно приводит embeddings к numpy
        (работает и с torch, и с numpy)
        """
        if hasattr(embeddings, "cpu"):   # torch tensor
            return embeddings.cpu().numpy()
        return np.array(embeddings)

    def cluster(self, n_clusters: int, show_results: bool = False) -> Dict[int, List[int]]:

        if n_clusters < 1:
            raise ValueError("n_clusters must be >= 1")

        if n_clusters > len(self.questions):
            raise ValueError("n_clusters cannot be greater than number of questions")

        # embeddings
        embeddings = self.encoder.encode(self.questions)
        embeddings = self._to_numpy(embeddings)

        # kmeans
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        labels = self.kmeans.fit_predict(embeddings)

        # grouping
        clusters: Dict[int, List[int]] = {}

        for idx, label in enumerate(labels):
            clusters.setdefault(label, []).append(idx)

        self.clusters = clusters

        if show_results:
            self.print_clusters()

        return clusters

    def print_clusters(self):

        if self.clusters is None:
            raise ValueError("Clusters are not created yet. Call cluster() first.")

        for cluster_id, indices in self.clusters.items():
            print(f"\nCluster {cluster_id}:")

            for idx in indices:
                print(f"  [{self.questions_idx[idx]}] {self.questions[idx]}")

qCluster = QCluster(questions_idx, questions, enc) 
cluster = qCluster.cluster(n_clusters=4, show_results=True)
