from sklearn.cluster import KMeans
from vectorize import Vectorize
from sklearn.metrics import pairwise_distances_argmin_min
from transcript import Transcript
from preprocessing import Preprocessor
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import numpy as np
class Summarizer:
    def __init__(self):
        self.vectorizer = Vectorize()
        self.transcript = Transcript()
        self.preprocessor = Preprocessor()
        
    
    
    def summarize(self,video_path,n_clusters=5,mode='clustering'):
        paragraph = self.transcript.extract_audio(video_path)
        sentences = self.preprocessor.tokenize(paragraph)
        
        X = self.vectorizer.vectorize(sentences)
        if mode == 'clustering':
            kmeans = KMeans(n_clusters=n_clusters)
            kmeans = kmeans.fit(X)
            avg = []
            for j in range(n_clusters):
                idx = np.where(kmeans.labels_ == j)[0]
                avg.append(np.mean(idx))
            closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, X)
            ordering = sorted(range(n_clusters), key=lambda k: avg[k])
            summary = ' '.join([sentences[closest[idx]] for idx in ordering])
        else:
            sim_mat = np.zeros([len(sentences), len(sentences)])
            for i in range(len(sentences)):
                for j in range(len(sentences)):
                    if i != j:
                        sim_mat[i][j] = cosine_similarity(X[i].reshape(1, -1),
                                                          X[j].reshape(1, -1))[0][0]

            nx_graph = nx.from_numpy_array(sim_mat)
            scores = list(nx.pagerank(nx_graph).values())
            top_sentences = np.argsort(scores)[-n_clusters:][::-1]
            top_sentences.sort()
            summary = " ".join([sentences[i] for i in top_sentences])

        print(summary)
        return summary
        

