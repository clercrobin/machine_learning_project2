import numpy as np
import gensim
import gensim.models.word2vec as w2v
from tqdm import tqdm
from gensim.models import Word2Vec
from gensim.similarities import MatrixSimilarity
from gensim.matutils import Dense2Corpus
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter


# Iterate through tweets
class MySentences():
    # Iterator through tweets
    def __init__(self,data ):
        self.data = data

    def __iter__(self):
        for line in self.data:
            yield line.split()

# Add Word2Vec features:
# - Load pre-defined or train word2vec model --> embedding of dimension 400
# - Average word2vec features for words in tweet 
# - Spectral Clustering on word2vec cosine similarity martix between words --> Default: 100 topics
class Tweets_w2v_Features():
    def __init__(self,pre_filled_feats,built_w2v=None,wdw_sz=5,vc_sz=512,neg_sampl=20,iter=50,wrkrs=96,min_ct=10,max_vs=70000):
        if built_w2v!=None:
            self.info=pre_filled_feats
            self.model = gensim.models.KeyedVectors.load_word2vec_format(built_w2v, binary=True,unicode_errors='ignore')
        else:
            sentences = MySentences(self.info.data.text) # a memory-friendly iterator
            model = w2v.Word2Vec(sentences,size=vc_sz,window=wdw_sz,negative=neg_sampl,min_count=min_ct,iter=iter,seed=1,workers=wrkrs,max_vocab_size=max_vs)
            model.save('.')

    def get_vector(self,word):
        if word not in self.model.vocab:
            return None
        return self.model[word]

    def get_similarity(self, word1, word2):
        if word1 not in self.model.vocab or word2 not in self.model.vocab:
            return None
        return self.model.similarity(word1, word2)

    def after_treatment_word2vec(self,min_times):
        treated_words=[]
        for word in self.model.wv.vocab:
            alpha_wrds=''.join(x for x in word if x.isalpha())
            if len(alpha_wrds)==len(word) and self.model.wv.vocab[word].count>min_times:
                treated_words.append(word)
        return treated_words
    
    def after_treatment_file(self,min_times):
        counted_words=Counter([word for line in self.info.data.text for word in line.split()])
        treated_words=[]
        for word,nb in counted_words.items():
            alpha_wrds=''.join(x for x in word if x.isalpha())
            if len(alpha_wrds)==len(word) and nb>min_times:
                treated_words.append(word)
        return treated_words

    def nlm_features(self):
        tw_sent_data = self.info.data
        w2v_fts=[]
        for tweet in tw_sent_data.text:
            inter=[self.get_vector(word) for word in tweet.split() if word in self.model]
            if len(inter)==0:
                w2v_fts.append(None)
            else:
                w2v_fts.append(np.mean(inter,axis=0))
        self.info.data['w2v']=w2v_fts

    def sp_features(self,min_times=500):
        tw_sent_data = self.info.data
        vocab = self.after_treatment_file(min_times)
        
        print("Nb words in vocabulary.... %d"%len(vocab))
        print("Computing similarity matrix .....")
        modelvect = [self.model[word] for word in vocab if word in self.model]
        
        A_sparse = sparse.csr_matrix(np.array([gensim.matutils.unitvec(i) for i in tqdm(modelvect)]))
        similarities = cosine_similarity(A_sparse,dense_output=True)
        threshold_for_bug = 0.000000001
        similarities[(similarities)<threshold_for_bug]= threshold_for_bug
        nb_clusts=100
        
        print("Spectral Clustering ....")
        from sklearn.cluster import spectral_clustering,SpectralClustering
        spectral = SpectralClustering(n_clusters=nb_clusts,affinity="precomputed",n_jobs=30)
        spectral.fit(similarities)
        
        print("SC labeling .... ")
        labels = spectral.fit_predict(similarities)
        word_clusters={i:[] for i in range(nb_clusts)}
        
        for it,lab in enumerate(labels):
            word_clusters[lab].append(vocab[it])
        
        word2cluster={word:cluster_nb 
                      for cluster_nb,cluster_words in word_clusters.items()
                      for word in cluster_words}
        clust_freq=[]
        for tweet in tw_sent_data.text:
            clust_freq.append((Counter([word2cluster[word] for word in tweet.split() if word in word2cluster ])))
        pre_cfd=[{k:(v+0.0)/(sum(dic_count.values()))
                     for k,v in dic_count.items()} for dic_count in clust_freq]
        self.info.data["cfd"]=[np.array(list({clus:(dic_count[clus] if clus in dic_count else 0)
                                            for clus in range(len(word_clusters))}.values())) for dic_count in pre_cfd]
    def get_embedding(self,embed_name):
        df=self.info.data
        dim_total=len((df.head(1)[embed_name].as_matrix())[0])
        return np.array(np.vstack([np.hstack(sample.as_matrix()).reshape((1,dim_total))
                                   for it,sample in (df[[embed_name]].iterrows())]))

