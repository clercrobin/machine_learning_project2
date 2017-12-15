import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
import os

all_but_text=pickle.load(open("/home/jlevyabi/seacabo/fempl_big_sim/sent_class/checkpt_data/dic_all.p","rb"))
def get_embedding(embed_name,df):
    dim_total=len((df.head(1)[embed_name][0]))
    return np.array(np.vstack([sample[0]
                      for it,sample in tqdm(df[[embed_name]].iterrows())]))

os.system("cd ~/seacabo/python_scripts/Librairies/Multicore-TSNE/")
relevant_samples=all_but_text[all_but_text['w2v'].isnull()==False]
shallow=["number_of_uppercases","number_of_exclamations","number_of_questions","number_of_sights",
         "number_of_hashtags","number_of_mentions","number_of_urls"]

w2v_fts=get_embedding("w2v",relevant_samples)
cfd_fts=get_embedding("cfd",relevant_samples)
shallow_fts=relevant_samples[shallow]
targets=relevant_samples["pos_sent"]

full_fts=np.hstack([w2v_fts,cfd_fts,shallow_fts])
print("TSNE: Word2vec features")
from MulticoreTSNE import MulticoreTSNE as TSNE
tsne = TSNE(n_jobs=60)
fts_target = tsne.fit_transform(full_fts)

dic_tsne={"fts_target":fts_target,"sent_target":targets}
pickle.dump(dic_tsne, open( "/home/jlevyabi/seacabo/fempl_big_sim/sent_class/checkpt_data/dic_tsne_full.p", "wb" ))
