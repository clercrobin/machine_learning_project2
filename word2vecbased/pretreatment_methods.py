import pandas as pd
import numpy as np
import nltk,re,logging
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from unidecode import unidecode
from nltk import ngrams

ps = PorterStemmer()
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Load positive and negative dataset: Different loading if training/testing dataset
class TwitterSentimentData():
    
    def __init__(self,loc_data,pre_cleaned,is_test_file):
        self.pre_cleaned=pre_cleaned
        self.is_test_file=is_test_file
        self.loc_data=loc_data
        if not is_test_file:
            d_neg=pd.read_csv(loc_data+"train_neg.txt",sep='\t',header=None);d_neg.columns=["text"]
            d_pos=pd.read_csv(loc_data+"train_pos.txt",sep='\t',header=None);d_pos.columns=["text"]
            d_pos['pos_sent']=1;d_neg['pos_sent']=-1
            self.data=pd.concat([d_pos,d_neg])
        else:
            f_test=open(loc_data+"test_data.txt")
            data_test=[(int(line.split(',')[0]),",".join(line.split(',')[1:]))
                       for line in (f_test.readlines())]
            f_test.close()
            d_test=pd.DataFrame(data_test,columns=["Id","text"],dtype='str')
            d_test['Id']=d_test['Id'].astype(str)
            d_results=pd.read_csv(loc_data+"sample_submission.csv", sep=",",header=0,dtype='str')
            finals=d_test.merge(d_results,on="Id").drop("Id",axis=1);finals.columns=["text","pos_sent"]
            self.data=finals
            
            
#Add information about textual markers in tweets (urls,mentions,hashtags,punctuation,uppercase)
class TwitterAddTextData():
    
    def __init__(self, tw_sent_data,upc,exc,qust,sght,hashs,mns,urls):
        self.info = tw_sent_data
        self.uppercase=upc
        self.exclamation=exc
        self.question=qust
        self.sight=sght
        self.hashtag=hashs
        self.mention=mns
        self.url=urls

    def add_text_features(self):
        if self.uppercase:
            uppercase = list(map(lambda txt: sum(1 for c in txt if c.isupper()),
                                 self.info.data.text))
            self.info.data["number_of_uppercases"]= uppercase
        if self.exclamation:
            exclamations = list(map(lambda txt: txt.count("!"),
                                 self.info.data.text))
            self.info.data["number_of_exclamations"]= exclamations
        if self.question:
            questions = list(map(lambda txt: txt.count("?"),
                                 self.info.data.text))
            self.info.data["number_of_questions"]= questions
        if self.sight:
            sghts = list(map(lambda txt: txt.count("..."),
                                 self.info.data.text))
            self.info.data["number_of_sights"]= sghts
        if self.hashtag:
            hashs = list(map(lambda txt: txt.count("#"),
                                 self.info.data.text))
            self.info.data["number_of_hashtags"]= hashs
        if self.mention:
            mens = list(map(lambda txt: txt.count("<user>"),
                                 self.info.data.text))
            self.info.data["number_of_mentions"]= mens
        if self.url:
            urls = list(map(lambda txt: txt.count("<url>"),
                                 self.info.data.text))
            self.info.data["number_of_urls"]= urls

# Remove annotations, useless_chars, numbers,           
class TwitterDataCleaning():

    def __init__(self, complete):
        self.info = complete.info
        self.cleanup()

    def iterate(self):
        for cleanup_method in [self.remove_annotations,
                               self.remove_nums,
                               self.remove_useless_chars,
                               self.remove_usernames,
                               self.remove_urls]:
            yield cleanup_method

    @staticmethod
    def correct(regexp,tweets):
        return re.sub(regexp, "",tweets)

    @staticmethod
    def lower_no_accs(tweets):
        return unidecode(tweets).lower()

    def remove_nums(self,tweets):
        return self.correct("[0-9]",tweets)

    def remove_urls(self, tweets):
        return self.correct("http[^ ]*",tweets)

    def remove_useless_chars(self, tweets):
        return self.correct("[;:,*^×“\(\).!?\{\}\[\]\/\|'&#=+<>-]"+'"',tweets)

    def remove_usernames(self, tweets):
        return self.correct("@[^ ]*",tweets)

    def remove_annotations(self,tweets):
        return self.correct("(<user>|<url>)",tweets)

    def cleanup(self):
        tw_sent_data = self.info.data
        replacement=[]
        for tweet in tw_sent_data.text:
            for cleanup_method in self.iterate():
                tweet=cleanup_method(tweet)
            replacement.append(tweet)
        tw_sent_data['text']=[self.lower_no_accs(tweet) for tweet in replacement]
        self.info.data = tw_sent_data

#Analysis of sentiment polarity of words and 3 grams
class SentimentAnalysis():

    def __init__(self,cleaned_and_complete):
        self.info = cleaned_and_complete.info
        self.polar_df_words=None
        self.polar_df_3grams=None
        self.polarity_analysis()

    def polarity_analysis(self):
        tw_sent_data = self.info.data
        polar_dic_wrd={};polar_dic_grams={}
        for it,tweet in tw_sent_data.iterrows():
            for word in tweet.text.split():
                usless=polar_dic_wrd.setdefault(word,[])
                polar_dic_wrd[word].append(tweet.pos_sent)
            for grams in ngrams(tweet.text.split(), 3):
              usless=polar_dic_grams.setdefault(grams,[])
              polar_dic_grams[grams].append(tweet.pos_sent)
        self.polar_df_words=pd.DataFrame([(k,v.count(1),v.count(-1)) for k,v in polar_dic_wrd.items()],columns=["word","pos_sent","neg_sent"])
        self.polar_df_words['total']=(self.polar_df_words.pos_sent+self.polar_df_words.neg_sent)
        self.polar_df_words['polarity']=self.polar_df_words.pos_sent/self.polar_df_words.total
        self.polar_df_3grams=pd.DataFrame([(k,v.count(1),v.count(-1)) for k,v in polar_dic_grams.items()],columns=["3gram","pos_sent","neg_sent"])
        self.polar_df_3grams['total']=(self.polar_df_3grams.pos_sent+self.polar_df_3grams.neg_sent)
        self.polar_df_3grams['polarity']=self.polar_df_3grams.pos_sent/self.polar_df_3grams.total

#Stemming, tokenizing tweets 
class TwitterDataTokenizing():

    def __init__(self, cleaned_and_complete,stemmer,tokenizer):
        self.info = cleaned_and_complete.info
        self.stemmer=stemmer
        self.tokenizer=tokenizer

    def lex_analysis(self):
        self.stem_transform()
        self.token_transform()

    def stem_transform(self):
        stem_tweet = lambda tw:self.stemmer.stem(tw)
        stemd_txt=[stem_tweet(tw) for tw in self.info.data.text]
        self.info.data.text=stemd_txt

    def token_transform(self):
        token_tweet = lambda tw:self.tokenizer(tw)
        tokd_tweet=[token_tweet(tw) for tw in self.info.data.text]
        self.info.data['tokens']=tokd_tweet
