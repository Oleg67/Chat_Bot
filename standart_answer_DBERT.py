"""
Maintained by IA Team
"""

from logging import getLogger

import numpy
import pandas
import re
import os
from bert_serving.client import BertClient
from sklearn.metrics import pairwise_distances

from utils import check_port
from config import (
    BERT_SERVER,
    BERT_PORT_IN,
    BERT_PORT_OUT,
    BERT_TIMEOUT,
    STANDARD_ANSWER_EMB_PATH,
    STANDARD_ANSWER_THRESHOLD,
)


log = getLogger(__name__)


class StandardAnswerModel:
    def __init__(self):
        assert check_port(BERT_SERVER, BERT_PORT_IN) == check_port(BERT_SERVER, BERT_PORT_OUT)
        self.model = BertClient(ip=BERT_SERVER, port=BERT_PORT_IN, port_out=BERT_PORT_OUT, timeout=BERT_TIMEOUT)
        log.info(
            f"{self.__class__.__name__}: BertClient connected to: "
            f"ip={BERT_SERVER}, port={BERT_PORT_IN}, port_out={BERT_PORT_OUT}"
        )
        self.df = pandas.read_pickle(STANDARD_ANSWER_EMB_PATH)
        self.emb_requests = numpy.array(self.df["request_embed"].to_list())
        log.info(f"{self.__class__.__name__}: embeddings loaded from {STANDARD_ANSWER_EMB_PATH}")
        self.threshold = STANDARD_ANSWER_THRESHOLD
        log.info(f"{self.__class__.__name__}: initialized with threshold {STANDARD_ANSWER_THRESHOLD}")

    def predict_standard(self, request, l_id=None):
        
        _request = re.sub(r'[^\w\s]',' ', request) # del all punctations
        _request = request.lower() # lower registr
        emb_request = self.model.encode([_request])  # embeding of the question

        similarity = pairwise_distances(self.emb_requests, emb_request, metric="cosine", n_jobs=1)
        min_similarity = similarity.min()
        if min_similarity > self.threshold:
            return request, l_id, False
        else:
            mask = similarity == min_similarity
            ind = numpy.arange(len(similarity)).reshape(-1, 1)[mask]
            while len(ind) > 0:
                r_ind = numpy.random.choice(ind, replace=False)
                answer = self.df.loc[r_ind, "answer"]
                log.info(f"{self.__class__.__name__}: threshold={self.threshold}, request={request}, answer={answer}")
                return answer, l_id, True    
                
 class DBERT():
    
    def __init__(self, model, tokenizer, max_len=100):
        
        self.model = model
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def emcode(self, texts):
        
        """
        bert embeddings from momel and tokenizer 
        not use padding
        calculate by one sequence
        """
        emb = np.full((len(texts), self.model.config.dim), 0., dtype=np.float32)
     
        for i, text in enumerate(texts):
            text = self.tokenizer.tokenize(text)
            text = text[:max_len-2]
            input_sequence = ["[CLS]"] + text + ["[SEP]"]
            tokens = tf.convert_to_tensor([tokenizer.convert_tokens_to_ids(input_sequence)], dtype=tf.int32)
            _emb = self.model.predict(tokens)[0]
            emb[i] = _emb.mean(axis=1)
        return emb

    def load_data(self, data_path, out_path='emb_df_'):
        """
        data_path - folder with csv files
        seperated by *  column 1 is request column 2 is answer
        model - use to get embeddings
        out_path - file to write the result
        """
        df = pd.DataFrame(columns=['request', 'answer', 'request_embed'])
        
        for f in os.listdir(data_path):
            if f.split('.')[-1] != 'csv':
                continue
            print('Read file %s'%f)
            f = os.path.join(data_path, f)
            df = df.append(self._get_embs(f))
        
        df = df.reset_index(drop=True)
        
        ## save DataFrame
        out_path = out_path + data_path.split('.')[0] + '.pkl'
        df.to_pickle(out_path)
        print('save data embeddings in ', out_path)
        return
    
    def _get_embs(f_path, model):
        """
        serve to get the sequebces embedings from csv file with two columns request and answer
        """
        df = pd.read_csv(f_path, sep='*', names=['request', 'answer'])
        df.dropna(axis=0, how='all', inplace=True)
        df = df.fillna(method='ffill').reset_index(drop=True) # Data farame with request and negative replise
        df['request'] = df['request'].apply()
        
        df['request_embed'] = None
        emb = self.encode(df['request'].tolist())
        for ind, row in tqdm(df.iterrows()):
            text =  re.sub(r'[^\w\s]',' ', row['request']).lower() # del all punctations & lower registr
            df.at[ind, 'request_embed'] = self.encode([text]).flatten()
        #print(df.shape)
        df['answer'] = df['answer'].str.replace('[', '').str.replace(']', '').str.replace('<', '').str.replace('>', '')
        return df       
