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
                
        
