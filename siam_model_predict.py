""""How to run:
1. git clone https://github.com/hanxiao/bert-as-service
2. Run BERT Server bert-serving-start -model_dir e:\models\cased_L-12_H-768_A-12 -num_worker=4 -max_seq_len=100
3. SIAM model 'siam_trained.pkl'
Then run the script

"""

import datatable as dt
import pandas as pd
import numpy as np
import pickle as pk
import json
import sys
from sklearn.metrics import pairwise_distances
from bert_serving.client import BertClient

import torch
from models.Siam.train_siam import SiamSimilaraty, get_free_gpu

from utils import log_execution_time, read_one_pkl_file, path_to_all_pkl_files
import logging
import timeit
logger = logging.getLogger(__name__)


#Cosine questions similarity-based model using datatable
class SiamModel:
    
    def __init__(self, path_to_data, model_path='siam_trained.pkl', threshold=100., max_dist=10000., 
                 bert_port=5555, bert_port_out=5556, 
                 model_type = 'siam', min_distance = 1., device='cuda:1'):
        self.threshold = threshold
        self.min_dist = min_distance # the distance from we find the nearest answer to escape the same text
        self.max_dist = max_dist
        self.model_type = model_type
        
        self.bert_wrapper = BertClient(port=bert_port, port_out=bert_port_out)
        logger.info("Initializing BERT")
        
        # Choice the device
        if device == 'free':
            device = 'cuda:' +str(get_free_gpu()) # Choice the free device
        self.device = device
        
        ## Load Siam model
        self.siam = SiamSimilaraty() 
        self.siam.load_state_dict(torch.load(model_path, map_location=device))
        logger.info("Load SiamModel from  %s"% model_path)
        logger.info("Preparing SiamModel, settings: min_dist=%.4f, threshold=%.4f, max_dist=%.4f"%(min_distance, threshold, max_dist))
        
        self.siam.eval()
        self.siam.to(device)
        logger.info("Load SiamModel to  %s"% device)
        
        logger.info("Reading dialogs dataset '%s'" % path_to_data)
        # We expect that dataframe is sorted in natural order - by dialog_owner, dialog_with, datatime
        self.create_model(path_to_data)
        self.df = self.df.sort(["dialog_owner","dialog_with", "epoch_time"])
        logger.info("Shape of the train data is [%d, %d]" %(self.df.shape))
        logger.info("SiamModel: Listerning...")
    
    def get_data(self):
        return self.df

    """Create model using data from selected folder"""    
    @log_execution_time
    def create_model(self, l_path: str) -> None:
        path_to_all_files = path_to_all_pkl_files(l_path, self.model_type)
        #Read file by file, convert to Pandas DataFrame (Datatable isn't good with dicts) then convert to Datatable and concatenate
        dataset = dt.rbind([dt.Frame(pd.DataFrame(read_one_pkl_file(p)['data'])) for p in path_to_all_files])
        
        dataset = dt.Frame(pd.DataFrame(read_one_pkl_file('/data/datasets/fb_test/subfolder3/test1.pkl')['data']))
        assert 'with_text' in dataset.names, 'Check datatable there is no with_text key or it is empty' 
        
        #Remove rows without text
        self.df = dataset[(dt.f.with_text==1),:]
        assert 'text' in self.df.names, 'Need the text in Data'
        assert 'text_siam_embed' in self.df.names, 'Need the siam embeddings in Data'
        
        
    """Remove user's virtual model data from memory"""    
    @log_execution_time
    def delete_from_model(self, c_user: int, model_id: int) -> None:
        del self.df[(dt.f.model_id == model_id) & (dt.f.dialog_owner == c_user), :]

    """Add data from single file to the model memory.
        Be careful! Calling function on a same data few times will produce duplicates!"""     
    @log_execution_time
    def add_to_model(self, path: str) -> None:
        # Read one pkl file to DataTable
        #  convert to Pandas DataFrame (Datatable isn't good with dicts) then convert to Datatable
        dataset = dt.Frame(pd.DataFrame(read_one_pkl_file(path)['data']))
        #Add data after removing rows without text
        self.df.rbind(dataset[(dt.f.with_text==1),:])

    """Remove user's model old data and uploads model new data"""    
    @log_execution_time
    def update_model(self, c_user: int, model_id: int, path: str) -> None:
        self.delete_from_model(c_user=c_user, model_id=model_id)  # delete first
        self.add_to_model(path=path)  # then load new data

        
    """Parses JSON and return important info.
       Returns error_code if JSON badly structured"""
    def parse_json(self, l_json_message):
        logger.info("CosineModel: Parsing JSON data '%s'" % l_json_message)
        try: #parse json structure
            parsed_data = json.loads(l_json_message)
        except Exception as e:
            logger.exception(sys.exc_info())
            l_error = "Incorrect JSON Structure"
            return None, l_error
        return parsed_data, None        

    
    """Method finds the closest answer in dataset on the question"""
    def predict(self, l_json_message, l_model_id, l_id="0000-00000-0000-0000-00000"):    
        """Find answer based on distance between Siam embeddings"""
        def find_closest_answer(l_question, l_data):
            
            if l_data.shape[0]==0:
                answer = "I'm not sure"
                dist = self.max_dist                      
                logger.debug("find_closest_answer: There is no data in dataset with selected questions")
                return dist, answer
            
            logger.debug('Shape l_question %d, %d'%(l_question.shape))
            logger.debug(l_data['text_siam_embed'].shape)
            v = pairwise_distances(l_question, l_data['text_siam_embed'].to_list()[0], metric='euclidean', n_jobs=1)
            v[v < self.min_dist] = np.inf # to escape the same text
            
            #Nearest answer
            min_index = int(np.argmin(v))
            dist = v.min()
            answer = l_data[min_index,'text']
            # If nearest answer is the the same text
            if dist == np.inf:
                dist = self.max_dist
                answer = "I'm not sure"
                logger.debug("find_closest_answer: There is the same text")
                return dist, answer
                    
            return dist, answer
        
            
        #Parse data
        l_error = None
        parsed_data, l_error = self.parse_json(l_json_message)
        if l_error is not None:
            return "I'm not sure", l_id, l_error        
        l_question = str(parsed_data["text"])
        l_persona_ask = parsed_data["message_sender"] #str()
        l_persona_answer = parsed_data["message_recipient"] #str()

    
        #Check if the text field has a content
        if l_question is None or len(l_question)==0:
            logger.info("Empty question")
            l_error = "Empty question"
            return "I'm not sure", l_id, l_error
        
        
        logger.debug("SiamModel: Creating embedding for sentence '%s'" % l_question)
        gold_question = self.bert_wrapper.encode([l_question]) # Bert embedings
        gold_question = self.siam(torch.tensor(gold_question).to(self.device)).cpu().detach().numpy() # Siam embedings
        l_set = self.df[(dt.f.with_text==1),:]
        
        if l_set.shape[0]==0:
            logger.debug("There is no data in dataset")
            return "I'm not sure", l_id, "Whole dataset is empty"
        
        #Choose answer based on data from current model
        l_set = l_set[(dt.f.model_id==l_model_id),:]
        if l_set.shape[0]==0:
            logger.debug("IN_MODEL_QUESTIONS: There is no data in model_id = %s" % l_model_id)
            l11_answer = "I'm not sure"
            l11_dist = self.max_dist
            l12_answer = "I'm not sure"
            l12_dist = self.max_dist
            l13_answer = "I'm not sure"
            l13_dist = self.max_dist            
        else:
            ###Find best answer from current user and current interlocutor
            data = l_set[(dt.f.message_sender==l_persona_answer) & (dt.f.message_recipient==l_persona_ask),:]
            if data.shape[0]==0:
                logger.debug("IN_MODEL_QUESTIONS (lvl 1.1), same asker, same answerer: No data for pair persona_ask = %s, persona_answer = %s" % (l_persona_ask,l_persona_answer))
                l11_answer = "I'm not sure"
                l11_dist = self.max_dist
            else:
                logger.debug("IN_MODEL_QUESTIONS (lvl 1.1), processing")
                l11_dist, l11_answer = find_closest_answer(gold_question, data)
                logger.debug('IN_MODEL_QUESTIONS (lvl 1.1), same asker, same answerer: answer found (q.dist=%.4f) "%s"'%(l11_dist, l11_answer))

            if l11_dist<=self.threshold:
                logger.debug('IN_MODEL_QUESTIONS (lvl 1.1), same asker, same answerer: answer selected (%.4f from %.4f)'%(l11_dist,self.threshold))
                return l11_answer, l_id, l_error
            #-----------------------------

            ###Find best answer from all current user's dialogs
            data = l_set[(dt.f.message_sender==l_persona_answer),:]
            if data.shape[0]==0:
                logger.debug("IN_MODEL_QUESTIONS (lvl 1.2), same answerer , any asker: No data for persona_asker = %s" % l_persona_answer)
                l12_answer = "I'm not sure"
                l12_dist = self.max_dist
            else:    
                logger.debug("IN_MODEL_QUESTIONS (lvl 1.2), processing")
                l12_dist, l12_answer = find_closest_answer(gold_question, data)    
                logger.debug('IN_MODEL_QUESTIONS (lvl 1.2), same asker, any answerer: answer found (q.dist=%.4f) "%s"'%(l12_dist, l12_answer))
        
            if l12_dist<=self.threshold:
                logger.debug('IN_MODEL_QUESTIONS (lvl 1.2), same asker, any answerer: answer selected (%.4f from %.4f)'%(l12_dist,self.threshold))                
                return l12_answer, l_id, l_error
            #-----------------------------

            ###Find best answer from all data            
            data = l_set
            if data.shape[0]==0:
                logger.debug("IN_MODEL_QUESTIONS (lvl 1.3), any asker, any answerer: No data")
                l13_answer = "I'm not sure"
                l13_dist = self.max_dist
            else:    
                logger.debug("IN_MODEL_QUESTIONS (lvl 1.3), processing")
                l13_dist, l13_answer = find_closest_answer(gold_question, data)    
                logger.debug('IN_MODEL_QUESTIONS (lvl 1.3), any asker, any answerer: answer found (q.dist=%.4f) "%s"'%(l13_dist, l13_answer))
        
            if l13_dist<=self.threshold:
                logger.debug('IN_MODEL_QUESTIONS (lvl 1.3), any asker, any answerer: answer selected (%.4f from %.4f)'%(l13_dist,self.threshold))                
                return l13_answer, l_id, l_error            
        #============================
        
        
        #Choose answer based on all data
        l_set = self.df[(dt.f.with_text==1),:]
        ###Find best answer from current user and current interlocutor
        data = l_set[(dt.f.message_sender==l_persona_answer) & (dt.f.message_recipient==l_persona_ask),:]
        if data.shape[0]==0:
            logger.debug("ALL_DATASET_QUESTIONS (lvl 2.1), same asker, same answerer: No data for pair persona_ask = %s, persona_answer = %s" % (l_persona_ask,l_persona_answer))
            l21_answer = "I'm not sure"
            l21_dist = self.max_dist
        else:
            logger.debug("ALL_DATASET_QUESTIONS (lvl 2.1), processing")
            l21_dist, l21_answer = find_closest_answer(gold_question, data)
            logger.debug('ALL_DATASET_QUESTIONS (lvl 2.1), same asker, same answerer: answer found (q.dist=%.4f) "%s"'%(l21_dist, l21_answer))

        if l21_dist<=self.threshold:
            logger.debug('ALL_DATASET_QUESTIONS (lvl 2.1), same asker, same answerer: answer selected (%.4f from %.4f)'%(l21_dist,self.threshold))
            return l21_answer, l_id, l_error
        #-----------------------------
        
        ###Find best answer from all current user's dialogs
        data = l_set[(dt.f.message_sender==l_persona_answer),:]
        if data.shape[0]==0:
            logger.debug("ALL_DATASET_QUESTIONS (lvl 2.2), same answerer, any asker: No data for persona_ask = %s" % l_persona_answer)
            l22_answer = "I'm not sure"
            l22_dist = self.max_dist
        else:    
            logger.debug("ALL_DATASET_QUESTIONS (lvl 2.2), processing")
            l22_dist, l22_answer = find_closest_answer(gold_question, data)    
            logger.debug('ALL_DATASET_QUESTIONS (lvl 2.2), same asker, any answerer: answer found (q.dist=%.4f) "%s"'%(l22_dist, l22_answer))

        if l22_dist<=self.threshold:
            logger.debug('ALL_DATASET_QUESTIONS (lvl 2.2), same asker, any answerer: answer selected (%.4f from %.4f)'%(l22_dist,self.threshold))                
            return l22_answer, l_id, l_error
        #-----------------------------
        
        ###Find best answer from all data            
        data = l_set
        logger.debug("ALL_DATASET_QUESTIONS (lvl 2.3), processing")
        l23_dist, l23_answer = find_closest_answer(gold_question, data)    
        logger.debug('ALL_DATASET_QUESTIONS (lvl 2.3), any asker, any answerer: answer found (q.dist=%.4f) "%s"'%(l23_dist, l23_answer))

        if l23_dist<=self.threshold:
            logger.debug('ALL_DATASET_QUESTIONS (lvl 2.3), any asker, any answerer: answer selected (%.4f from %.4f)'%(l23_dist,self.threshold))                
            return l23_answer, l_id, l_error  
        #============================
                
    
    
        #This logic we use in case when no answer were selected with distance lower than threshold
        distances = [l11_dist, l12_dist, l13_dist, l21_dist, l22_dist, l23_dist]
        answers = [l11_answer, l12_answer, l13_answer, l21_answer, l22_answer, l23_answer]
        smallest_index = np.argmin(distances)
        if distances[smallest_index]<=self.max_dist:
            logger.debug('ABOVE_THRESHOLD: seleted %dth (0-based) choice with distance %.4f'%(smallest_index,distances[smallest_index]))
            return answers[smallest_index], l_id, l_error

        logger.debug('ABOVE_THRESHOLD: seleted %dth choice with distance %.4f but it is above max dist %.4f'%(smallest_index,distances[smallest_index],self.max_dist))
        return "I'm not sure", l_id, l_error

    
    """Method gives user a possibility to have a dialog with himself.
       For this we disguise current user_id by imitation other user_id"""
    def predict_self_answer(self, l_message_sender, l_message_recipient, l_text, l_model_id, l_id="0000-00000-0000-0000-00000"):       
        #Data template
        l_json_message = """{
          "message_id": "cAAAAACvd3zF2GD0nlVv5vfNq1F66",
          "dialog_owner": 100043479992945,
          "dialog_with": 100042904853723,
          "epoch_time": 1579857721818,
          "message_sender": %d,
          "message_recipient": %d,
          "text": "%s",
          "sticker": null,
          "reactions": {},
          "quick_reply": null,
          "attachments": []
        }"""%(l_message_sender, l_message_recipient, l_text)
        
        #Parse data
        l_error = None
        parsed_data, l_error = self.parse_json(l_json_message)
        if l_error is not None:
            return "I'm not sure", l_id, l_error     

        l_persona_ask = parsed_data["message_sender"] #str()
        l_persona_answer = parsed_data["message_recipient"] #str()
        parsed_data["message_recipient"] = l_persona_ask
        parsed_data["message_sender"] = l_persona_answer
        l_json_message = json.dumps(parsed_data)
        
        return self.predict(l_json_message, l_model_id, l_id)
