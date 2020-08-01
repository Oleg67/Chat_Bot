import sys

import json
import numpy as np
from bert_serving.client import BertClient
from fasttext import FastText
import pandas as pd

# from tqdm import tqdm
# from tqdm.autonotebook import trange, tqdm
import logging
import os
import datetime
import pickle
import torch

import argparse

from utils import path_to_all_pkl_files


logger = logging.getLogger()

cols = [ #columns order
 'message_id',
 'model_id',
 'dialog_owner',
 'dialog_with',
 'epoch_time',
 'message_sender',
 'message_recipient',
 'sticker',
 'reactions',
 'text',
 'quick_reply',    
 'attachments',
 'lang',
 'lang_prob',
 'datatime',
 'with_text'
 ]


#Training module for Siam-based model using datatable
class SiamModelTrainer:
    def __init__(self, model_path='siam_trained.pkl', path_to_lang_model='lid.176.bin', 
                 bert_port=6665, bert_port_out=6666, device='cuda:1'):
        
        logger.info("Creating SiamTrain Module")
        logger.info("Initializing BERT")
        self.bert_wrapper = BertClient(port=bert_port, port_out=bert_port_out)
        
        logger.info("Initializing lang detect model")
        self.lang_model = FastText.load_model(path_to_lang_model)
        
        # Choice the device
        if device == 'free':
            device = 'cuda:' +str(get_free_gpu()) # Choice the free device
        self.device = device
        
        ## Load Siam model
        self.siam = SiamSimilaraty() 
        self.siam.load_state_dict(torch.load(model_path, map_location=device))
        logger.info("Load SiamModel from  %s"% model_path)
        self.siam.eval()
        self.siam.to(device)
        logger.info("Load SiamModel to  %s"% device)

    def _lang_detect(self, l_text):
        res = self.lang_model.predict(l_text.replace("\n"," "))
        return {'lang':res[0][0][9:], 'lang_prob':res[1][0]}
    
    #Calculate the Siam embeddings
    def _get_Siam_embeddings(self, d_f, w_b_emb=False):
        logger.info("Shape of the data is [%d, %d]" %(d_f.shape))
        
        assert 'text' in d_f.columns, 'Need the text in Data'
        assert ('text_embed' in d_f.columns) or bool(self.bert_wrapper), 'Need bert embedings or BertModel'
    
        # Add  embeddings into the dataframe
        if 'text_embed' in d_f.columns:
            logger.info("Start to calculate Siam embedings")
            d_f['text_siam_embed'] = None
            
            for i, r in d_f[d_f['with_text']].iterrows():
                d_f.at[i, 'text_siam_embed'] = self.siam(torch.tensor(r['text_embed']).to(self.device)).cpu().detach().numpy()
        else:
            logger.info("Start to calculate Siam & Bert embedings")
            if w_b_emb:
                d_f['text_embed'] = None
            d_f['text_siam_embed'] = None
            
            for i, r in d_f[d_f['with_text']].iterrows():
                #print(r['text'])
                b_emb = self.bert_wrapper.encode([r['text']]).flatten()
                if w_b_emb:
                    d_f.at[i, 'text_embed'] = b_emb
                d_f.at[i, 'text_siam_embed'] = self.siam(torch.tensor(b_emb).to(self.device)).cpu().detach().numpy()
    
        logger.info("Embeddings have been created")
        return d_f    
    
    def create_model(self, json_data):
        logger.info("Parsing JSON")
        try: #parse json structure
            parsed_data = json.loads(json_data)
        except Exception as e:
            logger.exception(sys.exc_info())
            return {'status':"Failed", 'data':None}
        
        logger.info("Enriching the data")
        try: #Fill all needed fields
            model_id = parsed_data['model_id']
            dialogs = parsed_data['dialogs']
            df = pd.DataFrame(dialogs)
            df['datatime'] = df['epoch_time'].apply(lambda x: datetime.datetime.fromtimestamp(x//1000).strftime('%Y-%m-%d %H:%M:%S'))
#             df['with_text'] = df['text'].apply(lambda x: len(x.strip())>0)
            df['with_text'] = df['text'].apply(lambda x: not (x is None or len(x.strip())==0))
            
            data_lang =  df[df['with_text']]['text'].apply(lambda x: pd.Series(self._lang_detect(x)))
            df['lang_prob'] = None
            df['lang'] = None
            df['model_id'] = model_id
            df.loc[df['with_text'],'lang_prob'] = data_lang['lang_prob']
            df.loc[df['with_text'],'lang'] = data_lang['lang']
            df = df[cols]
        except Exception as e:
            logger.exception(sys.exc_info())
            return {'status':"Failed", 'data':None}
        
        logger.info("Calculating embeddings")
        df = self._get_Siam_embeddings(df)
        
        return {'status':"OK", 'data':df.to_dict()} 
        
    def write_embedings_file(self, path_files, w_b_emb=False):
        """
        read the pkl files and write them with siam embedings
        """
        for f in os.listdir(path_files):
            print("Reading dialogs dataset from file '%s'" % f)
            with open(os.path.join(path_files, f), 'rb') as handle:
                data = pickle.load(handle)

            df = self._get_Siam_embeddings(pd.DataFrame(data['data']), w_b_emb)
            if (not w_b_emb) & ('text_embed' in df.columns):
                df = df.drop(axis=1, columns='text_embed')
            data['data'] = df
            with open(os.path.join(path_files, f), 'wb') as handle:
                pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print("Writing  dialogs dataset with embedings  to file '%s'" % f)
                
        
class SiamSimilaraty(torch.nn.Module):
    def __init__(self,  hid_size=768):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(SiamSimilaraty, self).__init__()
        
        self.linear1 = torch.nn.Linear(768, hid_size)
        
        

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        # First linear layer
        net = self.linear1(x)
        
        
        return net
        
    def fit(self, train_data, num_epochs=10, batch_size=32, val_data=None, early_stop=[5, 1e-4]):

        """
        to train the Pytorch Siam model
        train data has to has 3 lists of data [ancor, positive, negative]
        """
        # Construct our loss function and an Optimizer. Training this strange model with
        # vanilla stochastic gradient descent is tough, so we use momentum
        criterion = torch.nn.TripletMarginLoss(reduction='sum', margin=1)
        opt = torch.optim.Adam(self.parameters(), lr=1e-3, betas=[0.8, 0.95], weight_decay=0.1)
        # Decay LR by a factor of 0.1 every 7 epochs
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=3, gamma=0.5)
        
        # Choice the device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self = self.to(device)
    
        best_model_wts = copy.deepcopy(self.state_dict())
        best_loss = np.inf
        count = 0
    
        if early_stop:
            mode_loss = []
    
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch+1, num_epochs))
            print('-' * 10)
        
            running_loss = 0.0

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    scheduler.step()
                    self.train()  # Set model to training mode
                
                    for i in range(0, len(train_data[0]), batch_size):
                        input1 = torch.tensor(train_data[0][i:i+batch_size]).to(device)
                        input2 = torch.tensor(train_data[1][i:i+batch_size]).to(device)
                        input3 = torch.tensor(train_data[2][i:i+batch_size]).to(device)
                    
                        if input1.shape[0]<=1 or input2.shape[0]<=1 or input3.shape[0]<=1:
                            continue
                    
                        # zero the parameter gradients
                        opt.zero_grad()
                
                        # forward
                        # track history if only in train
                        with torch.set_grad_enabled(phase == 'train'):
                        
                            output1 = self(input1)
                            output2 = self(input2)
                            output3 = self(input3)
                        
                            loss = criterion(output1, output2, output3)
                      
                            loss.backward()
                            opt.step()    # Does the update
                 
                        # statistics
                        running_loss += loss.item()  
                    epoch_loss = running_loss / len(train_data[0])
                    print('{} Loss: {:10.4e} '.format( phase, epoch_loss))

                    
                else:
                    self.eval()   # Set model to evaluate mode
                
                    for i in range(0, len(val_data[0]), batch_size):
                        input1 = torch.tensor(val_data[0][i:i+batch_size]).to(device)
                        input2 = torch.tensor(val_data[1][i:i+batch_size]).to(device)
                        input3 = torch.tensor(val_data[2][i:i+batch_size]).to(device)
                    
                        if input1.shape[0]<=1 or input2.shape[0]<=1 or input3.shape[0]<=1:
                            continue
                
                    
                        # forward
                        output1 = self(input1)
                        output2 = self(input2)
                        output3 = self(input3)
                    
                        loss = criterion(output1, output2, output3)
                    
                        # statistics
                        running_loss += loss.item() 
                    
                    epoch_loss = running_loss / len(val_data[0])
                    print('{} Loss: {:10.4e} '.format( phase, epoch_loss))
                
                    mode_loss.append(epoch_loss)
                    
                    # deep copy the model
                    if phase == 'val' and epoch_loss < (best_loss - early_stop[1]):
                        best_loss = epoch_loss
                        count = 0
                        best_model_wts = copy.deepcopy(self.state_dict())
                    else:
                        count +=1
                    
            if early_stop  and epoch > early_stop[0]:
                if count > early_stop[0] or (np.array(mode_loss[-early_stop[0]:]).std() < early_stop[1]):
                    print ('Early Stop')
                    break
            print()

        print('Best val Acc: {:10.4e}'.format(best_loss))

        # load best model weights
        self.load_state_dict(best_model_wts)
        
def get_free_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return np.argmax(memory_available)
    


if __name__ =='main':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
      '--data',
      type=str,
      default = '/data/datasets/fb_test/subfolder3',
      help='Path to data for train as pkl')
     
    parser.add_argument(
      '--model',
      type=str,
      default = 'siam_trained.pkl',
      help='Path to trained model')
      
    kwargs, unparsed = parser.parse_known_args()
      
    mod_train = SiamModelTrainer(kwargs.model, path_to_lang_model='lid.176.bin', device='cpu')
    mod_train.write_embedings_file(kwargs.data)
    
    
