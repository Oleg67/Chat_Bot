import re
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import json



def pattern_srt(s, p, patterns):
    tmp = []
    for w in patterns:
        n_s = re.sub(p, w, s)
        tmp.append(n_s)
    return tmp
    
        
def split_placehojds(text):
    """
    split the text by placeholds
    """
    
    _int = []
    _parts = []
    a = (0, 0)
    for i,s in enumerate(text):
        if s == "<":
            a= i
        elif s == ">":
            _int.append((a, i))

    for i,j in _int:
        _parts.append(text[i:j+1])
                      
    return _parts  
    
def create_patterns(df):
    
    df_new = pd.DataFrame()
    _df = df.copy()
    empty_group = set()
    while len(_df):
        try:
            df_new = df_new.append(_df[_df['text'].str.find('<') == -1])
            df_new = df_new.reset_index(drop=True)
            _df = _df[_df['text'].str.find('<') != -1]
        except AttributeError:
            print("ERROR")
            return _df
        print('df_new', df_new.shape, '_df_old', _df.shape)
        l_tmp = []
        l_group = []
        for _, r in tqdm(_df.iterrows()):
            # iteration over patterns in the textdf
            tm = []
            s = r['text']
            for w in split_placehojds(s):
                empty_group.add(w)
                if len(tm):
                    _tmp = []
                    for s in tm:
                        l_patterns = df_new.loc[df_new['group'] == w, 'text'].tolist()
                        _tmp += pattern_srt(s, w, l_patterns)
                            
                    tm = _tmp
                else:
                    l_patterns = df_new.loc[df_new['group'] == w, 'text'].tolist()
                    tm = pattern_srt(s, w, l_patterns)
                    
            tg = [r['group']] * len(tm) 
            l_group += tg 
            l_tmp += tm
        _df = pd.DataFrame()
        #print('tmp', len(l_tmp), 'group', len(l_group))
        _df['text'] = l_tmp
        _df['group'] = l_group
        
    return df_new.reset_index(drop=True), empty_group.difference(set(df_new['group'].unique()))
    
    
def create_triggers(df_triggers, df_patterns):
    
    l_tmp = []
    l_group = []
    empty = set()
    for _, r in tqdm(df_triggers.iterrows()):
        # iteration over patterns in the textdf
        tm = []
        s = r['text']
        l_w = split_placehojds(s)
        if len(l_w) == 0:
            tm.append(s)
        else:
            for w in l_w:
                if len(tm):
                    empty.add(w)
                    _tmp = []
                    for s in tm:
                        l_patterns = df_patterns.loc[df_patterns['group'] == w, 'text'].tolist()
                        _tmp += pattern_srt(s, w, l_patterns)
                        
                    tm = _tmp
                else:
                    l_patterns = df_patterns.loc[df_patterns['group'] == w, 'text'].tolist()
                    tm = pattern_srt(s, w, l_patterns)

        tg = [r['group']] * len(tm) 
        l_group += tg 
        l_tmp += tm
        df = pd.DataFrame()
        #print('tmp', len(l_tmp), 'group', len(l_group))
        df['text'] = l_tmp
        df['group'] = l_group
        
        
    return df.reset_index(drop=True), empty.difference(set(df_patterns['group'].unique()))
    
    
    
def split_text(text):
    """
    split text by words or placeholds
    Ex:
    'Could <subject pronoun> give <object gg pronoun>' 
    -> ['Could', '<subject pronoun>', 'give', '<object gg pronoun>']
    'You’re a <liar>'
    ->  ['You’re', 'a', '<liar>']
    ' <Please forgive me>'
    -> ['<please forgive me>']
    'Please forgive me'
    -> ['please', 'forgive', 'me']
    """
    text = re.sub(r'[?!,.;:&@%/*]',' ', text) # del all punctations
    _text = text.lower().split()
    l_text = []
    f = False
    for i, w in enumerate(_text):
        if (w[0] == '<') & (w[-1] != '>'):
            j = i
            f = True
        elif (w[-1] == '>') & (w[0] != '<') & f:
            l_text.append( " ".join(_text[j: i+1]) )
            f = False
        elif not f:
            l_text.append(w)
    return l_text
    
    
def replace_digits(text):
    """
    replace digits by placehold
    """
    _text = re.sub(r'[?!,.;:&@%(){}<>/*]',' ', text) # del all punctations
    l_text = _text.lower().split()
    for i, w in enumerate(l_text):
        try:
            w = int(w)
            l_text[i] = 'int_number'
        except ValueError:
            continue    
    return l_text  
    
    
def check_patterns(l_patterns, l_text, k):
    """
    check if there is the any pattern from list  in text on k place
    l_patterns = list of patterns 
    l_text  = string as a list
    k index of l_text to compare with the pattern
    return the next number if the pattern was found 
    or -1 if not
    """
    #print('check_patterns', k)
    for p in l_patterns:
        n = len(p) # length of pattern
        #print(n, p)
        try:
            s = l_text[k: k+n] # slice of text the same length as the pattern
        # if text is the smaller length go to the next pattern 
        except IndexError:
            continue
        #print('p',p)
        #print('s',s)
        if p == s:
            # if pattern was found return the index of text where patterns end
            return k+n
        
    # if pattern was not found return -1
    return -1
    
    
def check_placehold(s):
    """
    check the string is the placehold
    """
    if (s[0] == '<') & (s[-1] == '>'):
        return True
    return False
    
    
    
def check_trigger_phraze(trigger, l_text, df):
    """
    check the trigger phraze in text
    """
    #print('check_trigger_phraze')
    # only triggers phrazes which no longer as the text
    if len(trigger) > len(l_text):
        return False
    
    for i, t in enumerate(trigger):
        if check_placehold(t):
            _t = df.get(t, [])
        else:
            _t = [[t]]
        #print(_t, i)
        if i == 0:
            # find the start of trigger phraze
            l_ind = [] # list of starts the trigger phraze
            for k, _ in enumerate(l_text):
                _k = check_patterns(_t, l_text, k)
                if _k != -1:
                    l_ind.append(_k)
            #print(l_ind)
        else:
            _l_ind = []
            for j,k in enumerate(l_ind):
                _k = check_patterns(_t, l_text, k)
                if _k != -1:
                    _l_ind.append(_k)
            if len(_l_ind) < 1:
                return False
            l_ind = _l_ind
            #print(_l_ind)
    return len(l_ind) > 0    
    

            
    
class Triggers():
    
    def __init__(self, placeholds='placeholds.json', triggers='triggers.csv'):
        """
        placeholds - file name with placeholds
        triggers - file with triggers phrazes
        """
        ## read and prepare placeholds set
        with open('placeholds.json', 'r') as fp:
            d = json.load(fp)
        self.placeholds = d
        
        ## read adn prepare triggers phrazes
        df_triggers = pd.read_csv(triggers, index_col=[0])
        df_triggers['text'] = df_triggers['text'].str.lower()
        df_triggers['group'] = df_triggers['group'].str.lower()
        df_triggers['triggers_list'] = df_triggers['text'].apply(split_text)
        df_triggers['n_placeholds'] = df_triggers['text'].apply(lambda x: len(re.findall(r'<', x)))
        df_triggers['len_phraze'] = df_triggers['triggers_list'].apply(len)
        df_triggers = df_triggers.sort_values(by=['n_placeholds', 'len_phraze'])
        df_triggers.drop(axis=1, columns='group', inplace=True)
        self.trigger = np.array(df_triggers['triggers_list'].tolist())
        self.length = np.array(df_triggers['len_phraze'].tolist())
        print('Read placeholds  from {} and triggers phrazes from {}'.format(placeholds, triggers))
        
        pl_set = set()
        for r in self.trigger:
            for w in r:
                if check_placehold(w):
                    pl_set.add(w)
        
        print('Empty placeholds', pl_set.difference(d.keys()))
        
        
    def find_pattern(self, text):
        """
        serve to find the triger pattens in text
        return True if triget was found & text
               False if triger was not found & text
        """
        l_text = replace_digits(text)
        n_max = len(l_text)
        
        
        for tr in self.trigger[self.length <= n_max]:
            #print(tr)
            f = check_trigger_phraze(tr, l_text, self.placeholds)
            #print(ts)
            #print(f)
            if f:
                print(tr)
                break
        return f, text
        
             
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
      '--data',
      type=str,
      default = 'Trigger phrasesv2.xlsx',
      help='Path to data for train as xlsx')  
    
    parser.add_argument(
      '--out_placeholds',
      type=str,
      default = 'placeholds.json',
      help='Path to out data of placeholds as json')  
      
    parser.add_argument(
      '--out_trigger',
      type=str,
      default = 'triggers.csv',
      help='Path to out data of triggers phrazes as csv')  
    
    
      
    kwargs, unparsed = parser.parse_known_args()

    df = pd.read_excel(kwargs.data, sheet_name='Sheet1', names=['text', 'group'])
    # creating patterns to replace the placeholds
    df_patterns, empty = create_patterns(df[df['group'] !='<trigger phrases>'])
    
    # creating placeholds dictionary
    df_patterns['text'] = df_patterns['text'].str.lower()
    df_patterns['group'] = df_patterns['group'].str.lower()
    df_patterns['l_text'] = df_patterns['text'].apply(split_text)
    df_patterns['length'] = df_patterns['l_text'].apply(len)
    df_patterns = df_patterns.sort_values(by='length')
    print('Create placeholds dictionary')
    d = {}
    for _,r in tqdm(df_patterns[['group', 'l_text']].iterrows()):
        d.setdefault(r['group'], []).append(r['l_text'])
        
    with open(kwargs.out_placeholds, 'w') as fp:
        json.dump(d, fp) 
 
    # writing the triggers phrazes
    df[df['group'] =='<trigger phrases>'].to_csv(kwargs.out_trigger)
    
    print('Write triggers phrazes to {} and placeholds to {}'.format(kwargs.out_trigger, kwargs.out_placeholds))
    print('Empty group', empty)
 
