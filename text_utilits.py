import re
import zipfile
import os
import pandas as pd

def is_time_stamp(l):
    if l[:2].isdigit() and l[2] == ':':
        return True
    return False

def has_letters(line):
    if re.search(r'[a-zA-Z]', line):
        return True
    return False

def has_no_text(line):
    l = line.strip()
    if not len(l):
        return True
    if l.isdigit():
        return True
    if is_time_stamp(l):
        return True
    if l[0] == '(' and l[-1] == ')':
        return True
    if not has_letters(line):
        return True
    return False

def is_lowercase_letter_or_comma(letter):

    if letter.isalpha() and letter.lower() == letter:
        return True
    if letter == ',':
        return True
    return False

def clean_up(lines):
    """
    Get rid of all non-text lines and
    try to combine text broken into multiple lines
    """
    new_lines = []
    for line in lines[1:]:
        if type(line) is not str:
            line = line.decode()
        line = re.sub('<br/>', ' ', line)
        line = re.sub("[\(\[].*?[\)\]]", "", line)
        if has_no_text(line):
            continue
        elif len(new_lines) and is_lowercase_letter_or_comma(line[0]):
            #combine with previous line
            new_lines[-1] = new_lines[-1].strip() + ' ' + line
        else:
            #append line
            line = line.lstrip('-').lstrip()
            if line[0] == '<':
                continue 
            new_lines.append(line)
    return new_lines

def read_zipSRT_to_TXT_list(file_name, write_path=None, verbose=False):
    """
    file name is zip file
    write_path is path to write the txt files
    """
    if not write_path:
        path = os.path.join(os.getcwd(), 'data', file_name[:-4])
    else:
        path = os.path.join(os.getcwd(), 'data', write_path)
    txt_list = []
    zf = zipfile.ZipFile(file_name)
    for i,f_name in enumerate(zf.namelist()):
        f = zf.open(f_name)
        lines = f.readlines()
        new_lines = clean_up(clean_up(lines))
        new_file_name = os.path.split(f_name)[-1][:-4] + '.txt'
        new_file_name = os.path.join(path, new_file_name)
        if not os.path.exists(path):
            os.makedirs(path)
        with open(new_file_name, 'w') as f:
            for line in new_lines:
                f.write(line)
        if verbose:
            print ('write text to file', new_file_name)
        txt_list.append(new_file_name)
    return txt_list
    
def read_listTXT_to_CSV(file_name_list, name=None, mode='dialog', answer=1, return_df=False, verbose=False):
    """
    list of txt files to csv file 
    file_name_list - list of txt files
    name - name of output csv file
    mode - if dialog use all text as questions and answers with shift 1
           if None or other all odd sequences are as questions, and all even sequences are as answers
    answer - from such number begain the first question
    return_df - ruturn DataFrame 
    verbose - flag debuging
    """
    df = pd.DataFrame()
    
    for file_name in file_name_list:
        df_temp = pd.DataFrame()
        with open(file_name, 'r') as f:
            lines = f.readlines()
        if verbose:
            print (file_name)
            print (len(lines))
        if mode == 'dialog':
            if len(lines)%2 ==0:
                lines = lines[:-1]
            
            df_temp['question'] = lines[answer-1:-1]
            df_temp['answer'] = lines[answer:]
        else:
            if len(lines)%2 !=0:
                lines = lines[:-1]
            df_temp['question'] = lines[answer-1:-1][::2]
            df_temp['answer'] = lines[answer:][::2]

            
        df = df.append(df_temp)
            
        if verbose:
            print (df.shape)
    df['question'] = df['question'].str.rstrip('\n')
    df['answer'] = df['answer'].str.rstrip('\n')
        
    df.index = range(1, len(df)+1)
    if not name:
        df.to_csv(file_name_list[0].split('/')[-2] +'.csv')
    else:
        df.to_csv(name+'.csv')
    return df if return_df else None
