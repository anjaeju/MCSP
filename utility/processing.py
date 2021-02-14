import os
import time
import pickle
import pandas as pd

import re
import numpy as np
import hashlib as h

from skimage import io
from nltk.util import ngrams

def make_opcode_lookup_table(path):
    return set(pd.read_csv(path).apply(lambda x: x.str.lower()).values.reshape(-1))

def make_line_start_lookup_table(path):
    DF = pd.read_excel(path)
    code_section = []
    
    for i in DF.values.reshape(-1):
        section_name, content = i.split('–')
        
        if 'Code Section' in content:
            code_section.append(section_name.replace(' ', '').lower())
    return set(code_section)

def is_code(line, start_text):
    temp = set(line.lower().split(':'))
    temp = temp.intersection(start_text)
    
    if temp == set():
        return False
    
    return True

def make_hash_768(opcode_list, bits, gram):
    
    hash_v = np.zeros([bits[0]])

    if gram == None:
        
        for opcode in opcode_list:
            
            binary1 = bin(int(h.sha256(opcode.encode()).hexdigest(), 16))[2:]
            binary1 = list(binary1)
            temp = bits[1] - len(binary1)
            for i in range(temp):
                binary1.insert(0, '0')

            binary2 = bin(int(h.sha512(opcode.encode()).hexdigest(), 16))[2:]
            binary2 = list(binary2)
            temp = bits[2] - len(binary2)
            for i in range(temp):
                binary2.insert(0, '0')

            binary = binary2 + binary1             # SHA-512 + SHA-256
            binary = np.array(list(map(lambda x: -1 if x == '0'else 1, binary)))
            hash_v = hash_v + binary
    
    elif gram == 2:
        
        opcode_ngram_tuple = list(ngrams(opcode_list, n=gram))
        
        for word1, word2 in opcode_ngram_tuple:
            
            opcode = word1+word2
            
            binary1 = bin(int(h.sha256(opcode.encode()).hexdigest(), 16))[2:]
            binary1 = list(binary1)
            temp = bits[1] - len(binary1)
            for i in range(temp):
                binary1.insert(0, '0')

            binary2 = bin(int(h.sha512(opcode.encode()).hexdigest(), 16))[2:]
            binary2 = list(binary2)
            temp = bits[2] - len(binary2)
            for i in range(temp):
                binary2.insert(0, '0')

            binary = binary2 + binary1             # SHA-512 + SHA-256
            binary = np.array(list(map(lambda x: -1 if x == '0'else 1, binary)))
            hash_v = hash_v + binary

    elif gram == 3:
        opcode_ngram_tuple = list(ngrams(opcode_list, n=gram))
        
        for word1, word2, word3 in opcode_ngram_tuple:
            
            opcode = word1+word2+word3
            
            binary1 = bin(int(h.sha256(opcode.encode()).hexdigest(), 16))[2:]
            binary1 = list(binary1)
            temp = bits[1] - len(binary1)
            for i in range(temp):
                binary1.insert(0, '0')

            binary2 = bin(int(h.sha512(opcode.encode()).hexdigest(), 16))[2:]
            binary2 = list(binary2)
            temp = bits[2] - len(binary2)
            for i in range(temp):
                binary2.insert(0, '0')

            binary = binary2 + binary1             # SHA-512 + SHA-256
            binary = np.array(list(map(lambda x: -1 if x == '0'else 1, binary)))
            hash_v = hash_v + binary
            
    elif gram == 4:
        opcode_ngram_tuple = list(ngrams(opcode_list, n=gram))
        
        for word1, word2, word3, word4 in opcode_ngram_tuple:
            
            opcode = word1+word2+word3+word4
            
            binary1 = bin(int(h.sha256(opcode.encode()).hexdigest(), 16))[2:]
            binary1 = list(binary1)
            temp = bits[1] - len(binary1)
            for i in range(temp):
                binary1.insert(0, '0')

            binary2 = bin(int(h.sha512(opcode.encode()).hexdigest(), 16))[2:]
            binary2 = list(binary2)
            temp = bits[2] - len(binary2)
            for i in range(temp):
                binary2.insert(0, '0')

            binary = binary2 + binary1             # SHA-512 + SHA-256
            binary = np.array(list(map(lambda x: -1 if x == '0'else 1, binary)))
            hash_v = hash_v + binary
            
    return hash_v


"""if __name__ == '__main__':
    ## 2. opcode 룩업 테이블/asm파일의 시작점 룩업 테이블 준비
    save_path = 'D:/virus/opcode/'
    file_path = 'D:/virus/asm_data/'

    empty = set()

    op_lookup_table = make_opcode_lookup_table()
    start_text = make_line_start_lookup_table()
    start_text.add('.tixt')
    start_text.add('.icode')
    start_text.add('seg')

    no_opcode_file = []
    file_list = os.listdir(file_path)
    
    ## 3. 각 파일로부터 opcode sequence를 추출하고 opcode sequence를 save_path에 저장
    check = lambda x: True if not start_text[0] in x and not start_text[1] in x else False
    for num, file in enumerate(file_list):

        file_opcode = []

        for line in open(file_path+file,'r',encoding='ISO-8859-1'):

            if not is_code(line, start_text):
                continue

            words = line.split()
            words = set(words)

            opcode = words.intersection(op_lookup_table)

            if opcode == empty:
                continue

            else:
                file_opcode.append(opcode.pop())
                continue

        if file_opcode == []:
            no_opcode_file.append(file)
            print(f"{file} doesn't have any opcode in CODE or TEXT section")
            continue

        file_name = file.replace('.asm', '.pkl')

        with open(save_path+file_name, 'wb') as fp:
            pickle.dump(file_opcode, fp)

        if num% 1000 == 0:
            print('Now processing the {}th file'.format(num+1))
            
            
    ## 4. opcode sequence를 읽어 들인 후, 이미지로 바꾼다음 저장
    bits = [768, 256, 512]
    bits1 = 256
    bits2 = 512

    width = 24
    height = 32
    save_path ='D:/virus/image/1gram_768/'

    start_time = time.time()
    make_img_768(bits, width, height, save_path)  # default: 1garm
    
    print(f'Took: {time.time() - start_time}s seconds')"""