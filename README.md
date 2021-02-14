# MCSP
Malware classification using Simhash encoding and PCA (MCSP) paper repository by Jaeju An



# 데이터셋
데이터셋 이름: Microsoft Malware Classification Challenge (BIG 2015)

url: https://www.kaggle.com/c/malware-classification/data

설명: 마이크로 소프트에서 2015년 공개한 바이러스 데이터셋이다. 해당 데이터셋은 9개의 클래스로 구성되어 있으며, 두 가지의 버전(bytes, asm)으로 제공되는 바이러스 파일이다. 이 중 asm파일 형태를 사용하였다.



# 전처리
전처리 과정은 개요에서 알 수 있듯이, 각각의 바이러스 파일마다 갖고있는 opcode를 뽑아내 이미지로 변환하였다. 다음은 바이러스 파일을 이미지로 변환하는 과정이다.
1. 바이러스파일로 부터 opcode를 추출한다. (opcode를 추출할 때 비교대상인 opcode mnemonic들은 opcode_mnemonic.csv를 사용하였다.)
2. 추출한 opcode를 simhash algorithm을 사용하여 고정된 크기의 binary 형태로 변환한다.
3. 변환된 bianry 형태를 모두 더해준다. (*이때, 0의 자리는 빼준다.)
4. 0 이하의 값은 0으로, 초과의 값은 1로 변환한다.
5. 이후, binary 형태를 사각형으로 reshape 해준다.

해당 과정을 sudo코드로 작성하면 다음과 같다.

```
file_list = all virus files
opcode_list = []

preprocessed_list = []

for each_file in file_list:
    for each_line in each_file:
        
        if each_line has opcode:
            opcode_list.append(the specific opcode)
        
    # After extracting a opcode seqeunce from a file, we apply simhash algorithm
    the static length of binary code list = SHA-algorithm(opcode_list)
    total = sum(the static length of binary code list)  # ---> axis for column
    
    for idx in range(len(total)):
    
        if total[idx] <= 0:
            total[idx] = 0
        else:
            total[idx] = 1
    
    preprocessed_list.append( reshape( total, row, column ) )
```




# 실행 방법
1. 바이러스 파일로부터 opcode를 추출한 뒤 이미지 파일로부터 변환  
Jupyter Notebook: [1-preprocessing] From extraction to converting  
Python FIle: Utiliy/Preprocessing.py  
  
2. 이미지 파일을 PCA를 통해 데이터 변환  
Jupyter Notebook: [2-converting] apply pca to img data  
Python File: utilty/converting.py  

위의 1단계, 2단계를 통해서 데이터를 준비한 후  
갖가지 Model을 이용해 실험을 진행  
  
모델에 따른 필요 데이터 목록  
1. SHA-768로 생성한 1-gram to 4-gram 이미지 데이터 (PCA 진행 하지 않은 것): Baseline, reluMCSC, tanhMCSC and MCSLT  
2. SHA-768로 생성한 1-gram to 4-gram 이미지 데이터에 PCA를 적용해 512의 크기로 선형 변환된 이미지: MCSP  
