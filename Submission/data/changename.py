import os 
import glob 
# 이미지폴더 내부에 S_DRR이 있는 파일을 S_DRG로 변경
cnt=0
for path in  glob.glob('/workspace/data/*/images/*.jpg'):
    name = path.split("/")[-1]
    if 'S_DRR' in name:
        print("There is filename erros")
        new_path = path.replace("S_DRR",'S_DRG')
        cnt+=1
        os.rename(path,new_path)
print(f"{cnt} images chaned filename!")
