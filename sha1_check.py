#数据下载
import hashlib
import pandas as pd

def sha1_check(filename,DATA_HUB):
    if filename not in DATA_HUB:
        raise ValueError(f"{filename} does not exist in {DATA_HUB}")
    
    sha1_hash=DATA_HUB[filename][1]
    
    sha1=hashlib.sha1()
    with open(DATA_HUB[filename][0],'rb') as f:
        while True:
            data=f.read(1048576)
            if not data:
                break
            sha1.update(data)
            
        if sha1.hexdigest()!=sha1_hash:
            raise ValueError(f"{filename} is downloaded but the hash value does not match. The file might be corrupted. You can try to delete the file and download it again.")
    
    return True


    
    