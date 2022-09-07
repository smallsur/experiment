import requests
import os
import sys
def download_from_url(url, path):
    """Download file, with logic (from tensor2tensor) for Google Drive"""
    if 'drive.google.com' not in url:
        print('Downloading %s; may take a few minutes' % url)
        r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        with open(path, "wb") as file:
            file.write(r.content)
        return
    print('Downloading from Google Drive; may take a few minutes')
    confirm_token = None
    session = requests.Session()
    response = session.get(url, stream=True)
    for k, v in response.cookies.items():
        if k.startswith("download_warning"):
            confirm_token = v

    if confirm_token:
        url = url + "&confirm=" + confirm_token
        response = session.get(url, stream=True)

    chunk_size = 16 * 1024
    with open(path, "wb") as f:
        for chunk in response.iter_content(chunk_size):
            if chunk:
                f.write(chunk)
def del_cache():
    path_ = os.getcwd()
    path_cache = []
    for root,dirs,files in os.walk(path_): 
        # print(root)  #遍历path,进入每个目录都调用visit函数，，有3个参数，root表示目录路径，dirs表示当前目录的目录名，files代表当前目录的文件名
        # for dir in dirs:
        #     #print(dir)             #文件夹名                 
        #     print(os.path.join(root,dir))
        
        # for file in files:
        #     print(os.path.join(root,file))
        # print(dirs)
        # print(files)
        for dir in dirs:
            if dir=="__pycache__":
                path_cache.append(os.path.join(root,dir))
    
    for dir in path_cache:
        for file in os.listdir(dir):
            os.remove(os.path.join(dir,file))

    

# if __name__ == '__main__':
#     del_cache()
