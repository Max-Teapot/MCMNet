import os

# 获取文件夹下最新的文件

def find_NewFile(path):
    #获取文件夹中的所有文件
    lists = os.listdir(path)
    #对获取的文件根据修改时间进行排序
    lists.sort(key=lambda x:os.path.getmtime(path +'/'+x))
    #把目录和文件名合成一个路径
    file_new = os.path.join(path,lists[-1])
    return file_new


if __name__ =='__main__':
    newfile = find_NewFile('../checkpoints/unbias_fusion_model')
    print(newfile)