import os


def find_NewFile(path):
    lists = os.listdir(path)
    lists.sort(key=lambda x:os.path.getmtime(path +'/'+x))
    file_new = os.path.join(path,lists[-1])
    return file_new


if __name__ =='__main__':
    newfile = find_NewFile('../checkpoints/unbias_fusion_model')
    print(newfile)