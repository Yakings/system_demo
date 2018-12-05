
import zipfile
import os



file_list = os.listdir(r'.')
def unzip_func(file_name,extract_path):
    name_lsit = os.path.splitext(file_name)
    if name_lsit[-1] == '.zip':
        print(file_name)
        file_zip = zipfile.ZipFile(file_name, 'r')
        for file in file_zip.namelist():
            # file_list = file.split('/')
            # if '.csv' in file_list[-1]:
            file_zip.extract(file, extract_path)
        file_zip.close()
        # os.remove(file_name)
        print('finished unzip')
    else:
        print('unzip error!')




if __name__=='__main__':
    unzip_func('./data/Bearing1.zip','./data/aaaa/')
