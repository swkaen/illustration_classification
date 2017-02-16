from zipfile import ZipFile
from glob import glob

file_list = glob("datasets/*.zip")
for file in file_list:
    print("unzipping" + file)
    with ZipFile("./" + file, 'r') as z:
        z.extractall('./datasets/')

