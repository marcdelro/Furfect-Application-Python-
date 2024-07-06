import os

folderPath = '.'
fileNum = 0

for filename in os.listdir(folderPath):
    if '.jpg' in filename:
        os.rename(filename, str(fileNum) + '.jpg')
        fileNum += 1