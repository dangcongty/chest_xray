from glob import glob

with open('datasets/test.txt', 'w') as f:
    for path in glob('datasets/test/images/*'):
        f.write(path+'\n')
