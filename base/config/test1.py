# import os
# base_dir = os.path.abspath(os.path.join(os.getcwd(),'../..'))+'/'
# print(base_dir)
# strs= (1, 2, 3,4)

# print('strs= %s' % strs,)
src_data_path = "machine_translation/data/IWSLT2014de-en/data/{}.de-en.de"
type = "test"
aaa = src_data_path.format(type)
print(aaa)