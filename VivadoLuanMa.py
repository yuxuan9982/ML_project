'''
遍历目录下所有文件，将UTF-8编码的verilog文件转换为GBK编码。
'''

import chardet
import os
import codecs

file_path = r'E:\desktop\sb\Lab4\Lab4.srcs\sources_1\new'


def findAllFile(base):
    for root, ds, fs in os.walk(base):
        for f in fs:
            fullname = os.path.join(root, f)
            yield fullname


def det_encoding(file_path):
    with open(file_path, 'rb') as f:
        s = f.read()
        chatest = chardet.detect(s)
    return chatest['encoding']


def convert(file, in_enc="GBK", out_enc="UTF-8"):
    """
    该程序用于将目录下的文件从指定格式转换到指定格式，默认的是GBK转到utf-8
    :param file:    文件路径
    :param in_enc:  输入文件格式
    :param out_enc: 输出文件格式
    :return:
    """
    in_enc = in_enc.upper()
    out_enc = out_enc.upper()
    try:
        print("convert [ " + file.split('\\')[-1] + " ].....From " + in_enc + " --> " + out_enc)
        f = codecs.open(file, 'r', in_enc)
        new_content = f.read()
        codecs.open(file, 'w', out_enc).write(new_content)
    # print (f.read())
    except IOError as err:
        print("I/O error: {0}".format(err))


file_list = findAllFile(file_path)
for i in file_list:
    if (i.split('.')[-1] == 'v'):  # verilog file
        file_encoding = det_encoding(i)
        print('{}: {}'.format(file_encoding, i))
        if (file_encoding == 'utf-8'):
            convert(i, in_enc='utf-8', out_enc='GBK')
        if(file_encoding=='Windows-1254'):
            convert(i,in_enc='Windows-1254',out_enc='GBK')