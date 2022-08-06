
#---------------------------------------------------
#读取yolo.cfg配置文件
#---------------------------------------------------
def parse_model_config(path):
    """Parses the yolo-v3 layer configuration file and returns module definitions"""
    file = open(path, 'r',encoding = "utf-8")
    lines = file.read().split('\n')
    #去掉空行以及注释
    lines = [x for x in lines if x and not x.startswith('#')]
    #去掉左右空格
    lines = [x.rstrip().lstrip() for x in lines] 
    module_defs = []
    #将每一个结构名和参数储存到一个字典中，最后都储存在一个列表中
    for line in lines:
        #取出结构名称
        if line.startswith('['): 
            module_defs.append({})
            module_defs[-1]['type'] = line[1:-1].rstrip()
            if module_defs[-1]['type'] == 'convolutional':
                module_defs[-1]['batch_normalize'] = 0
        #取出结构定义的参数
        else:
            key, value = line.split("=")
            value = value.strip()
            module_defs[-1][key.rstrip()] = value.strip()

    return module_defs

#读取coco.date
def parse_data_config(path):
    """Parses the data configuration file"""
    options = dict()
    options['gpus'] = '0,1,2,3'
    options['num_workers'] = '10'
    with open(path, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.strip()
        if line == '' or line.startswith('#'):
            continue
        key, value = line.split('=')
        options[key.strip()] = value.strip()
    return options
