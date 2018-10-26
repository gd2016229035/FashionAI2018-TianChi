# coding: utf-8

import os, shutil, random

# Read label.csv
# For each task, make folders, and copy picture to corresponding folders

label_train_dir = './data2/train/Annotations/label.csv'
label_base_dir = './data1/base/Annotations/label.csv'
label_dir_testa = './data1/fashionAI_attributes_answer_a_20180428.csv'
label_dir_testb = './data1/fashionAI_attributes_answer_b_20180428.csv'

label_dict = {'coat_length_labels': [],
              'lapel_design_labels': [],
              'neckline_design_labels': [],
              'skirt_length_labels': [],
              'collar_design_labels': [],
              'neck_design_labels': [],
              'pant_length_labels': [],
              'sleeve_length_labels': []}

task_list = label_dict.keys()

def mkdir_if_not_exist(path):
    if not os.path.exists(os.path.join(*path)):
        os.makedirs(os.path.join(*path))

all_lable = {}
with open(label_train_dir, 'r') as f:
    lines = f.readlines()
    tokens = [l.rstrip().split(',') for l in lines]
    for path, task, label in tokens:
        if path in all_lable:
            continue
        else:
            all_lable.setdefault(path,[])
            path1='data2/train/'+path
            label_dict[task].append((path1, label))

with open(label_base_dir, 'r') as f:
    lines = f.readlines()
    tokens = [l.rstrip().split(',') for l in lines]
    for path, task, label in tokens:
        if path in all_lable:
            continue
        else:
            all_lable.setdefault(path,[])
            path1='data1/base/'+path
            label_dict[task].append((path1, label))

with open(label_dir_testa, 'r') as f:
    lines = f.readlines()
    tokens = [l.rstrip().split(',') for l in lines]
    for path, task, label in tokens:
        if path in all_lable:
            continue
        else:
            all_lable.setdefault(path,[])
            path1='data1/rank/'+path
            label_dict[task].append((path1, label))

with open(label_dir_testb, 'r') as f:
    lines = f.readlines()
    tokens = [l.rstrip().split(',') for l in lines]
    for path, task, label in tokens:
        if path in all_lable:
            continue
        else:
            all_lable.setdefault(path,[])
            path1='data1/z_rank/'+path
            label_dict[task].append((path1, label))

mkdir_if_not_exist(['data2/train_valid_allset'])

for task, path_label in label_dict.items(): 
    mkdir_if_not_exist(['data2/train_valid_allset',  task])
    train_count = 0 # 对每一类都要重新置0
    n = len(path_label) # 每个task有多少条数据
    m = len(list(path_label[0][1])) # 每个task有几类

    for mm in range(m):
        mkdir_if_not_exist(['data2/train_valid_allset', task, 'train', str(mm)])
        mkdir_if_not_exist(['data2/train_valid_allset', task, 'val', str(mm)])
        
    random.seed(2018)
    random.shuffle(path_label)
    for path, label in path_label:
        label_index = list(label).index('y')
        # if 'm' in label:    # 如果存在m标签,就寻找m标签
        #     m_index = list(label).index('m')
        #     label_index = label_index if random.randint(1,5)>2 else m_index # 60%选择y对应的label,40%选择m对应的label
        src_path = os.path.join(path)
        if train_count < n * 0.95:
            shutil.copy(src_path,
                        os.path.join('data2/train_valid_allset', task, 'train', str(label_index)))
        else:
            shutil.copy(src_path,
                        os.path.join('data2/train_valid_allset', task, 'val', str(label_index)))
        train_count += 1
    print( ' finished ' + task)
print( ' all finished!')


# Add warmup data to skirt task

# label_dict = {'skirt_length_labels': []}
#
# with open(warmup_label_dir, 'r') as f:
#     lines = f.readlines()
#     tokens = [l.rstrip().split(',') for l in lines]
#     for path, task, label in tokens:
#         label_dict[task].append((path, label))
#
# for task, path_label in label_dict.items():
#     train_count = 0
#     n = len(path_label)
#     m = len(list(path_label[0][1]))
#
#     random.shuffle(path_label)
#     for path, label in path_label:
#         label_index = list(label).index('y')
#         src_path = os.path.join('data/web', path)
#         if train_count < n * 0.9:
#             shutil.copy(src_path,
#                         os.path.join('data/train_valid', task, 'train', str(label_index)))
#         else:
#             shutil.copy(src_path,
#                         os.path.join('data/train_valid', task, 'val', str(label_index)))
#         train_count += 1

