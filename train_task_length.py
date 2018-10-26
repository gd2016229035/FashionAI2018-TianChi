# coding: utf-8
import mxnet as mx
import numpy as np
import os, time, logging, math, argparse
from mxnet.gluon.data.vision import transforms
from mxnet import gluon, image, init, nd
from mxnet import autograd as ag
from mxnet.gluon import nn
from mxnet.gluon.model_zoo import vision as models
import random


mAP_thresh_step = -0.0001
mAP_output_dir = 'log/'

def parse_args():
    parser = argparse.ArgumentParser(description='Gluon for FashionAI Competition',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--task', required=True, type=str,
                        help='name of the classification task')
    parser.add_argument('--model', required=True, type=str,
                        help='name of the pretrained model from model zoo.')
    parser.add_argument('-j', '--workers', dest='num_workers', default=4, type=int,
                        help='number of preprocessing workers')
    parser.add_argument('--num-gpus', default=0, type=int,
                        help='number of gpus to use, 0 indicates cpu only')
    parser.add_argument('--epochs', default=40, type=int,
                        help='number of training epochs')
    parser.add_argument('-b', '--batch-size', default=64, type=int,
                        help='mini-batch size')
    parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', dest='wd', default=1e-4, type=float,
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--lr-factor', default=0.1, type=float,
                        help='learning rate decay ratio')
    parser.add_argument('--lr-steps', default='10,20,30', type=str,
                        help='list of learning rate decay epochs as in str')
    args = parser.parse_args()
    return args

def calculate_ap(labels, outputs):
    cnt = 0
    ap = 0.
    for label, output in zip(labels, outputs):
        for lb, op in zip(label.asnumpy().astype(np.int),
                          output.asnumpy()):
            op_argsort = np.argsort(op)[::-1]
            lb_int = int(lb)
            ap += 1.0 / (1+list(op_argsort).index(lb_int))
            cnt += 1
    return ((ap, cnt))

def two_crop(img):
    img_flip = img[:, :, ::-1]
    crops = nd.stack(
        img,
        img_flip,
    )
    return (crops)

def transform_predict(im, size):
    im = im.astype('float32') / 255
    #im = image.resize_short(im, size, interp=1)
    im = image.imresize(im, size, size, interp=1)
    # im = image.resize_short(im, 331)
    im = nd.transpose(im, (2,0,1))
    im = mx.nd.image.normalize(im, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    # im = forty_crop(im, (352, 352))
    im = two_crop(im)
    return (im)


def progressbar(i, n, bar_len=40):
    percents = math.ceil(100.0 * i / float(n))
    filled_len = int(round(bar_len * i / float(n)))
    prog_bar = '=' * filled_len + '-' * (bar_len - filled_len)
    print('[%s] %s%s\r' % (prog_bar, percents, '%'))

def validate(net, val_data, ctx):
    metric = mx.metric.Accuracy()
    L = gluon.loss.SoftmaxCrossEntropyLoss()
    AP = 0.
    AP_cnt = 0
    val_loss = 0
    all_softmax_output = []
    mAP_name = task+model_name+'.npy'
    for i, batch in enumerate(val_data):
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
        #data = transform_predict(data, scale)
        outputs = [net(X) for X in data]    # 将图片输入,得到16X5维的结果
        metric.update(label, outputs)
        loss = [L(yhat, y) for yhat, y in zip(outputs, label)]  # 输出16个数,代表loss
        val_loss += sum([l.mean().asscalar() for l in loss]) / len(loss)    # loss相加求和
        #ap, cnt = calculate_ap(label, outputs)
        # softmax_output和label本身是list，但是softmax_output[0]和label[0]是NDarray格式，注意这个NDarray是mxnet的，不是numpy的NDarray
        softmax_output = [nd.SoftmaxActivation(output) for output in outputs]  # 取softmax,然后对十个结果取平均
        sm_out_label = zip(softmax_output[0].asnumpy(), label[0].asnumpy())
        all_softmax_output += sm_out_label
        np.save(mAP_name, all_softmax_output)
        #AP += ap
        #AP_cnt += cnt
    this_AP = cal_mAP(mAP_name) # 得到当前的AP
    _, val_acc = metric.get()
    return ((this_AP, val_acc, val_loss / len(val_data)))

def train():
    logging.info('Start Training for Task: %s\n' % (task))

    # Initialize the net with pretrained model
    pretrained_net = gluon.model_zoo.vision.get_model(model_name, pretrained=True)

    finetune_net = gluon.model_zoo.vision.get_model(model_name, classes=task_num_class)
    finetune_net.features = pretrained_net.features
    finetune_net.output.initialize(init.Xavier(), ctx = ctx)
    finetune_net.collect_params().reset_ctx(ctx)
    finetune_net.hybridize()

    train_transform = transforms.Compose([
        transforms.Resize(input_scale),
        #transforms.RandomResizedCrop(448,scale=(0.76, 1.0),ratio=(0.999, 1.001)),
        transforms.RandomFlipLeftRight(),
        transforms.RandomBrightness(0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = gluon.data.vision.ImageFolderDataset(os.path.join('data2/train_valid_allset', task, 'train'))
    train_data = gluon.data.DataLoader(train_dataset.transform_first(train_transform),
        batch_size=batch_size, shuffle=True, num_workers=num_workers, last_batch='discard')


    val_transform = transforms.Compose([
        transforms.Resize(input_scale),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_dataset = gluon.data.vision.ImageFolderDataset(os.path.join('data2/train_valid_allset', task, 'val'))
    val_data = gluon.data.DataLoader(val_dataset.transform_first(val_transform),
        batch_size=batch_size, shuffle=False, num_workers = num_workers, last_batch='discard')

    trainer = gluon.Trainer(finetune_net.collect_params(), 'adam', {
        'learning_rate': lr})
    metric = mx.metric.Accuracy()
    L = gluon.loss.SoftmaxCrossEntropyLoss()
    lr_counter = 0
    num_batch = len(train_data)

    # Start Training
    best_AP = 0
    best_acc = 0
    for epoch in range(epochs):
        if epoch == lr_steps[lr_counter]:
            finetune_net.collect_params().load(best_path, ctx= ctx)
            trainer.set_learning_rate(trainer.learning_rate*lr_factor)
            lr_counter += 1

        tic = time.time()
        train_loss = 0
        metric.reset()
        AP = 0.
        AP_cnt = 0

        for i, batch in enumerate(train_data):
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
            label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
            with ag.record():
                outputs = [finetune_net(X) for X in data]
                loss = [L(yhat, y) for yhat, y in zip(outputs, label)]
            for l in loss:
                l.backward()

            trainer.step(batch_size)
            train_loss += sum([l.mean().asscalar() for l in loss]) / len(loss)

            metric.update(label, outputs)
            #ap, cnt = calculate_ap(label, outputs)
            #AP += ap
            #AP_cnt += cnt
            #progressbar(i, num_batch-1)

        #train_map = AP / AP_cnt
        _, train_acc = metric.get()
        train_loss /= num_batch

        this_AP, val_acc, val_loss = validate(finetune_net, val_data, ctx)

        logging.info('[Epoch %d] Train-acc: %.3f, loss: %.3f | Val-acc: %.3f, mAP: %.3f, loss: %.3f | time: %.1f | learning_rate %.6f' %
                 (epoch, train_acc,  train_loss, val_acc, this_AP, val_loss, time.time() - tic, trainer.learning_rate))
        f_val.writelines('[Epoch %d] Train-acc: %.3f, , loss: %.3f | Val-acc: %.3f, mAP: %.3f, loss: %.3f | time: %.1f | learning_rate %.6f\n' %
                 (epoch, train_acc, train_loss, val_acc, this_AP, val_loss, time.time() - tic, trainer.learning_rate))
        if val_acc > best_acc:
            best_AP = this_AP
            best_acc = val_acc
            best_path = './models/%s_%s_%s_%s.params' % (task, model_name, epoch, best_acc)
            finetune_net.collect_params().save(best_path)

    logging.info('\n')
    finetune_net.collect_params().load(best_path, ctx= ctx)
    f_val.writelines('Best val acc is :[Epoch %d] Train-acc: %.3f, loss: %.3f | Best-val-acc: %.3f, Best-mAP: %.3f, loss: %.3f | time: %.1f | learning_rate %.6f\n' %
         (epoch, train_acc, train_loss, best_acc, best_AP, val_loss, time.time() - tic, trainer.learning_rate))
    return (finetune_net)

def predict(task):
    logging.info('Training Finished. Starting Prediction.\n')
    f_out = open('submission/%s.csv'%(task), 'w')
    with open('data2/week-rank/Tests/question.csv', 'r') as f_in:
        lines = f_in.readlines()
    tokens = [l.rstrip().split(',') for l in lines]
    task_tokens = [t for t in tokens if t[1] == task]
    n = len(task_tokens)
    cnt = 0
    for path, task, _ in task_tokens:
        img_path = os.path.join('data2/week-rank', path)
        with open(img_path, 'rb') as f:
            img = image.imdecode(f.read())
        data = transform_predict(img, input_scale)
        with ag.predict_mode():
            out = net(data.as_in_context(mx.gpu(0)))
            out = nd.SoftmaxActivation(out).mean(axis=0)

        pred_out = ';'.join(["%.8f"%(o) for o in out.asnumpy().tolist()])
        line_out = ','.join([path, task, pred_out])
        f_out.write(line_out + '\n')
        cnt += 1
        progressbar(cnt, n)
    f_out.close()

def cal_mAP(file_name):
    output_file = open(os.path.join(mAP_output_dir, model_name + '_' + task + '.txt'), 'a+')
    origin_data = np.load(file_name, encoding='latin1')  # 必须使用此编码模式
    # 读取最大的预测值，并用这个label和真实label对比，得到一个1X2的向量
    handle_data = np.array(
        [[origin_data[0][0].max(),
          np.where(origin_data[0][0] == origin_data[0][0].max())[0][0] == int(origin_data[0][1])]])

    GT_COUNT = origin_data.shape[0] - 1  # 因为规则上BLOCK_COUNT遍历时，GT_COUNT是开集，所以减一
    for i in range(1, GT_COUNT):
        # 读取最大的预测值，并用这个label和真实label对比，得到一个1X2的向量
        tmp_data = np.array([[origin_data[i][0].max(),
                              np.where(origin_data[i][0] == origin_data[i][0].max())[0][0] == int(
                                  origin_data[i][1])]])
        handle_data = np.concatenate([handle_data, tmp_data])

    handle_data.sort(axis=0)  # 按第一列进行排序。有小到大
    sort_array = handle_data[::-1]  # 由大到小排序

    PRED_COUNT = 0.0
    PRED_CORRECT_COUNT = 0.0
    AP_sum = 0
    count_time = 0  # 计算的次数
    cur_data = 0  # 当前索引
    for thresh in np.arange(1, float(sort_array[-1][0]), mAP_thresh_step):
        while (sort_array[cur_data][0] > thresh):  # 进入下一个step，更新大于thresh的数量
            PRED_COUNT += 1
            if sort_array[cur_data][1] == 1:
                PRED_CORRECT_COUNT += 1
            cur_data += 1
        # 每一个thresh都需要计算
        if PRED_COUNT != 0:
            AP_sum += PRED_CORRECT_COUNT / PRED_COUNT
            count_time += 1
    AP = AP_sum / count_time
    output_file.writelines('the AP of ' + task + ' is: ' + str(AP) + '\n')  # 写入
    print('the AP of ' + task + ' is: ' + str(AP))
    os.system('rm ' + file_name)
    return AP



# Preparation
args = parse_args()

task_list = {
    'collar_design_labels': 5,
    'skirt_length_labels': 6,
    'lapel_design_labels': 5,
    'neckline_design_labels': 10,
    'coat_length_labels': 8,
    'neck_design_labels': 5,
    'pant_length_labels': 6,
    'sleeve_length_labels': 9
}
task = args.task
task_num_class = task_list[task]

model_name = args.model

epochs = args.epochs
lr = args.lr
batch_size = args.batch_size
momentum = args.momentum
wd = args.wd

lr_factor = args.lr_factor
lr_steps = [int(s) for s in args.lr_steps.split(',')] + [np.inf]

num_gpus = args.num_gpus
num_workers = args.num_workers
ctx = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]
batch_size = batch_size * max(num_gpus, 1)

logging.basicConfig(level=logging.INFO,
                    handlers = [
                        logging.StreamHandler(),
                        logging.FileHandler('training.log')
                    ])

logpath_train=os.path.join('./log/',task+'_'+model_name+'_'+str(batch_size)+'_'+str(epochs)+'_5.30')
txtname='val.csv'
if not os.path.exists(logpath_train):
    os.makedirs(logpath_train)
if os.path.exists(os.path.join(logpath_train,txtname)):
    os.remove(os.path.join(logpath_train,txtname))
global f_val
f_val=file(os.path.join(logpath_train,txtname),'a+')


global random_scale 
input_scale = 480
if __name__ == "__main__":
    net = train()
    net.collect_params().save('models/%s_%s_%s_%s_final.params' % (task, model_name, batch_size, epochs))
    predict(task)

