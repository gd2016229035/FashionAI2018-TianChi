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
import cutomdataset

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

 ## Random earsing by ourselves ##    
def random_mask(img, probability = 0.5, sl = 0.02, sh = 0.4, r1 = 0.3, mean=[0.485, 0.456, 0.406]):
    if random.uniform(0, 1) > probability:
        return img

    for attempt in range(100):
        img_w, img_h = img.shape[:2]
        area = img_w * img_h
   
        target_area = random.uniform(sl, sh) * area
        aspect_ratio = random.uniform(r1, 1/r1)

        h = int(round(math.sqrt(target_area * aspect_ratio)))
        w = int(round(math.sqrt(target_area / aspect_ratio)))

        if w < img_w and h < img_h:
            x1 = random.randint(0, img_h - h)
            y1 = random.randint(0, img_w - w)
            if img.shape[2] == 3:
                img[x1:x1+h, y1:y1+w,0] = mean[0]
                img[x1:x1+h, y1:y1+w,1] = mean[1]
                img[x1:x1+h, y1:y1+w,2] = mean[2]
            else:
                img[x1:x1+h, y1:y1+w] = mean[0]
            return img

    return img

## Defined by hetong007 But maybe wrong
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

### Expand ten crop to nighteen crop by ourselves ### 
def ten_crop(img, size):
    H, W = size
    iH, iW = img.shape[1:3]

    if iH < H or iW < W:
        raise ValueError('image size is smaller than crop size')

    img_flip = img[:, :, ::-1]

    crops = nd.stack(
        img[:, (iH - H) // 2:(iH + H) // 2, (iW - W) // 2:(iW + W) // 2],#center crop
        img[:, 0:H, 0:W],#left top corner
        img[:, iH - H:iH, 0:W],#left bottom corner
        img[:, 0:H, iW - W:iW],#right top corner
        img[:, iH - H:iH, iW - W:iW],#right bottom corner

        ## new define
        img[:, 0:H, (iW - W) // 2:(iW + W) // 2], #middle top
        img[:, iH - H:iH, (iW - W) // 2:(iW + W) // 2],#middle bottom    
        img[:, (iH - H) // 2:(iH + H) // 2, 0:W],#left middle
        img[:, (iH - H) // 2:(iH + H) // 2, iW - W:iW],#right middle


        img_flip[:, (iH - H) // 2:(iH + H) // 2, (iW - W) // 2:(iW + W) // 2],
        img_flip[:, 0:H, 0:W],
        img_flip[:, iH - H:iH, 0:W],
        img_flip[:, 0:H, iW - W:iW],
        img_flip[:, iH - H:iH, iW - W:iW],

        img_flip[:, 0:H, (iW - W) // 2:(iW + W) // 2], #middle top
        img_flip[:, iH - H:iH, (iW - W) // 2:(iW + W) // 2],#middle bottom    
        img_flip[:, (iH - H) // 2:(iH + H) // 2, 0:W],#left middle
        img_flip[:, (iH - H) // 2:(iH + H) // 2, iW - W:iW],#right middle

    )
    return (crops)

## Test time transform function , 19-crop  ##
def transform_predict(im, size):
    im = im.astype('float32') / 255
    im = image.resize_short(im, size, interp=1)
    # im = image.resize_short(im, 331)
    im = nd.transpose(im, (2,0,1))
    im = mx.nd.image.normalize(im, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    # im = forty_crop(im, (352, 352))
    im = ten_crop(im, (448, 448))
    return (im)

def progressbar(i, n, bar_len=40):
    percents = math.ceil(100.0 * i / float(n))
    filled_len = int(round(bar_len * i / float(n)))
    prog_bar = '=' * filled_len + '-' * (bar_len - filled_len)
    print('[%s] %s%s\r' % (prog_bar, percents, '%'))

def accuracy(output, labels):
    return nd.mean(nd.argmax(output[0], axis=1) == labels[0][:,0]).asscalar()

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
        metric.update([label[0][:,0]], outputs)
        with ag.predict_mode():
            outputs = [net(X) for X in data]
            loss = []
            for yhat, y in zip(outputs[0], label[0]):
                loss_1 = 0
                if y[1] == 99: # only have y [4,0,0,0,0]
                    loss_1 += L(yhat, y[0])
                elif y[2] == 99: #have one m [4,1,0,0,0]
                    loss_1 = 0.8 * L(yhat, y[0]) + 0.2 * L(yhat, y[1])
                elif y[3] == 99: #have two m [4,1,3,0,0]
                    loss_1 = 0.7 * L(yhat, y[0]) + 0.15 * L(yhat, y[1]) + 0.15 * L(yhat, y[2])
                else: # have many m [4,1,3,2,0]
                    loss_1 = 0.6 * L(yhat, y[0]) + 0.13 * L(yhat, y[1]) + 0.13 * L(yhat, y[2]) + 0.13 * L(yhat, y[3])

                loss += [loss_1]
        val_loss += sum([l.mean().asscalar() for l in loss]) / len(loss)    # loss相加求和
        #ap, cnt = calculate_ap(label, outputs)
        # softmax_output和label本身是list，但是softmax_output[0]和label[0]是NDarray格式，注意这个NDarray是mxnet的，不是numpy的NDarray
        #softmax_output = [nd.SoftmaxActivation(output) for output in outputs]  # 取softmax,然后对十个结果取平均
        #sm_out_label = zip(softmax_output[0].asnumpy(), label[0].asnumpy())
        #all_softmax_output += sm_out_label
        #np.save(mAP_name, all_softmax_output)

    #this_AP = cal_mAP(mAP_name) # 得到当前的AP
    _, val_acc = metric.get()
    #return ((this_AP, val_acc, val_loss / len(val_data)))
    return ((val_acc, val_loss / len(val_data)))

def train():
    logging.info('Start Training for Task: %s\n' % (task))

    # Initialize the net with pretrained model
    pretrained_net = gluon.model_zoo.vision.get_model(model_name, pretrained=True)

    finetune_net = gluon.model_zoo.vision.get_model(model_name, classes=task_num_class)
    finetune_net.features = pretrained_net.features
    finetune_net.output.initialize(init.Xavier(), ctx = ctx)
    finetune_net.collect_params().reset_ctx(ctx)
    finetune_net.hybridize()
    
    # Carefully set the 'scale' parameter to make the 'muti-scale train' and 'muti-scale test'
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(448,scale=(0.76, 1.0),ratio=(0.999, 1.001)),
        transforms.RandomFlipLeftRight(),
        transforms.RandomBrightness(0.20),
        #transforms.RandomColorJitter(brightness=jitter_param, contrast=jitter_param,
        #                             saturation=jitter_param),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset =cutomdataset.custom_dataset2(root='./data2/crop_lapel2',filename=os.path.join('data2/', task+'_train.txt'))
    train_data = gluon.data.DataLoader(train_dataset.transform_first(train_transform),
        batch_size=batch_size, shuffle=True, num_workers=num_workers, last_batch='discard')

    val_transform = transforms.Compose([
        transforms.Resize(480),
        transforms.CenterCrop(448),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_dataset = cutomdataset.custom_dataset2(root='./data2/crop_lapel2',filename=os.path.join('data2/', task+'_val.txt'))
    val_data = gluon.data.DataLoader(val_dataset.transform_first(val_transform),
        batch_size=batch_size, shuffle=False, num_workers = num_workers)
    
    # Define Trainer use ADam to make mdoel converge quickly
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
        train_acc = 0.
        #### Load the best model when go to the next training stage
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
                loss = []
                ###### Handle 'm' label by soft-softmax function ######
                for yhat, y in zip(outputs[0], label[0]):
                    loss_1 = 0
                    if y[1] == 99: # only have y [4,0,0,0,0]
                        loss_1 += L(yhat, y[0])
                    elif y[2] == 99: #have one m [4,1,0,0,0]
                        loss_1 = 0.8 * L(yhat, y[0]) + 0.2 * L(yhat, y[1])
                    elif y[3] == 99: #have two m [4,1,3,0,0]
                        loss_1 = 0.7 * L(yhat, y[0]) + 0.15 * L(yhat, y[1]) + 0.15 * L(yhat, y[2])
                    else: # have many m [4,1,3,2,0]
                        loss_1 = 0.6 * L(yhat, y[0]) + 0.13 * L(yhat, y[1]) + 0.13 * L(yhat, y[2]) + 0.13 * L(yhat, y[3])

                    loss += [loss_1]

                #loss = [L(yhat, y) for yhat, y in zip(outputs, label)
            # for l in loss:
            #     l.backward()
            ag.backward(loss)# for soft-softmax

            trainer.step(batch_size)
            train_loss += sum([l.mean().asscalar() for l in loss]) / len(loss)
            #train_acc += accuracy(outputs, label)
            metric.update([label[0][:,0]], outputs)
            #ap, cnt = calculate_ap(label, outputs)
            #AP += ap
            #AP_cnt += cnt
            #progressbar(i, num_batch-1)

        #train_map = AP / AP_cnt
        _, train_acc = metric.get()
        train_loss /= num_batch

        val_acc, val_loss = validate(finetune_net, val_data, ctx)

        logging.info('[Epoch %d] Train-acc: %.3f, loss: %.3f | Val-acc: %.3f, loss: %.3f | time: %.1f | learning_rate %.6f' %
                 (epoch, train_acc,  train_loss, val_acc, val_loss, time.time() - tic, trainer.learning_rate))
        f_val.writelines('[Epoch %d] Train-acc: %.3f, , loss: %.3f | Val-acc: %.3f,  loss: %.3f | time: %.1f | learning_rate %.6f\n' %
                 (epoch, train_acc, train_loss, val_acc, val_loss, time.time() - tic, trainer.learning_rate))
        ### Save the best model every stage
        if val_acc > best_acc:
            #best_AP = this_AP
            best_acc = val_acc
            best_path = '/usr/data/fashionai/models/%s_%s_%s_%s.params' % (task, model_name, epoch, best_acc)
            finetune_net.collect_params().save(best_path)

    logging.info('\n')
    finetune_net.collect_params().load(best_path, ctx= ctx)
    f_val.writelines('Best val acc is :[Epoch %d] Train-acc: %.3f, loss: %.3f | Best-val-acc: %.3f, loss: %.3f | time: %.1f | learning_rate %.6f\n' %
         (epoch, train_acc, train_loss, best_acc,  val_loss, time.time() - tic, trainer.learning_rate))
    return (finetune_net)

#### Model test 
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
        out_all = np.zeros([task_list[task],])
        ###### Test Time augmentation (muti-scale test) ######
        for scale in input_scale:
            data = transform_predict(img, scale)
            with ag.predict_mode():
                out = net(data.as_in_context(mx.gpu(0)))  # 随机crop十张图片,所以此处是10张图片的结果
                out = nd.SoftmaxActivation(out).mean(axis=0)  # 取softmax,然后对十个结果取平均  
                out_all += out.asnumpy()
        out = out_all / len(input_scale)

        pred_out = ';'.join(["%.8f"%(o) for o in out.tolist()])
        line_out = ','.join([path, task, pred_out])
        f_out.write(line_out + '\n')
        cnt += 1
        #progressbar(cnt, n)
    f_out.close()

### Define mAP by ourselves according the competetion illustrate
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

## log
logpath_train=os.path.join('./log/',task+'_'+model_name+'_'+str(batch_size)+'_'+str(epochs)+'_5.26')
txtname='val.csv'
if not os.path.exists(logpath_train):
    os.makedirs(logpath_train)
if os.path.exists(os.path.join(logpath_train,txtname)):
    os.remove(os.path.join(logpath_train,txtname))
global f_val
f_val=file(os.path.join(logpath_train,txtname),'a+')

#### Muti-scale train & test  scale parameter
global random_scale 
input_scale = [448,480,512]


if __name__ == "__main__":
    net = train()
    net.collect_params().save('models/%s_%s_%s_%s_final.params' % (task, model_name, batch_size, epochs))
    predict(task)

