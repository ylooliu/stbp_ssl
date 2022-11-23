import logging

import matplotlib.pyplot as plt
import torch.optim.lr_scheduler

import loadNMNIST_Spiking
import torch.nn as nn

from all_conv_rossum import SCNN
from torch.nn.modules.loss import _Loss
from get_dataset import *
import global_v as glv

glv.init()

batch_size=100
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
time_window=glv.n_steps
num_class=10
torch.autograd.set_detect_anomaly(True)


def psp(inputs):
    shape = inputs.shape

    tau_s = glv.tau_s

    syn = torch.zeros(shape[0], shape[1]).cuda()
    syns = torch.zeros(shape[0], shape[1], time_window).cuda()

    for t in range(time_window):
        syn = syn - syn / tau_s + inputs[..., t]
        syns[..., t] = syn / tau_s

    return syns


def vrdistance(f, g): #f是网络输出 g是label  f (batch,10,T)
    # Computes the Van Rossum distance between f and g

    assert f.shape == g.shape, 'Spike trains must have the same shape, had f: ' + str(f.shape) + ', g: ' + str(g.shape)

    return nn.MSELoss()(psp(f), psp(g))



class VRDistance(_Loss):
    # Van Rossum distance loss
    def __init__(self,  size_average=None, reduce=None, reduction='mean'):
        super(VRDistance, self).__init__(size_average, reduce, reduction)


    def forward(self, input, target):
        return vrdistance(input, target)




if __name__=='__main__':
    data_path='../NMNIST_data/2312_3000_stable/'

    train_loader, test_loader = loadNMNIST_Spiking.get_nmnist(data_path,n_steps=time_window,batch_size=batch_size)


    net = SCNN().cuda()
    torch.save(net.state_dict(),'./pth/last_state.pth')

    logging.basicConfig(filename='log/result_rossum.log', level=logging.INFO)


    loss_function=nn.MSELoss()
    # 使用Adam优化器
    lr=0.0003
    optimizer = torch.optim.AdamW(net.parameters(), lr, weight_decay=5e-4)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.5, last_epoch=-1)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [25], gamma=0.5, last_epoch=-1)
    rossum_distance = VRDistance()   #好像tau小点效果好

    best_acc = 0.0
    train_epoch=50

    logging.info('dataset name:{}, epoch:{}, batch size:{}, lr:{}'.format(' NMNIST', train_epoch, batch_size, lr))
    logging.info(f'tau: {glv.tau}, tau_s: {glv.tau_s}, vth: {glv.vth}, time window:{glv.n_steps}')


    for epoch in range(train_epoch):
        # logging.info('epoch:{}'.format(epoch))
        last_state=torch.load('./pth/last_state.pth')

        net.train()
        losses = []
        correct = 0.0
        total = 0.0

        scheduler.step()
        for batch, (img, label) in enumerate(train_loader):
            img = img.to(device)
            y_one_hot = torch.zeros(batch_size, 10).scatter_(1, label.reshape((label.shape[0],1)), 1)
            label_seq=y_one_hot.repeat(time_window,1,1).permute(1,2,0)

            optimizer.zero_grad()
            out_mem,out_spike = net(img)  #out shape[batch,10,time_window]

            loss = rossum_distance(out_spike, label_seq.to(device))  #loss的大小为【batch】
            #loss=loss_function(out_spike, label_seq.to(device))
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            out=torch.sum(out_spike,dim=-1)   ##[batch,10]
            correct += (out.max(dim=1)[1] == label.to(device)).float().sum().item()

            total += out.shape[0]
            if batch % 100 == 0 and batch!=0:
                acc = correct / total

                print('Epoch %d [%d/%d] ANN Training Loss:%f Accuracy:%f' % (epoch,
                                                                                 batch + 1,
                                                                                 len(train_loader),
                                                                                 np.array(losses).mean(),
                                                                                 acc))
                logging.info('Epoch %d [%d/%d] ANN Training Loss:%f Accuracy:%f' % (epoch,
                                                                                 batch + 1,
                                                                                 len(train_loader),
                                                                                 np.array(losses).mean(),
                                                                                 acc))
                correct = 0.0
                total = 0.0


        torch.save(net.state_dict(),'./pth/last_state.pth')
        logging.info('Train Loss:%f Accuracy:%f'%(np.array(losses).mean(),acc))


        net.eval()
        correct = 0.0
        total = 0.0

        with torch.no_grad():
            for batch, (img, label) in enumerate(test_loader):
                img = img.to(device)

                out_mem,out_spike = net(img)  # out shape[time_window,batch,10]

                out = torch.sum(out_spike, dim=-1)  #[batch,10]
                correct += (out.max(dim=1)[1] == label.to(device)).float().sum().item()

                total += out.shape[0]

            acc = correct / total
            if epoch == None:
                print('ANN Validating Accuracy:%f' % (acc))
            else:
                print('Epoch %d [%d/%d]  Accuracy:%f ' % (epoch,
                                                          batch + 1,
                                                          len(test_loader),
                                                          acc))

        logging.info('Accuracy:%f' % acc)



        if best_acc <= acc:
            best_acc=acc
            torch.save(net, 'pth/best_nminst_snn_rossum.pkl')
