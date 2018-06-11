"""
    2018 Spring EE898
    Advanced Topics in Deep Learning
    for Robotics and Computer Vision

    Programming Assignment 2
    Neural Style Transfer

    Author : Jinsun Park (zzangjinsun@gmail.com)

    References
    [1] Gatys et al., "Image Style Transfer using Convolutional
        Neural Networks", CVPR 2016.
    [2] Huang and Belongie, "Arbitrary Style Transfer in Real-Time
        with Adaptive Instance Normalization", ICCV 2017.
"""
from common import *

def train():
    gc.disable()

    # Parameters
    path_snapshot = 'snapshots/'
    path_content = 'dataset/train/content'
    path_style = 'dataset/train/style'

    if not os.path.exists(path_snapshot):
        os.makedirs(path_snapshot)

    batch_size = 16
    weight_decay = 1.0e-5
    num_epoch = 200
    lr_init = 0.0001
    lr_decay_step = 100
    momentum = 0.9
    w_style = 10

    # Data loader
    dm = DataManager(path_content, path_style, random_crop=True)
    dl = DataLoader(dm, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=False)

    num_train = dm.num
    num_batch = np.ceil(num_train / batch_size)
    loss_train_avg = np.zeros(num_epoch)

    net = StyleTransferNet(w_style)
    net = net.cuda()
    net.train()

    trainable = filter(lambda x: x.requires_grad, net.parameters())
    optimizer = optim.Adam(trainable,
                           lr=lr_init,
                           weight_decay=weight_decay)

    # For visualization
    vis = visdom.Visdom()

    vis_loss = VisdomLine(vis, dict(title='Training Loss', markers=True))
    vis_image = VisdomImage(vis, dict(title='Content / Style / Result'))

    # Start training
    for epoch in range(0, num_epoch):
        running_loss_train = 0
        np.random.shuffle(dl.dataset.list_style)

        for i, data in enumerate(dl, 0):
            img_con = data['content']
            img_sty = data['style']

            img_con = Variable(img_con, requires_grad=False).cuda()
            img_sty = Variable(img_sty, requires_grad=False).cuda()

            optimizer.zero_grad()

            loss, img_result = net(img_con, img_sty)

            loss = torch.mean(loss)
            loss.backward()

            optimizer.step()

            running_loss_train += loss

            print('[%s] Epoch %3d / %3d, Batch %5d / %5d, Loss = %12.8f' %
                  (str(datetime.now())[:-3], epoch + 1, num_epoch,
                   i + 1, num_batch, loss))

        loss_train_avg[epoch] = running_loss_train / num_batch

        print('[%s] Epoch %3d / %3d, Avg Loss = %12.8f' % \
              (str(datetime.now())[:-3], epoch + 1, num_epoch,
               loss_train_avg[epoch]))

        optimizer = LearningRateScheduler(optimizer, epoch + 1, lr_decay_step=lr_decay_step)

        # Display using visdom
        vis_loss.Update(np.arange(epoch + 1) + 1, loss_train_avg[0:epoch + 1])

        img_cat = torch.cat((img_con, img_sty, img_result), dim=3)
        img_cat = torch.unbind(img_cat, dim=0)
        img_cat = torch.cat(img_cat, dim=1)
        img_cat = dm.restore(img_cat.data.cpu())
        vis_image.Update(torch.clamp(img_cat, 0, 1))

        # Snapshot
        if (epoch + 1) % 10 == 0:
            torch.save(net, '%s/epoch_%06d.pth' % (path_snapshot, epoch + 1))

        gc_collected = gc.collect()
        gc.disable()

    print('Training finished.')



if __name__ == '__main__':
    train()
