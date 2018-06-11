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
import argparse
import imageio

parser = argparse.ArgumentParser(description='SR benchmark')
parser.add_argument('-m', '--model-path', type=str, default='./snapshots/best_model.pth',
                    help='path to trained model')
parser.add_argument('--content-path', type=str, default='./dataset/test/content',
                    help='path to content image')
parser.add_argument('--style-path', type=str, default='./dataset/test/style',
                    help='path to style image')
parser.add_argument('--result-path', type=str, default='./results',
                    help='path to save result')
parser.add_argument('--alpha', type=float, default='1.0',
                    help='content and style trade-off at test time')
args = parser.parse_args()

def convert_img(img, dm):
    img = img.squeeze(0)
    img = dm.restore(img.data.cpu())
    img = torch.clamp(img, 0, 1)*255
    img = img.numpy().transpose(1, 2, 0).astype(np.uint8)
    return img

def test():
    # Data loader
    dm = DataManager(args.content_path, args.style_path, random_crop=False)
    dl = DataLoader(dm, batch_size=1, shuffle=False, num_workers=4)

    # load model
    net = StyleTransferNet()
    net = torch.load(args.model_path)
    net = net.cuda()
    net.eval()

    # result
    result_path = os.path.join(args.result_path, 'alpha{}'.format(args.alpha))
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    # Start Testing
    print('Start testing')
    with torch.no_grad():
        for i, data in enumerate(dl, 0):
            img_con = data['content']
            img_sty = data['style']

            img_con = Variable(img_con, requires_grad=False).cuda()
            img_sty = Variable(img_sty, requires_grad=False).cuda()

            _, img_result = net(img_con, img_sty, args.alpha)
            
            file_names = [os.path.join(result_path, name + str(i) + '.png') 
                         for name in ['con', 'sty', 'result']]
            imgs = [img_con, img_sty, img_result]
            imgs_converted = [convert_img(img, dm) for img in imgs]
            for file_name, img in zip(file_names, imgs_converted):
                imageio.imwrite(file_name, img)
    print('Test finished.')



if __name__ == '__main__':
    test()
