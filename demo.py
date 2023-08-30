import sys
import torch
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from RBSFormer import RBSFormer
from datasets import *
from utils import pad, unpad


def test(args):

    model_restoration = RBSFormer()
    if args.cuda:
        model_restoration = model_restoration.cuda()

    # testing dataset
    datasetTest = MyTestDataSet(args.input_dir)
    testLoader = DataLoader(dataset=datasetTest, batch_size=1, shuffle=False, drop_last=False,
                            num_workers=4, pin_memory=True)

    print('--------------------------------------------------------------')
    if args.cuda:
        model_restoration.load_state_dict(torch.load(args.resume_state))
    else:
        model_restoration.load_state_dict(torch.load(args.resume_state, map_location=torch.device('cpu')))
    model_restoration.eval()

    with torch.no_grad():
        for index, (x, name) in enumerate(tqdm(testLoader, desc='Testing !!! ', file=sys.stdout), 0):

            input_ = x.cuda() if args.cuda else x

            input_, pad_size = pad(input_, factor=16)
            restored = model_restoration(input_)
            restored = unpad(restored, pad_size)

            save_image(restored, args.result_dir + name[0])


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='./Test1/input/')
    parser.add_argument('--result_dir', type=str, default='./Test1/result/')
    parser.add_argument('--resume_state', type=str, default='./model_best.pth')
    parser.add_argument('--cuda', type=bool, default=True)
    args = parser.parse_args()

    test(args)