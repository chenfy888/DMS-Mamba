"""
Compare with state-of-the-art methods.
Load models from the folder networks/compare_models.
"""

import os

from skimage import io
import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import time
import torch
import numpy as np
import torch.optim as optim
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from networks.compare_models import build_model_from_name
# from networks.wavelet import build_model_from_name
# from dataloaders.BRATS_dataloader_new import Hybrid as MyDataset
# from dataloaders.Knee_dataloader_800 import KneeDataset_Cartesian as MyDataset
from dataloaders.Knee_dataloader import KneeDataset as MyDataset
from dataloaders.Knee_dataloader import RandomPadCrop, ToTensor, AddNoise

from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
from metric import nmse, psnr, ssim, AverageMeter
from collections import defaultdict

from utils import bright, trunc
from matplotlib import pyplot as plt
import ast  # 用于解析字符串中的列表


# from networks.DCAMSR import DCAMSR
# from plot_learnablefilter import visualize_filter

# torch.backends.cudnn.benchmark = True

# 自定义转换函数
def str_to_list(s):
    return ast.literal_eval(s)  # 将字符串转换为列表

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/public/home/chenfurong/2025data/fastmri_knee/')
parser.add_argument('--MRIDOWN', type=str, default='4X', help='MRI down-sampling rate')
parser.add_argument('--low_field_SNR', type=int, default=0, help='SNR of the simulated low-field image')
parser.add_argument('--phase', type=str, default='test', help='Name of phase')
parser.add_argument('--gpu', type=str, default='1', help='GPU to use')
parser.add_argument('--exp', type=str, default='MyModelX13_lr001_4X_knee', help='model_name')
parser.add_argument('--max_iterations', type=int, default=100000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')
parser.add_argument('--base_lr', type=float, default=0.001, help='maximum epoch numaber to train')
parser.add_argument('--seed', type=int, default=1337, help='random seed')

# parser.add_argument('--input_dim', type=int, default=1, help='number of channels of the input image')
# parser.add_argument('--output_dim', type=int, default=1, help='number of channels of the reconstructed image')
# mmr_mamba 
parser.add_argument('--model_name', type=str, default='mymodelX13', help='model_name')
parser.add_argument('--use_multi_modal', type=str, default='False', help='whether use multi-modal data for MRI reconstruction')
parser.add_argument('--modality', type=str, default='t2', help='MRI modality')
parser.add_argument('--input_modality', type=str, default='t2', help='input MRI modality')

parser.add_argument('--relation_consistency', type=str, default='False', help='regularize the consistency of feature relation')

parser.add_argument('--clip_grad', type=str, default='True', help='clip gradient of the network parameters')


parser.add_argument('--norm', type=str, default='False', help='Norm Layer between UNet and Transformer')
parser.add_argument('--input_normalize', type=str, default='mean_std', help='choose from [min_max, mean_std, divide]')

parser.add_argument('--kspace_refine', type=str, default='False', \
                    help='use the original under-sampled input or the kspace-interpolated input')

parser.add_argument('--kspace_round', type=str, default='round4', help='use which round of kspace_recon as model input')


parser.add_argument('--MASKTYPE', type=str, default='random', help='mask type for kspace sampling') # "random" or "equispaced"
parser.add_argument('--CENTER_FRACTIONS', type=list, default=[0.08], help='center fraction for kspace sampling')
parser.add_argument('--ACCELERATIONS', type=list, default=[4], help='acceleration for kspace sampling')


args = parser.parse_args()
train_data_path = args.root_path
test_data_path = args.root_path
# snapshot_path = "../model_knee/" + args.exp + "/"
# snapshot_path = "../model_241005cvpr_knee/" + args.exp + "/"
snapshot_path = "/home/chenfurong/CODE2025/testsave/" + args.exp + "/"
snapshot_path1 = "/home/chenfurong/CODE2025/modelsave/" + args.exp + "/"


# os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = args.batch_size * len(args.gpu.split(','))
max_iterations = args.max_iterations
base_lr = args.base_lr



def cc(img1, img2):
    eps = torch.finfo(torch.float32).eps
    """Correlation coefficient for (N, C, H, W) image; torch.float32 [0.,1.]."""
    N, C, _, _ = img1.shape
    img1 = img1.reshape(N, C, -1)
    img2 = img2.reshape(N, C, -1)
    img1 = img1 - img1.mean(dim=-1, keepdim=True)
    img2 = img2 - img2.mean(dim=-1, keepdim=True)
    cc = torch.sum(img1 * img2, dim=-1) / (eps + torch.sqrt(torch.sum(
        img1 **2, dim=-1)) * torch.sqrt(torch.sum(img2**2, dim=-1)))
    cc = torch.clamp(cc, -1., 1.)
    return cc.mean()



if __name__ == "__main__":
    ## make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    # if os.path.exists(snapshot_path + '/code'):
    #     shutil.rmtree(snapshot_path + '/code')
    # shutil.copytree('.', snapshot_path + '/code', shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))


    
    if args.model_name == 'DCAMSR':
        network = DCAMSR().cuda()
    else:
        network = build_model_from_name(args).cuda()
        
    if len(args.gpu.split(',')) > 1:
        network = nn.DataParallel(network)
        # network = nn.SyncBatchNorm.convert_sync_batchnorm(network)
    
    # print("network architecture:", network)
    
    n_parameters = sum(p.numel() for p in network.parameters() if p.requires_grad)
    print('number of params: %.2f M' % (n_parameters / 1024 / 1024))
    
    db_test = MyDataset(kspace_refine=args.kspace_refine, kspace_round = args.kspace_round,
                        split='val', MRIDOWN=args.MRIDOWN, SNR=args.low_field_SNR, 
                        transform=transforms.Compose([ToTensor()]),
                        base_dir=test_data_path, input_normalize = args.input_normalize,
                        CENTER_FRACTIONS=args.CENTER_FRACTIONS, ACCELERATIONS=args.ACCELERATIONS)

    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)


    if args.phase == 'test':
        
        save_mode_path = os.path.join(snapshot_path1, 'best_checkpoint.pth')  #  best_checkpoint.pth    iter_2000
        print('load weights from ' + save_mode_path)
        checkpoint = torch.load(save_mode_path)
        network.load_state_dict(checkpoint['network'], strict=False)
        network.eval()
        cnt = 0
        
        output_dic = defaultdict(dict)
        target_dic = defaultdict(dict)
        # input_dic = defaultdict(dict)
        nmse_meter = AverageMeter()
        psnr_meter = AverageMeter()
        ssim_meter = AverageMeter()
        # print()
        t2_MSE_all, t2_PSNR_all, t2_SSIM_all = [], [], []

        for (pd, pdfs, _) in testloader:
            cnt+=1
            t1_in, t1, t2_in, t2 = pd[1].unsqueeze(1).cuda(), pd[1].unsqueeze(1).cuda(), \
                                    pdfs[0].unsqueeze(1).cuda(), pdfs[1].unsqueeze(1).cuda()   
            
            mean = pdfs[2].unsqueeze(1).unsqueeze(2).cuda()
            std = pdfs[3].unsqueeze(1).unsqueeze(2).cuda()
    
            fname = pdfs[4]
            slice_num = pdfs[5]               
            name = os.path.basename(pdfs[4][0]).split('.')[0]

            with torch.no_grad():
                t2_out = network(t2_in, t1_in)
                t1_out = None
                # 检查网络输出后的数据范围

            outputs = t2_out[0].squeeze(1)
            target = t2.squeeze(1)
            
            if cnt < 100:

                t2_out_img = ((np.clip(t2_out[0].squeeze().detach().cpu().numpy() / 6.0, -1, 1) + 1) * 127).astype(np.uint8)
                t2_in_img  = ((np.clip(t2_in.squeeze().detach().cpu().numpy()      / 6.0, -1, 1) + 1) * 127).astype(np.uint8)
                t2_img     = ((np.clip(t2.squeeze().detach().cpu().numpy()         / 6.0, -1, 1) + 1) * 127).astype(np.uint8)
                t1_img     = ((np.clip(t1.squeeze().detach().cpu().numpy()         / 6.0, -1, 1) + 1) * 127).astype(np.uint8)

                save_path = os.path.join(snapshot_path, "results")
                os.makedirs(save_path, exist_ok=True)

                MSE_i  = mean_squared_error(t2_img, t2_out_img)
                PSNR_i = peak_signal_noise_ratio(t2_img, t2_out_img, data_range=255)
                SSIM_i = structural_similarity(t2_img, t2_out_img, data_range=255)

                tag = f"psnr{PSNR_i:.2f}_ssim{SSIM_i:.4f}"

                io.imsave(os.path.join(save_path, f"{cnt:04d}_{tag}_t2_gt.png"),  t2_img)
                io.imsave(os.path.join(save_path, f"{cnt:04d}_{tag}_t2_out.png"), t2_out_img)
                io.imsave(os.path.join(save_path, f"{cnt:04d}_{tag}_t2_in.png"),  t2_in_img)
                io.imsave(os.path.join(save_path, f"{cnt:04d}_{tag}_t1.png"),     t1_img)

                fig = plt.figure()
                plt.imshow(t2_in_img, cmap='gray')
                plt.axis('off')
                fig.savefig(os.path.join(save_path, f"{cnt:04d}_{tag}_t2_in_mpl.png"), bbox_inches='tight', pad_inches=0)
                plt.close(fig)

                fig = plt.figure()
                plt.imshow(t2_img, cmap='gray')
                plt.axis('off')
                fig.savefig(os.path.join(save_path, f"{cnt:04d}_{tag}_t2_gt_mpl.png"), bbox_inches='tight', pad_inches=0)
                plt.close(fig)

                fig = plt.figure()
                plt.imshow(t2_out_img, cmap='gray')
                plt.axis('off')
                fig.savefig(os.path.join(save_path, f"{cnt:04d}_{tag}_t2_out_mpl.png"), bbox_inches='tight', pad_inches=0)
                plt.close(fig)

                fig = plt.figure()
                plt.imshow(t1_img, cmap='gray')
                plt.axis('off')
                fig.savefig(os.path.join(save_path, f"{cnt:04d}_{tag}_t1_mpl.png"), bbox_inches='tight', pad_inches=0)
                plt.close(fig)

                diff = (target * std + mean).detach().cpu().numpy()[0] - (outputs * std + mean).detach().cpu().numpy()[0]
                diff = np.abs(diff) * 255
                print(f"Diff range: {diff.min()} to {diff.max()}")

                fig = plt.figure()
                plt.imshow(diff, cmap='jet', vmin=1e-10, vmax=0.01)
                plt.axis('off')
                fig.savefig(os.path.join(save_path, f"{cnt:04d}_{tag}_errormap01.png"), bbox_inches='tight', pad_inches=0)
                plt.close(fig)

                fig = plt.figure()
                plt.imshow(diff, cmap='jet', vmin=1e-10, vmax=0.012)
                plt.axis('off')
                fig.savefig(os.path.join(save_path, f"{cnt:04d}_{tag}_errormap012.png"), bbox_inches='tight', pad_inches=0)
                plt.close(fig)

                fig = plt.figure()
                plt.imshow(diff, cmap='jet', vmin=1e-10, vmax=0.015)
                plt.axis('off')
                fig.savefig(os.path.join(save_path, f"{cnt:04d}_{tag}_errormap015.png"), bbox_inches='tight', pad_inches=0)
                plt.close(fig)

                diff_origin = (target * std + mean).detach().cpu().numpy()[0] - ((t2_in[0] * std + mean).detach().cpu().numpy()[0])
                diff_origin = np.abs(diff_origin) * 255

                fig = plt.figure()
                plt.imshow(diff_origin, cmap='jet')
                plt.axis('off')
                fig.savefig(os.path.join(save_path, f"{cnt:04d}_{tag}_errormap_origin.png"), bbox_inches='tight', pad_inches=0)
                plt.close(fig)

                gt_float  = target[0].detach().cpu().numpy()
                out_float = outputs[0].detach().cpu().numpy()

                our_nmse = nmse(gt_float, out_float)
                our_psnr = psnr(gt_float, out_float)
                our_ssim = ssim(gt_float, out_float)


                t2_MSE_all.append(our_nmse)
                t2_PSNR_all.append(our_psnr)
                t2_SSIM_all.append(our_ssim)

                # fig = plt.figure()
                # plt.imshow(t2_in.data.cpu().numpy()[0, 0], cmap='gray')
                # # plt.set_title('t2_in')
                # plt.axis('off')
                # fig.savefig(save_path + str(name) + '_' + str(slice_num[0].cpu().numpy()) + '_t2_in', bbox_inches='tight', pad_inches=0)
                
                # fig = plt.figure() 
                # plt.imshow(t2.data.cpu().numpy()[0, 0], cmap='gray')
                # # print('t2:', t2.data.cpu().numpy()[0, 0].shape)
                # # plt.set_title('t2_gt')
                # plt.axis('off')
                # fig.savefig(save_path + str(name) + '_' + str(slice_num[0].cpu().numpy()) + '_t2_gt', bbox_inches='tight', pad_inches=0)

                # fig = plt.figure()
                # plt.imshow(outputs[0].cpu().numpy(), cmap='gray')
                # # plt.set_title('output')
                # plt.axis('off')
                # fig.savefig(save_path + str(name) + '_' + str(slice_num[0].cpu().numpy()) + '_output', bbox_inches='tight', pad_inches=0)
                
                # # diff = t2.data.cpu().numpy()[0, 0] - outputs[0].cpu().numpy()
                
                # # diff = t2.data.cpu().numpy()[0, 0]*std[0]+mean[0] - (outputs[0].cpu().numpy()*std[0]+mean[0])
                # diff = (target * std + mean).cpu().numpy()[0] - (outputs * std + mean).cpu().numpy()[0]
                # diff = np.abs(diff) * 255
                # # print('diff: max and min', diff.max(), diff.min())
                # fig = plt.figure()
                # plt.imshow(diff, cmap='jet', vmin=0.0000000001, vmax=0.01)
                # # plt.set_title('diff')
                # plt.axis('off')  
                # fig.savefig(save_path + str(name) + '_' + str(slice_num[0].cpu().numpy()) + '_errormap01', bbox_inches='tight', pad_inches=0)
                
                # fig = plt.figure()
                # plt.imshow(diff, cmap='jet', vmin=0.0000000001, vmax=0.012)
                # # plt.set_title('diff')
                # plt.axis('off')  
                # fig.savefig(save_path + str(name) + '_' + str(slice_num[0].cpu().numpy()) + '_errormap012', bbox_inches='tight', pad_inches=0)
                
                # fig = plt.figure()
                # plt.imshow(diff, cmap='jet', vmin=0.0000000001, vmax=0.015)
                # # plt.set_title('diff')
                # plt.axis('off')  
                # fig.savefig(save_path + str(name) + '_' + str(slice_num[0].cpu().numpy()) + '_errormap015', bbox_inches='tight', pad_inches=0)
                
                # diff_origin = (target * std + mean).cpu().numpy()[0] - (t2_in[0] * std + mean).cpu().numpy()[0]
                # diff_origin = np.abs(diff_origin) * 255
                # fig = plt.figure()
                # plt.imshow(diff_origin, cmap='jet')
                # # plt.set_title('diff_origin')
                # plt.axis('off')
                # fig.savefig(save_path + str(name) + '_' + str(slice_num[0].cpu().numpy()) + '_errormap_origin', bbox_inches='tight', pad_inches=0)
                # 这一部分为原始可视化
            
            # if cnt == 1:
            #     save_path_filter = snapshot_path + '/result_case_best_filter_coolwarm2/'
            #     if not os.path.exists(save_path_filter):
            #         os.makedirs(save_path_filter)   
            #     visualize_filter(network.wfca1.filter_hl, save_path_filter + str(cnt) +'_wfca1_' + '_filterhl.png')
            #     visualize_filter(network.wfca1.filter_lh, save_path_filter + str(cnt) +'_wfca1_' + '_filterlh.png')
            #     visualize_filter(network.wfca1.filter_hh, save_path_filter + str(cnt) +'_wfca1_' + '_filterhh.png')
            #     visualize_filter(network.wfca2.filter_hl, save_path_filter + str(cnt) +'_wfca2_' + '_filterhl.png')
            #     visualize_filter(network.wfca2.filter_lh, save_path_filter + str(cnt) +'_wfca2_' + '_filterlh.png')
            #     visualize_filter(network.wfca2.filter_hh, save_path_filter + str(cnt) +'_wfca2_' + '_filterhh.png')
            #     visualize_filter(network.wfca3.filter_hl, save_path_filter + str(cnt) +'_wfca3_' + '_filterhl.png')
            #     visualize_filter(network.wfca3.filter_lh, save_path_filter + str(cnt) +'_wfca3_' + '_filterlh.png')
            #     visualize_filter(network.wfca3.filter_hh, save_path_filter + str(cnt) +'_wfca3_' + '_filterhh.png')
                
            outputs = outputs * std + mean
            # print("outputs:", outputs.max(), outputs.min())
            target = target * std + mean
            inputs = t2_in.squeeze(1) * std + mean
            
            # print('target:', target.max(), target.min())
            # print('output:', outputs.max(), outputs.min())
            
            # fig = plt.figure()
            # plt.axis('off')
            # plt.imshow(outputs[0].cpu().numpy(), cmap = 'gray')
            # fig.savefig('./test_images/output.png', bbox_inches='tight', pad_inches=0)

            output_dic[fname[0]][slice_num[0]] = outputs[0]
            target_dic[fname[0]][slice_num[0]] = target[0]
            # input_dic[fname[0]][slice_num[0]] = inputs[0]
                    
            our_nmse = nmse(target[0].cpu().numpy(), outputs[0].cpu().numpy())
            our_psnr = psnr(target[0].cpu().numpy(), outputs[0].cpu().numpy())
            our_ssim = ssim(target[0].cpu().numpy(), outputs[0].cpu().numpy())
                
            our_nmse = nmse(target[0].detach().cpu().numpy(), outputs[0].detach().cpu().numpy())
            our_psnr = psnr(target[0].detach().cpu().numpy(), outputs[0].detach().cpu().numpy())
            our_ssim = ssim(target[0].detach().cpu().numpy(), outputs[0].detach().cpu().numpy())
                
            # print('name:{}, slice:{}, psnr:{}, ssim:{}'.format(name, slice_num[0], our_psnr, our_ssim))
            # breakpoint()  
            
        t2_MSE_all.append(our_nmse)
        t2_PSNR_all.append(our_psnr)
        t2_SSIM_all.append(our_ssim)
        print('Evaluation by slice:', 'MSE:', np.array(t2_MSE_all).mean(), 'PSNR:', np.array(t2_PSNR_all).mean(), 'SSIM:', np.array(t2_SSIM_all).mean()) 

        for name in output_dic.keys():
            f_output = torch.stack([v for _, v in output_dic[name].items()])
            f_target = torch.stack([v for _, v in target_dic[name].items()])
            our_nmse = nmse(f_target.detach().cpu().numpy(), f_output.detach().cpu().numpy())
            our_psnr = psnr(f_target.detach().cpu().numpy(), f_output.detach().cpu().numpy())
            our_ssim = ssim(f_target.detach().cpu().numpy(), f_output.detach().cpu().numpy())
            # print('name:{}, psnr:{}, ssim:{}'.format(name, our_psnr, our_ssim))
            nmse_meter.update(our_nmse, 1)
            psnr_meter.update(our_psnr, 1)
            ssim_meter.update(our_ssim, 1)
        

        print("==> Evaluate Metric")
        print("Results ----------")
        # print("NMSE: {:.4}".format(nmse_meter.avg))
        # print("PSNR: {:.4}".format(psnr_meter.avg))
        # print("SSIM: {:.4}".format(ssim_meter.avg))
        print('mean',"NMSE: {:.4}".format(np.array(nmse_meter.score).mean()),"PSNR: {:.4}".format(np.array(psnr_meter.score).mean()),"SSIM: {:.4}".format(np.array(ssim_meter.score).mean()))
        print('std',"NMSE: {:.4}".format(np.array(nmse_meter.score).std()),"PSNR: {:.4}".format(np.array(psnr_meter.score).std()),"SSIM: {:.4}".format(np.array(ssim_meter.score).std()))
        print("------------------")
        
        # return {'NMSE': nmse_meter.avg, 'PSNR': psnr_meter.avg, 'SSIM':ssim_meter.avg}                  

        t2_nmse = np.array(nmse_meter.score).mean()
        t2_psnr = np.array(psnr_meter.score).mean()
        t2_ssim = np.array(ssim_meter.score).mean()
        
        t2_nmse_std = np.array(nmse_meter.score).std()
        t2_psnr_std = np.array(psnr_meter.score).std()
        t2_ssim_std = np.array(ssim_meter.score).std()
          
        print(f"[T2 MRI:] average NMSE: {t2_nmse}, std: {t2_nmse_std} average PSNR: {t2_psnr}, std: {t2_psnr_std}, average SSIM: {t2_ssim}, std: {t2_ssim_std}")
        