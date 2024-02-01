# coding:utf-8
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from tqdm import tqdm
import argparse
import warnings
import json
from utils import *
from model.utils import DataLoader
from model.DLAN_AC import *
from Evaluate import auc_cal

warnings.filterwarnings("ignore")
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


def AC_Pre_Clustering():
    # Model setting
    model = convAE(args.c, args.t_length, args.fdim[0], args.pdim[0], args.AC_clustering, args.side_length_som,
                   args.sigma_som, args.lr_som, args.neighborhood_function_som)
    params_encoder = list(model.encoder.parameters())
    params_decoder = list(model.decoder.parameters())
    params_AE = params_encoder + params_decoder

    optimizer = torch.optim.Adam(params_AE, lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    model.to(device)


    if os.path.exists(args.resume_AC_model):
        print('Resume model from ' + args.resume_AC_model)
        ckpt = args.resume_AC_model
        checkpoint = torch.load(ckpt)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'].state_dict())
        optimizer.load_state_dict(checkpoint['optimizer'])

    # Report the training process
    AC_log_dir = os.path.join('./exp', args.dataset_type,'AC_results')
    if not os.path.exists(AC_log_dir):
        os.makedirs(AC_log_dir)

    model_save_dir = os.path.join('./exp', args.dataset_type, 'AC_results/model')
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    sys.stdout = Logger(f'{AC_log_dir}/AC_train_{args.AC_log_name}.txt')


    loss_func_mse = nn.MSELoss(reduction='none')
    loss_pix = AverageMeter()
    gradient_loss = Gradient_Loss(3)
    loss_gd = AverageMeter()

    # Training
    model.train()
    # obtain initial cluster number and  initialize DLAN weights; 10 epoch
    start_epoch = 0
    for epoch in range(start_epoch, args.epochs_AC_clustering):
        ac_cluster_num = AverageMeter()
        pbar = tqdm(total=len(train_batch))
        for j, (imgs) in enumerate(train_batch):
            imgs = Variable(imgs).to(device)
            imgs = imgs.view(args.batch_size, -1, imgs.shape[-2], imgs.shape[-1])

            outputs, AC_cluster_number, AC_weights = model.forward(args, imgs[:, 0:12], True)

            optimizer.zero_grad()
            loss_pixel = torch.mean(loss_func_mse(outputs, imgs[:, 12:]))
            loss_gradient = gradient_loss(outputs, imgs[:, 12:])
            loss = args.loss_fra_reconstruct * loss_pixel + args.loss_gd * loss_gradient

            loss.backward(retain_graph=True)
            optimizer.step()

            loss_pix.update(loss_pixel.item(), 1)
            loss_gd.update(loss_gradient.item(), 1)
            ac_cluster_num.update(AC_cluster_number, 1)
            pbar.set_postfix({
                'Epoch': '{0} {1}'.format(epoch + 1, args.AC_log_name),
                'Lr': '{:.6f}'.format(optimizer.param_groups[-1]['lr']),
                'PRe': '{:.6f}({:.6f})'.format(loss_pixel.item(), loss_pix.avg),
                'Gd': '{:.6f}({:.6f})'.format(loss_gradient.item(), loss_gd.avg),
                'Ac_cluster_num': '{:.6f}({:.6f})'.format(AC_cluster_number, ac_cluster_num.avg),
            })
            pbar.update(1)
        scheduler.step()
        print('----------------------------------------')
        print('Epoch:', epoch + 1)
        print('Lr: {:.6f}'.format(optimizer.param_groups[-1]['lr']))
        print('PRe: {:.6f}({:.6f})'.format(loss_pixel.item(), loss_pix.avg))
        print('Gd: {:.6f}({:.6f})'.format(loss_gradient.item(), loss_gd.avg))
        print('Ac_cluster_num:{:.6f}({:.6f})'.format(AC_cluster_number, ac_cluster_num.avg))
        print('----------------------------------------')
        cluster_num = ac_cluster_num.avg
        pbar.close()

        loss_pix.reset()
        ac_cluster_num.reset()
        loss_gd.reset()
        # Save the model
        if epoch % 1 == 0:

            if len(args.gpus[0]) > 1:
                model_save = model.module
            else:
                model_save = model

            state = {
                'epoch': epoch,
                'state_dict': model_save,
                'optimizer': optimizer.state_dict(),
            }
            torch.save(state, os.path.join('./exp', args.dataset_type,'AC_results/model', 'model_AC_' + str(epoch) + '.pth'))
            AC_Cluster_Num = cluster_num
            # print(AC_Cluster_Num)
            np.save(f"exp/{args.dataset_type}/AC_results/ac_weights_{args.AC_log_name}_epoch{epoch}.npy", AC_weights)
    AC_cluster_num = {"AC_Cluster_Num":int(AC_Cluster_Num)}
    with open(f"exp/{args.dataset_type}/AC_results/ac_cluster_results_{args.AC_log_name}.json", "w") as f:
        json.dump(AC_cluster_num, f)
        print("AC clustering completed...")

    return AC_Cluster_Num, AC_weights


def Formal_Training():

    args.AC_clustering = False
    # Model setting
    model = convAE(args.c, args.t_length, args.fdim[0], args.pdim[0], args.AC_clustering, args.side_length_som,
                       args.sigma_som, args.lr_som, args.neighborhood_function_som, AC_Cluster_Num, AC_Weights)
    model.to(device)

    params_encoder = list(model.encoder.parameters())
    params_decoder = list(model.decoder.parameters())
    params_ohead = list(model.ohead.parameters())
    params_netvlad_drcs = list(model.netvlad_drcs.parameters())

    params = params_encoder + params_decoder + params_netvlad_drcs + params_ohead

    optimizer = torch.optim.Adam(params, lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch = 0
    if os.path.exists(args.resume):
        print('Resume model from ' + args.resume)
        ckpt = args.resume
        checkpoint = torch.load(ckpt)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'].state_dict())
        optimizer.load_state_dict(checkpoint['optimizer'])

    if len(args.gpus[0]) > 1:
        model = nn.DataParallel(model)

    # Report the training process
    log_dir = os.path.join('./exp', args.dataset_type, args.exp_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    loss_func_mse = nn.MSELoss(reduction='none')
    gradient_loss = Gradient_Loss(3)
    loss_gd = AverageMeter()
    loss_pix = AverageMeter()
    loss_compactness = AverageMeter()
    loss_separateness = AverageMeter()
    # Training
    model.train()
    best_auc = 0
    writer = SummaryWriter(f'tensorboard_log/{args.dataset_type}/{args.exp_dir}')
    for epoch in range(start_epoch, args.epochs):
        model.train()
        pbar = tqdm(total=len(train_batch))
        for j, (imgs) in enumerate(train_batch):
            imgs = Variable(imgs).cuda()
            imgs = imgs.view(args.batch_size, -1, imgs.shape[-2], imgs.shape[-1])

            outputs, compactness_loss, separateness_loss = model.forward( args , imgs[:, 0:12], True)

            optimizer.zero_grad()
            loss_pixel = torch.mean(loss_func_mse(outputs, imgs[:, 12:]))
            loss_gradient = gradient_loss(outputs, imgs[:, 12:])
            compactness_loss = compactness_loss.mean()
            separateness_loss = separateness_loss.mean()

            loss = args.loss_fra_reconstruct * loss_pixel + args.loss_compact * compactness_loss + args.loss_separate * separateness_loss
            loss.backward(retain_graph=True)
            optimizer.step()

            loss_pix.update(args.loss_fra_reconstruct * loss_pixel.item(), 1)
            loss_gd.update(args.loss_gd * loss_gradient.item(), 1)
            loss_compactness.update(args.loss_compact * compactness_loss.item(), 1)
            loss_separateness.update(args.loss_separate * separateness_loss.item(), 1)

            pbar.set_postfix({
                'Epoch': '{0} {1}'.format(epoch + 1, args.exp_dir),
                'Lr': '{:.6f}'.format(optimizer.param_groups[-1]['lr']),
                'RC_loss': '{:.6f}({:.6f})'.format(loss_pixel.item(), loss_pix.avg),
                'Gd_loss': '{:.6f}({:.6f})'.format(loss_gradient.item(), loss_gd.avg),
                'CP_loss': '{:.8f}({:.8f})'.format(compactness_loss.item(), loss_compactness.avg),
                'SP_loss': '{:.8f}({:.8f})'.format(separateness_loss.item(), loss_separateness.avg),
            })
            pbar.update(1)
        scheduler.step()

        print('----------------------------------------')
        print('Epoch:', epoch + 1)
        print('Lr: {:.6f}'.format(optimizer.param_groups[-1]['lr']))
        print('RC_loss: {:.6f}({:.6f})'.format(loss_pixel.item(), loss_pix.avg))
        print('Gd_loss: {:.6f}({:.6f})'.format(loss_gradient.item(), loss_gd.avg))
        print('CP_loss: {:.6f}({:.6f})'.format(compactness_loss.item(), loss_compactness.avg))
        print('SP_loss: {:.10f}({:.10f})'.format(separateness_loss.item(), loss_separateness.avg))
        print('----------------------------------------')


        writer.add_scalar('loss/RC_loss', loss_pix.avg, global_step=epoch)
        writer.add_scalar('loss/Gd_loss', loss_gd.avg, global_step=epoch)
        writer.add_scalar('loss/CP_loss', loss_compactness.avg, global_step=epoch)
        writer.add_scalar('loss/SP_loss', loss_separateness.avg, global_step=epoch)

        pbar.close()

        loss_pix.reset()
        loss_compactness.reset()
        loss_separateness.reset()

        # Save the model
        if epoch % 10 == 0:

            if len(args.gpus[0]) > 1:
                model_save = model.module
            else:
                model_save = model

            state = {
                'epoch': epoch,
                'state_dict': model_save,
                'optimizer': optimizer.state_dict(),
            }
            torch.save(state, os.path.join(log_dir, 'model_' + str(epoch) + '.pth'))
            auc = auc_cal(epoch)
            writer.add_scalar('results/auc', auc, global_step=epoch)
            if auc >= 96.0:
                torch.save(state, os.path.join(log_dir, 'model_' + str(epoch) + '_' + str(args.log_name) + '_' + str(auc)+'_.pth'))
            if auc > best_auc:
                best_auc = auc
                best_epoch = epoch
                writer.add_scalar('results/best_auc', best_auc, global_step=epoch)
                print(f"Best auc is epoch {str(best_epoch)}:{best_auc}")

    print('Training is finished')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DLAN-AC")
    parser.add_argument('--gpus', nargs='+', type=str, default='0', help='gpus')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size for training')
    parser.add_argument('--test_batch_size', type=int, default=1, help='batch size for test')
    parser.add_argument('--epochs', type=int, default=300, help='number of epochs for training')
    parser.add_argument('--epochs_AC_clustering', type=int, default=10, help='number of epochs for AC pre-clustering')
    parser.add_argument('--loss_fra_reconstruct', type=float, default=1.00, help='weight of the frame reconstruction loss')
    parser.add_argument('--loss_gd', type=float, default=1.00, help='weight of the gradient loss')
    parser.add_argument('--loss_compact', type=float, default=0.01, help='weight of the feature compactness loss')
    parser.add_argument('--loss_separate', type=float, default=0.01, help='weight of the feature separateness loss')
    parser.add_argument('--h', type=int, default=256, help='height of input images')
    parser.add_argument('--w', type=int, default=256, help='width of input images')
    parser.add_argument('--c', type=int, default=3, help='channel of input images')
    parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')
    parser.add_argument('--t_length', type=int, default=5, help='length of the frame sequences')
    parser.add_argument('--fdim', type=list, default=[512], help='channel dimension of the features')
    parser.add_argument('--pdim', type=list, default=[512], help='channel dimension of the prototypes')
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers for the train loader')
    parser.add_argument('--num_workers_test', type=int, default=1, help='number of workers for the test loader')
    parser.add_argument('--dataset_type', type=str, default='ped2', help='type of dataset: ped2, avenue, shanghai')
    parser.add_argument('--dataset_path', type=str, default='data/', help='directory of data')
    parser.add_argument('--exp_dir', type=str, default='log1', help='directory of log')
    parser.add_argument('--log_name', type=str, default='log1', help='directory of log')
    parser.add_argument('--resume', type=str, default='', help='file path of resume pth')
    parser.add_argument('--resume_AC_model', type=str, default='', help='file path of resume AC pth')
    parser.add_argument('--AC_clustering', type=bool, default=False, help='if AC_clustering')
    parser.add_argument('--side_length_som', type=int, default=5, help='side length of SOM in AC')
    parser.add_argument('--sigma_som', type=float, default=0.3, help='sigma of SOM in AC')
    parser.add_argument('--lr_som', type=float, default=0.5, help='learning rate for AC')
    parser.add_argument('--AC_log_name', type=str, default='log_1_test', help='directory of AC clustering log')
    parser.add_argument('--neighborhood_function_som', type=str, default='gaussian', help='neighborhood_function of SOM in AC')
    args = parser.parse_args()

    manual_seed(2022)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if args.gpus is None:
        gpus = "0"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    else:
        gpus = ""
        for i in range(len(args.gpus)):
            gpus = gpus + args.gpus[i] + ","
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus[:-1]
    device = torch.device('cuda' if args.gpus is not None else 'cpu')

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    # ***********Loading dataset***********
    print("Loading dataset")
    train_folder = args.dataset_path + args.dataset_type + "/training/frames"

    train_dataset = DataLoader(train_folder, transforms.Compose([transforms.ToTensor(), ]),
                               resize_height=args.h, resize_width=args.w, time_step=args.t_length - 1)
    train_batch = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                  drop_last=True)

    # ***********AC pre-clustering***********

    if args.AC_clustering:
        print("AC pre-clustering")
        AC_Weights, AC_Cluster_Num = AC_Pre_Clustering()
    else:
        AC_Cluster_Num = 13
        AC_Weights = np.load(f'exp/{args.dataset_type}/AC_results/ac_weights_log_1_test_epoch0.npy')
    # ***********Formal training*************
    print("Formal training")
    Formal_Training()

