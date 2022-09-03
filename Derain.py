from Unet import UNet
import numpy as np
import time
import datetime
import torch
from torch import nn
import torchvision
from torchvision import transforms
import os
from tqdm import tqdm
import torch.optim as optim
import utils
import losses
from torch.optim import lr_scheduler
from data_RGB import get_training_data, get_validation_data
from warmup_scheduler import GradualWarmupScheduler

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    model = UNet().to(device)
    # print(model)

    '''##############################################加载数据#########################################################################'''
    NUM_EPOCHS = 100
    BATCH_SIZE = 16
    patch_size = 128
    train_dir = "C:/project/pro/Gan/dataset/rain_data_train_Heavy/"
    ######### DataLoaders ###########
    train_dataset = get_training_data(train_dir, {'patch_size':patch_size})
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=False, pin_memory=True)
    val_dir = "C:/project/pro/Gan/dataset/rain_data_test_Heavy/"
    val_dataset = get_validation_data(val_dir, {'patch_size':patch_size})
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=16, shuffle=False, num_workers=0, drop_last=False, pin_memory=True)

    '''##############################################优化器#########################################################################'''
    optimizer = optim.Adam(model.parameters(), lr=2e-4, betas=(0.9, 0.999), eps=1e-8)

    ######### Scheduler ###########
    warmup_epochs = 3
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, NUM_EPOCHS - warmup_epochs,
                                                            eta_min=1e-6)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs,
                                       after_scheduler=scheduler_cosine)
    scheduler.step()

    ######### Loss ###########
    criterion_char = losses.CharbonnierLoss()
    criterion_edge = losses.EdgeLoss()
    criterion = nn.MSELoss()


    best_psnr = 0
    best_epoch = 0
    model_dir = "C:/project/pro//Gan/model/"
    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start_time = time.time()
        epoch_loss = 0
        train_id = 1

        model.train()
        for i, data in enumerate(tqdm(train_loader), 0):
            # zero_grad
            for param in model.parameters():
                param.grad = None

            target = data[0].to(device)
            input_ = data[1].to(device)

            restored = model(input_)

            # Compute loss at each stage
            # loss_char = np.sum([criterion_char(restored[j], target) for j in range(len(restored))])
            # loss_edge = np.sum([criterion_edge(restored[j], target) for j in range(len(restored))])
            # loss = (loss_char) + (0.05 * loss_edge)
            loss = criterion(restored, target)
            print("loss: ", loss)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        #### Evaluation ####
        if epoch % 1 == 0:
            model.eval()
            psnr_val_rgb = []
            for ii, data_val in enumerate((val_loader), 0):
                target = data_val[0].to(device)
                input_ = data_val[1].to(device)

                with torch.no_grad():
                    restored = model(input_)
                restored = restored[0]

                for res, tar in zip(restored, target):
                    psnr_val_rgb.append(utils.torchPSNR(res, tar))

            psnr_val_rgb = torch.stack(psnr_val_rgb).mean().item()

            if psnr_val_rgb > best_psnr:
                best_psnr = psnr_val_rgb
                best_epoch = epoch
                torch.save({'epoch': epoch,
                            'state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict()
                            }, os.path.join(model_dir, "model_best.pth"))

            print("[epoch %d PSNR: %.4f --- best_epoch %d Best_PSNR %.4f]" % (epoch, psnr_val_rgb, best_epoch, best_psnr))

            torch.save({'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict()
                        }, os.path.join(model_dir, f"model_epoch_{epoch}.pth"))

        scheduler.step()

        print("------------------------------------------------------------------")
        print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.8f}".format(epoch, time.time() - epoch_start_time,
                                                                                  epoch_loss, scheduler.get_lr()[0]))
        print("------------------------------------------------------------------")

        torch.save({'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                    }, os.path.join(model_dir, "model_latest.pth"))















