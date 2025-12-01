import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
import argparse
from torch.optim.lr_scheduler import StepLR


from models_multiscale_dinov2 import DINOv2_Encoder
from models_DualDomainFusion_SF2 import *
from datasets import *
from utils import *
from eval import evaluate_transformer


def train(args, train_loader, encoder_image,encoder_feat, decoder, criterion, encoder_image_optimizer,encoder_image_lr_scheduler,encoder_feat_optimizer,encoder_feat_lr_scheduler, decoder_optimizer, decoder_lr_scheduler, epoch):
    encoder_image.train()
    encoder_feat.train()
    decoder.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top5accs_our = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()

    best_bleu4 = 0.

    for i, (img_pairs, caps, caplens) in enumerate(train_loader):
        data_time.update(time.time() - start)

        img_pairs = img_pairs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)

        imgs_A = img_pairs[:, 0, :, :, :]
        imgs_B = img_pairs[:, 1, :, :, :]
        imgs_A = encoder_image(imgs_A)
        imgs_B = encoder_image(imgs_B)

        feat_out = encoder_feat(imgs_A,imgs_B)
        scores, caps_sorted, decode_lengths, sort_ind = decoder(feat_out, caps, caplens)

        targets = caps_sorted[:, 1:]

        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

        loss = criterion(scores, targets)

        decoder_optimizer.zero_grad()
        encoder_feat_optimizer.zero_grad()
        if encoder_image_optimizer is not None:
            encoder_image_optimizer.zero_grad()
        loss.backward()

        if args.grad_clip is not None:
            clip_gradient(decoder_optimizer, args.grad_clip)
            if encoder_image_optimizer is not None:
                clip_gradient(encoder_image_optimizer, args.grad_clip)

        decoder_optimizer.step()
        decoder_lr_scheduler.step()
        encoder_feat_optimizer.step()
        encoder_feat_lr_scheduler.step()
        if encoder_image_optimizer is not None:
            encoder_image_optimizer.step()
            encoder_image_lr_scheduler.step()

        top5 = accuracy(scores, targets, 1)
        losses.update(loss.item(), sum(decode_lengths))
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)

        start = time.time()
        if i % args.print_freq == 0:
            print("Epoch: {}/{} step: {}/{} Loss: {} AVG_Loss: {} Top-5 Accuracy: {} Batch_time: {}s".format(epoch+0, args.epochs, i+0, len(train_loader), losses.val, losses.avg, top5accs.val, batch_time.val))



def main(args):

    print(time.strftime("%m-%d  %H : %M : %S", time.localtime(time.time())))

    start_epoch = 0
    best_bleu4 = 0.
    epochs_since_improvement = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = True

    word_map_file = os.path.join(args.data_folder, 'WORDMAP_' + args.data_name + '.json')
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)

    encoder_image = DINOv2_Encoder(NetType=args.encoder_image, method=args.decoder)
    encoder_image.fine_tune(args.fine_tune_encoder)

    encoder_image_dim = 1024


    if args.encoder_feat == 'DualDomainTransformer':
        encoder_feat = DualDomainTransformer(feature_dim=encoder_image_dim, dropout=0.5, h=14, w=14, d_model=512, n_head=args.n_heads,
                               n_layers=args.n_layers)

    if args.decoder == 'trans':
        decoder = DecoderTransformer(feature_dim=args.feature_dim_de,
                                     vocab_size=len(word_map),
                                     n_head=args.n_heads,
                                     n_layers=args.decoder_n_layers,
                                     dropout=args.dropout)


    encoder_image_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder_image.parameters()),
                                         lr=args.encoder_lr) if args.fine_tune_encoder else None
    encoder_image_lr_scheduler = StepLR(encoder_image_optimizer, step_size=900, gamma=1) if args.fine_tune_encoder else None

    encoder_feat_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder_feat.parameters()),
                                         lr=args.encoder_lr)
    encoder_feat_lr_scheduler = StepLR(encoder_feat_optimizer, step_size=900, gamma=1)

    decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                         lr=args.decoder_lr)
    decoder_lr_scheduler = StepLR(decoder_optimizer,step_size=900,gamma=1)

    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        if isinstance(checkpoint['bleu-4'], dict):
            best_bleu4 = checkpoint['bleu-4'].get('Bleu_4', 0.)
        else:
            best_bleu4 = checkpoint['bleu-4']
        
        if isinstance(checkpoint['encoder_image'], CNN_Encoder):
            encoder_image = checkpoint['encoder_image']
        else:
            encoder_image.load_state_dict(checkpoint['encoder_image'])
            
        if isinstance(checkpoint['encoder_feat'], DualDomainTransformer):
            encoder_feat = checkpoint['encoder_feat']
        else:
            encoder_feat.load_state_dict(checkpoint['encoder_feat'])
            
        if isinstance(checkpoint['decoder'], DecoderTransformer):
            decoder = checkpoint['decoder']
        else:
            decoder.load_state_dict(checkpoint['decoder'])
        
        if encoder_image_optimizer is not None:
            if 'encoder_image_optimizer' in checkpoint:
                if isinstance(checkpoint['encoder_image_optimizer'], torch.optim.Optimizer):
                    encoder_image_optimizer = checkpoint['encoder_image_optimizer']
                else:
                    encoder_image_optimizer.load_state_dict(checkpoint['encoder_image_optimizer'])
                for param_group in encoder_image_optimizer.param_groups:
                    param_group['lr'] = args.encoder_lr
        
        if 'encoder_feat_optimizer' in checkpoint:
            if isinstance(checkpoint['encoder_feat_optimizer'], torch.optim.Optimizer):
                encoder_feat_optimizer = checkpoint['encoder_feat_optimizer']
            else:
                encoder_feat_optimizer.load_state_dict(checkpoint['encoder_feat_optimizer'])
            for param_group in encoder_feat_optimizer.param_groups:
                param_group['lr'] = args.encoder_lr
        
        if 'decoder_optimizer' in checkpoint:
            if isinstance(checkpoint['decoder_optimizer'], torch.optim.Optimizer):
                decoder_optimizer = checkpoint['decoder_optimizer']
            else:
                decoder_optimizer.load_state_dict(checkpoint['decoder_optimizer'])
            for param_group in decoder_optimizer.param_groups:
                param_group['lr'] = args.decoder_lr
        
        print(f"Loaded checkpoint from {args.checkpoint}")
        print(f"Resuming training from epoch {start_epoch}")
        print(f"Best BLEU-4 score so far: {best_bleu4}")


    encoder_image = encoder_image.to(device)
    encoder_feat = encoder_feat.to(device)
    decoder = decoder.to(device)

    print("Checkpoint_savepath:{}".format(args.savepath))
    print("Encoder_image_mode:{}   Encoder_feat_mode:{}   Decoder_mode:{}".format(args.encoder_image,args.encoder_feat,args.decoder))
    print("encoder_layers {} decoder_layers {} n_heads {} dropout {} encoder_lr {} "
          "decoder_lr {}".format(args.n_layers, args.decoder_n_layers, args.n_heads, args.dropout,
                                         args.encoder_lr, args.decoder_lr))

    criterion = nn.CrossEntropyLoss(ignore_index=0).to(device)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_loader = torch.utils.data.DataLoader(
        CaptionDataset(args.data_folder, args.data_name, 'TRAIN', transform=transforms.Compose([normalize])),
        batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)

    for epoch in range(start_epoch, args.epochs):

        if epochs_since_improvement == args.stop_criteria:
            print("the model has not improved in the last {} epochs".format(args.stop_criteria))
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 3 == 0:
            adjust_learning_rate(decoder_optimizer, 0.7)
            if args.fine_tune_encoder and encoder_image_optimizer is not None:
                print(encoder_image_optimizer)

        print(time.strftime("%m-%d  %H : %M : %S", time.localtime(time.time())))
        train(args,
              train_loader=train_loader,
              encoder_image=encoder_image,
              encoder_feat=encoder_feat,
              decoder=decoder,
              criterion=criterion,
              encoder_image_optimizer=encoder_image_optimizer,
              encoder_image_lr_scheduler=encoder_image_lr_scheduler,
              encoder_feat_optimizer=encoder_feat_optimizer,
              encoder_feat_lr_scheduler=encoder_feat_lr_scheduler,
              decoder_optimizer=decoder_optimizer,
              decoder_lr_scheduler=decoder_lr_scheduler,
              epoch=epoch)

        metrics = evaluate_transformer(args,
                            encoder_image=encoder_image,
                            encoder_feat=encoder_feat,
                           decoder=decoder)

        recent_bleu4 = metrics["Bleu_4"]
        is_best = recent_bleu4 > best_bleu4
        best_bleu4 = max(recent_bleu4, best_bleu4)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        checkpoint_name = args.encoder_image + '_'+args.encoder_feat + '_' + args.decoder
        save_checkpoint(args, checkpoint_name, epoch, epochs_since_improvement, encoder_image,encoder_feat, decoder,
                        encoder_image_optimizer,encoder_feat_optimizer,decoder_optimizer, metrics, is_best)


if __name__ == '__main__':
    torch.cuda.set_device(0)
    parser = argparse.ArgumentParser(description='Image_Change_Captioning')

    # Data parameters
    parser.add_argument('--data_folder', default="./data/",help='folder with data files saved by create_input_files.py.')
    parser.add_argument('--data_name', default="LEVIR_CC_5_cap_per_img_5_min_word_freq",help='base name shared by data files.')

    parser.add_argument('--encoder_image', default='dinov2_vitb14', help='which model does encoder use?')
    parser.add_argument('--encoder_feat', default='DualDomainTransformer')
    parser.add_argument('--decoder', default='trans')
    parser.add_argument('--n_heads', type=int, default=8, help='Multi-head attention in Transformer.')
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--decoder_n_layers', type=int, default=1)
    parser.add_argument('--feature_dim_de', type=int, default=1024)
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout')

    parser.add_argument('--epochs', type=int, default=40, help='number of epochs to train for (if early stopping is not triggered).')
    parser.add_argument('--stop_criteria', type=int, default=10, help='training stop if epochs_since_improvement == stop_criteria')
    parser.add_argument('--batch_size', type=int, default=2, help='batch_size')
    parser.add_argument('--print_freq', type=int, default=100, help='print training/validation stats every __ batches.')
    parser.add_argument('--workers', type=int, default=0, help='for data-loading; right now, only 0 works with h5pys in windows.')
    parser.add_argument('--encoder_lr', type=float, default=1e-4, help='learning rate for encoder if fine-tuning.')
    parser.add_argument('--decoder_lr', type=float, default=1e-4, help='learning rate for decoder.')
    parser.add_argument('--grad_clip', type=float, default=5., help='clip gradients at an absolute value of.')
    parser.add_argument('--fine_tune_encoder', type=bool, default=False, help='whether fine-tune encoder or not')
    parser.add_argument('--checkpoint', default=None, help='path to checkpoint, None if none.')

    parser.add_argument('--Split', default="VAL", help='which')
    parser.add_argument('--beam_size', type=int, default=32, help='beam_size.')
    parser.add_argument('--savepath', default="./saved_model/")

    args = parser.parse_args()
    main(args)