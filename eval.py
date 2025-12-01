import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from datasets import *
from utils import *
from models_multiscale_dinov2 import DINOv2_Encoder
from models_DualDomainFusion_SF2 import DualDomainTransformer
from models_DualDomainFusion_SF2 import DecoderTransformer
import torch.nn.functional as F
from tqdm import tqdm
import argparse
import time
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_captions(args, word_map, hypotheses, references):
    result_json_file = {}
    reference_json_file = {}
    kkk = -1
    for item in hypotheses:
        kkk += 1
        line_hypo = ""

        for word_idx in item:
            word = get_key(word_map, word_idx)
            line_hypo += word[0] + " "

        result_json_file[str(kkk)] = []
        result_json_file[str(kkk)].append(line_hypo)

        line_hypo += "\r\n"

    kkk = -1
    for item in tqdm(references):
        kkk += 1

        reference_json_file[str(kkk)] = []

        for sentence in item:
            line_repo = ""
            for word_idx in sentence:
                word = get_key(word_map, word_idx)
                line_repo += word[0] + " "
            reference_json_file[str(kkk)].append(line_repo)
            line_repo += "\r\n"

    with open('eval_results_fortest/' + args.Split +'/'+args.encoder_image + "_" +args.encoder_feat+"_" +args.decoder + '_res.json', 'w') as f:
        json.dump(result_json_file, f)

    with open('eval_results_fortest/' + args.Split +'/'+ args.encoder_image + "_" +args.encoder_feat+"_" +args.decoder + '_gts.json', 'w') as f:
        json.dump(reference_json_file, f)

def get_key(dict_, value):
  return [k for k, v in dict_.items() if v == value]

def evaluate_transformer(args,encoder_image,encoder_feat,decoder):
    encoder_image = encoder_image.to(device)
    encoder_image.eval()
    encoder_feat = encoder_feat.to(device)
    encoder_feat.eval()
    decoder = decoder.to(device)
    decoder.eval()

    word_map_file = os.path.join(args.data_folder, 'WORDMAP_' + args.data_name + '.json')
    with open(word_map_file, 'r') as f:
        word_map = json.load(f)

    rev_word_map = {v: k for k, v in word_map.items()}
    vocab_size = len(word_map)

    beam_size = args.beam_size
    Caption_End = False
    loader = torch.utils.data.DataLoader(
        CaptionDataset(args.data_folder, args.data_name, args.Split, transform=transforms.Compose([normalize])),
        batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    references = list()
    hypotheses = list()
    change_references = list()
    change_hypotheses = list()
    nochange_references = list()
    nochange_hypotheses = list()
    change_acc=0
    nochange_acc=0

    with torch.no_grad():
        for i, (image_pairs, caps, caplens, allcaps) in enumerate(
                tqdm(loader, desc=args.Split + " EVALUATING AT BEAM SIZE " + str(beam_size))):
            if (i + 1) % 5 != 0:
                continue
            k = beam_size

            image_pairs = image_pairs.to(device)

            imgs_A = image_pairs[:, 0, :, :, :]
            imgs_B = image_pairs[:, 1, :, :, :]
            imgs_A = encoder_image(imgs_A)
            imgs_B = encoder_image(imgs_B)
            encoder_out = encoder_feat(imgs_A, imgs_B)

            tgt = torch.zeros(52, k).to(device).to(torch.int64)
            tgt_length = tgt.size(0)
            mask = (torch.triu(torch.ones(tgt_length, tgt_length)) == 1).transpose(0, 1)
            mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
            mask = mask.to(device)

            tgt[0, :] = torch.LongTensor([word_map['<start>']]*k).to(device)
            seqs = torch.LongTensor([[word_map['<start>']]*1] * k).to(device)
            top_k_scores = torch.zeros(k, 1).to(device)
            complete_seqs = []
            complete_seqs_scores = []
            step = 1

            k_prev_words = tgt.permute(1,0)
            S = encoder_out.size(0)
            encoder_dim = encoder_out.size(-1)

            encoder_out = encoder_out.expand(S,k, encoder_dim)
            encoder_out = encoder_out.permute(1,0,2)

            while True:
                tgt = k_prev_words.permute(1,0)
                tgt_embedding = decoder.vocab_embedding(tgt)
                tgt_embedding = decoder.position_encoding(tgt_embedding)

                encoder_out = encoder_out.permute(1, 0, 2)
                pred = decoder.transformer(tgt_embedding, encoder_out, tgt_mask=mask)
                encoder_out = encoder_out.permute(1, 0, 2)
                pred = decoder.wdc(pred)
                scores = pred.permute(1,0,2)
                scores = scores[:, step - 1, :].squeeze(1)
                scores = F.log_softmax(scores, dim=1)
                scores = top_k_scores.expand_as(scores) + scores
                if step == 1:
                    top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)
                else:
                    top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)

                prev_word_inds = torch.div(top_k_words, vocab_size, rounding_mode='floor')
                next_word_inds = top_k_words % vocab_size

                seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)
                incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                                   next_word != word_map['<end>']]
                complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))
                if len(complete_inds) > 0:
                    Caption_End = True
                    complete_seqs.extend(seqs[complete_inds].tolist())
                    complete_seqs_scores.extend(top_k_scores[complete_inds])
                k -= len(complete_inds)
                if k == 0:
                    break
                seqs = seqs[incomplete_inds]
                encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
                top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
                k_prev_words = k_prev_words[incomplete_inds]
                k_prev_words[:, :step + 1] = seqs
                if step > 50:
                    break
                step += 1

            if (len(complete_seqs_scores) == 0):
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            if (len(complete_seqs_scores) > 0):
                assert Caption_End
                indices = complete_seqs_scores.index(max(complete_seqs_scores))
                seq = complete_seqs[indices]
                img_caps = allcaps[0].tolist()
                img_captions = list(
                    map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}],
                        img_caps))

                references.append(img_captions)
                new_sent = [w for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}]
                hypotheses.append(new_sent)
                assert len(references) == len(hypotheses)

                nochange_list = ["the scene is the same as before ", "there is no difference ",
                                 "the two scenes seem identical ", "no change has occurred ",
                                 "almost nothing has changed "]
                ref_sentence = img_captions[1]
                ref_line_repo = ""
                for ref_word_idx in ref_sentence:
                    ref_word = get_key(word_map, ref_word_idx)
                    ref_line_repo += ref_word[0] + " "

                hyp_sentence = new_sent
                hyp_line_repo = ""
                for hyp_word_idx in hyp_sentence:
                    hyp_word = get_key(word_map, hyp_word_idx)
                    hyp_line_repo += hyp_word[0] + " "
                if ref_line_repo not in nochange_list:
                    change_references.append(img_captions)
                    change_hypotheses.append(new_sent)
                    if hyp_line_repo not in nochange_list:
                        change_acc = change_acc+1
                else:
                    nochange_references.append(img_captions)
                    nochange_hypotheses.append(new_sent)
                    if hyp_line_repo in nochange_list:
                        nochange_acc = nochange_acc+1

    print('len(nochange_references):', len(nochange_references))
    print('len(change_references):', len(change_references))
    if len(nochange_references)>0:
        print('nochange_metric:')
        nochange_metric = get_eval_score(nochange_references, nochange_hypotheses)
        print("nochange_acc:", nochange_acc / len(nochange_references))
    if len(change_references)>0:
        print('change_metric:')
        change_metric = get_eval_score(change_references, change_hypotheses)
        print("change_acc:", change_acc / len(change_references))
    print(".......................................................")
    metrics = get_eval_score(references, hypotheses)

    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Change_Captioning')
    parser.add_argument('--data_folder', default="./data/",help='folder with data files saved by create_input_files.py.')
    parser.add_argument('--data_name', default="LEVIR_CC_5_cap_per_img_5_min_word_freq",help='base name shared by data files.')

    parser.add_argument('--encoder_image', default="dinov2_vitl14")
    parser.add_argument('--encoder_feat', default="DualDomainTransformer")
    parser.add_argument('--decoder', default="trans", help="decoder img2txt")
    parser.add_argument('--Split', default="TEST", help='which')
    parser.add_argument('--epoch', default="epoch", help='which')
    parser.add_argument('--beam_size', type=int, default=1, help='beam_size.')
    parser.add_argument('--path', default="./saved_model/", help='model checkpoint.')
    args = parser.parse_args()
    word_map_file = os.path.join(args.data_folder, 'WORDMAP_' + args.data_name + '.json')
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)
    filename = os.listdir(args.path)
    for i in range(len(filename)):
        print(time.strftime("%m-%d  %H : %M : %S", time.localtime(time.time())))

        checkpoint_path = os.path.join(args.path, filename[i])
        print(args.path + filename[i])

        encoder_image = DINOv2_Encoder(model_name='dinov2_vitl14').to(device)
        checkpoint = torch.load(checkpoint_path, map_location=str(device),weights_only=False)
        encoder_image.load_state_dict(checkpoint['encoder_image'].state_dict())

        encoder_feat = checkpoint['encoder_feat']
        decoder = checkpoint['decoder']

        if args.decoder == "trans":
            metrics = evaluate_transformer(args, encoder_image, encoder_feat, decoder)

        print("{} - beam size {}: BLEU-1 {} BLEU-2 {} BLEU-3 {} BLEU-4 {} METEOR {} ROUGE_L {} CIDEr {}".format
              (args.decoder, args.beam_size, metrics["Bleu_1"], metrics["Bleu_2"], metrics["Bleu_3"],
               metrics["Bleu_4"],
               metrics["METEOR"], metrics["ROUGE_L"], metrics["CIDEr"]))
        print("\n")
        print("\n")
        time.sleep(10)