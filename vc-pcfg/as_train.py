import os
import re
import time, pickle, argparse, logging
import numpy as np
import torch
#from transformers import get_scheduler, AutoTokenizer, AutoModelForCausalLM, AutoConfig, OPTConfig

import vpcfg.as_dataloader as data
from vpcfg.utils import Vocabulary, save_checkpoint
from vpcfg.as_evaluation import AverageMeter, LogCollector, semantic_bootstrapping_test, syntactic_bootstrapping_test
from vpcfg.as_vocab import get_vocab

def train(opt, train_loader, model, epoch, val_loader=None):
    # average meters to record the training statistics
    train_logger = LogCollector()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    nbatch = len(train_loader)
    # switch to train mode
    end = time.time()
    model.n_word = 0
    model.n_sent = 0
    model.s_time = end
    model.all_stats = [[0., 0., 0.]]

    if getattr(opt, 'use_temperature_annealing', False) and hasattr(model, 'parser'):
        anneal_span = max(1.0, opt.num_epochs * opt.temp_anneal_frac)
        frac = min(1.0, max(0.0, epoch / anneal_span))
        new_temp = opt.temp_start + (opt.temp_end - opt.temp_start) * frac
        model.parser.temperature = float(new_temp)
        model.parser.temp_mode = getattr(opt, 'temp_mode', 'all')
        logger.info(
            f"Epoch [{epoch}] PCFG temperature set to {new_temp:.4f} "
            f"(mode={model.parser.temp_mode})"
        )

    if getattr(opt, 'use_entropy_bonus', False):
        anneal_span = max(1.0, opt.num_epochs * opt.entropy_anneal_frac)
        frac = min(1.0, max(0.0, epoch / anneal_span))
        current_w = opt.entropy_weight * (1.0 - frac)
        model.entropy_weight = float(current_w)
        logger.info(
            f"Epoch [{epoch}] entropy bonus weight set to {current_w:.4f} "
            f"(base={opt.entropy_weight}, anneal_frac={opt.entropy_anneal_frac})"
        )
    for i, train_data in enumerate(train_loader):
        # Always reset to train mode
        model.train()
        # measure data loading time
        data_time.update(time.time() - end)
        # make sure train logger is used
        model.logger = train_logger
        # Update the model
        info = model.forward(*train_data, epoch=epoch)
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # Print log info
        if model.niter % opt.log_step == 0:
            logger.info(
                'Epoch: [{0}][{1}/{2}] {e_log} {info}'
                .format(
                    epoch, i, nbatch, e_log=str(model.logger), info=info
                )
            )
        #break
        # validate at every val_step
        if model.niter % opt.val_step == 0 and val_loader is not None:
            semantic_bootstrapping_test(opt, val_loader, model, logger, epoch, save=False)

    if "MT-Loss" in train_logger.meters:
        mt = train_logger.meters["MT-Loss"].avg
        ll = train_logger.meters["LL-Loss"].avg
        kl = train_logger.meters["KL-Loss"].avg
        extra = ""
        if "StructNeg-Loss" in train_logger.meters:
            extra += f", StructNeg: {train_logger.meters['StructNeg-Loss'].avg:.4f}"
        if "MI-Loss" in train_logger.meters:
            extra += f", MI: {train_logger.meters['MI-Loss'].avg:.4f}"
        if "Entropy" in train_logger.meters:
            extra += f", Entropy: {train_logger.meters['Entropy'].avg:.4f}"
        if "EntropyWeight" in train_logger.meters:
            extra += f", EntW: {train_logger.meters['EntropyWeight'].val:.4f}"
        if "Temp" in train_logger.meters:
            extra += f", Temp: {train_logger.meters['Temp'].val:.3f}"
        logger.info(
            "Epoch [%d] raw (unweighted) train losses — "
            "MT(semantic): %.4f, NLL(syntax): %.4f, KL(syntax): %.4f, "
            "Syntax_total(NLL+KL): %.4f%s",
            epoch, mt, ll, kl, ll + kl, extra
        )

if __name__ == '__main__':
    # hyper parameters
    parser = argparse.ArgumentParser()

    # Parser: Generative model parameters
    
    parser.add_argument('--z_dim', default=64, type=int, help='latent dimension')
    parser.add_argument('--t_states', default=60, type=int, help='number of preterminal states')
    parser.add_argument('--nt_states', default=30, type=int, help='number of nonterminal states')
    parser.add_argument('--state_dim', default=256, type=int, help='symbol embedding dimension')
    # Parser: Inference network parameters
    parser.add_argument('--h_dim', default=768, type=int, help='hidden dim for variational LSTM')
    parser.add_argument('--w_dim', default=768, type=int, help='embedding dim for variational LSTM')
    parser.add_argument('--gpu', default=1, type=int, help='which gpu to use')

    #
    parser.add_argument('--seed', default=527, type=int, help='random seed')
    parser.add_argument('--model_init', default=None, type=str, help='checkpoint to initialize model with')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from latest checkpoint in logger_name/checkpoints')
    #parser.add_argument('--w2vec_file', default=None, type=str, help='word vector file')
    parser.add_argument('--max_length', default=40, type=int, help='max sentence length')
    parser.add_argument('--prefix', default="all", type=str, help='prefix')
    #parser.add_argument('--parser_type', default='2nd', type=str, help='model name (1st/2nd)')
    #parser.add_argument('--share_w2vec', default=False, type=bool, help='shared embeddings')
    parser.add_argument('--visual_mode', action='store_true', help='run visual model')
    
    parser.add_argument('--encoder_file', default=None, help='image representations file name to use')
    parser.add_argument('--tiny', action='store_true', help='if testing will create tiny dataloaders')
    parser.add_argument('--one_shot', action='store_true', help='If split for ssyntactic bootstrapping test should be one-shot')
    parser.add_argument('--shuffle', action='store_true', help='shuffle training data')

    #
    parser.add_argument('--sem_dim', default=768, type=int, help='semantic rep. dim')
    parser.add_argument('--syn_dim', default=768, type=int, help='syntactic rep. dim')
    parser.add_argument('--word_dim', default=768, type=int,
                        help='dimensionality of the word embedding')
    parser.add_argument('--lstm_dim', default=768, type=int,
                        help='dimensionality of the lstm hidden embedding')
    
    parser.add_argument('--vocab_size', default=2000, type=int,
                        help='tokenizer/vocabulary size')
    parser.add_argument('--data_path', default='../preprocessed-data/abstractscenes', help='path to datasets')
    #parser.add_argument('--tokenizer_path', default='../../babylm-models/test/', help='path to pretrained tokenizer')
    #parser.add_argument('--save_model_path', default='../../babylm-models/test/', help='path to directory with model configs')
    parser.add_argument('--logger_name', default='../../../scratch/vcpcfg/runs/test', help='location for model outputs and logfiles to be saved')
    
    parser.add_argument('--margin', default=0.2, type=float,
                        help='rank loss margin')
    parser.add_argument('--num_epochs', default=100, type=int,
                        help='number of training epochs')
    parser.add_argument('--batch_size', default=5, type=int,
                        help='size of a training mini-batch')
    parser.add_argument('--grad_clip', default=3., type=float,
                        help='gradient clipping threshold')
    parser.add_argument('--lr', default=.001, type=float,
                        help='initial learning rate')
    parser.add_argument('--workers', default=1, type=int,
                        help='number of data loader workers')
    #
    parser.add_argument('--log_step', default=500, type=int,
                        help='number of steps to print and record the log')
    parser.add_argument('--val_step', default=float("inf"), type=int,
                        help='number of steps to run validation')
    #
    parser.add_argument('--img_dim', default=2048, type=int,
                        help='dimensionality of the image embedding')
    parser.add_argument('--no_imgnorm', action='store_true',
                        help='Do not normalize the image embeddings.')
    #
    parser.add_argument('--optimizer', type=str, default='Adam',
                        help='optimizer, can be Adam, SGD, etc.')
    parser.add_argument('--beta1', default=0.75, type=float, help='beta1 for adam')
    parser.add_argument('--beta2', default=0.999, type=float, help='beta2 for adam')
    #
    parser.add_argument('--vse_mt_alpha', type=float, default=1.0, help='weight parameter for matching loss')
    parser.add_argument('--vse_lm_alpha', type=float, default=1.0,  help='weight parameter for  loss')
    parser.add_argument('--sem_first', action='store_true', help='Run semantics first model')
    parser.add_argument('--syn_first', action='store_true', help='Run syntax first model')
    parser.add_argument('--switch_epoch', type=int, default=-1,
                        help='Epoch at which staged training switches phase. -1 means num_epochs/2.')
    parser.add_argument('--phase1_vse_mt_alpha', type=float, default=None,
                        help='Optional phase-1 matching loss weight for sem_first/syn_first.')
    parser.add_argument('--phase1_vse_lm_alpha', type=float, default=None,
                        help='Optional phase-1 syntax/LM loss weight for sem_first/syn_first.')
    parser.add_argument('--phase2_vse_mt_alpha', type=float, default=None,
                        help='Optional phase-2 matching loss weight for sem_first/syn_first.')
    parser.add_argument('--phase2_vse_lm_alpha', type=float, default=None,
                        help='Optional phase-2 syntax/LM loss weight for sem_first/syn_first.')
    parser.add_argument('--skip_syntactic_bootstrapping', action='store_true',
                        help='Skip syntactic bootstrapping evaluation and data split')

    parser.add_argument('--use_structural_negatives', action='store_true',
                        help='Add structural-negative term: the matching loss with a shuffled parse '
                             'should be higher than with the real parse by --struct_neg_margin.')
    parser.add_argument('--struct_neg_margin', type=float, default=0.1,
                        help='Margin for the structural-negatives hinge loss.')
    parser.add_argument('--struct_neg_weight', type=float, default=1.0,
                        help='Weight on the structural-negatives loss term.')

    parser.add_argument('--use_mi_regularizer', action='store_true',
                        help='Add MI-style regularizer: the matching loss with the real parse '
                             'should beat the uniform-marginals baseline by --mi_margin.')
    parser.add_argument('--mi_margin', type=float, default=0.1,
                        help='Margin for the MI regularizer (only used when --mi_style=hinge).')
    parser.add_argument('--mi_weight', type=float, default=1.0,
                        help='Weight on the MI regularizer loss term.')
    parser.add_argument('--mi_style', type=str, default='ratio',
                        choices=['hinge', 'ratio', 'infonce'],
                        help='MI regularizer form. "hinge": relu(margin+real-uniform) (old, inactive). '
                             '"ratio": real/(real+uniform) in (0,1), always active. '
                             '"infonce": -log(exp(-real)/(exp(-real)+exp(-uniform))), always active.')

    parser.add_argument('--use_temperature_annealing', action='store_true',
                        help='Anneal the PCFG softmax temperature from --temp_start to --temp_end '
                             'linearly over the first half of training.')
    parser.add_argument('--temp_start', type=float, default=2.0,
                        help='Initial PCFG softmax temperature (higher = flatter tree distribution).')
    parser.add_argument('--temp_end', type=float, default=1.0,
                        help='Final PCFG softmax temperature.')
    parser.add_argument('--temp_anneal_frac', type=float, default=0.5,
                        help='Fraction of num_epochs over which to linearly anneal the temperature.')
    parser.add_argument('--temp_mode', type=str, default='all',
                        choices=['all', 'rule_only'],
                        help='Which PCFG distributions to apply temperature to. '
                             '"all": root+term+rule (old behavior). '
                             '"rule_only": only the binary-rule distribution (preserves NLL, '
                             'only perturbs tree shape).')

    parser.add_argument('--use_entropy_bonus', action='store_true',
                        help='Add a decaying entropy bonus on the parser span-category distribution. '
                             'Prevents the parser from crystallizing too early.')
    parser.add_argument('--entropy_weight', type=float, default=0.1,
                        help='Initial weight on the entropy bonus (annealed to 0 over --entropy_anneal_frac).')
    parser.add_argument('--entropy_anneal_frac', type=float, default=0.5,
                        help='Fraction of num_epochs over which to linearly anneal the entropy weight to 0.')

    opt = parser.parse_args()
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    # setup logger
    if os.path.exists(opt.logger_name):
        print(f'Warning: the folder {opt.logger_name} exists.')
    else:
        print('Creating {}'.format(opt.logger_name))
        os.mkdir(opt.logger_name)
        os.mkdir(opt.logger_name+'/syntactic_bootstrapping_results')
        os.mkdir(opt.logger_name+'/semantic_bootstrapping_results')
        os.mkdir(opt.logger_name+'/checkpoints')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    log_path = os.path.join(opt.logger_name, 'train.log')
    log_mode = 'a' if opt.resume and os.path.exists(log_path) else 'w'
    handler = logging.FileHandler(log_path, log_mode)
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)
    logger.propagate = False
    logger.info('cuda:{}@{}'.format(opt.gpu, os.uname().nodename))
    logger.info(opt)

    # load predefined vocabulary and pretrained word embeddings if applicable
    #vocab = pickle.load(open(os.path.join(opt.data_path, opt.vocab_name), 'rb'))
    ## EP to get Abstract Scenes vocabulary
    vocab = get_vocab(opt.data_path)
    opt.vocab_size = len(vocab)
    logger.info("|vocab|={}".format(len(vocab)))

    # construct the model
    if not opt.visual_mode:
        from vpcfg.model import VGCPCFGs 
    else:
        logger.info("using visually-grounded version of model.")
        from vpcfg.model_vis import VGCPCFGs 
    sampler = True
    model = VGCPCFGs(opt, vocab, logger)
    start_epoch = 0
    best_rsum = float('inf')
    resumed = False
    if opt.resume:
        ckpt_dir = os.path.join(opt.logger_name, 'checkpoints')
        if os.path.isdir(ckpt_dir):
            candidates = []
            for name in os.listdir(ckpt_dir):
                match = re.match(r'^(\d+)\.pth\.tar$', name)
                if match:
                    candidates.append((int(match.group(1)), os.path.join(ckpt_dir, name)))
            if candidates:
                _, ckpt_path = max(candidates, key=lambda x: x[0])
                logger.info(f"Resuming from checkpoint: {ckpt_path}")
                checkpoint = torch.load(ckpt_path, map_location='cpu')
                model.set_state_dict(checkpoint['model'])
                start_epoch = checkpoint['epoch'] + 1
                best_rsum = checkpoint.get('best_rsum', best_rsum)
                model.niter = checkpoint.get('Eiters', 0)
                resumed = True
            else:
                logger.info("No numeric checkpoints found; starting from scratch.")
        else:
            logger.info("No checkpoint directory found; starting from scratch.")

    if not resumed and opt.model_init:
        logger.info("override parser's params.")
        checkpoint = torch.load(opt.model_init, map_location='cpu')
        parser_params = checkpoint['model'][VGCPCFGs.NS_PARSER]
        model.parser.load_state_dict(parser_params)
        start_epoch = checkpoint['epoch'] + 1

    if not resumed and not opt.model_init:
        save_checkpoint({
        'epoch': -1,
        'model': model.get_state_dict(),
        'best_rsum': -1,
        'opt': opt,
        'Eiters': -1 }, False, -1, prefix=opt.logger_name)

    # Load data loaders
    data.set_constant(opt.visual_mode, opt.max_length)
    train_loader, syn_test_loader, sem_test_loader = data.get_data_iters(
        opt.data_path, opt.prefix, vocab, opt.batch_size, opt.workers,
        load_img=opt.visual_mode, encoder_file=opt.encoder_file, img_dim=opt.img_dim,
        shuffle=opt.shuffle, sampler=sampler, tiny=opt.tiny, one_shot=opt.one_shot,
        use_syntactic_bootstrapping=not opt.skip_syntactic_bootstrapping
    )
    syn_items = 0 if syn_test_loader is None else syn_test_loader.dataset.length
    logger.info("Number of train items: {}, semantic test items: {}, syntactic test items: {}".format(train_loader.dataset.length, sem_test_loader.dataset.length, syn_items))
    
    if start_epoch == 0:
        semantic_bootstrapping_test(opt, sem_test_loader, model, logger, -1)
        if not opt.skip_syntactic_bootstrapping:
            syntactic_bootstrapping_test(opt, syn_test_loader, model, logger, -1)
    best_rsum = float('inf')
    switch_epoch = int(opt.num_epochs / 2) if opt.switch_epoch < 0 else opt.switch_epoch
    switch_epoch = max(0, min(opt.num_epochs, switch_epoch))
    logger.info(f"Loss schedule switch_epoch={switch_epoch}")
    
    if opt.sem_first:
        p1_mt = 1.0 if opt.phase1_vse_mt_alpha is None else opt.phase1_vse_mt_alpha
        p1_lm = 0.0 if opt.phase1_vse_lm_alpha is None else opt.phase1_vse_lm_alpha
        p2_mt = 1.0 if opt.phase2_vse_mt_alpha is None else opt.phase2_vse_mt_alpha
        p2_lm = 1.0 if opt.phase2_vse_lm_alpha is None else opt.phase2_vse_lm_alpha
        model.vse_mt_alpha = p1_mt
        model.vse_lm_alpha = p1_lm
        logger.info(
            f"Training model on phase-1 loss (sem-first): "
            f"vse_mt_alpha={model.vse_mt_alpha}, vse_lm_alpha={model.vse_lm_alpha}"
        )
        for epoch in range(switch_epoch):
            current_epoch = start_epoch + epoch
            # train for one epoch
            train(opt, train_loader, model, epoch, sem_test_loader)
            # evaluate on validation set using VSE metrics
            rsum = semantic_bootstrapping_test(opt, sem_test_loader, model, logger, current_epoch)
            if not opt.skip_syntactic_bootstrapping:
                syntactic_bootstrapping_test(opt, syn_test_loader, model, logger, current_epoch)
            # remember best R@ sum and save checkpoint
            is_best = rsum < best_rsum
            best_rsum = max(rsum, best_rsum)
            save_checkpoint({
                'epoch': current_epoch,
                'model': model.get_state_dict(),
                'best_rsum': best_rsum,
                'opt': opt,
                'Eiters': model.niter,
            }, is_best, current_epoch, prefix=opt.logger_name)
        model.vse_mt_alpha = p2_mt
        model.vse_lm_alpha = p2_lm
        logger.info(
            f"Training model on phase-2 loss (sem-first): "
            f"vse_mt_alpha={model.vse_mt_alpha}, vse_lm_alpha={model.vse_lm_alpha}"
        )
        for epoch in range(switch_epoch, opt.num_epochs):
            current_epoch = start_epoch + epoch
            # train for one epoch
            train(opt, train_loader, model, epoch, sem_test_loader)
            # evaluate on validation set using VSE metrics
            rsum = semantic_bootstrapping_test(opt, sem_test_loader, model, logger, current_epoch)
            if not opt.skip_syntactic_bootstrapping:
                syntactic_bootstrapping_test(opt, syn_test_loader, model, logger, current_epoch)
            # remember best R@ sum and save checkpoint
            is_best = rsum < best_rsum
            best_rsum = max(rsum, best_rsum)
            save_checkpoint({
                'epoch': current_epoch,
                'model': model.get_state_dict(),
                'best_rsum': best_rsum,
                'opt': opt,
                'Eiters': model.niter,
            }, is_best, current_epoch, prefix=opt.logger_name)
        
    elif opt.syn_first:
        p1_mt = 0.0 if opt.phase1_vse_mt_alpha is None else opt.phase1_vse_mt_alpha
        p1_lm = 1.0 if opt.phase1_vse_lm_alpha is None else opt.phase1_vse_lm_alpha
        p2_mt = 1.0 if opt.phase2_vse_mt_alpha is None else opt.phase2_vse_mt_alpha
        p2_lm = 1.0 if opt.phase2_vse_lm_alpha is None else opt.phase2_vse_lm_alpha
        model.vse_mt_alpha = p1_mt
        model.vse_lm_alpha = p1_lm
        logger.info(
            f"Training model on phase-1 loss (syn-first): "
            f"vse_mt_alpha={model.vse_mt_alpha}, vse_lm_alpha={model.vse_lm_alpha}"
        )
        for epoch in range(switch_epoch):
            current_epoch = start_epoch + epoch
            # train for one epoch
            train(opt, train_loader, model, epoch, sem_test_loader)
            # evaluate on validation set using VSE metrics
            rsum = semantic_bootstrapping_test(opt, sem_test_loader, model, logger, current_epoch)
            if not opt.skip_syntactic_bootstrapping:
                syntactic_bootstrapping_test(opt, syn_test_loader, model, logger, current_epoch)
            # remember best R@ sum and save checkpoint
            is_best = rsum < best_rsum
            best_rsum = max(rsum, best_rsum)
            save_checkpoint({
                'epoch': current_epoch,
                'model': model.get_state_dict(),
                'best_rsum': best_rsum,
                'opt': opt,
                'Eiters': model.niter,
            }, is_best, current_epoch, prefix=opt.logger_name)
        model.vse_mt_alpha = p2_mt
        model.vse_lm_alpha = p2_lm
        logger.info(
            f"Training model on phase-2 loss (syn-first): "
            f"vse_mt_alpha={model.vse_mt_alpha}, vse_lm_alpha={model.vse_lm_alpha}"
        )
        for epoch in range(switch_epoch, opt.num_epochs):
            current_epoch = start_epoch + epoch
            # train for one epoch
            train(opt, train_loader, model, epoch, sem_test_loader)
            # evaluate on validation set using VSE metrics
            rsum = semantic_bootstrapping_test(opt, sem_test_loader, model, logger, current_epoch)
            if not opt.skip_syntactic_bootstrapping:
                syntactic_bootstrapping_test(opt, syn_test_loader, model, logger, current_epoch)
            # remember best R@ sum and save checkpoint
            is_best = rsum < best_rsum
            best_rsum = max(rsum, best_rsum)
            save_checkpoint({
                'epoch': current_epoch,
                'model': model.get_state_dict(),
                'best_rsum': best_rsum,
                'opt': opt,
                'Eiters': model.niter,
            }, is_best, current_epoch, prefix=opt.logger_name)
    else:
        model.vse_mt_alpha = opt.vse_mt_alpha
        model.vse_lm_alpha = opt.vse_lm_alpha
        logger.info(
            f"Training model with joint loss: "
            f"vse_mt_alpha={model.vse_mt_alpha}, vse_lm_alpha={model.vse_lm_alpha}"
        )
        for epoch in range(opt.num_epochs):
            current_epoch = start_epoch + epoch
            # train for one epoch
            train(opt, train_loader, model, epoch, syn_test_loader)
            # evaluate on validation set using VSE metrics
            rsum = semantic_bootstrapping_test(opt, sem_test_loader, model, logger, current_epoch)
        if syn_test_loader is not None:
            score = syntactic_bootstrapping_test(opt, syn_test_loader, model, logger, current_epoch)
            # remember best R@ sum and save checkpoint
            is_best = rsum < best_rsum
            best_rsum = max(rsum, best_rsum)
            save_checkpoint({
                'epoch': current_epoch,
                'model': model.get_state_dict(),
                'best_rsum': best_rsum,
                'opt': opt,
                'Eiters': model.niter,
            }, is_best, current_epoch, prefix=opt.logger_name)
