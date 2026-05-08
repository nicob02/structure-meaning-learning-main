import time
import numpy as np

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch_struct import SentCFG

from . import utils
from .as_module import CompoundCFG, ContrastiveLoss, ImageEncoder, TextEncoder

class VGCPCFGs(object):
    NS_PARSER = 'parser'
    NS_TXT_ENCODER = 'txt_enc'
    NS_IMG_ENCODER = 'img_enc'
    NS_OPTIMIZER = 'optimizer'

    def __init__(self, opt, vocab, logger):
        self.niter = 0
        self.vocab = vocab
        self.logger = logger
        self.log_step = opt.log_step
        self.grad_clip = opt.grad_clip

        self.vse_mt_alpha = opt.vse_mt_alpha
        self.vse_lm_alpha = opt.vse_lm_alpha
        self.sem_first = opt.sem_first

        self.use_structural_negatives = getattr(opt, 'use_structural_negatives', False)
        self.struct_neg_margin = getattr(opt, 'struct_neg_margin', 0.1)
        self.struct_neg_weight = getattr(opt, 'struct_neg_weight', 1.0)
        self.struct_neg_style = getattr(opt, 'struct_neg_style', 'hinge')
        self.use_mi_regularizer = getattr(opt, 'use_mi_regularizer', False)
        self.mi_margin = getattr(opt, 'mi_margin', 0.1)
        self.mi_weight = getattr(opt, 'mi_weight', 1.0)
        self.mi_style = getattr(opt, 'mi_style', 'ratio')

        self.use_entropy_bonus = getattr(opt, 'use_entropy_bonus', False)
        self.entropy_weight = getattr(opt, 'entropy_weight', 0.1)
        self.entropy_mode = getattr(opt, 'entropy_mode', 'category')

        # Anti-crystallization knobs (zero by default; opt-in).
        self.parser_grad_noise = getattr(opt, 'parser_grad_noise', 0.0)

        # Anti-left-branching architectural priors (zero by default; opt-in).
        # `branching_weight` penalizes mass on spans starting at 0 (left-bias).
        # `right_prior_weight` rewards mass on spans ending at last token (right-bias).
        self.branching_weight = getattr(opt, 'branching_weight', 0.0)
        self.right_prior_weight = getattr(opt, 'right_prior_weight', 0.0)
        # Cache of (start, end, width) for each flat span index, keyed by N.
        self._span_idx_cache = {}

        self.loss_criterion = ContrastiveLoss(margin=opt.margin)

        self.parser = CompoundCFG(
            opt.vocab_size, opt.nt_states, opt.t_states,
            h_dim = opt.h_dim,
            w_dim = opt.w_dim,
            z_dim = opt.z_dim,
            s_dim = opt.state_dim
        )
        # Optional architectural prior on the rule_mlp bias to bias the
        # parser toward right-branching trees at initialization. Training
        # can override; this just shifts the starting point so the parser
        # doesn't fall straight into the left-branching attractor.
        branching_init = getattr(opt, 'branching_init', 0.0)
        branching_init_mode = getattr(opt, 'branching_init_mode', 'right')
        if branching_init != 0.0 and branching_init_mode != 'none':
            self.parser.apply_branching_init(
                init_bias=float(branching_init),
                mode=str(branching_init_mode),
            )
            logger.info(
                f"Applied branching init bias={branching_init} "
                f"mode={branching_init_mode} to rule_mlp."
            )
        word_emb = torch.nn.Embedding(len(vocab), opt.word_dim)
        torch.nn.init.xavier_uniform_(word_emb.weight)

        self.all_params = []
        self.img_enc = ImageEncoder(opt)
        self.txt_enc = TextEncoder(opt, word_emb)
        self.all_params += list(self.txt_enc.parameters())
        self.all_params += list(self.parser.parameters())
        self.all_params += list(self.img_enc.parameters())

        # Per-module learning rates. Default: each falls back to opt.lr,
        # matching the original single-LR Adam exactly.
        lr_parser = getattr(opt, 'lr_parser', None) or opt.lr
        lr_txt_enc = getattr(opt, 'lr_txt_enc', None) or opt.lr
        lr_img_enc = getattr(opt, 'lr_img_enc', None) or opt.lr
        param_groups = [
            {'params': list(self.parser.parameters()), 'lr': lr_parser, 'name': 'parser'},
            {'params': list(self.txt_enc.parameters()), 'lr': lr_txt_enc, 'name': 'txt_enc'},
            {'params': list(self.img_enc.parameters()), 'lr': lr_img_enc, 'name': 'img_enc'},
        ]
        self.optimizer = torch.optim.Adam(
            param_groups, lr=opt.lr, betas=(opt.beta1, opt.beta2)
        )
        self.logger.info(
            f"Adam per-group LRs: parser={lr_parser}, "
            f"txt_enc={lr_txt_enc}, img_enc={lr_img_enc}"
        )

        if torch.cuda.is_available():
            cudnn.benchmark = False
            self.img_enc.cuda()
            self.txt_enc.cuda()
            self.parser.cuda()
        self.logger.info(self.parser)

    def train(self):
        self.img_enc.train()
        self.txt_enc.train()
        self.parser.train()

    def eval(self):
        self.img_enc.eval()
        self.txt_enc.eval()
        self.parser.eval()

    def get_state_dict(self):
        state_dict = {
            self.NS_PARSER: self.parser.state_dict(),
            self.NS_IMG_ENCODER: self.img_enc.state_dict(),
            self.NS_TXT_ENCODER: self.txt_enc.state_dict(),
            self.NS_OPTIMIZER: self.optimizer.state_dict(),
        }
        return state_dict

    def set_state_dict(self, state_dict):
        self.parser.load_state_dict(state_dict[self.NS_PARSER])
        self.img_enc.load_state_dict(state_dict[self.NS_IMG_ENCODER])
        self.txt_enc.load_state_dict(state_dict[self.NS_TXT_ENCODER])
        self.optimizer.load_state_dict(state_dict[self.NS_OPTIMIZER])

    def norms(self):
        p_norm = sum([p.norm() ** 2 for p in self.all_params]).item() ** 0.5
        g_norm = sum([p.grad.norm() ** 2 for p in self.all_params if p.grad is not None]).item() ** 0.5
        return p_norm, g_norm

    def forward_parser(self, captions, lengths):
        params, kl = self.parser(captions)
        dist = SentCFG(params, lengths=lengths)
        the_spans = dist.argmax[-1]
        lengths_list = lengths.tolist() if hasattr(lengths, "tolist") else list(lengths)
        argmax_spans, trees, lprobs = utils.extract_parses(the_spans, lengths_list, inc=0)
        ll, span_margs = dist.inside_im
        nll = -ll
        kl = torch.zeros_like(nll) if kl is None else kl
        return nll, kl, span_margs, argmax_spans, trees, lprobs

    def forward_encoder(self, images, captions, lengths, spans, require_grad=True):
        if torch.cuda.is_available():
            images = images.cuda()
            lengths = lengths.cuda()
            captions = captions.cuda()
        with torch.set_grad_enabled(require_grad):
            img_emb = self.img_enc(images)
            parser_outs = self.forward_parser(captions, lengths)
            txt_outputs = self.txt_enc(captions, lengths, parser_outs[-3])
        return (img_emb, txt_outputs) + parser_outs

    def _get_span_idx(self, N, device):
        """Return (starts, ends, widths) tensors of shape (nstep,) for max length N.
        Indexing matches `_forward_srnn` in as_module.py: width-major (k=1..N-1),
        then start-major within each width. Used by the branching/right-prior
        regularizers to map flat span index -> structural position.
        """
        Ni = int(N)
        cache = self._span_idx_cache.get(Ni)
        if cache is not None:
            if cache[0].device == device:
                return cache
            cache = tuple(t.to(device) for t in cache)
            self._span_idx_cache[Ni] = cache
            return cache
        starts, ends, widths = [], [], []
        for k in range(1, Ni):  # k = width - 1, so width = k+1
            for s in range(Ni - k):
                starts.append(s)
                ends.append(s + k)
                widths.append(k + 1)
        starts_t = torch.tensor(starts, device=device, dtype=torch.long)
        ends_t = torch.tensor(ends, device=device, dtype=torch.long)
        widths_t = torch.tensor(widths, device=device, dtype=torch.long)
        cache = (starts_t, ends_t, widths_t)
        self._span_idx_cache[Ni] = cache
        return cache

    def _expected_matching_loss(self, img_emb, cap_span_features, span_margs, nstep):
        """Compute the expected span-level matching loss given arbitrary span_margs.
        Reused for the regular matching loss, structural negatives (shuffled margs),
        and the MI regularizer (uniform margs)."""
        b = img_emb.size(0)
        matching_loss_matrix = torch.zeros(b, nstep, device=img_emb.device)
        for k in range(nstep):
            cap_emb = cap_span_features[:, k]
            cap_marg = span_margs[:, k].softmax(-1).unsqueeze(-2)
            cap_emb = torch.matmul(cap_marg, cap_emb).squeeze(-2)
            cap_emb = utils.l2norm(cap_emb)
            loss = self.loss_criterion(img_emb, cap_emb)
            matching_loss_matrix[:, k] = loss
        span_margs_summed = span_margs.sum(-1)
        expected = (span_margs_summed[:, :nstep] * matching_loss_matrix).sum(-1)
        return expected

    def forward_loss(self, base_img_emb, cap_span_features, lengths, span_bounds, span_margs):
        b = base_img_emb.size(0)
        N = lengths.max(0)[0]
        nstep = int(N * (N - 1) / 2)
        mstep = (lengths * (lengths - 1) / 2).int()
        img_emb = base_img_emb
        # If doing semantics first only consider the matching loss between complete caption embedding and images (not intermediate spans as well)
        if self.sem_first and self.vse_lm_alpha == 0.0:
            cap_emb = torch.cat([cap_span_features[j][k - 1].unsqueeze(0) for j, k in enumerate(mstep)], dim=0)
            cap_emb = cap_emb.sum(-2)
            cap_emb = utils.l2norm(cap_emb)
            loss = self.loss_criterion(img_emb, cap_emb)
            expected_loss = loss.sum(-1)
        else:
            expected_loss = self._expected_matching_loss(img_emb, cap_span_features, span_margs, nstep)
        return expected_loss

    def forward(self, images, captions, lengths, ids=None, spans=None, epoch=None, *args):
        self.niter += 1
        self.logger.update('Eit', self.niter)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        lengths = torch.tensor(lengths).long() if isinstance(lengths, list) else lengths

        img_emb, cap_span_features, nll, kl, span_margs, argmax_spans, trees, lprobs = \
            self.forward_encoder(
                images, captions, lengths, spans
            )
        matching_loss = self.forward_loss(
            img_emb, cap_span_features, lengths, argmax_spans, span_margs
        )

        bsize = captions.size(0)

        rl_loss = torch.tensor(0.0, device=nll.device)
        mt_loss = matching_loss.sum()

        kl.clamp_(max=20) # avoid kl explosion
        if self.vse_lm_alpha <=0.:
            kl = torch.zeros_like(kl)
            nll = torch.zeros_like(nll)

        ll_loss = nll.sum()
        kl_loss = kl.sum()

        struct_neg_term = torch.tensor(0.0, device=nll.device)
        mi_term = torch.tensor(0.0, device=nll.device)
        entropy_term = torch.tensor(0.0, device=nll.device)
        only_joint_path = not (self.sem_first and self.vse_lm_alpha == 0.0)
        if only_joint_path and (self.use_structural_negatives or self.use_mi_regularizer):
            N = lengths.max(0)[0]
            nstep = int(N * (N - 1) / 2)
            if self.use_structural_negatives and bsize > 1:
                perm = torch.randperm(bsize, device=span_margs.device)
                while bool((perm == torch.arange(bsize, device=span_margs.device)).all()):
                    perm = torch.randperm(bsize, device=span_margs.device)
                shuffled_margs = span_margs[perm]
                shuffled_loss = self._expected_matching_loss(
                    img_emb, cap_span_features, shuffled_margs, nstep
                )
                if self.struct_neg_style == 'ratio':
                    # Always-active continuous ratio form. Bounded in (0, 1).
                    # Minimized when real << shuffled.
                    struct_neg_term = (
                        matching_loss / (matching_loss + shuffled_loss + 1e-8)
                    ).sum()
                else:  # 'hinge' (original)
                    struct_neg_term = F.relu(
                        self.struct_neg_margin + matching_loss - shuffled_loss
                    ).sum()
            if self.use_mi_regularizer:
                uniform_margs = torch.ones_like(span_margs) / span_margs.size(-1)
                uniform_loss = self._expected_matching_loss(
                    img_emb, cap_span_features, uniform_margs, nstep
                )
                if self.mi_style == 'hinge':
                    mi_term = F.relu(
                        self.mi_margin + matching_loss - uniform_loss
                    ).sum()
                elif self.mi_style == 'infonce':
                    # -log(exp(-real) / (exp(-real) + exp(-uniform)))
                    # Equivalent to softplus(uniform_loss - real_loss) semantically
                    mi_per_sample = matching_loss + torch.logsumexp(
                        torch.stack([-matching_loss, -uniform_loss], dim=0), dim=0
                    )
                    mi_term = mi_per_sample.sum()
                else:  # 'ratio' (default): always in (0, 1)
                    mi_per_sample = matching_loss / (matching_loss + uniform_loss + 1e-8)
                    mi_term = mi_per_sample.sum()

        if self.use_entropy_bonus and self.entropy_weight > 0.0:
            if self.entropy_mode == 'boundary':
                # Entropy over which spans the parser thinks are constituents.
                # span_existence[b, k] = total mass on span k being a constituent of
                # any nonterminal label.
                # The root span (full sentence) is *always* a constituent in any
                # binary tree, so its existence mass is exactly 1.0 by construction
                # — independent of the parser. If we don't mask it, the normalized
                # distribution collapses onto the root and entropy reads ~0.
                # Mask by value: only the root has span_existence ≈ 1.0 deterministically.
                span_existence_raw = span_margs.sum(-1).clamp(min=1e-8)  # (b, nstep)
                if not getattr(self, '_dbg_boundary_done', False):
                    self._dbg_boundary_done = True
                    se0 = span_existence_raw[0].detach().cpu()
                    sorted_se, _ = se0.sort(descending=True)
                    self.logger.info(
                        f"DEBUG boundary[0]: len={se0.size(0)}, "
                        f"min={se0.min().item():.6f}, "
                        f"max={se0.max().item():.6f}, "
                        f"mean={se0.mean().item():.6f}, "
                        f"sum={se0.sum().item():.4f}, "
                        f"top5={[round(x, 4) for x in sorted_se[:5].tolist()]}, "
                        f"bot5={[round(x, 6) for x in sorted_se[-5:].tolist()]}"
                    )
                root_mask = (span_existence_raw < 0.999).to(span_existence_raw.dtype)
                span_existence = span_existence_raw * root_mask
                Z = span_existence.sum(-1, keepdim=True) + 1e-8
                p = span_existence / Z  # (b, nstep), sums to ~1 per sentence
                logp = (p + 1e-10).log()  # 1e-10 keeps masked cells finite
                entropy_per_sent = -(p * logp).sum(-1)  # (b,)
                mean_entropy = entropy_per_sent.mean()
            else:  # 'category' (original)
                # Entropy over nonterminal categories per span.
                log_probs = F.log_softmax(span_margs, dim=-1)
                probs = log_probs.exp()
                entropy_per_span = -(probs * log_probs).sum(-1)  # (b, nstep)
                mean_entropy = entropy_per_span.mean()
            # Subtract to maximize entropy.
            entropy_term = -mean_entropy * bsize

        # ---- Architectural priors against the left-branching attractor ----
        # `branching_term`: penalize soft mass on spans starting at position 0
        #   (left-edge constituents). Push the parser away from "everything
        #   attaches to the leftmost prefix" trees.
        # `right_prior_term`: reward soft mass on spans ending at the last
        #   token (right-edge constituents). A right-branching tree has
        #   exactly L-1 such spans (incl. root); a left-branching tree has 1.
        # Both use the same (start, end, width) cache and the same masking
        # logic that excludes (a) out-of-range spans for short sentences in
        # the batch, and (b) the trivial root span whose marginal is 1.0.
        branching_term = torch.tensor(0.0, device=nll.device)
        right_prior_term = torch.tensor(0.0, device=nll.device)
        if (self.branching_weight > 0.0) or (self.right_prior_weight > 0.0):
            span_existence = span_margs.sum(-1).clamp(min=1e-8)  # (b, nstep)
            Nmax = int(lengths.max().item())
            starts_t, ends_t, widths_t = self._get_span_idx(Nmax, span_existence.device)
            # Per-batch valid mask: ends_t[k] < lengths[b]  -> span k fits in sent b.
            valid = ends_t.unsqueeze(0) < lengths.unsqueeze(1).to(ends_t.device)
            # Exclude the root span of each sentence (width == lengths[b]).
            not_root = widths_t.unsqueeze(0) < lengths.unsqueeze(1).to(widths_t.device)
            nontrivial = valid & not_root  # (b, nstep)
            ntf = nontrivial.to(span_existence.dtype)

            if self.branching_weight > 0.0:
                # Normalize span_existence over non-trivial spans, then sum
                # the mass on left-edge spans. p_left ∈ [0, 1] per sentence.
                se_masked = span_existence * ntf
                Z = se_masked.sum(-1, keepdim=True) + 1e-8
                p_span = se_masked / Z
                left_mask = (starts_t.unsqueeze(0) == 0) & nontrivial
                leftness = (p_span * left_mask.to(p_span.dtype)).sum(-1)  # (b,)
                # Sum (mean later by /bsize) so it scales like the other terms.
                branching_term = leftness.sum()

            if self.right_prior_weight > 0.0:
                # For each sentence b, right-edge non-root spans are those
                # ending at lengths[b]-1 with width < lengths[b]. There are
                # exactly lengths[b]-2 such spans for a length>=3 sentence.
                last_idx = (lengths - 1).to(ends_t.device).unsqueeze(1)  # (b, 1)
                right_edge = (ends_t.unsqueeze(0) == last_idx) & nontrivial
                ref = right_edge.to(span_existence.dtype)
                # Mean span_existence on right-edge spans -> in [0, 1].
                right_mass = (span_existence * ref).sum(-1)
                right_count = ref.sum(-1).clamp(min=1.0)
                right_avg = right_mass / right_count  # (b,)
                # Maximize right_avg => loss = -log(right_avg).
                right_prior_term = -(right_avg + 1e-8).log().sum()

        loss = (
            self.vse_mt_alpha * mt_loss
            + self.vse_lm_alpha * (ll_loss + kl_loss)
            + self.struct_neg_weight * struct_neg_term
            + self.mi_weight * mi_term
            + self.entropy_weight * entropy_term
            + self.branching_weight * branching_term
            + self.right_prior_weight * right_prior_term
        ) / bsize

        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm_(self.all_params, self.grad_clip)
        # Gradient noise injection on the parser only.
        # Adds Gaussian noise to parser gradients to keep them stochastic and
        # discourage early commitment to a single tree topology. Noise scale
        # decays with global step so it's strong early and gentle late.
        if getattr(self, 'parser_grad_noise', 0.0) > 0.0:
            decay = 1.0 / ((1 + getattr(self, '_step', 0)) ** 0.55)
            sigma = float(self.parser_grad_noise) * decay
            for p in self.parser.parameters():
                if p.grad is not None:
                    p.grad.add_(torch.randn_like(p.grad) * sigma)
            self._step = getattr(self, '_step', 0) + 1
        self.optimizer.step()

        self.logger.update('Loss', loss.item(), bsize)
        self.logger.update('MT-Loss', mt_loss.item() / bsize, bsize)
        self.logger.update('KL-Loss', kl_loss.item() / bsize, bsize)
        self.logger.update('LL-Loss', ll_loss.item() / bsize, bsize)
        if self.use_structural_negatives:
            self.logger.update('StructNeg-Loss', struct_neg_term.item() / bsize, bsize)
        if self.use_mi_regularizer:
            self.logger.update('MI-Loss', mi_term.item() / bsize, bsize)
        if self.use_entropy_bonus:
            # Log positive entropy (higher = more uncertain = good early in training).
            self.logger.update('Entropy', -entropy_term.item() / bsize, bsize)
            self.logger.update('EntropyWeight', float(self.entropy_weight))
        if hasattr(self.parser, 'temperature'):
            self.logger.update('Temp', float(self.parser.temperature))
        if self.branching_weight > 0.0:
            self.logger.update('Leftness', branching_term.item() / bsize, bsize)
        if self.right_prior_weight > 0.0:
            self.logger.update('RightPrior', right_prior_term.item() / bsize, bsize)

        self.n_word += (lengths + 1).sum().item()
        self.n_sent += bsize

        for b in range(bsize):
            max_len = lengths[b].item()
            pred = [(a[0], a[1]) for a in argmax_spans[b] if a[0] != a[1]]
            pred_set = set(pred[:-1])
            gold = [(spans[b][i][0].item(), spans[b][i][1].item()) for i in range(max_len - 1)]
            gold_set = set(gold[:-1])
            utils.update_stats(pred_set, [gold_set], self.all_stats)

        info = ''
        if self.niter % self.log_step == 0:
            p_norm, g_norm = self.norms()
            all_f1 = utils.get_f1(self.all_stats)
            train_kl = self.logger.meters["KL-Loss"].sum
            train_ll = self.logger.meters["LL-Loss"].sum
            info = '|Pnorm|: {:.6f}, |Gnorm|: {:.2f}, ReconPPL: {:.2f}, KL: {:.2f}, ' + \
                   'PPLBound: {:.2f}, CorpusF1: {:.2f}, Speed: {:.2f} sents/sec'
            info = info.format(
                p_norm, g_norm, np.exp(train_ll / self.n_word), train_kl / self.n_sent,
                np.exp((train_ll + train_kl) / self.n_word), all_f1[0],
                self.n_sent / (time.time() - self.s_time)
            )
            pred_action = utils.get_actions(trees[0])
            sent_s = [self.vocab.idx2word[wid] for wid in captions[0].cpu().tolist()]
            pred_t = utils.get_tree(pred_action, sent_s)
            gold_t = utils.span_to_tree(spans[0].tolist(), lengths[0].item())
            gold_action = utils.get_actions(gold_t)
            gold_t = utils.get_tree(gold_action, sent_s)
            info += "\nPred T: {}\nGold T: {}".format(pred_t, gold_t)
        if epoch > 0: #
            del img_emb, cap_span_features, nll, kl, span_margs, argmax_spans, trees, lprobs, matching_loss
        return info
