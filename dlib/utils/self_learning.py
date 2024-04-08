import operator
import sys
import os
import warnings
from os.path import dirname, abspath
import time
from typing import Callable, Tuple

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

from kornia.morphology import dilation
from kornia.morphology import erosion


_NBINS = 256

__all__ = ['MBSeederSLCAMS',
           'MBProbSeederSLCAMS',
           'MBProbSeederSLCamsWithROI',
           'BinaryRegionsProposal']


_FG = 'foreground'
_BG = 'background'

_UNIFORM = 'uniform'
_WEIGHTED = 'weighted'

_TRGS = [_FG, _BG]


def rv1d(t: torch.Tensor) -> torch.Tensor:
    assert t.ndim == 1
    return torch.flip(t, dims=(0, ))


class _STOtsu(nn.Module):
    def __init__(self):
        super(_STOtsu, self).__init__()

        self.bad_egg = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.bad_egg = False

        min_x = x.min()
        max_x = x.max()

        if min_x == max_x:
            self.bad_egg = True
            return torch.tensor(min_x)

        bins = int(max_x - min_x + 1)
        bin_centers = torch.arange(min_x, max_x + 1, 1, dtype=torch.float32,
                                   device=x.device)

        hist = torch.histc(x, bins=bins)
        weight1 = torch.cumsum(hist, dim=0)
        _weight2 = torch.cumsum(rv1d(hist), dim=0)
        weight2_r = _weight2
        weight2 = rv1d(_weight2)
        mean1 = torch.cumsum(hist * bin_centers, dim=0) / weight1
        mean2 = rv1d(torch.cumsum(rv1d(hist * bin_centers), dim=0) / weight2_r)
        diff_avg_sq = torch.pow(mean1[:-1] - mean2[1:], 2)
        variance12 = weight1[:-1] * weight2[1:] * diff_avg_sq

        idx = torch.argmax(variance12)
        threshold = bin_centers[:-1][idx]

        return threshold


class _STFG(nn.Module):
    def __init__(self, max_):
        super(_STFG, self).__init__()
        self.max_ = max_

    def forward(self, roi: torch.Tensor, fg: torch.Tensor) -> torch.Tensor:
        # roi: h,w
        idx_fg = torch.nonzero(roi, as_tuple=True)  # (idx, idy)
        n_fg = idx_fg[0].numel()
        if (n_fg > 0) and (self.max_ > 0):
            probs = torch.ones(n_fg, dtype=torch.float)
            selected = probs.multinomial(
                num_samples=min(self.max_, n_fg), replacement=False)
            fg[idx_fg[0][selected], idx_fg[1][selected]] = 1

        return fg


class _STBG(nn.Module):
    def __init__(self, nbr_bg, min_):
        super(_STBG, self).__init__()

        self.nbr_bg = nbr_bg
        self.min_ = min_

    def forward(self,
                cam: torch.Tensor,
                bg: torch.Tensor,
                roi: torch.Tensor = None) -> torch.Tensor:
        """

        :param cam: cam.
        :param bg: bg holder.
        :param roi: foreground roi. we are not allowed to sample from it
        background pixels.
        :return:
        """
        assert cam.ndim == 2
        # cam: h, w
        h, w = cam.shape
        _cam = cam
        if roi is not None:
            _cam = _cam + roi * 1000.  # increase activation inside roi so we
            # cant sample from it.
        val, idx_bg_ = torch.sort(cam.view(h * w), dim=0, descending=False)

        tmp = torch.zeros_like(bg)
        if self.nbr_bg > 0:
            tmp = tmp.view(h * w)
            tmp[idx_bg_[:self.nbr_bg]] = 1
            tmp = tmp.view(h, w)

            idx_bg = torch.nonzero(tmp, as_tuple=True)  #
            # (idx, idy)
            n_bg = idx_bg[0].numel()
            if (n_bg > 0) and (self.min_ > 0):
                probs = torch.ones(n_bg, dtype=torch.float)
                selected = probs.multinomial(
                    num_samples=min(self.min_, n_bg),
                    replacement=False)
                bg[idx_bg[0][selected], idx_bg[1][selected]] = 1

        return bg


class _STOneSample(nn.Module):
    def __init__(self, min_, max_, nbr_bg):
        super(_STOneSample, self).__init__()

        self.min_ = min_
        self.max_ = max_

        self.otsu = _STOtsu()
        self.fg_capture = _STFG(max_=max_)
        self.bg_capture = _STBG(nbr_bg=nbr_bg, min_=min_)

    def forward(self, cam: torch.Tensor,
                erode: Callable[[torch.Tensor], torch.Tensor]) -> Tuple[
        torch.Tensor, torch.Tensor]:

        assert cam.ndim == 2

        # cam: h, w
        h, w = cam.shape
        fg = torch.zeros((h, w), dtype=torch.long, device=cam.device,
                         requires_grad=False)
        bg = torch.zeros((h, w), dtype=torch.long, device=cam.device,
                         requires_grad=False)

        # otsu
        cam_ = torch.floor(cam * 255)
        th = self.otsu(x=cam_)

        if th == 0:
            th = 1.
        if th == 255:
            th = 254.

        if self.otsu.bad_egg:
            return fg, bg

        # ROI
        roi = (cam_ > th).long()
        roi = erode(roi.unsqueeze(0).unsqueeze(0)).squeeze()

        fg = self.fg_capture(roi=roi, fg=fg)
        bg = self.bg_capture(cam=cam, bg=bg)
        return fg, bg


class MBSeederSLCAMS(nn.Module):
    def __init__(self,
                 min_: int = 1,
                 max_: int = 1,
                 min_p: float = 0.3,
                 fg_erode_k: int = 11,
                 fg_erode_iter: int = 1,
                 ksz: int = 3,
                 seg_ignore_idx: int = -255
                 ):
        """
        Sample seeds from CAM.
        Requires defining: min_p. region to sample background pixels.
        foreground pixels are determined using otsu threshold.
        Supports batch.

        :param min_: int. number of pixels to sample from background.
        :param max_: int. number of pixels to sample from foreground.
        :param min_p: float [0, 1]. percentage (from entire image) of pixels
        to be considered background. min_ pixels will be sampled from these
        pixels. IMPORTANT.
        :param fg_erode_k: int. erosion kernel of the cam before sampling.
        11 is fine.
        :param fg_erode_iter: int. number iteration to perform the erosion.
        1 is fine.
        :param ksz: int. kernel size to dilate the seeds.
        :param seg_ignore_idx: int. index for unknown seeds. 0: background.
        1: foreground. seg_ignore_idx: unknown.
        """
        super(MBSeederSLCAMS, self).__init__()

        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')#torch.device('cuda')

        assert isinstance(ksz, int)
        assert ksz > 0
        self.ksz = ksz
        self.kernel = None
        if self.ksz > 1:
            self.kernel = torch.ones((self.ksz, self.ksz), dtype=torch.long,
                                     device=self._device)

        assert isinstance(min_, int)
        assert isinstance(max_, int)
        assert min_ >= 0
        assert max_ >= 0
        assert min_ + max_ > 0

        self.min_ = min_
        self.max_ = max_

        assert isinstance(min_p, float)
        assert 0. <= min_p <= 1.
        self.min_p = min_p

        # fg
        assert isinstance(fg_erode_k, int)
        assert fg_erode_k >= 1
        self.fg_erode_k = fg_erode_k

        # fg
        assert isinstance(fg_erode_iter, int)
        assert fg_erode_iter >= 0
        self.fg_erode_iter = fg_erode_iter

        self.fg_kernel_erode = None
        if self.fg_erode_iter > 0:
            assert self.fg_erode_k > 1
            self.fg_kernel_erode = torch.ones(
                (self.fg_erode_k, self.fg_erode_k), dtype=torch.long,
                device=self._device)

        self.ignore_idx = seg_ignore_idx

    def mb_erosion_n(self, x):
        assert self.fg_erode_k > 1
        assert x.ndim == 4
        assert x.shape[1] == 1

        out: torch.Tensor = x

        for i in range(self.fg_erode_iter):
            out = erosion(out, self.fg_kernel_erode)

        # expensive
        # assert 0 <= torch.min(out) <= 1
        # assert 0 <= torch.max(out) <= 1

        assert out.shape == x.shape

        return out

    def mb_dilate(self, x):
        assert self.ksz > 1
        assert x.ndim == 4
        assert x.shape[1] == 1

        out: torch.Tensor = dilation(x, self.kernel)

        # expensive
        # assert 0 <= torch.min(out) <= 1
        # assert 0 <= torch.max(out) <= 1

        assert out.shape == x.shape

        return out

    def identity(self, x):
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: torch.Tensor. average cam: batchs, 1, h, w. Detach it first.
        :return: pseudo labels. batchsize, h, w
        """
        assert isinstance(x, torch.Tensor)
        assert x.ndim == 4

        # expensive
        # assert 0 <= torch.min(x) <= 1
        # assert 0 <= torch.max(x) <= 1

        if self.ksz == 1:
            dilate = self.identity
        elif self.ksz > 1:
            dilate = self.mb_dilate
        else:
            raise ValueError

        if self.fg_erode_iter > 0:
            erode = self.mb_erosion_n
        else:
            erode = self.identity

        b, d, h, w = x.shape
        assert d == 1  # todo multilabl.

        out = torch.zeros((b, h, w), dtype=torch.long, requires_grad=False,
                          device=self._device) + self.ignore_idx

        nbr_bg = int(self.min_p * h * w)

        all_fg = torch.zeros((b, h, w), dtype=torch.long, device=x.device,
                             requires_grad=False)
        all_bg = torch.zeros((b, h, w), dtype=torch.long, device=x.device,
                             requires_grad=False)

        opx = _STOneSample(min_=self.min_, max_=self.max_, nbr_bg=nbr_bg)

        for i in range(b):
            all_fg[i], all_bg[i] = opx(cam=x[i].squeeze(), erode=erode)

        # fg
        all_fg = dilate(all_fg.unsqueeze(1)).squeeze(1)
        # b, h, w

        # bg
        all_bg = dilate(all_bg.unsqueeze(1)).squeeze(1)
        # b, h, w

        # sanity
        outer = all_fg + all_bg
        all_fg[outer == 2] = 0
        all_bg[outer == 2] = 0

        # assign
        out[all_fg == 1] = 1
        out[all_bg == 1] = 0

        out = out.detach()

        assert out.dtype == torch.long

        return out

    def extra_repr(self):
        return 'min_={}, max_={}, ' \
               'ksz={}, min_p: {}, fg_erode_k: {}, fg_erode_iter: {}, ' \
               'seg_ignore_idx={}'.format(
                self.min_, self.max_, self.ksz, self.min_p, self.fg_erode_k,
                self.fg_erode_iter,
                self.ignore_idx)


class _ProbaSampler(nn.Module):
    def __init__(self, nbr: int, trg: str):
        super(_ProbaSampler, self).__init__()
        assert trg in _TRGS

        assert nbr >= 0

        self.trg = trg
        self.nbr = nbr

        self.eps = 1e-6

    def forward(self,
                cam: torch.Tensor,
                roi: torch.Tensor = None) -> torch.Tensor:
        """

        :param cam: cam. >= 0. sum(cam) > 0.
        :param roi: Foreground binary map to indicate where we are allowed to
        sample. inside that region, the sampling is based on cam activations.
        :return:
        """
        assert cam.ndim == 2
        if roi is not None:
            assert roi.shape == cam.shape

        _cam: torch.Tensor = cam + self.eps
        _cam = _cam / _cam.sum()

        if self.trg == _BG:
            _cam = 1. - _cam

        if roi is not None:
            if self.trg == _FG:
                _cam = _cam * roi  # sample FG only from inside ROI.
            elif self.trg == _BG:
                _cam = _cam * (1. - roi)  # do not sample bg from inside ROI.
            else:
                raise ValueError(self.trg)

        # cam: h, w
        h, w = _cam.shape
        probs = _cam.contiguous().view(h * w)

        tmp = torch.zeros((h, w), dtype=torch.long, device=cam.device,
                          requires_grad=False).contiguous().view(h * w)

        if self.nbr == 0:
            return tmp.contiguous().view(h, w)

        sumit = probs.sum()
        if sumit <= 0.:
            warnings.warn('Some ROI maps have full 0.')
            return tmp.contiguous().view(h, w)

        selected = probs.multinomial(num_samples=min(self.nbr, h * w),
                                     replacement=False)
        tmp[selected] = 1
        return tmp.contiguous().view(h, w)


class _ProbFG(nn.Module):
    def __init__(self, max_p: float, max_: int, seed_tech: str):
        super(_ProbFG, self).__init__()

        self.max_ = max_
        self.max_p = max_p
        self.seed_tech = seed_tech

    def forward(self,
                cam: torch.Tensor,
                roi: torch.Tensor,
                fg: torch.Tensor) -> torch.Tensor:

        assert cam.shape == roi.shape
        h, w = cam.shape

        n = roi.sum()

        n = int(self.max_p * n)
        _cam = cam * roi
        _cam = _cam + 1e-8

        _cam_flatten = _cam.view(h * w)

        val, idx_ = torch.sort(_cam_flatten, dim=0, descending=True,
                               stable=True)

        if (n > 0) and (self.max_ > 0):

            tmp = _cam_flatten * 0.
            tmp[idx_[:n]] = 1
            tmp = tmp.view(h, w)

            _idx = torch.nonzero(tmp, as_tuple=True)  # (idx, idy)

            if self.seed_tech == _UNIFORM:
                probs = torch.ones(n, dtype=torch.float, device=cam.device)

            elif self.seed_tech == _WEIGHTED:
                probs = _cam[_idx[0], _idx[1]]  # 1d array. numel: n
                assert probs.numel() == n

            else:
                raise NotImplementedError(self.seed_tech)

            selected = probs.multinomial(
                num_samples=min(self.max_, n), replacement=False)

            fg[_idx[0][selected], _idx[1][selected]] = 1

        return fg


class _ProbBG(nn.Module):
    def __init__(self, min_p: float, min_: int, seed_tech: str):
        super(_ProbBG, self).__init__()

        self.min_ = min_
        self.min_p = min_p
        self.seed_tech = seed_tech

    def forward(self, cam: torch.Tensor, bg: torch.Tensor) -> torch.Tensor:
        assert cam.ndim == 2

        # cam: h, w
        h, w = cam.shape

        n = int(self.min_p * h * w)

        _cam = cam + 1e-8
        _cam_flatten = _cam.view(h * w)
        val, idx_ = torch.sort(_cam_flatten, dim=0, descending=False,
                               stable=True)

        if (n > 0) and (self.min_ > 0):

            tmp = _cam_flatten * 0.
            tmp[idx_[:n]] = 1
            tmp = tmp.view(h, w)

            _idx = torch.nonzero(tmp, as_tuple=True)  # (idx, idy)

            if self.seed_tech == _UNIFORM:
                probs = torch.ones(n, dtype=torch.float, device=cam.device)

            elif self.seed_tech == _WEIGHTED:
                probs = 1. - _cam[_idx[0], _idx[1]]  # 1d array. numel: n
                probs = torch.relu(probs) + 1e-8
                assert probs.numel() == n

            else:
                raise NotImplementedError(self.seed_tech)

            selected = probs.multinomial(
                num_samples=min(self.min_, n),
                replacement=False)
            bg[_idx[0][selected], _idx[1][selected]] = 1

        return bg


class _ProbaOneSample(nn.Module):
    def __init__(self, min_: int, max_: int, min_p: float, max_p: float,
                 seed_tech: str, bg_seed_tech: str):
        super(_ProbaOneSample, self).__init__()

        self.min_ = min_
        self.max_ = max_
        self.min_p = min_p
        self.max_p = max_p
        self.seed_tech = seed_tech
        self.bg_seed_tech = bg_seed_tech

        self.fg_capture = _ProbFG(max_p=max_p, max_=max_, seed_tech=seed_tech)
        self.bg_capture = _ProbBG(min_p=min_p, min_=min_,
                                  seed_tech=bg_seed_tech)

    def forward(self,
                cam: torch.Tensor,
                roi: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:

        assert cam.ndim == 2

        # cam: h, w
        if roi is not None:
            assert roi.shape == cam.shape

        # cam: h, w
        h, w = cam.shape
        bg = torch.zeros((h, w), dtype=torch.long, device=cam.device,
                         requires_grad=False)
        fg = torch.zeros((h, w), dtype=torch.long, device=cam.device,
                         requires_grad=False)

        fg = self.fg_capture(cam=cam, roi=roi, fg=fg)
        bg = self.bg_capture(cam=cam, bg=bg)
        return fg, bg


class MBProbSeederSLCAMS(MBSeederSLCAMS):
    def __init__(self,
                 min_: int = 1,
                 max_: int = 1,
                 ksz: int = 3,
                 seg_ignore_idx: int = -255
                 ):
        """
        Sample seeds from CAM.
        Does the same job as 'MBSeederSLCAMS' but, it does not require to
        define foreground and background sampling regions.
        We use a probabilistic approach to sample foreground and background.

        Supports batch.

        :param min_: int. number of pixels to sample from background.
        :param max_: int. number of pixels to sample from foreground.
        :param ksz: int. kernel size to dilate the seeds.
        :param seg_ignore_idx: int. index for unknown seeds. 0: background.
        1: foreground. seg_ignore_idx: unknown.
        """
        super(MBProbSeederSLCAMS, self).__init__()

        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')#torch.device('cuda')

        assert isinstance(ksz, int)
        assert ksz > 0
        self.ksz = ksz
        self.kernel = None
        if self.ksz > 1:
            self.kernel = torch.ones((self.ksz, self.ksz), dtype=torch.long,
                                     device=self._device)

        assert isinstance(min_, int)
        assert isinstance(max_, int)
        assert min_ >= 0
        assert max_ >= 0
        assert min_ + max_ > 0

        self.min_ = min_
        self.max_ = max_

        self.ignore_idx = seg_ignore_idx

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: torch.Tensor. average cam: batchs, 1, h, w. Detach it first.
        :return: pseudo labels. batchsize, h, w
        """
        assert isinstance(x, torch.Tensor)
        assert x.ndim == 4

        # expensive
        # assert 0 <= torch.min(x) <= 1
        # assert 0 <= torch.max(x) <= 1

        if self.ksz == 1:
            dilate = self.identity
        elif self.ksz > 1:
            dilate = self.mb_dilate
        else:
            raise ValueError

        b, d, h, w = x.shape
        assert d == 1  # todo multilabl.

        out = torch.zeros((b, h, w), dtype=torch.long, requires_grad=False,
                          device=self._device) + self.ignore_idx

        all_fg = torch.zeros((b, h, w), dtype=torch.long, device=x.device,
                             requires_grad=False)
        all_bg = torch.zeros((b, h, w), dtype=torch.long, device=x.device,
                             requires_grad=False)

        opx = _ProbaOneSample(min_=self.min_, max_=self.max_)

        for i in range(b):
            all_fg[i], all_bg[i] = opx(cam=x[i].squeeze())

        # fg
        all_fg = dilate(all_fg.unsqueeze(1)).squeeze(1)
        # b, h, w

        # bg
        all_bg = dilate(all_bg.unsqueeze(1)).squeeze(1)
        # b, h, w

        # sanity
        outer = all_fg + all_bg
        all_fg[outer == 2] = 0
        all_bg[outer == 2] = 0

        # assign
        out[all_fg == 1] = 1
        out[all_bg == 1] = 0

        out = out.detach()

        assert out.dtype == torch.long

        return out

    def extra_repr(self):
        return 'min_={}, max_={}, ' \
               'ksz={}, fg_erode_k: {}, fg_erode_iter: {}, ' \
               'seg_ignore_idx={}'.format(
                self.min_, self.max_, self.ksz, self.fg_erode_k,
                self.fg_erode_iter,
                self.ignore_idx)


class MBProbSeederSLCamsWithROI(MBSeederSLCAMS):
    def __init__(self,
                 min_: int = 1,
                 max_: int = 1,
                 max_p: float = 1.,
                 min_p: float = 1.,
                 ksz: int = 3,
                 seg_ignore_idx: int = -255,
                 seed_tech: str = _WEIGHTED,
                 bg_seed_tech: str = _WEIGHTED
                 ):
        """

        Sample seeds from CAM.

        New option 1: We sample foreground pixels only from a
        predefined ROI indicated via an input mask which differs from
        'MBProbSeederSLCAMS' that can sample foreground from entire CAM.

        New option 2: we added a way to control the portion from the images
        to sample BACKGROUND using 'min_p'. It is percentage (from entire
        image) of pixels to be considered background. min_ pixels will be
        sampled from these pixels. when set to 1., we are allowed to sample
        from entire image. when set for example to .2, we sort the
        activations in the 1 - CAM, and consider the top 20% pixels. Then,
        sample from them min_ pixels.

        New option 3: 'max_p'
        'max_p' similar to 'min_p' but for foreground. and the percentage is
        over the ROI not the image. for example, max_p = 0.1 means that 10%
        of the area of the ROI will be considered as potential foreground.

        New option 4: 'seed_tech'
        among the pixels considered FOREGROUND (before sampling
        the seeds), how to sample from them: - uniform (_UNIFORM) or - using
        CAM activation (_WEIGHTED).

        New option 4: 'bg_seed_tech'
        among the pixels considered BACKGROUND(before sampling
        the seeds), how to sample from them: - uniform (_UNIFORM) or - using
        CAM activation (_WEIGHTED). _UNIFORM is recommended.

        Notes:
        - set min_ = max_ to avoid unbalanced learning.
        - try to set min_p = max_p.
        - use sedd_tech = _WEIGHTED. [FG]
        - use bg_sedd_tech = _UNIFORM. [BG]

        Values to test:
        - min = max_ in {1, 10, 50, 100, 1k, 10k}
        - min_p = max_p in {0.1, 0.2, 0.3}. avoid alrge values. this is a ver
        critical hyper-parameter. use small values. the higher the value,
        the higher the risk of wron-sampling.
        - ksz=3.


        Does the same job as 'MBSeederSLCAMS' but, it does not require to
        define foreground and background sampling regions.
        We use a probabilistic approach to sample foreground and background.

        Supports batch.

        :param min_: int. number of pixels to sample from background.
        :param max_: int. number of pixels to sample from foreground.
        :param min_p: float [0, 1]. percentage (from entire image) of pixels
        to be considered background. min_ pixels will be sampled from these
        pixels. when set to 1., we are allowed to sample from entire image.
        when set for example to .2, we sort the activations in the 1 - CAM,
        and consider the top 20% pixels. Then, sample from them min_ pixels.
        :param ksz: int. kernel size to dilate the seeds.
        :param seg_ignore_idx: int. index for unknown seeds. 0: background.
        1: foreground. seg_ignore_idx: unknown.
        """
        super(MBProbSeederSLCamsWithROI, self).__init__()

        self._device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        assert isinstance(ksz, int)
        assert ksz > 0
        self.ksz = ksz
        self.kernel = None
        if self.ksz > 1:
            self.kernel = torch.ones((self.ksz, self.ksz), dtype=torch.long,
                                     device=self._device)

        assert isinstance(min_, int)
        assert isinstance(max_, int)
        assert min_ >= 0
        assert max_ >= 0
        assert min_ + max_ > 0

        assert isinstance(min_p, float)
        assert 0. <= min_p <= 1.
        self.min_p = min_p

        assert isinstance(max_p, float)
        assert 0. <= max_p <= 1.
        self.max_p = max_p

        assert seed_tech in [_UNIFORM, _WEIGHTED]
        self.seed_tech = seed_tech

        assert bg_seed_tech in [_UNIFORM, _WEIGHTED]
        self.bg_seed_tech = bg_seed_tech


        self.min_ = min_
        self.max_ = max_

        self.ignore_idx = seg_ignore_idx

    def forward(self, x: torch.Tensor, roi: torch.Tensor) -> torch.Tensor:
        """
        :param x: torch.Tensor. average cam: batchs, 1, h, w. Detach it first.
        :param roi: torch.Tensor. bianry mask that indicated foreground
        regions we are allowed to sample from. size: batch_sz, 1, h, w.
        :return: pseudo labels. batchsize, h, w
        """
        assert isinstance(x, torch.Tensor)
        assert x.ndim == 4
        assert isinstance(roi, torch.Tensor)
        assert x.shape == roi.shape

        # expensive
        # assert 0 <= torch.min(x) <= 1
        # assert 0 <= torch.max(x) <= 1

        if self.ksz == 1:
            dilate = self.identity
        elif self.ksz > 1:
            dilate = self.mb_dilate
        else:
            raise ValueError

        b, d, h, w = x.shape
        assert d == 1  # todo multilabl.

        out = torch.zeros((b, h, w), dtype=torch.long, requires_grad=False,
                          device=self._device) + self.ignore_idx

        all_fg = torch.zeros((b, h, w), dtype=torch.long, device=x.device,
                             requires_grad=False)
        all_bg = torch.zeros((b, h, w), dtype=torch.long, device=x.device,
                             requires_grad=False)

        nbr_bg = int(self.min_p * h * w)

        opx = _ProbaOneSample(min_=self.min_, max_=self.max_,
                              min_p=self.min_p, max_p=self.max_p,
                              seed_tech=self.seed_tech,
                              bg_seed_tech=self.bg_seed_tech
                              )

        for i in range(b):
            all_fg[i], all_bg[i] = opx(cam=x[i].squeeze(), roi=roi[i].squeeze())

        # fg
        all_fg = dilate(all_fg.unsqueeze(1)).squeeze(1)
        # b, h, w

        # bg
        all_bg = dilate(all_bg.unsqueeze(1)).squeeze(1)
        # b, h, w

        # sanity
        outer = all_fg + all_bg
        all_fg[outer == 2] = 0
        all_bg[outer == 2] = 0

        # assign
        out[all_fg == 1] = 1
        out[all_bg == 1] = 0

        out = out.detach()

        assert out.dtype == torch.long

        return out

    def extra_repr(self):
        return 'min_={}, max_={}, ' \
               'ksz={}, fg_erode_k: {}, fg_erode_iter: {}, ' \
               'seg_ignore_idx={}'.format(
                self.min_, self.max_, self.ksz, self.fg_erode_k,
                self.fg_erode_iter,
                self.ignore_idx)


class BinaryRegionsProposal(nn.Module):
    def __init__(self):
        super(BinaryRegionsProposal, self).__init__()

        self.otsu = _STOtsu()

    def _process_one_sample(self, cam: torch.Tensor) -> torch.Tensor:
        assert cam.ndim == 2  # h, w

        # cam: h, w
        h, w = cam.shape
        fg = torch.zeros((h, w), dtype=torch.long, device=cam.device,
                         requires_grad=False)

        # otsu
        cam_ = torch.floor(cam * 255.)
        th = self.otsu(x=cam_)

        if th == 0:
            th = 1.
        if th == 255:
            th = 254.

        if self.otsu.bad_egg:
            return fg

        # FG: ROI
        roi = (cam_ > th).long()  # 1: foureground. 0: rest.

        return roi

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: btahc_size, n, h, w.
        assert x.ndim == 4

        b, n, h, w = x.shape

        out = torch.zeros((b, n, h, w), dtype=torch.long, device=x.device,
                          requires_grad=False)

        for i in range(b):
            for j in range(n):
                out[i, j] = self._process_one_sample(cam=x[i, j])

        assert out.dtype == torch.long
        return out


def test_MBSeederSLCAMS():
    import cProfile

    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.colors import ListedColormap
    from torch.profiler import profile, record_function, ProfilerActivity
    import time

    def get_cm():
        col_dict = dict()
        for i in range(256):
            col_dict[i] = 'k'

        col_dict[0] = 'k'
        col_dict[int(255 / 2)] = 'b'
        col_dict[255] = 'r'
        colormap = ListedColormap([col_dict[x] for x in col_dict.keys()])

        return colormap

    def cam_2Img(_cam):
        return _cam.squeeze().cpu().numpy().astype(np.uint8) * 255

    def plot_limgs(_lims, title):
        nrows = 1
        ncols = len(_lims)

        him, wim = _lims[0][0].shape
        r = him / float(wim)
        fw = 10
        r_prime = r * (nrows / float(ncols))
        fh = r_prime * fw

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=False,
                                 sharey=False, squeeze=False, figsize=(fw, fh))
        for i, (im, tag) in enumerate(_lims):
            axes[0, i].imshow(im, cmap=get_cm())
            axes[0, i].text(3, 40, tag,
                            bbox={'facecolor': 'white', 'pad': 1, 'alpha': 0.8})
        plt.suptitle(title)
        plt.show()

    cuda = 1
    torch.cuda.set_device(cuda)

    seed = 0
    min_ = 10
    max_ = 10
    min_p = .2
    fg_erode_k = 11
    fg_erode_iter = 1

    batchs = 1

    cam = torch.rand((batchs, 1, 224, 224), dtype=torch.float,
                     device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), requires_grad=False)

    cam = cam * 0
    for i in range(batchs):
        cam[i, 0, 100:150, 100:150] = 1
    limgs_conc = [(cam_2Img(cam), 'CAM')]

    for ksz in [3]:
        set_seed(seed)
        module_conc = MBSeederSLCAMS(
            min_=min_,
            max_=max_,
            min_p=min_p,
            fg_erode_k=fg_erode_k,
            fg_erode_iter=fg_erode_iter,
            ksz=ksz,
            seg_ignore_idx=-255)
        print('Testing {}'.format(module_conc))

        # cProfile.runctx('module_conc(cam)', globals(), locals())

        t0 = time.perf_counter()
        out = module_conc(cam)

        print('time CONC: {}s'.format(time.perf_counter() - t0))

        out[out == 1] = 255
        out[out == 0] = int(255 / 2)
        out[out == module_conc.ignore_idx] = 0

        if batchs == 1:
            limgs_conc.append((out.squeeze().cpu().numpy().astype(np.uint8),
                               'pseudo_ksz_{}_conc'.format(ksz)))

    if batchs == 1:
        plot_limgs(limgs_conc, 'CONCURRENT-THREH')


def test_MBProbSeederSLCAMS():
    import cProfile

    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.colors import ListedColormap
    from torch.profiler import profile, record_function, ProfilerActivity
    import time

    def get_cm():
        col_dict = dict()
        for i in range(256):
            col_dict[i] = 'k'

        col_dict[0] = 'k'
        col_dict[int(255 / 2)] = 'b'
        col_dict[255] = 'r'
        colormap = ListedColormap([col_dict[x] for x in col_dict.keys()])

        return colormap

    def cam_2Img(_cam):
        return _cam.squeeze().cpu().numpy().astype(np.uint8) * 255

    def plot_limgs(_lims, title):
        nrows = 1
        ncols = len(_lims)

        him, wim = _lims[0][0].shape
        r = him / float(wim)
        fw = 10
        r_prime = r * (nrows / float(ncols))
        fh = r_prime * fw

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=False,
                                 sharey=False, squeeze=False, figsize=(fw, fh))
        for i, (im, tag) in enumerate(_lims):
            axes[0, i].imshow(im, cmap=get_cm())
            axes[0, i].text(3, 40, tag,
                            bbox={'facecolor': 'white', 'pad': 1, 'alpha': 0.8})
        plt.suptitle(title)
        plt.show()

    cuda = 1
    torch.cuda.set_device(cuda)

    seed = 0
    min_ = 10
    max_ = 10

    batchs = 1

    cam = torch.rand((batchs, 1, 224, 224), dtype=torch.float,
                     device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), requires_grad=False)

    cam = cam * 0
    for i in range(batchs):
        cam[i, 0, 100:150, 100:150] = 1
    limgs_conc = [(cam_2Img(cam), 'CAM')]

    for ksz in [3]:
        set_seed(seed)
        module_conc = MBProbSeederSLCAMS(
            min_=min_,
            max_=max_,
            ksz=ksz,
            seg_ignore_idx=-255)
        print('Testing {}'.format(module_conc))

        # cProfile.runctx('module_conc(cam)', globals(), locals())

        t0 = time.perf_counter()
        out = module_conc(cam)

        print('time CONC: {}s'.format(time.perf_counter() - t0))

        out[out == 1] = 255
        out[out == 0] = int(255 / 2)
        out[out == module_conc.ignore_idx] = 0

        if batchs == 1:
            limgs_conc.append((out.squeeze().cpu().numpy().astype(np.uint8),
                               'pseudo_ksz_{}_conc'.format(ksz)))

    if batchs == 1:
        plot_limgs(limgs_conc, 'CONCURRENT-PROB')


def test_MBProbSeederSLCamsWithROI():
    import warnings
    import sys
    import os
    from os.path import dirname, abspath, join, basename

    root_dir = dirname(dirname(abspath(__file__)))
    sys.path.append(root_dir)
    print(root_dir)

    import cProfile
    from os.path import join

    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.colors import ListedColormap
    from torch.profiler import profile, record_function, ProfilerActivity
    import time

    def get_cm():
        col_dict = dict()
        for i in range(256):
            col_dict[i] = 'k'

        col_dict[0] = 'k'
        col_dict[int(255 / 2)] = 'b'
        col_dict[255] = 'r'
        colormap = ListedColormap([col_dict[x] for x in col_dict.keys()])

        return colormap

    def cam_2Img(_cam):
        return (_cam.squeeze().cpu().numpy() * 255).astype(np.uint8)

    def plot_limgs(_lims, title):
        nrows = 1
        ncols = len(_lims)

        him, wim = _lims[0][0].shape
        r = him / float(wim)
        fw = 10
        r_prime = r * (nrows / float(ncols))
        fh = r_prime * fw

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=False,
                                 sharey=False, squeeze=False, figsize=(fw, fh))
        for i, (im, tag) in enumerate(_lims):
            if tag != 'CAM':
                axes[0, i].imshow(im, cmap=get_cm())
            else:
                axes[0, i].imshow(im, cmap='jet')
            axes[0, i].text(3, 40, tag,
                            bbox={'facecolor': 'white', 'pad': 1, 'alpha': 0.8})
        plt.suptitle(title)
        plt.show()

    cuda = 1
    torch.cuda.set_device(cuda)

    seed = 0
    min_ = 100000
    max_ = 100000
    min_p = .1
    max_p = .1
    seed_tech = _WEIGHTED
    bg_seed_tech = _UNIFORM
    batchs = 1

    cam = torch.rand((batchs, 1, 224, 224), dtype=torch.float,
                     device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), requires_grad=False)
    path_cam = join(root_dir,
                    'aux/data/train_n02229544_n02229544_2648.JPEG.npy')
    np_cam = np.load(path_cam)
    np_cam = np_cam.squeeze()
    device = torch.device(f'cuda:{cuda}' if torch.cuda.is_available() else
                          'cpu')
    cam = torch.from_numpy(np_cam).to(device)
    cam: torch.Tensor = cam.float()
    cam.requires_grad = False
    cam = cam.view(1, 224, 224)
    cam = cam.repeat(batchs, 1, 1, 1)

    # cam = cam * 0
    roi = cam * 0.
    for i in range(batchs):
        # cam[i, 0, 100:150, 100:150] = 1
        roi[i, 0, 10:170, 30:200] = 1


    limgs_conc = [(cam_2Img(cam), 'CAM'), (cam_2Img(roi), 'ROI')]

    for ksz in [3]:
        set_seed(seed)
        module_conc = MBProbSeederSLCamsWithROI(
            min_=min_,
            max_=max_,
            min_p=min_p,
            max_p=max_p,
            ksz=ksz,
            seg_ignore_idx=-255,
            seed_tech=seed_tech,
            bg_seed_tech=bg_seed_tech
        )
        print('Testing {}'.format(module_conc))

        # cProfile.runctx('module_conc(cam)', globals(), locals())

        t0 = time.perf_counter()
        out = module_conc(cam, roi)

        print('time CONC: {}s'.format(time.perf_counter() - t0))

        out[out == 1] = 255
        out[out == 0] = int(255 / 2)
        out[out == module_conc.ignore_idx] = 0

        if batchs == 1:
            limgs_conc.append((out.squeeze().cpu().numpy().astype(np.uint8),
                               'pseudo_ksz_{}_conc'.format(ksz)))

    if batchs == 1:
        plot_limgs(limgs_conc, f'CONCURRENT-PROB-{seed_tech}')

def mkdir(fd):
    if not os.path.isdir(fd):
        os.makedirs(fd, exist_ok=True)


def set_seed(seed: int = 0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def test_BinaryRegionsProposal():
    cuda = 1
    device = torch.device(f'cuda:{cuda}' if torch.cuda.is_available() else
                          'cpu')
    torch.cuda.set_device(cuda)

    brpn = BinaryRegionsProposal()
    x = torch.rand(32, 10, 256, 256).to(device)
    out = brpn(x)
    print(f'input: {out.shape}(dtype: {x.dtype}) output: {out.shape} '
          f'(dtype: {out.dtype})')


if __name__ == "__main__":
    # install:
    # pip install kornia==0.6.1
    # pytorch 1.10.0
    import datetime as dt

    set_seed(0)
    # test_MBSeederSLCAMS()
    # test_MBProbSeederSLCAMS()
    # test_BinaryRegionsProposal()
    test_MBProbSeederSLCamsWithROI()
