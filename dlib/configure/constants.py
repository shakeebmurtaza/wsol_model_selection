# possible tasks
STD_CL = "STD_CL"  # standard classification using only the encoder features.
# ouputs: logits, cams.
F_CL = 'F_CL'  # standard classification but using the decoder features.
# outputs: logits, cams.
SEG = "SEGMENTATION"  # standard supervised segmentation. outputs:
# segmentation masks.

TASKS = [STD_CL, F_CL, SEG]


# name of the classifier head (pooling operation)
WILDCATHEAD = "WildCatCLHead"

GAP = 'GAP'
WGAP = 'WGAP'
MAXPOOL = 'MaxPool'
LSEPOOL = 'LogSumExpPool'
NONEPOOL = 'NONE'

SPATIAL_POOLINGS = [WILDCATHEAD, GAP, WGAP, MAXPOOL, LSEPOOL, NONEPOOL]

# methods
METHOD_WILDCAT = 'WILDCAT'  # pooling: WILDCATHEAD
METHOD_GAP = 'GAP'  # pooling: GAP

METHOD_MAXPOOL = 'MaxPOL'  # pooling: MAXPOOL
METHOD_LSE = 'LogSumExp'  # pooling: logsumexp.

# -- all methods below use WGAP.

METHOD_CAM = 'CAM'
METHOD_SCORECAM = 'ScoreCAM'
METHOD_SSCAM = 'SSCAM'
METHOD_ISCAM = 'ISCAM'

METHOD_GRADCAM = 'GradCam'
METHOD_GRADCAMPP = 'GradCAMpp'
METHOD_SMOOTHGRADCAMPP = 'SmoothGradCAMpp'
METHOD_XGRADCAM = 'XGradCAM'
METHOD_LAYERCAM = 'LayerCAM'

METHOD_HAS = 'HAS'

METHOD_ADL = 'ADL'
METHOD_SPG = 'SPG'
METHOD_ACOL = 'ACOL'
METHOD_TSCAM = 'TSCAM'

METHOD_NLCCAM = 'NL-CCAM'
METHOD_SAT = 'SAT'
METHOD_SCM = 'SCM'


METHODS = [METHOD_WILDCAT,
           METHOD_GAP,
           METHOD_MAXPOOL,
           METHOD_LSE,
           METHOD_CAM,
           METHOD_SCORECAM,
           METHOD_SSCAM,
           METHOD_ISCAM,
           METHOD_GRADCAM,
           METHOD_GRADCAMPP,
           METHOD_SMOOTHGRADCAMPP,
           METHOD_XGRADCAM,
           METHOD_LAYERCAM,
           METHOD_HAS,
           METHOD_ADL,
           METHOD_SPG,
           METHOD_ACOL,
           METHOD_TSCAM,
           METHOD_NLCCAM,
           METHOD_SAT,
           METHOD_SCM]

METHOD_2_POOLINGHEAD = {
        METHOD_WILDCAT: WILDCATHEAD,
        METHOD_GAP: GAP,
        METHOD_MAXPOOL: MAXPOOL,
        METHOD_LSE: LSEPOOL,
        METHOD_CAM: WGAP,
        METHOD_SCORECAM: WGAP,
        METHOD_SSCAM: WGAP,
        METHOD_ISCAM: WGAP,
        METHOD_GRADCAM: WGAP,
        METHOD_GRADCAMPP: WGAP,
        METHOD_SMOOTHGRADCAMPP: WGAP,
        METHOD_XGRADCAM: WGAP,
        METHOD_LAYERCAM: WGAP,
        METHOD_ACOL: NONEPOOL,
        METHOD_ADL: NONEPOOL,
        METHOD_SPG: NONEPOOL,
        METHOD_HAS: WGAP,
        METHOD_TSCAM: GAP,
        METHOD_NLCCAM: GAP,
        METHOD_SAT: GAP,
        METHOD_SCM: GAP
    }

METHOD_LITERAL_NAMES = {
        METHOD_WILDCAT: 'WILDCAT',
        METHOD_GAP: 'GAP',
        METHOD_MAXPOOL: 'MaxPool',
        METHOD_LSE: 'LSEPool',
        METHOD_CAM: 'CAM*',
        METHOD_SCORECAM: 'ScoreCAM',
        METHOD_SSCAM: 'SSCAM',
        METHOD_ISCAM: 'ISCAM',
        METHOD_GRADCAM: 'GradCAM',
        METHOD_GRADCAMPP: 'GradCam++',
        METHOD_SMOOTHGRADCAMPP: 'Smooth-GradCAM++',
        METHOD_XGRADCAM: 'XGradCAM',
        METHOD_LAYERCAM: 'LayerCAM',
        METHOD_ACOL: 'ACOsL',
        METHOD_ADL: 'ADL',
        METHOD_SPG: 'SPG',
        METHOD_HAS: 'HAS',
        METHOD_TSCAM: 'TS-CAM',
        METHOD_NLCCAM: 'NL-CCAM',
        METHOD_SAT: 'SAT',
        METHOD_SCM: 'SCM'
}
# datasets mode
DS_TRAIN = "TRAIN"
DS_EVAL = "EVAL"

dataset_modes = [DS_TRAIN, DS_EVAL]

# Tags for samples
L = 0  # Labeled samples

samples_tags = [L]  # list of possible sample tags.

# pixel-wise supervision:
ORACLE = "ORACLE"  # provided by an oracle.
SELF_LEARNED = "SELF-LEARNED"  # self-learned.
VOID = "VOID"  # None

# segmentation modes.
#: Loss binary mode suppose you are solving binary segmentation task.
#: That mean yor have only one class which pixels are labled as **1**,
#: the rest pixels are background and labeled as **0**.
#: Target mask shape - (N, H, W), model output mask shape (N, 1, H, W).
BINARY_MODE: str = "binary"

#: Loss multiclass mode suppose you are solving multi-**class** segmentation task.
#: That mean you have *C = 1..N* classes which have unique label values,
#: classes are mutually exclusive and all pixels are labeled with theese values.
#: Target mask shape - (N, H, W), model output mask shape (N, C, H, W).
MULTICLASS_MODE: str = "multiclass"

#: Loss multilabel mode suppose you are solving multi-**label** segmentation task.
#: That mean you have *C = 1..N* classes which pixels are labeled as **1**,
#: classes are not mutually exclusive and each class have its own *channel*,
#: pixels in each channel which are not belong to class labeled as **0**.
#: Target mask shape - (N, C, H, W), model output mask shape (N, C, H, W).
MULTILABEL_MODE: str = "multilabel"


# pretraining
IMAGENET = "imagenet"

# archs
STDCLASSIFIER = "STDClassifier"
TSCAMCLASSIFIER = 'TSCAMClassifier'
NLCCAMCLASSIFIER = 'NLCCAMClassifier'
SATCLASSIFIER = 'SATClassifier'
SCMCLASSIFIER = 'SCMClassifier'

UNETFCAM = 'UnetFCAM'  # USED

ACOLARCH = 'ACOL'
SPGARCH = 'SPG'
ADLARCH = 'ADL'

UNET = "Unet"
UNETPLUPLUS = "UnetPlusPlus"
MANET = "MAnet"
LINKNET = "Linknet"
FPN = "FPN"
PSPNET = "PSPNet"
DEEPLABV3 = "DeepLabV3"
DEEPLABV3PLUS = "DeepLabV3Plus"
PAN = "PAN"

ARCHS = [STDCLASSIFIER, UNETFCAM, TSCAMCLASSIFIER, NLCCAMCLASSIFIER,
         SATCLASSIFIER, SCMCLASSIFIER,
         ACOLARCH, ADLARCH, SPGARCH,]

# std cld method to arch.
STD_CL_METHOD_2_ARCH = {
    METHOD_WILDCAT: STDCLASSIFIER,
    METHOD_GAP: STDCLASSIFIER,
    METHOD_MAXPOOL: STDCLASSIFIER,
    METHOD_LSE: STDCLASSIFIER,
    METHOD_CAM: STDCLASSIFIER,
    METHOD_SCORECAM: STDCLASSIFIER,
    METHOD_SSCAM: STDCLASSIFIER,
    METHOD_ISCAM: STDCLASSIFIER,
    METHOD_GRADCAM: STDCLASSIFIER,
    METHOD_GRADCAMPP: STDCLASSIFIER,
    METHOD_SMOOTHGRADCAMPP: STDCLASSIFIER,
    METHOD_XGRADCAM: STDCLASSIFIER,
    METHOD_LAYERCAM: STDCLASSIFIER,
    METHOD_ACOL: ACOLARCH,
    METHOD_ADL: ADLARCH,
    METHOD_SPG: SPGARCH,
    METHOD_HAS: STDCLASSIFIER,
    METHOD_TSCAM: TSCAMCLASSIFIER,
    METHOD_NLCCAM: NLCCAMCLASSIFIER,
    METHOD_SAT: SATCLASSIFIER,
    METHOD_SCM: SCMCLASSIFIER
}

# ecnoders

#  resnet
RESNET50 = 'resnet50'

# vgg
VGG16 = 'vgg16'

# inceptionv3
INCEPTIONV3 = 'inceptionv3'

# DEIT TSCAM
DEIT_TSCAM_BASE_P16_224 = 'deit_tscam_base_patch16_224'
DEIT_TSCAM_SMALL_P16_224 = 'deit_tscam_small_patch16_224'
DEIT_TSCAM_TINY_P16_224 = 'deit_tscam_tiny_patch16_224'

# DEIT SAT
DEIT_SAT_BASE_P16_224 = 'deit_sat_base_patch16_224'
DEIT_SAT_SMALL_P16_224 = 'deit_sat_small_patch16_224'
DEIT_SAT_TINY_P16_224 = 'deit_sat_tiny_patch16_224'

# DEIT SCM
DEIT_SCM_BASE_P16_224 = 'deit_scm_base_patch16_224'
DEIT_SCM_SMALL_P16_224 = 'deit_scm_small_patch16_224'
DEIT_SCM_TINY_P16_224 = 'deit_scm_tiny_patch16_224'

# NL-CCAM
NLCCAM_VGG16 = 'nlccam_vgg16'


BACKBONES = [RESNET50,
             VGG16,
             INCEPTIONV3,
             DEIT_TSCAM_SMALL_P16_224,
             DEIT_TSCAM_BASE_P16_224,
             DEIT_TSCAM_TINY_P16_224,
             NLCCAM_VGG16,
             DEIT_SAT_SMALL_P16_224,
             DEIT_SAT_BASE_P16_224,
             DEIT_SAT_TINY_P16_224,
             DEIT_SCM_SMALL_P16_224,
             DEIT_SCM_BASE_P16_224,
             DEIT_SCM_TINY_P16_224,
             ]

TSCAM_BACKBONES = [DEIT_TSCAM_SMALL_P16_224,
                   DEIT_TSCAM_BASE_P16_224,
                   DEIT_TSCAM_TINY_P16_224]

SAT_BACKBONES = [DEIT_SAT_SMALL_P16_224,
                 DEIT_SAT_BASE_P16_224,
                 DEIT_SAT_TINY_P16_224]

SCM_BACKBONES = [DEIT_SCM_SMALL_P16_224,
                 DEIT_SCM_BASE_P16_224,
                 DEIT_SCM_TINY_P16_224]

# ------------------------------------------------------------------------------

# datasets
DEBUG = False

ILSVRC = "ILSVRC"
CUB = "CUB"
OpenImages = 'OpenImages'

FORMAT_DEBUG = 'DEBUG_{}'
if DEBUG:
    CUB = FORMAT_DEBUG.format(CUB)
    ILSVRC = FORMAT_DEBUG.format(ILSVRC)
    OpenImages = FORMAT_DEBUG.format(OpenImages)


datasets = [CUB, ILSVRC, OpenImages]

RELATIVE_META_ROOT = './folds/wsol-done-right-splits'

NUMBER_CLASSES = {
    ILSVRC: 1000,
    CUB: 200,
    OpenImages: 100
}

CROP_SIZE = 224
RESIZE_SIZE = 256

# ================= check points
LAST = 'last'
BEST_LOC = 'best_localization'
BEST_CL = 'best_classification'
BEST_LOC_RAW_IOU = 'best_loc_raw_iou'
BEST_LOC_MULTI_IOU = 'best_loc_multi_iou'

BEST_BBOXS_SIZE_VAR = 'best_bboxs_size_variance'
BEST_WEIGHTS_NORM_L2 = 'best_weights_norm_l2'

# metrics names
LOCALIZATION_MTR = 'localization'
CLASSIFICATION_MTR = 'classification'

LOCALIZATION_MTR_MULTI_IOU = 'localization_multi_iou'
LOCALIZATION_MTR_RAW_IOU = 'raw_iou'

BBOXS_SIZE_MTR = 'bboxs_size'
BBOXS_SIZE_VAR_MTR = 'bboxs_size_variance'
WEIGHTS_NORM_L2_MTR = 'l2_weights_norm'
MTRS_ABLATION_STUDIES = [BBOXS_SIZE_MTR, BBOXS_SIZE_VAR_MTR, WEIGHTS_NORM_L2_MTR]
# ==============================================================================

# Colours
COLOR_WHITE = "white"
COLOR_BLACK = "black"

# backbones.

# =================================================
NCOLS = 80  # tqdm ncols.

# stages:
STGS_TR = "TRAIN"
STGS_EV = "EVAL"


# datasets:
TRAINSET = 'train'
VALIDSET = 'val'
TESTSET = 'test'

SPLITS = [TRAINSET, VALIDSET, TESTSET]

# image range: [0, 1] --> Sigmoid. [-1, 1]: TANH
RANGE_TANH = "tanh"
RANGE_SIGMOID = 'sigmoid'

# ==============================================================================
# cams extractor
TRG_LAYERS = {
            RESNET50: 'encoder.layer4.2.relu3',
            VGG16: 'encoder.relu',
            INCEPTIONV3: 'encoder.SPG_A3_2b.2'
        }
FC_LAYERS = {
    RESNET50: 'classification_head.fc',
    VGG16: 'classification_head.fc',
    INCEPTIONV3: 'classification_head.fc'
}

# EXPs
OVERRUN = False

# cam_curve_interval: for bbox. use high interval for validation (not test).
# high number of threshold slows down the validation because of
# `cv2.findContours`. this gets when cams are bad leading to >1k contours per
# threshold. default evaluation: .001.
VALID_FAST_CAM_CURVE_INTERVAL = .004

# data: name of the folder where cams will be stored.
DATA_CAMS = 'data_cams'

FULL_BEST_EXPS = 'full_best_exps'

# folder
FOLDER_PRETRAINED_IMAGENET = 'pretrained-imgnet'

#Configuration for ablations
ABL_WINOW_MAP_SIZE_VAR = {
    ILSVRC: 2,
    CUB: 5,
}
