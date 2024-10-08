import datetime as dt
import sys
from copy import deepcopy

from dlib.process.parseit import parse_input

from dlib.process.instantiators import get_model
from dlib.process.instantiators import get_optimizer
from dlib.utils.tools import get_device
from dlib.utils.tools import bye

from dlib.configure import constants
from dlib.learning.train_wsol import Trainer
from dlib.process.instantiators import get_loss
from dlib.process.instantiators import get_pretrainde_classifier
from dlib.utils.shared import fmsg
from dlib.utils.shared import is_cc

import dlib.dllogger as DLLogger


args, args_dict = parse_input(eval=False)


if __name__ == '__main__':
    device = get_device(args)
    model = get_model(args)
    model.to(device)
    best_state_dict = deepcopy(model.state_dict())

    optimizer, lr_scheduler = get_optimizer(args, model)
    loss = get_loss(args, device)

    inter_classifier = None
    if args.task == constants.F_CL:
        inter_classifier = get_pretrainde_classifier(args)
        inter_classifier.to(device)

    trainer: Trainer = Trainer(
        args=args, model=model, optimizer=optimizer,
        lr_scheduler=lr_scheduler, loss=loss, device=device,
        classifier=inter_classifier)

    DLLogger.log(fmsg("Start epoch 0 ..."))

    trainer.evaluate(epoch=0, split=constants.VALIDSET)
    trainer.print_performances()
    trainer.report(epoch=0, split=constants.VALIDSET)
    trainer.model_selection(epoch=0, split=constants.VALIDSET)

    DLLogger.log(fmsg("Epoch 0 done."))

    for epoch in range(trainer.args.max_epochs):
        zepoch = epoch + 1
        DLLogger.log(fmsg(("Start epoch {} ...".format(zepoch))))

        train_performance = trainer.train(
            split=constants.TRAINSET, epoch=zepoch)
        trainer.report_train(train_performance, zepoch,
                             split=constants.TRAINSET)
        trainer.evaluate(zepoch, split=constants.VALIDSET)
        trainer.print_performances()
        trainer.report(zepoch, split=constants.VALIDSET)
        trainer.model_selection(epoch=zepoch, split=constants.VALIDSET)
        DLLogger.log(fmsg(("Epoch {} done.".format(zepoch))))

        trainer.adjust_learning_rate()
        DLLogger.flush()

    # trainer.save_checkpoints(split=constants.VALIDSET)
    trainer.capture_perf_meters()

    DLLogger.log(fmsg("Final epoch evaluation on test set ..."))
    
    
    eval_checkpoints = []
    if args.dataset == constants.CUB:
        eval_checkpoints = [constants.BEST_LOC, constants.BEST_CL]
    #use these checkpoints to eval on pseduo bboxs
    #best_localization_using_metadata_by_clip, best_localization_using_metadata_by_ss, best_localization_using_metadata_by_rpn
    
    #########ABALATION ONLY#########  
    if args.ablation_model_sele_map_area:
        eval_checkpoints += [constants.BEST_BBOXS_SIZE_VAR]
    
    if args.ablation_model_sele_l2_norm:
        eval_checkpoints += [constants.BEST_WEIGHTS_NORM_L2]
    #########ABALATION ONLY#########  

    for eval_checkpoint_type in eval_checkpoints:#[constants.BEST_LOC, constants.BEST_CL]:#.LAST, constants.BEST_CL]:
        t0 = dt.datetime.now()

        DLLogger.log(fmsg('EVAL TEST SET. CHECKPOINT: {}'.format(
            eval_checkpoint_type)))

        if eval_checkpoint_type == constants.BEST_LOC:
            epoch = trainer.args.best_epoch_loc
        elif eval_checkpoint_type == constants.BEST_CL:
            epoch = trainer.args.best_epoch_cl
        elif eval_checkpoint_type == constants.LAST:
            epoch = trainer.args.max_epochs
        elif eval_checkpoint_type == constants.BEST_BBOXS_SIZE_VAR:
            epoch = trainer.args.best_epoch_bboxs_size_var
        elif eval_checkpoint_type == constants.BEST_WEIGHTS_NORM_L2:
            epoch = trainer.args.best_epoch_weights_norm_l2
        else:
            raise NotImplementedError

        trainer.load_checkpoint(checkpoint_type=eval_checkpoint_type)

        argmax = [False]
        if args.task == constants.F_CL:
            pass
            # argmax += [True]

        for fcam_argmax in argmax:
            trainer.evaluate(epoch, split=constants.TESTSET,
                             checkpoint_type=eval_checkpoint_type,
                             fcam_argmax=fcam_argmax)
            
            trainer.plot_valdation_acc_with_pseudo_bbox(eval_checkpoint_type=eval_checkpoint_type)

            trainer.print_performances(checkpoint_type=eval_checkpoint_type)
            trainer.report(epoch, split=constants.TESTSET,
                           checkpoint_type=eval_checkpoint_type)
            trainer.save_performances(
                epoch=epoch, checkpoint_type=eval_checkpoint_type)
            trainer.switch_perf_meter_to_captured()

            tagargmax = 'Argmax: True' if fcam_argmax else ''

            DLLogger.log("EVAL time TESTSET - CHECKPOINT {} {}: {}".format(
                eval_checkpoint_type, tagargmax, dt.datetime.now() - t0))
            DLLogger.flush()

    trainer.save_args()
    # trainer.plot_perfs_meter()
    trainer.plot_perfs_meter(checkpoint_type=constants.LOCALIZATION_MTR)
    trainer.plot_perfs_meter(checkpoint_type=constants.CLASSIFICATION_MTR)
    bye(trainer.args)
