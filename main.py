# ------------------------------------------------------------------
# SimpleNet: A Simple Network for Image Anomaly Detection and Localization (https://openaccess.thecvf.com/content/CVPR2023/papers/Liu_SimpleNet_A_Simple_Network_for_Image_Anomaly_Detection_and_Localization_CVPR_2023_paper.pdf)
# Github source: https://github.com/DonaldRR/SimpleNet
# Licensed under the MIT License [see LICENSE for details]
# The script is based on the code of PatchCore (https://github.com/amazon-science/patchcore-inspection)
# ------------------------------------------------------------------

import pandas as pd
import wandb
import argparse
import json
from trainer.trainer_simplenet import Trainer_SimpleNet
from trainer.trainer_ours_score import Trainer_Ours_Score
from trainer.trainer_ours_score_iv import Trainer_Ours_Score_Iv
from trainer.trainer_ours_score_patchlevel import Trainer_Ours_Score_PatchLevel
from trainer.trainer_ours_score_patchlevel_interpretable import Trainer_Ours_Score_PatchLevel_Interpretable
from trainer.trainer_ours_score_patchlevel_iv import Trainer_Ours_Score_PatchLevel_Iv
from trainer.trainer_ours_score_patchlevel_mid_few import Trainer_Ours_Score_PatchLevel_Mid_Few
from trainer.trainer_ours_score_att import Trainer_Ours_Score_Att
from trainer.trainer_ours_gans_img import Trainer_Ours_GANs_IMG
from trainer.trainer_ours_nogans_img import Trainer_Ours_NoGANs_IMG
from trainer.trainer_ours_gans_img_coreset import Trainer_Ours_GANs_IMG_Coreset
from trainer.trainer_ours_nogans_img_coreset import Trainer_Ours_NoGANs_IMG_Coreset
from trainer.trainer_patchcore import Trainer_PatchCore
from trainer.trainer_patchcore_few import Trainer_PatchCore_Few
from trainer.trainer_patchcore_few_att import Trainer_PatchCore_Few_Att
from trainer.trainer_ours_att import Trainer_Ours_Attention
from trainer.trainer_ours_att_score import Trainer_Ours_Attention_Score
from trainer.trainer_graphcore import Trainer_GraphCore
from trainer.trainer_interpretable_signet import Trainer_Interpretable_SIGNET
from intra_class_variance.plot_var import PlotVar
import utils
import backbones
import logging
import os
import sys
import pandas as pd

import numpy as np
import torch
import warnings
from loguru import logger
from torch.utils.data import DataLoader

sys.path.append("src")

_DATASETS = {
    "mvtec": ["datasets.mvtec", "MVTecDataset"],
    "lg": ["datasets.lgdata", "LGDataset"],
}
warnings.filterwarnings('ignore')
with warnings.catch_warnings():
    warnings.filterwarnings(
        'ignore', r'RuntimeWarning: invalid value encountered')


def print_stat(dataset, name):
    targets = []
    for data in dataset.data_to_iterate:
        targets.append(int(data[1] != "good"))

    classes, nums = torch.tensor(targets).unique(return_counts=True)
    for classes, num in zip(classes, nums):
        logger.info(f"{name} class {classes} : {num}")
    print()


def run(
    methods,
    args
):
    results_path, gpu, seed_list, log_group, log_project, run_name, save_segmentation_images = args.results_path, args.gpu, args.seed_list, args.log_group, args.log_project, args.run_name, args.save_segmentation_images

    methods = {key: item for (key, item) in methods}

    if log_group != "":
        run_save_path = utils.create_storage_folder(
            results_path, log_project, log_group, run_name, mode="overwrite"
        )
    else:
        run_save_path = None

    pid = os.getpid()
    # NOTE 확인 결과, seed 는 data 에는 사용되지 않음
    list_of_dataloaders = methods["get_dataloaders"](0)

    device = utils.set_torch_device(gpu)

    if args.check_intra_class_var:  # intra class variance 를 확인하는 경우
        df_list = []
        for dataloader_count, dataloaders in enumerate(list_of_dataloaders):
            dataset_name = dataloaders["training"].name
            imagesize = dataloaders["training"].dataset.imagesize
            simplenet_list = methods["get_simplenet"](imagesize, device)
            for i, SimpleNet in enumerate(simplenet_list):
                df = SimpleNet.eval_intra_class_variance(
                    dataloaders["training"], dataloaders["validation"], dataloaders["testing"], dataset_name)
                df_list.append(df)

        df = pd.concat(df_list).sort_values(by=["metric_name", "mean"])
        # df.to_csv("intra_class_variance_analysis.csv", index=False)
        df.to_csv(os.path.join("results_intra_class_variance",
                  f"intra_class_variance_analysis_{args.name}.csv"), index=False)

        print(df)
        PlotVar().plot_intra_class_var(df)
        exit(0)  # NOTE 종료됨

    result_collect = []
    # seed_i = 0  # NOTE TEMP
    for seed in seed_list:
        for dataloader_count, dataloaders in enumerate(list_of_dataloaders):
            logger.info(
                "Evaluating dataset [{}] ({}/{})...".format(
                    dataloaders["training"].name,
                    dataloader_count + 1,
                    len(list_of_dataloaders),
                )
            )

            # NOTE dataset 는 고정 seed 0 으로 처리하고, model 은 각각의 seed 로 처리함
            utils.fix_seeds(seed, device)

            dataset_name = dataloaders["training"].name

            imagesize = dataloaders["training"].dataset.imagesize
            simplenet_list = methods["get_simplenet"](imagesize, device)

            if run_save_path is not None:
                models_dir = os.path.join(run_save_path, "models")
                os.makedirs(models_dir, exist_ok=True)
            else:
                models_dir = None

            for i, SimpleNet in enumerate(simplenet_list):
                # torch.cuda.empty_cache()
                if SimpleNet.backbone.seed is not None:
                    utils.fix_seeds(SimpleNet.backbone.seed, device)
                logger.info(
                    "Training models ({}/{})".format(i + 1,
                                                     len(simplenet_list))
                )
                # torch.cuda.empty_cache()

                if models_dir is not None:
                    SimpleNet.set_model_dir(os.path.join(
                        models_dir, f"{i}"), dataset_name)

                i_auroc, p_auroc, pro_auroc, segmentations, labels_gt, scores = SimpleNet.train(
                    dataloaders["training"], dataloaders["validation"], dataloaders["testing"], dataset_name)
                # i_auroc = seed_i
                # seed_i += 1

                logger.info(
                    f"----- {dataset_name} : TEST I-AUROC:{round(i_auroc, 4)}")

                result_collect.append(
                    {
                        "dataset_name": dataset_name,
                        "instance_auroc": i_auroc,  # auroc,
                        # "full_pixel_auroc": p_auroc,  # full_pixel_auroc,
                        # "anomaly_pixel_auroc": pro_auroc,
                        "seed": seed
                    }
                )

            # convert result_collect into a pandas dataframe
            result_df = pd.DataFrame(result_collect).set_index("dataset_name")

            logger.info("\n\n-----\n")
            logger.info({f"{dataset_name}_test_auroc": i_auroc})
            if args.wan:
                wandb.log(
                    {f"{dataset_name}_test_auroc": i_auroc})

    logger.info("\n\n-----\n")
    logger.info("\n\n-----\n")
    logger.info("\n\n-----\n")

    if args.wan:
        each_dataset_df = result_df.groupby("dataset_name").mean()
        for i in range(len(each_dataset_df)):
            wandb.log(
                {f"{each_dataset_df.index[i]}_dataset_auroc_all_seed": each_dataset_df.iloc[i]["instance_auroc"]})

        wandb.log(
            {"Average_auroc_all_seed": result_df.mean(axis=0)["instance_auroc"]})

    # NOTE each_dataset_df 에 Average 가 들어가지 않도록 뒤에서 넣음
    result_df.loc["Average"] = result_df.mean(axis=0)
    print(result_df)


def net(
    args
):

    backbone_names, layers_to_extract_from, pretrain_embed_dimension, target_embed_dimension, patchsize, meta_epochs, aed_meta_epochs, gan_epochs, noise_std, dsc_layers, dsc_hidden, dsc_margin, dsc_lr, auto_noise, train_backbone, cos_lr, pre_proj, proj_layer_type, mix_noise, onnx = args.backbone_names, args.layers_to_extract_from, args.pretrain_embed_dimension, args.target_embed_dimension, args.patchsize, args.meta_epochs, args.aed_meta_epochs, args.gan_epochs, args.noise_std, args.dsc_layers, args.dsc_hidden, args.dsc_margin, args.dsc_lr, args.auto_noise, args.train_backbone, args.cos_lr, args.pre_proj, args.proj_layer_type, args.mix_noise, args.onnx

    backbone_names = list(backbone_names)
    if len(backbone_names) > 1:
        layers_to_extract_from_coll = [[] for _ in range(len(backbone_names))]
        for layer in layers_to_extract_from:
            idx = int(layer.split(".")[0])
            layer = ".".join(layer.split(".")[1:])
            layers_to_extract_from_coll[idx].append(layer)
    else:
        layers_to_extract_from_coll = [layers_to_extract_from]

    def get_simplenet(input_shape, device):
        simplenets = []
        for backbone_name, layers_to_extract_from in zip(
            backbone_names, layers_to_extract_from_coll
        ):
            backbone_seed = None
            if ".seed-" in backbone_name:
                backbone_name, backbone_seed = backbone_name.split(".seed-")[0], int(
                    backbone_name.split("-")[-1]
                )
            backbone = backbones.load(backbone_name)
            backbone.name, backbone.seed = backbone_name, backbone_seed

            if args.mainmodel == "vig_score":
                simplenet_inst = Trainer_Ours_Score(device)
            elif args.mainmodel == "vig_score_patchlevel":
                simplenet_inst = Trainer_Ours_Score_PatchLevel(device)
            elif args.mainmodel == "vig_score_patchlevel_interpretable":
                simplenet_inst = Trainer_Ours_Score_PatchLevel_Interpretable(
                    device)
            elif args.mainmodel == "vig_score_patchlevel_mid_few":
                simplenet_inst = Trainer_Ours_Score_PatchLevel_Mid_Few(device)
            elif args.mainmodel == "vig_score_patchlevel_iv":
                simplenet_inst = Trainer_Ours_Score_PatchLevel_Iv(device)
            elif args.mainmodel == "vig_score_iv":
                simplenet_inst = Trainer_Ours_Score_Iv(device)
            elif args.mainmodel == "vig_score_att":
                simplenet_inst = Trainer_Ours_Score_Att(device)
            elif args.mainmodel == "vig_att":
                simplenet_inst = Trainer_Ours_Attention(device)
            elif args.mainmodel == "vig_att_score":
                simplenet_inst = Trainer_Ours_Attention_Score(device)
            elif args.mainmodel == "vig_gans_img":
                simplenet_inst = Trainer_Ours_GANs_IMG(device)
            elif args.mainmodel == "vig_nogans_img":
                simplenet_inst = Trainer_Ours_NoGANs_IMG(device)
            elif args.mainmodel == "vig_gans_img_coreset":
                simplenet_inst = Trainer_Ours_GANs_IMG_Coreset(device)
            elif args.mainmodel == "vig_nogans_img_coreset":
                simplenet_inst = Trainer_Ours_NoGANs_IMG_Coreset(device)
            elif args.mainmodel == "simple":
                simplenet_inst = Trainer_SimpleNet(device)
            elif args.mainmodel == "patchcore":
                simplenet_inst = Trainer_PatchCore(device)
            elif args.mainmodel == "patchcore_few":
                simplenet_inst = Trainer_PatchCore_Few(device)
            elif args.mainmodel == "patchcore_few_att":
                simplenet_inst = Trainer_PatchCore_Few_Att(device)
            elif args.mainmodel == "graphcore":
                simplenet_inst = Trainer_GraphCore(device)
            elif args.mainmodel == "interpretable_signet":
                simplenet_inst = Trainer_Interpretable_SIGNET(device)
            else:
                raise ValueError(f"Unknown mainmodel: {args.mainmodel}")

            simplenet_inst.load(
                backbone=backbone,
                layers_to_extract_from=layers_to_extract_from,
                device=device,
                input_shape=input_shape,
                pretrain_embed_dimension=pretrain_embed_dimension,
                target_embed_dimension=target_embed_dimension,
                patchsize=patchsize,
                meta_epochs=meta_epochs,
                aed_meta_epochs=aed_meta_epochs,
                gan_epochs=gan_epochs,
                noise_std=noise_std,
                dsc_layers=dsc_layers,
                dsc_hidden=dsc_hidden,
                dsc_margin=dsc_margin,
                dsc_lr=dsc_lr,
                auto_noise=auto_noise,
                train_backbone=train_backbone,
                cos_lr=cos_lr,
                pre_proj=pre_proj,
                proj_layer_type=proj_layer_type,
                mix_noise=mix_noise,
                onnx=onnx,
                args=args,
            )
            simplenets.append(simplenet_inst)
        return simplenets

    return ("get_simplenet", get_simplenet)


def dataset(
    args
):
    def _get_dataloaders(seed, subdataset):
        name, data_path, subdatasets, train_val_split, batch_size, resize, imagesize, num_workers, rotate_degrees, translate, scale, brightness, contrast, saturation, gray, hflip, vflip, augment = args.name, args.data_path, args.subdatasets, args.train_val_split, args.batch_size, args.resize, args.imagesize, args.num_workers, args.rotate_degrees, args.translate, args.scale, args.brightness, args.contrast, args.saturation, args.gray, args.hflip, args.vflip, args.augment
        # few-shot learning setting 인 경우 추가
        dataset_info = _DATASETS[name]
        dataset_library = __import__(
            dataset_info[0], fromlist=[dataset_info[1]])

        train_dataset = dataset_library.__dict__[dataset_info[1]](
            data_path,
            classname=subdataset,
            resize=resize,
            train_val_split=train_val_split,
            imagesize=imagesize,
            split=dataset_library.DatasetSplit.TRAIN,
            seed=seed,
            rotate_degrees=rotate_degrees,
            translate=translate,
            brightness_factor=brightness,
            contrast_factor=contrast,
            saturation_factor=saturation,
            gray_p=gray,
            h_flip_p=hflip,
            v_flip_p=vflip,
            scale=scale,
            augment=augment,
            args=args,
        )
        print_stat(train_dataset, f"{subdataset}_train")

        val_dataset = dataset_library.__dict__[dataset_info[1]](
            data_path,
            classname=subdataset,
            resize=resize,
            imagesize=imagesize,
            split=dataset_library.DatasetSplit.VAL,
            seed=seed,
            subtest=True,  # NOTE val data 는 class 당 몇 개만 사용함 (HPO 속도 때문에)
            args=args,
        )
        print_stat(val_dataset, f"{subdataset}_val")

        test_dataset = dataset_library.__dict__[dataset_info[1]](
            data_path,
            classname=subdataset,
            resize=resize,
            imagesize=imagesize,
            split=dataset_library.DatasetSplit.TEST,
            seed=seed,
            args=args,
        )
        print_stat(test_dataset, f"{subdataset}_test")

        logger.info(
            f"Dataset: train={len(train_dataset)} test={len(test_dataset)}")

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            prefetch_factor=2,
            pin_memory=True,
        )

        if val_dataset is not None and val_dataset.data_to_iterate:
            val_dataloader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=num_workers,
                prefetch_factor=2,
                pin_memory=True,
            )
        else:
            val_dataloader = None

        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            prefetch_factor=2,
            pin_memory=True,
        )

        train_dataloader.name = name
        if subdataset is not None:
            train_dataloader.name += "_" + "+".join(subdataset)

        dataloader_dict = {
            "training": train_dataloader,
            "validation": val_dataloader,
            "testing": test_dataloader,
        }
        return dataloader_dict

    def get_dataloaders(seed):
        dataloaders = []
        for subdataset in args.subdatasets:
            # NOTE 적어도 MVTEC 과 PCBNG 를 + 로 연결하지는 않는다는 전제가 있음
            # mvtec 이면 변환
            if subdataset.split("+")[0] in ["bottle", "cable", "capsule", "capsule_0.02", "carpet", "grid", "hazelnut", "leather", "metal_nut", "pill", "pill_0.02", "screw", "tile", "toothbrush", "transistor", "wood", "zipper"]:
                setattr(args, "name", "mvtec")
                setattr(args, "data_path", args.mvtec_data_path)
            elif subdataset.split("+")[0] in ["SeoulOKNG", "TRAIN_corner_0.01", "TRAIN_corner", 'SIDE_ESOC4_0.01', 'SIDE_ESOC4', 'PCBNG_0.01_M', 'PCBNG', 'PCBNG_0.1', 'PCBNG_0.01', 'Washing240523']:
                setattr(args, "name", "lg")
                setattr(args, "data_path", args.lg_data_path)
            else:
                raise NotImplementedError(
                    'You need to set proper dataset path')

            # NOTE + 로 연결된 dataset 은 병합하여 load 하고 아니라면 단일 dataset 을 load 함
            dataloaders.append(_get_dataloaders(seed, subdataset.split("+")))
        return dataloaders

    return ("get_dataloaders", get_dataloaders)


def process_args_dependency(args):
    # if args.true_false_edge == 0: # NOTE true_false_edge 가 0 이어도 reverse edge 사용이 필요한 경우가 생김
    #    setattr(args, "include_reverse", "no")
    #    logger.critical("true_false edge is disabled. include_reverse >> no")

    if args.check_intra_class_var == 1:
        setattr(args, "mainmodel", "simple")
        setattr(args, "wan", 0)
        setattr(args, "batch_size", 20)
        logger.critical(
            "check_intra_class_var is True. mainmodel >> simple, wan >> 0, batch_size >> 5")

    if args.mainmodel != 'vig_att':
        setattr(args, "lambda_t", -999)
        setattr(args, "lambda_f", -999)
        setattr(args, "lambda_tf", -999)
        setattr(args, "sample_len", -999)
        logger.critical(
            "mainmodel is not vig_att. lambda_t, lambda_f, lambda_tf, sample_len >> -999")

    if args.vig_backbone_pooling:
        setattr(args, "target_embed_dimension", 512)
        logger.critical(
            "vig_backbone_pooling is True. target_embed_dimension >> 512")

    if args.wan:
        setattr(args, "seed_mode", "wandb")
        logger.critical("wan is True. seed_mode >> wandb")

    if args.test_arg_path != "":
        # NOTE 즉, wan == True 이어도, test_arg_path 가 있으면 test 로 변경됨
        setattr(args, "seed_mode", "test")
        logger.critical("test_arg_path is not empty. seed_mode >> test")

    if args.seed_mode == "test":
        setattr(args, "seed_list", [0, 123, 321])
        setattr(args, "skip_ea", 1)
        logger.critical(
            "seed_mode is test. seed_list >> [0, 123, 321], skip_ea >> 1, YOU NEED TO SET PROPER META_EPOCHS AND GAN_EPOCHS")
    elif args.seed_mode == "wandb":
        setattr(args, "seed_list", [0, 123])
        logger.critical("seed_mode is wandb. seed_list >> [0, 123]")
    elif args.seed_mode == "temp":
        setattr(args, "seed_list", [0])
        logger.critical("seed_mode is temp. seed_list >> [0]")
    else:
        raise ValueError(f"Unknown seed_mode: {args.seed_mode}")

    if args.mainmodel == "patchcore":
        setattr(args, "patchsize", 3)
        setattr(args, "resize", 256)
        setattr(args, "imagesize", 224)
        logger.critical(
            "mainmodel is patchcore, patchsize >> 3, resize >> 256, imagesize >> 224")

    if args.mainmodel == "simple":
        setattr(args, "pretrain_embed_dimension", 1536)
        setattr(args, "target_embed_dimension", 1536)
        setattr(args, "patchsize", 3)
        setattr(args, "meta_epochs", 40)
        setattr(args, "gan_epochs", 4)
        setattr(args, "embedding_size", 256)
        setattr(args, "dsc_layers", 2)
        setattr(args, "dsc_hidden", 1024)
        setattr(args, "dsc_margin", 0.5)
        setattr(args, "dsc_lr", 0.0002)
        setattr(args, "lr", 0.00001)
        setattr(args, "mix_noise", 1)
        setattr(args, "noise_std", 0.015)
        setattr(args, "skip_ea", 1)
        # setattr(args, "resize", 329)
        # setattr(args, "imagesize", 288)
        logger.critical("mainmodel is simple, pretrained_embed_dimension >> 1536, target_embed_dimension >> 1536, patchsize >> 3, meta_epochs >> 40, gan_epochs >> 4, embedding_size >> 256, noise_std >> 0.015, dsc_layers >> 2, dsc_hidden >> 1024, dsc_margin >> 0.5, dsc_lr >> 0.0002, lr >> 0.00001, mix_noise >> 1, skip_ea >> 1")

    if args.mainmodel == "graphcore":
        setattr(args, "imagesize", 224)
        setattr(args, "layers_to_extract_from", [
                "backbone.0", "backbone.1", "backbone.2"])
        # , "backbone.3", "backbone.4", "backbone.5", "backbone.6", "backbone.7", "backbone.8", "backbone.9", "backbone.10", "backbone.11"])
        logger.critical(
            f"mainmodel is graphcore. imagesize >> {args.imagesize}, layers_to_extract_from >> {args.layers_to_extract_from}")

    return args


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Command line arguments: {}".format(" ".join(sys.argv)))

    # convert click into argparse
    parser = argparse.ArgumentParser()
    main_arg = parser.add_argument_group("main_arg")
    main_arg.add_argument("--results_path", type=str, default='lg_results')
    main_arg.add_argument("--gpu", type=int, default=[0], nargs="+")
    main_arg.add_argument("--seed_mode", type=str, default="temp",
                          choices=["temp", "wandb", "test"])
    main_arg.add_argument("--test_arg_path", default="")
    main_arg.add_argument("--log_group", type=str, default="")
    main_arg.add_argument("--log_project", type=str, default="project")
    main_arg.add_argument("--run_name", type=str, default="test")
    main_arg.add_argument("--save_segmentation_images", action="store_true")
    main_arg.add_argument("--save_patch_scores", action="store_true")

    dataset_arg = parser.add_argument_group("dataset_arg")
    dataset_arg.add_argument("--name", type=str,
                             default='mvtec')
    # default='mvtec')
    dataset_arg.add_argument("--mvtec_data_path", type=str,
                             default='../datasets/mvtec')
    dataset_arg.add_argument("--lg_data_path", type=str,
                             default='./config.json')
    # default='/home/robert.lim/datasets/mvtec')
    # NOTE USAGE EXAMPLE --subdatasets bottle cable
    # NOTE USAGE EXAMPLE --subdatasets bottle+cable cable 이면, 1번은 bottle과 cable 을 병합한 dataset 이 됨
    dataset_arg.add_argument("--subdatasets", nargs="+", default=[
        #    "capsule+pill", "PCBNG_0.01_M"])
        "capsule"])
    dataset_arg.add_argument("--train_val_split", type=float, default=0.8)
    dataset_arg.add_argument("--batch_size", type=int, default=1)
    dataset_arg.add_argument("--num_workers", type=int, default=2)
    dataset_arg.add_argument("--resize", type=int, default=256)
    dataset_arg.add_argument("--imagesize", type=int, default=224)
    dataset_arg.add_argument("--rotate_degrees", type=int, default=0)
    dataset_arg.add_argument("--translate", type=float, default=0)
    dataset_arg.add_argument("--scale", type=float, default=0.0)
    dataset_arg.add_argument("--brightness", type=float, default=0.0)
    dataset_arg.add_argument("--contrast", type=float, default=0.0)
    dataset_arg.add_argument("--saturation", type=float, default=0.0)
    dataset_arg.add_argument("--gray", type=float, default=0.0)
    dataset_arg.add_argument("--hflip", type=float, default=0.0)
    dataset_arg.add_argument("--vflip", type=float, default=0.0)
    dataset_arg.add_argument("--augment", action="store_true")

    net_arg = parser.add_argument_group("net_arg")
    net_arg.add_argument("--backbone_names", type=str,
                         nargs="+", default=["wideresnet50"])
    net_arg.add_argument("--layers_to_extract_from", type=str,
                         nargs="+", default=["layer2", "layer3"])
    net_arg.add_argument("--pretrain_embed_dimension", type=int, default=1536)
    net_arg.add_argument("--target_embed_dimension", type=int, default=1536)
    net_arg.add_argument("--dsc_layers", type=int, default=2)
    net_arg.add_argument("--dsc_hidden", type=int, default=1024)
    net_arg.add_argument("--dsc_margin", type=float, default=0.5)
    net_arg.add_argument("--dsc_lr", type=float, default=0.0002)
    net_arg.add_argument("--lr", type=float, default=0.0002)
    net_arg.add_argument("--auto_noise", type=float, default=0)
    net_arg.add_argument("--train_backbone", action="store_true")
    net_arg.add_argument("--cos_lr", action="store_true")
    net_arg.add_argument("--pre_proj", type=int, default=2)
    net_arg.add_argument("--proj_layer_type", type=int, default=0)
    net_arg.add_argument("--onnx", type=str, default="no")
    net_arg.add_argument("--mainmodel", type=str, default="vig_score_patchlevel")
    net_arg.add_argument("--aed_meta_epochs", type=int, default=1)
    net_arg.add_argument("--skip_ea", type=int, default=0)

    # few-shot
    net_arg.add_argument("--n_abnormal", type=int, default=5)
    net_arg.add_argument("--n_support", type=int, default=2)
    net_arg.add_argument("--few_iterations", type=int, default=500)

    # visualization
    net_arg.add_argument("--save_fault_images", type=int, default=0)

    # vig fixed
    net_arg.add_argument("--use_stochastic", type=bool, default=False)
    net_arg.add_argument("--n_classes", type=int, default=2)

    # gans
    net_arg.add_argument("--gan_beta", type=float, default=1.0)
    net_arg.add_argument("--gan_coreset_p", type=float, default=0.01)
    net_arg.add_argument("--gan_lr", type=float, default=0.0002)
    net_arg.add_argument("--ood_metric", type=str, default="odin")

    # invariant feature modeling
    net_arg.add_argument("--la_v", type=float, default=0.1)
    net_arg.add_argument("--intervention_times", type=int, default=50)
    net_arg.add_argument("--la_penalty", type=float, default=0.1)

    # bam
    # NOTE bam 은 너무 느려서 cbam 만 사용
    net_arg.add_argument("--bam_type", type=str, default="cbam_spatial")
    net_arg.add_argument("--bam_att_type", type=str, default="sigmoid")
    net_arg.add_argument("--bam_reduction_ratio", type=int, default=16)

    # vig
    net_arg.add_argument("--vig_backbone_pooling", type=int, default=0)
    net_arg.add_argument("--check_intra_class_var", type=int, default=0)
    net_arg.add_argument("--patchsize", type=int, default=3)
    net_arg.add_argument("--meta_epochs", type=int, default=200)
    net_arg.add_argument("--gan_epochs", type=int, default=4)
    net_arg.add_argument("--mix_noise", type=int, default=1)
    net_arg.add_argument("--noise_std", type=float, default=0.015)
    net_arg.add_argument("--k", type=int, default=9)
    net_arg.add_argument("--conv", type=str, default="mr",
                         choices=["edge", "mr", "sage", "gin"])
    net_arg.add_argument("--act", type=str, default="gelu",
                         choices=["relu", "prelu", "leakyrelu", "gelu", "hswish"])
    net_arg.add_argument("--norm", type=str, default="batch",
                         choices=["batch", "instance"])
    net_arg.add_argument("--bias", type=int, default=1)
    net_arg.add_argument("--n_blocks", type=int, default=1)
    net_arg.add_argument("--n_filters", type=int, default=256)
    net_arg.add_argument("--dropout", type=float, default=0.0)
    net_arg.add_argument("--use_dilation", type=int, default=0)
    net_arg.add_argument("--epsilon", type=float, default=0.2)
    net_arg.add_argument("--drop_path", type=float, default=0.0)
    net_arg.add_argument("--ea_patience", type=int, default=3)
    net_arg.add_argument("--ea_delta", type=float, default=0.01)
    net_arg.add_argument("--ea_warmup", type=int, default=40)
    net_arg.add_argument("--true_false_edge", type=int, default=0)
    net_arg.add_argument("--include_reverse", type=str,
                         default="no", choices=["no", "training_only", "always", "only_reverse"])
    net_arg.add_argument("--lambda_t", type=float, default=0)
    net_arg.add_argument("--lambda_f", type=float, default=0)
    net_arg.add_argument("--lambda_tf", type=float, default=0)
    net_arg.add_argument("--sample_ratio", type=float, default=0)

    net_arg.add_argument("--wan", action="store_false")

    args = parser.parse_args()

    # load args from json file
    if args.test_arg_path != "":
        with open(args.test_arg_path, "r") as f:
            test_arg_path = json.load(f)
        for key, value in test_arg_path.items():
            if key[0] != "_":  # NOTE wandb 기본 속성은 제외
                setattr(args, key, value['value'])

    args = process_args_dependency(args)

    if args.wan:
        wandb.init(project="ofcfelight", config=vars(args))

    print(args)

    methods = [dataset(args),  net(args)]
    run(methods, args)
