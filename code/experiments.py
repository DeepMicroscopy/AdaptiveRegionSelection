import argparse
from AL_select import select_regions
from FROC import FROC_analysis
from inference import predict
from result_analysis import calculate_IoU, calculate_AUC
from train import train
from user_define import experiment_config, Log
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device = {device}")

if __name__ == '__main__':
    cf = experiment_config()
    parser = argparse.ArgumentParser(description='Camelyon16 AL')
    parser.add_argument('--exp_id', default=cf.exp_id, type=int)

    parser.add_argument('--label_dict', default=cf.label_dict, type=json.loads, help="{'Normal': 0, 'Tumor': 1}")
    parser.add_argument('--label_dict_inv', default=cf.label_dict_inv, type=json.loads, help="{0: 'Normal', 1 :'Tumor'}")
    parser.add_argument('--classes', default=cf.classes, type=list)

    parser.add_argument('--patch_size', default=cf.patch_size, type=int)
    parser.add_argument('--max_lr', default=cf.max_lr, type=float, help='max learning rate for one-cycle training')
    parser.add_argument('--level', default=cf.level, type=int, help='magnification level')
    parser.add_argument('--train_batch_size', default=cf.train_batch_size, type=int)
    parser.add_argument('--inference_batch_size', default=cf.inference_batch_size, type=int)
    parser.add_argument('--epochs', default=cf.epochs, type=int)
    parser.add_argument('--num_workers', default=cf.num_workers, type=int)
    parser.add_argument('--pin_memory', default=cf.pin_memory, type=bool)

    parser.add_argument('--dataset_path', default=cf.dataset_path, type=str)
    parser.add_argument('--train_list', default=cf.train_list, type=list)
    parser.add_argument('--valid_list', default=cf.valid_list, type=list)
    parser.add_argument('--test_list', default=cf.test_list, type=list)
    parser.add_argument('--tumor_test_list', default=cf.tumor_test_list, type=list)
    parser.add_argument('--macro_tumor_test_list', default=cf.macro_tumor_test_list, type=list)
    parser.add_argument('--micro_tumor_test_list', default=cf.micro_tumor_test_list, type=list)
    parser.add_argument('--normal_test_list', default=cf.normal_test_list, type=list)

    parser.add_argument('--stain_vector_pool', default=cf.stain_vector_pool, type=list)

    parser.add_argument('--region_size', default=cf.region_size, type=int, help='size of selected regions')
    parser.add_argument('--n_query', default=cf.n_query, type=int, help='number of regions selected per AL cycle')
    parser.add_argument('--sampling_strategy', default=cf.sampling_strategy, type=str,
                        help='full/random/uncertain_standard/uncertain_non_square/uncertain_adapt')
    parser.add_argument('--initial_sampling_strategy', default=cf.initial_sampling_strategy, type=str,
                        help='sampling strategy for the initial labeled set')

    parser.add_argument('--log_file', default=cf.log_file, type=str, help='path to log file')
    parser.add_argument('--res_path', default=cf.res_path, type=str, help='path to experimental results')

    args = parser.parse_args()
    os.makedirs(args.res_path, exist_ok=True)

    SELECT_LEVEL = 7
    PREDICT_LEVEL = 8
    # use only 1/5 of WSIs at each AL cycle to accelerate selection, as slide inference for generating the selection
    # priority map is time-consuming
    N_split = 5

    if args.sampling_strategy == 'full':
        CYCLES = 1
        train_idx_list = np.array([0, len(args.train_list)])
        valid_idx_list = np.array([0, len(args.valid_list)])
    else:
        CYCLES = 10
        train_idx_list = np.array([0] + [len(a) for a in np.array_split(np.arange(len(args.train_list)), N_split)])
        valid_idx_list = np.array([0] + [len(a) for a in np.array_split(np.arange(len(args.valid_list)), N_split)])

    print(f"train_idx_list = {train_idx_list}")
    print(f"valid_idx_list = {valid_idx_list}")

    for cycle in range(1, CYCLES+1):
        args.exp = f"cycle_{cycle}_{args.res_path.split('/')[-1]}"
        Log(f"#==============EXPERIMENT {args.exp} ======================", args.log_file)

        # file preparation
        learner_path = os.path.join(args.exp, args.exp + ".pkl")  # for storing and loading the model
        # valid_hdf5 = os.path.join(args.exp, args.exp + '_inference_valid.hdf5')  # for calibration
        test_hdf5 = os.path.join(args.exp, args.exp + '_inference_test.hdf5')
        select_hdf5 = None  # storing predictions of training and validation slides for selection
        AL_annotations = os.path.join(args.exp, args.exp + '_select.json')
        os.makedirs(os.path.join(args.exp), exist_ok=True)
        os.makedirs(os.path.join(args.exp, 'plots'), exist_ok=True)
        os.makedirs(os.path.join(args.exp, 'inference'), exist_ok=True)
        os.makedirs(os.path.join(args.exp, 'inference', 'test_macro_tumor'), exist_ok=True)
        os.makedirs(os.path.join(args.exp, 'inference', 'test_micro_tumor'), exist_ok=True)
        os.makedirs(os.path.join(args.exp, 'inference', 'test_normal'), exist_ok=True)
        os.makedirs(os.path.join(args.exp, 'inference', 'train'), exist_ok=True)
        os.makedirs(os.path.join(args.exp, 'inference', 'valid'), exist_ok=True)
        os.makedirs(os.path.join(args.exp, 'selected'), exist_ok=True)
        os.makedirs(os.path.join(args.exp, 'selected', 'train'), exist_ok=True)
        os.makedirs(os.path.join(args.exp, 'selected', 'valid'), exist_ok=True)

        # dataset preparation
        idx = cycle % N_split
        if idx == 0: idx = N_split
        train_idx = np.arange(np.sum(train_idx_list[:idx]), np.sum(train_idx_list[:(idx + 1)]))
        valid_idx = np.arange(np.sum(valid_idx_list[:idx]), np.sum(valid_idx_list[:(idx + 1)]))
        Log(f"training slides: \n{np.array(args.train_list)[train_idx]}", args.log_file)
        Log(f"validation slides: \n{np.array(args.valid_list)[valid_idx]}", args.log_file)
        Log(f"test slides (macro): \n{np.array(args.macro_tumor_test_list)}", args.log_file)
        Log(f"test slides (micro): \n{np.array(args.micro_tumor_test_list)}", args.log_file)

        mode = ['select', 'anno_cost_calculation', 'training', 'inference', 'result_analysis']

        for m in mode:
            Log(f"-----------------------------------------------------------------------{m}", args.log_file)
            if m == 'select':
                # prepare the annotation file by copying from last AL cycle
                if cycle == 1:
                    last_exp = None
                    last_AL_annotations = None
                else:
                    last_exp = args.exp.replace(f"cycle_{cycle}", f"cycle_{cycle - 1}")
                    if cycle == 2 and args.initial_sampling_strategy == "random":
                        last_exp = last_exp.replace(args.sampling_strategy, "random")
                    last_AL_annotations = os.path.join(last_exp, last_exp + '_select.json')

                prepare_AL_annotations(AL_annotations, last_AL_annotations)

                # predict slides for generating the selection priority maps
                if args.sampling_strategy.__contains__("uncertainty"):
                    last_learner_path = os.path.join(last_exp, last_exp + ".pkl")
                    select_hdf5 = os.path.join(last_exp, last_exp + '_inference_select.hdf5')

                    predict(learner_path=last_learner_path, slide_list=np.array(args.train_list)[train_idx],
                            output_hdf5_filename=select_hdf5, dataset_path=args.dataset_path, slide_level=args.level,
                            patch_size=args.patch_size, predict_level=PREDICT_LEVEL, class_dict=args.label_dict_inv,
                            batch_size=args.inference_batch_size, num_workers=args.num_workers,
                            pin_memory=args.pin_memory, device=device,
                            save_path=os.path.join(args.exp, 'inference', 'train'))

                    predict(learner_path=last_learner_path, slide_list=np.array(args.valid_list)[valid_idx],
                            output_hdf5_filename=select_hdf5, dataset_path=args.dataset_path, slide_level=args.level,
                            patch_size=args.patch_size, predict_level=PREDICT_LEVEL, class_dict=args.label_dict_inv,
                            batch_size=args.inference_batch_size, num_workers=args.num_workers,
                            pin_memory=args.pin_memory, device=device,
                            save_path=os.path.join(args.exp, 'inference', 'valid'))

                # select regions for training slides
                select_regions(sampling_strategy=args.sampling_strategy, dataset_path=args.dataset_path,
                               slide_name_list=np.array(args.train_list)[train_idx], predict_level=PREDICT_LEVEL,
                               select_level=SELECT_LEVEL, region_size=args.region_size, n_query=args.n_query,
                               pred_h5py_file=select_hdf5, AL_annotations=AL_annotations, set_name="train",
                               save_path=os.path.join(args.exp, "selected", "train"))

                # select regions for validation slides
                select_regions(sampling_strategy=args.sampling_strategy, dataset_path=args.dataset_path,
                               slide_name_list=np.array(args.valid_list)[valid_idx], predict_level=PREDICT_LEVEL,
                               select_level=SELECT_LEVEL, region_size=args.region_size, n_query=args.n_query,
                               pred_h5py_file=select_hdf5, AL_annotations=AL_annotations, set_name="valid",
                               save_path=os.path.join(args.exp, "selected", "valid"))

            if m == 'anno_cost_calculation':
                annotated_area, clicks, cost_a, tumor_ratio = annotation_cost(AL_annotations)
                anno_n_regions = annotated_area / ((args.region_size * (2 ** SELECT_LEVEL)) ** 2)
                Log(f"annotated {anno_n_regions:.1f} * ({args.region_size}x{args.region_size}) regions "
                    f"at level {SELECT_LEVEL}", args.log_file)
                Log("[int(c_p), int(c_i), int(c_b), int(c_c), int(anno_pixels)]", args.log_file)
                Log(str(clicks), args.log_file)
                Log(f"cost_a = {cost_a:.2f}%, annotated tumor ratio={tumor_ratio:.2f}%", args.log_file)

            if m == 'training':
                train(dataset_path=args.dataset_path, slide_name_list=args.train_list + args.valid_list,
                      level=args.level, patch_size=args.patch_size, AL_annotations=AL_annotations,
                      stain_aug_p=args.stain_aug_p, stain_vector_pool=args.stain_vector_pool,
                      class_dict=args.label_dict_inv, batch_size=args.train_batch_size,
                      num_workers=args.num_workers, pin_memory=args.pin_memory, device=device,
                      max_lr=args.max_lr, epochs=args.epochs, save_path=args.exp)

            if m == 'inference':
                predict(learner_path=learner_path, slide_list=args.macro_tumor_test_list,
                        output_hdf5_filename=test_hdf5, dataset_path=args.dataset_path, slide_level=args.level,
                        patch_size=args.patch_size, predict_level=PREDICT_LEVEL, class_dict=args.label_dict_inv,
                        batch_size=args.inference_batch_size, num_workers=args.num_workers, pin_memory=args.pin_memory,
                        device=device, save_path=os.path.join(args.exp, 'inference', 'test_macro_tumor'))

                predict(learner_path=learner_path, slide_list=args.micro_tumor_test_list,
                        output_hdf5_filename=test_hdf5, dataset_path=args.dataset_path, slide_level=args.level,
                        patch_size=args.patch_size, predict_level=PREDICT_LEVEL, class_dict=args.label_dict_inv,
                        batch_size=args.inference_batch_size, num_workers=args.num_workers, pin_memory=args.pin_memory,
                        device=device, save_path=os.path.join(args.exp, 'inference', 'test_micro_tumor'))

                # predict(learner_path=learner_path, slide_list=args.normal_test_list,
                #         output_hdf5_filename=test_hdf5, dataset_path=args.dataset_path, slide_level=args.level,
                #         patch_size=args.patch_size, predict_level=PREDICT_LEVEL, class_dict=args.label_dict_inv,
                #         batch_size=args.inference_batch_size, num_workers=args.num_workers, pin_memory=args.pin_memory,
                #         device=device, save_path=os.path.join(args.exp, 'inference', 'test_normal'))

            if m == 'result_analysis':

                IoU = calculate_IoU(dataset_path=args.dataset_path, slide_list=args.tumor_test_list,
                                    class_dict=args.label_dict_inv, test_hdf5_file=test_hdf5, p=0.5,
                                    log_file=args.log_file, predict_level=PREDICT_LEVEL)

                # AUC = calculate_AUC(dataset_path=args.dataset_path, slide_list=args.test_list,
                #                     class_dict=args.label_dict_inv, test_hdf5_file=test_hdf5, p=0.5,
                #                     log_file=args.log_file, predict_level=PREDICT_LEVEL)
                #
                # FROC_score = FROC_analysis(dataset_path=args.dataset_path, test_hdf5_file=test_hdf5,
                #                            predict_level=PREDICT_LEVEL, p=0.9, pooling='sum',
                #                            save_path=args.exp, log_file=args.log_file)
                # FROC_score = FROC_analysis(dataset_path=args.dataset_path, test_hdf5_file=test_hdf5,
                #                            predict_level=PREDICT_LEVEL, p=0.9, pooling='max',
                #                            save_path=args.exp, log_file=args.log_file)


