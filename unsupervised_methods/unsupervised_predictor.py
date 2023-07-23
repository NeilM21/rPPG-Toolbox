"""Unsupervised learning methods including POS, GREEN, CHROME, ICA, LGI and PBV."""

import logging
import os
from collections import OrderedDict

import torch
import wandb
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from evaluation.post_process import *
from unsupervised_methods.methods.CHROME_DEHAAN import *
from unsupervised_methods.methods.GREEN import *
from unsupervised_methods.methods.ICA_POH import *
from unsupervised_methods.methods.LGI import *
from unsupervised_methods.methods.PBV import *
from unsupervised_methods.methods.POS_WANG import *
from tqdm import tqdm
from metrics import df_highlight_outliers, visualize_ppg_pred_gt


def unsupervised_predict(config, data_loader, method_name, datetime_str=None, eval_table=None,
                         box_plot_grp=None, best_model_name="Unsupervised"):
    """ Model evaluation on the testing dataset."""
    if data_loader["unsupervised"] is None:
        raise ValueError("No data for unsupervised method predicting")
    print("===Unsupervised Method ( " + method_name + " ) Predicting ===")
    predict_hr_fft_all = list()
    gt_hr_fft_all = list()
    predict_hr_peak_all = list()
    gt_hr_peak_all = list()
    clip_names = list()
    clip_eval_df = pd.DataFrame(columns=["SubjectName", "MAE_Peak", "MAE_FFT", "RMSE_PEAK", "RMSE_FFT", "MAPE_PEAK", "MAPE_FFT"])
    clip_eval_df = clip_eval_df.round(decimals=4)

    dataset_name = config.UNSUPERVISED.DATA.DATASET

    eval_path = f"C://NIVS Project/NIVS Data/EvalCompare/Evaluation/{best_model_name}/{datetime_str}__{dataset_name}"
    eval_plots_save_path = f"{eval_path}/Plots"
    hilbert_eval_plots_save_path = f"{eval_path}/Hilbert"
    eval_hrs_save_path = f"{eval_path}/Data"

    eval_table = wandb.Table(columns=["type", "value", "algorithm"])

    sbar = tqdm(data_loader["unsupervised"], ncols=80)
    for _, test_batch in enumerate(sbar):
        batch_size = test_batch[0].shape[0]
        for idx in range(batch_size):
            clip_name = test_batch[2][idx]
            data_input, labels_input = test_batch[0][idx].cpu().numpy(), test_batch[1][idx].cpu().numpy()
            if method_name == "POS":
                BVP = POS_WANG(data_input, config.UNSUPERVISED.DATA.FS)
            elif method_name == "CHROM":
                BVP = CHROME_DEHAAN(data_input, config.UNSUPERVISED.DATA.FS)
            elif method_name == "ICA":
                BVP = ICA_POH(data_input, config.UNSUPERVISED.DATA.FS)
            elif method_name == "GREEN":
                BVP = GREEN(data_input)
            elif method_name == "LGI":
                BVP = LGI(data_input)
            elif method_name == "PBV":
                BVP = PBV(data_input)
            else:
                raise ValueError("unsupervised method name wrong!")

            visualize_ppg_pred_gt(gt_ppg=labels_input, pred_ppg=BVP, dataset_name=config.UNSUPERVISED.DATA.DATASET,
                                  subj_clip=clip_name, data_mode=config.UNSUPERVISED.DATA.PREPROCESS.LABEL_TYPE,
                                  ppg_save_path=eval_plots_save_path, ppg_fs=config.UNSUPERVISED.DATA.FS,
                                  generate_hilbert=True, hilbert_path=hilbert_eval_plots_save_path,
                                  method_name=method_name, unsupervised_mode=True)

            #if config.INFERENCE.EVALUATION_METHOD == "peak detection":
            if True:
                gt_hr_peak, pred_hr_peak = calculate_metric_per_video(BVP, labels_input, diff_flag=False,
                                                                fs=config.UNSUPERVISED.DATA.FS, hr_method='Peak')
                predict_hr_peak_all.append(pred_hr_peak)
                gt_hr_peak_all.append(gt_hr_peak)
            if True:
            #if config.INFERENCE.EVALUATION_METHOD == "FFT":
                gt_hr_fft, pred_hr_fft = calculate_metric_per_video(BVP, labels_input, diff_flag=False,
                                                                   fs=config.UNSUPERVISED.DATA.FS, hr_method='FFT')
                predict_hr_fft_all.append(pred_hr_fft)
                gt_hr_fft_all.append(gt_hr_fft)
                clip_names.append(clip_name)

            eval_metrics = [clip_name, np.mean(np.abs(pred_hr_peak - gt_hr_peak)), np.mean(np.abs(pred_hr_fft - gt_hr_fft)),
                            np.sqrt(np.mean(np.square(pred_hr_peak - gt_hr_peak))),
                            np.sqrt(np.mean(np.square(pred_hr_fft - gt_hr_fft))),
                            np.mean(np.abs((pred_hr_peak - gt_hr_peak) / gt_hr_peak)) * 100,
                            np.mean(np.abs((pred_hr_fft - gt_hr_fft) / gt_hr_fft)) * 100]
            eval_metrics[1:] = np.round(eval_metrics[1:], 4)

            clip_eval_df.loc[len(clip_eval_df)] = eval_metrics

    df_highlight_outliers(clip_eval_df)

    clip_names = np.array(clip_names)
    predict_hr_peak_all = np.array(predict_hr_peak_all).astype(float).round(4)
    predict_hr_fft_all = np.array(predict_hr_fft_all).astype(float).round(4)
    gt_hr_peak_all = np.array(gt_hr_peak_all).astype(float).round(4)
    gt_hr_fft_all = np.array(gt_hr_fft_all).astype(float).round(4)
    print("Used Unsupervised Method: " + method_name)

    if not os.path.exists(eval_hrs_save_path):
        os.makedirs(eval_hrs_save_path)

    # Generate box plot of model evaluation for heart rate estimates

    num_methods = len(config.UNSUPERVISED.METHOD)
    nrows = 2
    if num_methods % nrows != 0:
        ncols = (num_methods // nrows) + 1
    else:
        ncols = num_methods // nrows

    if box_plot_grp is None:
        box_plot_fig, box_plot_axs = plt.subplots(nrows, ncols)
        box_plot_fig.set_size_inches(18, 14)
        box_plot_fig.tight_layout(rect=[0, 0.03, 1, 0.9])
        box_plot_fig.suptitle(f"Box Plots for unsupervised methods on {dataset_name}: \n\n\n",
                              fontsize='xx-large', weight='extra bold')
    else:
        box_plot_fig, box_plot_axs = box_plot_grp

    #plt.show()
    metric_idx = config.UNSUPERVISED.METHOD.index(method_name)
    metric_row = metric_idx // ncols
    metric_col = metric_idx % ncols

    box_plot_axs[metric_row, metric_col].boxplot([clip_eval_df[x] for x in clip_eval_df.columns[1:]], labels=clip_eval_df.columns[1:])
    box_plot_axs[metric_row, metric_col].set_title(f"Evaluation Box Plot - {method_name} on {dataset_name}")

    box_plot_fig.savefig(f"{eval_path}/eval_stats_all.png", dpi=150)
    # plt.close(fig)

    # Output HR Estimation as a CSV file
    hr_estimates = np.array([clip_names, gt_hr_peak_all, predict_hr_peak_all, gt_hr_fft_all, predict_hr_fft_all]).transpose()
    hr_estimates_df = pd.DataFrame(columns=["Subject Name", "Gt_hr_peak", "Pred_hr_peak", "Gt_hr_fft", "Pred_hr_fft"],
                                   data=hr_estimates)

    wandb_metrics_dict = {}

    #if config.INFERENCE.EVALUATION_METHOD == "peak detection":
    if True:
        predict_hr_peak_all = np.array(predict_hr_peak_all)
        gt_hr_peak_all = np.array(gt_hr_peak_all)
        for metric in config.UNSUPERVISED.METRICS:
            if metric == "MAE":
                MAE_PEAK = np.mean(np.abs(predict_hr_peak_all - gt_hr_peak_all))
                print("Peak MAE (Peak Label):{0}".format(MAE_PEAK))
                eval_table.add_data("Peak", MAE_PEAK, method_name)
                wandb_metrics_dict[f"{method_name} MAE (Peak)"] = MAE_PEAK

            elif metric == "RMSE":
                RMSE_PEAK = np.sqrt(
                    np.mean(np.square(predict_hr_peak_all - gt_hr_peak_all)))
                print("PEAK RMSE (Peak Label):{0}".format(RMSE_PEAK))
                eval_table.add_data(f"Peak", RMSE_PEAK, method_name)
                wandb_metrics_dict[f"{method_name} RMSE (Peak)"] = RMSE_PEAK

            elif metric == "MAPE":
                MAPE_PEAK = np.mean(
                    np.abs((predict_hr_peak_all - gt_hr_peak_all) / gt_hr_peak_all)) * 100
                print("PEAK MAPE (Peak Label):{0}".format(MAPE_PEAK))
                eval_table.add_data("Peak", MAPE_PEAK, method_name)
                wandb_metrics_dict[f"{method_name} MAPE (Peak)"] = MAPE_PEAK

            elif metric == "Pearson":
                Pearson_PEAK = np.corrcoef(predict_hr_peak_all, gt_hr_peak_all)
                print("PEAK Pearson  (Peak Label):{0}".format(Pearson_PEAK[0][1]))
                eval_table.add_data("Peak", Pearson_PEAK[0][1], method_name)
                wandb_metrics_dict[f"{method_name} Pearson (Peak)"] = Pearson_PEAK[0][1]

            else:
                raise ValueError("Wrong Test Metric Type")

    # if config.INFERENCE.EVALUATION_METHOD == "FFT":
    if True:
        predict_hr_fft_all = np.array(predict_hr_fft_all)
        gt_hr_fft_all = np.array(gt_hr_fft_all)
        for metric in config.UNSUPERVISED.METRICS:
            if metric == "MAE":
                MAE_FFT = np.mean(np.abs(predict_hr_fft_all - gt_hr_fft_all))
                print("FFT MAE (FFT Label):{0}".format(MAE_FFT))
                eval_table.add_data("FFT", MAE_FFT, method_name)
                wandb_metrics_dict[f"{method_name} MAE (FFT)"] = MAE_FFT

            elif metric == "RMSE":
                RMSE_FFT = np.sqrt(
                    np.mean(np.square(predict_hr_fft_all - gt_hr_fft_all)))
                print("FFT RMSE (FFT Label):{0}".format(RMSE_FFT))
                eval_table.add_data("FFT", RMSE_FFT, method_name)
                wandb_metrics_dict[f"{method_name} RMSE (FFT)"] = RMSE_FFT

            elif metric == "MAPE":
                MAPE_FFT = np.mean(
                    np.abs((predict_hr_fft_all - gt_hr_fft_all) / gt_hr_fft_all)) * 100
                print("FFT MAPE (FFT Label):{0}".format(MAPE_FFT))
                eval_table.add_data("FFT", MAPE_FFT, method_name)
                wandb_metrics_dict[f"{method_name} MAPE (FFT)"] = MAPE_FFT

            elif metric == "Pearson":
                Pearson_FFT = np.corrcoef(predict_hr_fft_all, gt_hr_fft_all)
                print("FFT Pearson (FFT Label):{0}".format(Pearson_FFT[0][1]))
                eval_table.add_data("FFT", Pearson_FFT[0][1], method_name)
                wandb_metrics_dict[f"{method_name} Pearson (FFT)"] = Pearson_FFT[0][1]
            else:
                raise ValueError("Wrong Test Metric Type")

    # Group all method name metrics for plotting onto a single chart in wandb.ai
    #wandb.log(wandb_metrics_dict)
    # Optional TODO: Plot Table to Custom Charts
    #wandb.log({"multiline": wandb.plot_table(
    #            "wandb/line/v0", metrics_wandb_table_dict[metric], {"x": "algorithm", "y": "value", "groupKeys": "type"},
    #            {"title": f"{metric} test HR Estimate Evaluation"}})

    clip_eval_df.to_csv(f"{eval_hrs_save_path}/clip_eval_{dataset_name}_{method_name}.csv", index=False, sep=' ')

    clip_eval_df = clip_eval_df.style.apply(df_highlight_outliers, axis=None)
    clip_eval_df.to_excel(f"{eval_hrs_save_path}/clip_eval_{dataset_name}_{method_name}.xlsx", sheet_name='evaluation',
                          index=False)

    hr_estimates_df.to_csv(f"{eval_hrs_save_path}/clip_hrs_{dataset_name}_{method_name}.csv", index=False, sep=' ')

    return box_plot_fig, box_plot_axs