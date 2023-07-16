import numpy as np
import pandas as pd
import torch, os
from evaluation.post_process import *
from matplotlib import pyplot as plt
from scipy.signal import hilbert
import datetime


def read_label(dataset):
    """Read manually corrected labels."""
    df = pd.read_csv("label/{0}_Comparison.csv".format(dataset))
    out_dict = df.to_dict(orient='index')
    out_dict = {str(value['VideoID']): value for key, value in out_dict.items()}
    return out_dict


def read_hr_label(feed_dict, index):
    """Read manually corrected UBFC labels."""
    # For UBFC only
    if index[:7] == 'subject':
        index = index[7:]
    video_dict = feed_dict[index]
    if video_dict['Preferred'] == 'Peak Detection':
        hr = video_dict['Peak Detection']
    elif video_dict['Preferred'] == 'FFT':
        hr = video_dict['FFT']
    else:
        hr = video_dict['Peak Detection']
    return index, hr


def _reform_data_from_dict(data):
    """Helper func for calculate metrics: reformat predictions and labels from dicts. """
    sort_data = sorted(data.items(), key=lambda x: x[0])
    sort_data = [i[1] for i in sort_data]
    sort_data = torch.cat(sort_data, dim=0)
    return np.reshape(sort_data.cpu(), (-1))


def calculate_metrics(predictions, labels, config, best_model_name=""):
    """Calculate rPPG Metrics (MAE, RMSE, MAPE, Pearson Coef.)."""
    predict_hr_fft_all = list()
    gt_hr_fft_all = list()
    predict_hr_peak_all = list()
    gt_hr_peak_all = list()
    clip_names = list()
    clip_eval_df = pd.DataFrame(columns=["SubjectName", "MAE_Peak", "MAE_FFT", "RMSE_PEAK", "RMSE_FFT", "MAPE_PEAK", "MAPE_FFT"])
    clip_eval_df = clip_eval_df.round(decimals=4)

    datetime_str = '{date:%Y-%m-%d__%H-%M-%S}'.format(date=datetime.datetime.now())
    dataset_name = config.TEST.DATA.DATASET

    eval_path = f"C://NIVS Project/NIVS Data/EvalCompare/Evaluation/{best_model_name}/{datetime_str}__{dataset_name}"
    eval_plots_save_path = f"{eval_path}/Plots"
    hilbert_eval_plots_save_path = f"{eval_path}/Hilbert"
    eval_hrs_save_path = f"{eval_path}/Data"

    if not os.path.exists(eval_plots_save_path):
        os.makedirs(eval_plots_save_path)

    for index in predictions.keys():
        prediction = _reform_data_from_dict(predictions[index])
        label = _reform_data_from_dict(labels[index])

        # [NEIL] Visualize PPG Signals
        visualize_ppg_pred_gt(gt_ppg=label, pred_ppg=prediction, dataset_name=config.TEST.DATA.DATASET,
                              subj_clip=index, data_mode=config.TEST.DATA.PREPROCESS.LABEL_TYPE,
                              ppg_save_path=eval_plots_save_path, ppg_fs=config.TEST.DATA.FS, generate_hilbert=True,
                              hilbert_path=hilbert_eval_plots_save_path)

        if config.TEST.DATA.PREPROCESS.LABEL_TYPE == "Standardized" or \
                config.TEST.DATA.PREPROCESS.LABEL_TYPE == "Raw":
            diff_flag_test = False
        elif config.TEST.DATA.PREPROCESS.LABEL_TYPE == "DiffNormalized":
            diff_flag_test = True
        else:
            raise ValueError("Not supported label type in testing!")
        gt_hr_fft, pred_hr_fft = calculate_metric_per_video(
            prediction, label, diff_flag=diff_flag_test, fs=config.TEST.DATA.FS, hr_method='FFT')
        gt_hr_peak, pred_hr_peak = calculate_metric_per_video(
            prediction, label, diff_flag=diff_flag_test, fs=config.TEST.DATA.FS, hr_method='Peak')
        gt_hr_fft_all.append(gt_hr_fft)
        predict_hr_fft_all.append(pred_hr_fft)
        predict_hr_peak_all.append(pred_hr_peak)
        gt_hr_peak_all.append(gt_hr_peak)
        clip_names.append(index)
        # Append individual evaluation criteria
        eval_metrics = [index, np.mean(np.abs(pred_hr_peak - gt_hr_peak)), np.mean(np.abs(pred_hr_fft - gt_hr_fft)),
         np.sqrt(np.mean(np.square(pred_hr_peak - gt_hr_peak))), np.sqrt(np.mean(np.square(pred_hr_fft - gt_hr_fft))),
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

    if not os.path.exists(eval_hrs_save_path):
        os.makedirs(eval_hrs_save_path)

    # Generate box plot of model evaluation for heart rate estimates
    plt.boxplot([clip_eval_df[x] for x in clip_eval_df.columns[1:]], labels=clip_eval_df.columns[1:])
    plt.title(f"Evaluation Box Plot - {best_model_name} on {dataset_name}")
    plt.tight_layout()
    fig = plt.gcf()
    fig.set_size_inches(12, 6)
    plt.savefig(f"{eval_path}/eval_stats.png", dpi=150)

    # Output HR Estimation as a CSV file
    hr_estimates = np.array([clip_names, gt_hr_peak_all, predict_hr_peak_all, gt_hr_fft_all, predict_hr_fft_all]).transpose()
    hr_estimates_df = pd.DataFrame(columns=["Subject Name", "Gt_hr_peak", "Pred_hr_peak", "Gt_hr_fft", "Pred_hr_fft"],
                                   data=hr_estimates)

    for metric in config.TEST.METRICS:
        if metric == "MAE":
            #if config.INFERENCE.EVALUATION_METHOD == "FFT":
            MAE_FFT = np.mean(np.abs(predict_hr_fft_all - gt_hr_fft_all))
            print("FFT MAE (FFT Label):{0}".format(MAE_FFT))
            #elif config.INFERENCE.EVALUATION_METHOD == "peak detection":
            MAE_PEAK = np.mean(np.abs(predict_hr_peak_all - gt_hr_peak_all))
            print("Peak MAE (Peak Label):{0}".format(MAE_PEAK))
            #else:
            #    raise ValueError("Your evaluation method is not supported yet! Support FFT and peak detection now ")

        elif metric == "RMSE":
            #if config.INFERENCE.EVALUATION_METHOD == "FFT":
            RMSE_FFT = np.sqrt(np.mean(np.square(predict_hr_fft_all - gt_hr_fft_all)))
            print("FFT RMSE (FFT Label):{0}".format(RMSE_FFT))
            #elif config.INFERENCE.EVALUATION_METHOD == "peak detection":
            RMSE_PEAK = np.sqrt(np.mean(np.square(predict_hr_peak_all - gt_hr_peak_all)))
            print("PEAK RMSE (Peak Label):{0}".format(RMSE_PEAK))
            #else:
            #    raise ValueError("Your evaluation method is not supported yet! Support FFT and peak detection now ")

        elif metric == "MAPE":
            #if config.INFERENCE.EVALUATION_METHOD == "FFT":
            MAPE_FFT = np.mean(np.abs((predict_hr_fft_all - gt_hr_fft_all) / gt_hr_fft_all)) * 100
            print("FFT MAPE (FFT Label):{0}".format(MAPE_FFT))
            #elif config.INFERENCE.EVALUATION_METHOD == "peak detection":
            MAPE_PEAK = np.mean(np.abs((predict_hr_peak_all - gt_hr_peak_all) / gt_hr_peak_all)) * 100
            print("PEAK MAPE (Peak Label):{0}".format(MAPE_PEAK))
            #else:
            #    raise ValueError("Your evaluation method is not supported yet! Support FFT and peak detection now ")

        elif metric == "Pearson":
            #if config.INFERENCE.EVALUATION_METHOD == "FFT":
            Pearson_FFT = np.corrcoef(predict_hr_fft_all, gt_hr_fft_all)
            print("FFT Pearson (FFT Label):{0}".format(Pearson_FFT[0][1]))
           #elif config.INFERENCE.EVALUATION_METHOD == "peak detection":
            Pearson_PEAK = np.corrcoef(predict_hr_peak_all, gt_hr_peak_all)
            print("PEAK Pearson  (Peak Label):{0}".format(Pearson_PEAK[0][1]))
            #else:
            #    raise ValueError("Your evaluation method is not supported yet! Support FFT and peak detection now ")

        else:
            raise ValueError("Wrong Test Metric Type")

    clip_eval_df.to_csv(f"{eval_hrs_save_path}/clip_eval_{dataset_name}.csv", index=False, sep=' ')
    
    clip_eval_df = clip_eval_df.style.apply(df_highlight_outliers, axis=None)
    clip_eval_df.to_excel(f"{eval_hrs_save_path}/clip_eval_{dataset_name}.xlsx", sheet_name='evaluation', index=False)

    hr_estimates_df.to_csv(f"{eval_hrs_save_path}/clip_hrs_{dataset_name}.csv", index=False, sep=' ')


def visualize_ppg_pred_gt(gt_ppg, pred_ppg, dataset_name, subj_clip, data_mode, ppg_save_path, ppg_fs,
                          generate_hilbert=False, hilbert_path=None):

    gt_ppg = gt_ppg.numpy()
    pred_ppg = pred_ppg.numpy()

    # New Viz
    fig_eval, axs_eval = plt.subplot_mosaic([["GT_Pred_PPG_Overlay", "GT_Pred_PPG_Overlay"],
                                                   ["GT_PPG", "Pred_PPG"],
                                                   ], layout='constrained')
    fig_eval.set_size_inches(12, 9)

    axs_eval["GT_Pred_PPG_Overlay"].plot(gt_ppg, label='gt', color="tab:orange")
    axs_eval["GT_Pred_PPG_Overlay"].plot(pred_ppg, label='pred', color="tab:blue")
    axs_eval["GT_Pred_PPG_Overlay"].legend(loc='best')
    axs_eval["GT_Pred_PPG_Overlay"].set_title(f'{subj_clip} PPG Overlay - Ground Truth vs. Prediction')

    axs_eval["GT_PPG"].plot(gt_ppg, label='gt', color="tab:orange")
    axs_eval["GT_PPG"].set_title('Ground Truth WaveForm')

    axs_eval["Pred_PPG"].plot(pred_ppg, label='pred', color="tab:blue")
    axs_eval["Pred_PPG"].set_title('Prediction WaveForm')

    # Time axis for segmented plots (samples - no time axis from rPPG-Toolbox evaluation pipeline)
    seg_length = 20 * ppg_fs
    plot_len = len(gt_ppg)

    if plot_len % seg_length != 0:
        video_segments = (plot_len // seg_length) + 1
    else:
        video_segments = plot_len // seg_length

    # If max segments is larger than this value, restrict number of plots generated to this value
    max_sample_plots = 10

    num_plots = int(video_segments) if video_segments < max_sample_plots else max_sample_plots

    for i in range(num_plots):
        interval_start = np.round((i * seg_length), 3)
        interval_end = np.round(((i+1) * seg_length), 3)
        for _, ax in axs_eval.items():
            ax.set_xlim(interval_start, interval_end)
            ax.set_xlabel("Samples")
            plt.savefig(f"{ppg_save_path}/{dataset_name}_{subj_clip}_{i}_{data_mode}.png", dpi=200)

    plt.close(fig_eval)

    # Hilbert Transform Visualization
    if generate_hilbert:
        gt_analytic_signal = hilbert(gt_ppg)
        gt_amp = np.abs(gt_analytic_signal)
        gt_phase = np.unwrap(np.angle(gt_analytic_signal))
        gt_frequency = (np.diff(gt_phase) / (2.0 * np.pi) * ppg_fs)

        pred_analytic_signal = hilbert(pred_ppg)
        pred_amp = np.abs(pred_analytic_signal)
        pred_phase = np.unwrap(np.angle(pred_analytic_signal))
        pred_frequency = (np.diff(pred_phase) / (2.0 * np.pi) * ppg_fs)

        fig_hilbert, axs_hilbert = plt.subplot_mosaic([["GT_Envelope_Overlay"],
                                       ["Pred_Envelope_Overlay"],
                                       ["GT_Instantaneous_Frequency"],
                                       ["Pred_Instantaneous_Frequency"]
                                       ], layout='constrained')
        fig_hilbert.set_size_inches(12, 9)

        axs_hilbert["GT_Envelope_Overlay"].plot(gt_ppg, label='signal')
        axs_hilbert["GT_Envelope_Overlay"].plot(gt_amp, label='amp envelope', color='tab:red')
        axs_hilbert["GT_Envelope_Overlay"].legend()
        axs_hilbert["GT_Envelope_Overlay"].set_title(f"Ground Truth Signal with Amplitude Envelope Overlay")

        axs_hilbert["Pred_Envelope_Overlay"].plot(pred_ppg, label='signal')
        axs_hilbert["Pred_Envelope_Overlay"].plot(pred_amp, label='amp envelope', color='tab:red')
        axs_hilbert["Pred_Envelope_Overlay"].legend()
        axs_hilbert["Pred_Envelope_Overlay"].set_title(f"Prediction Signal with Amplitude Envelope Overlay")

        axs_hilbert["GT_Instantaneous_Frequency"].plot(np.real(gt_frequency))
        axs_hilbert["GT_Instantaneous_Frequency"].set_title(f"Ground Truth Instantaneous Frequency")

        axs_hilbert["Pred_Instantaneous_Frequency"].plot(np.real(pred_frequency))
        axs_hilbert["Pred_Instantaneous_Frequency"].set_title(f"Prediction Instantaneous Frequency")

        # plt.show()

        if not os.path.exists(hilbert_path):
            os.makedirs(hilbert_path, exist_ok=True)

        # Visualize and save Hilbert Plots
        for i in range(num_plots):
            interval_start = np.round((i * seg_length), 3)
            interval_end = np.round(((i + 1) * seg_length), 3)
            # Snip the main axis into ten 20-second segments for visual clarity of comparisons
            for _, ax in axs_hilbert.items():
                ax.set_xlim([interval_start, interval_end])
                ax.set_xlabel("samples")
            plt.suptitle(f"\nHilbert Transform for {dataset_name}-{subj_clip}: segment {i}\n",
                         fontsize='xx-large', weight='extra bold')
            plt.savefig(f"{hilbert_path}/{dataset_name}_{subj_clip}_{i}_{data_mode}.png", dpi=200)

        plt.close(fig_hilbert)


def df_highlight_outliers(df):
    df_copy = df.copy()
    q1 = df_copy.quantile(0.25)
    q3 = df_copy.quantile(0.75)
    iqr = q3-q1

    # Get Box Plot Outliers (lower, upper)
    lower_bounds = q1 - (1.5*iqr)
    upper_bounds = q3 + (1.5*iqr)
    df1 = pd.DataFrame('', index=df_copy.index, columns=df_copy.columns)

    c1 = 'background-color: yellow'
    numeric_cols = df_copy.columns[1:]

    # Highlight outliers in df
    for colname in numeric_cols:
        df1.loc[df[colname] < lower_bounds[colname], colname] = c1
        df1.loc[df[colname] > upper_bounds[colname], colname] = c1

    return df1
