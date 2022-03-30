from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
from pathlib import Path


class MetricEvaluator:
    """
    Generic class which helps with evaluation.
    Can do:
    - run selected metrics on predictions vs ground truths
    """

    def __init__(self,
                 plot_directory=None,
                 prc_plot_name="prc_plot.png",
                 cor_plot_name="cor_plot.png",
                 selection_top_frac=0.1):

        self.prc_plot_name = prc_plot_name
        self.cor_plot_name = cor_plot_name
        self.selection_top_frac = selection_top_frac

        if plot_directory is None:
            self.plot_directory = \
                "/home/vit.ruzicka/branches/w4/change-detection"
        else:
            self.plot_directory = plot_directory

    def evaluate_pixel_based(self,
                             predicted_anomaly_maps,
                             true_anomaly_maps,
                             invalid_masks_maps):

        # ignore masked pixels and flatten
        predicted_anomaly_maps_masked = \
            predicted_anomaly_maps[~invalid_masks_maps]  # N
        true_anomaly_maps_masked = true_anomaly_maps[~invalid_masks_maps]  # N

        precision, recall, thresholds = \
            precision_recall_curve(true_anomaly_maps_masked,
                                   predicted_anomaly_maps_masked)

        area_under_precision_curve = auc(recall, precision)
        precision_at_100_recall = precision[0]
        efficiency_over_manual_vetting = \
            precision_at_100_recall / true_anomaly_maps_masked.mean()

        plot = self.plot_precision_recall_curve(precision, recall)

        return area_under_precision_curve, precision_at_100_recall, \
            efficiency_over_manual_vetting, plot

    def evaluate_window_based(self,
                              predicted_anomaly_scores,
                              true_anomaly_fractions):

        # We want monotone ordering relationship between the predictions and
        # true values ~ use spearman correlation
        spearman_cor, p = stats.spearmanr(predicted_anomaly_scores,
                                          true_anomaly_fractions,
                                          nan_policy='raise')

        # If we select top k=100 windows to downlink, how many true anomaly
        # pixels would we get? Higher is better. Doesn't provide info on how
        # many useless pixels we send down though.
        sorted_true_score = \
            [x[1] for x in sorted(zip(predicted_anomaly_scores, true_anomaly_fractions), reverse=True)]
        selected_samples = \
            int(self.selection_top_frac * len(predicted_anomaly_scores))
        selected_top_frac_score = np.sum(sorted_true_score[0:selected_samples])

        plot = self.plot_correlation(predicted_anomaly_scores,
                                     true_anomaly_fractions, spearman_cor)

        return spearman_cor, selected_top_frac_score, plot

    def plot_precision_recall_curve(self, precision, recall):
        fig = plt.figure()
        plt.plot(recall, precision)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        return fig

    def plot_correlation(self,
                         predicted_anomaly_scores,
                         true_anomaly_fractions,
                         spearman_cor):

        fig = plt.figure()
        plt.scatter(true_anomaly_fractions, predicted_anomaly_scores)
        plt.xlabel('True fraction of anomalous pixels')
        plt.ylabel('Anomaly score')
        plt.suptitle("Spearman cor = "+str(spearman_cor))
        return fig

