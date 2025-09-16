# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict
import logging

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.stats import beta
from sklearn.metrics import auc, roc_curve


logger: logging.Logger = logging.getLogger(__name__)


class MIAResults:
    """
    Class for implenting Membership Inference (MIA) Attack analysis results

    Args:
        scores_train: 1 dimensional tensor of train values
        scores_test: 1 dimensional tensor of test values
        Length of tensors is 1xN, where N is specified in the attack config,
        and corresponds to n_users_for_eval
    """

    def __init__(self, scores_train: torch.Tensor, scores_test: torch.Tensor) -> None:
        assert scores_train.ndim == scores_test.ndim == 1

        self._scores_train = scores_train
        self._scores_test = scores_test

    def _get_indices_of_error_at_thresholds(
        self,
        error_rates: NDArray[float],
        error_thresholds: NDArray[float],
        error_type: str,
    ) -> NDArray[int]:
        """
        Get indices where error values are greater/smaller than error thresholds.
        Assumes that error_rates is sorted in increasing order for error_type = "tpr" and "fpr"
        and sorted in decreasing order for error_type = "tnr" and "fnr"
        """

        # find rightmost index where tpr >= threshold
        if error_type == "tpr":
            return np.minimum(
                np.searchsorted(error_rates, error_thresholds, side="left"),
                len(error_rates) - 1,
            )
        # find leftmost index where fpr <= threshold
        elif error_type == "fpr":
            return np.maximum(
                0, np.searchsorted(error_rates, error_thresholds, side="right") - 1
            )
        # find leftmost index where fnr <= threshold
        # search on tpr array instead since tpr has increasing order and fnr = 1 - tpr
        elif error_type == "fnr":
            return np.minimum(
                np.searchsorted(1.0 - error_rates, 1.0 - error_thresholds, side="left"),
                len(error_rates) - 1,
            )
        # find rightmost index where tnr >= threshold
        # search on fpr array instead since fpr has increasing order and tnr = 1 - fpr
        elif error_type == "tnr":
            return np.maximum(
                0,
                np.searchsorted(1.0 - error_rates, 1.0 - error_thresholds, side="right")
                - 1,
            )
        else:
            raise ValueError(f"Invalid error type: {error_type}")

    def get_tpr_fpr(self) -> tuple[NDArray[float], NDArray[float]]:
        """
        Computes true positive rate and true negative rate given scores and labels indicating membership.
        """

        scores: torch.Tensor = torch.cat([self._scores_train, self._scores_test])
        labels = torch.cat(
            [torch.ones_like(self._scores_train), torch.zeros_like(self._scores_test)]
        )
        fpr, tpr, _ = roc_curve(labels, scores)
        return tpr, fpr

    def compute_acc_auc_epsilon(self, delta: float) -> tuple[float, float, float]:
        """
        Compute accuracy, AUC and empirical epsilon for MIA attack WITHOUT Copper Pearson CI
        """

        tpr, fpr = self.get_tpr_fpr()

        fnr = 1 - tpr
        tnr = 1 - fpr

        accuracy = np.max(1 - (fpr + (1 - tpr)) / 2)
        auc_value = auc(fpr, tpr)

        # Divide by zero and invalid value warnings are expectd and occur at certain threshold values
        # We suppress these warnings to avoid disruptive output logs
        with np.errstate(divide="ignore", invalid="ignore"):
            eps1 = np.log(1 - fnr - delta) - np.log(fpr)
            eps2 = np.log(tnr - delta) - np.log(fnr)

        # filter out extreme values in eps1 and eps2
        eps_ub = np.log(self._scores_train.shape[0])
        eps1[eps1 > eps_ub] = 0.0
        eps2[eps2 > eps_ub] = 0.0

        # select max eps
        max_eps1 = np.nanmax(eps1)
        max_eps2 = np.nanmax(eps2)
        emp_eps = max(max_eps1, max_eps2)

        return accuracy, auc_value, emp_eps

    def compute_acc_auc_ci_epsilon(self, delta: float) -> tuple[float, float, float]:
        """
        Compute accuracy, AUC and empirical epsilon for MIA attack
        """

        labels, scores = self._get_scores_and_labels_ordered()

        labels = labels.numpy().astype(bool)
        scores = scores.numpy()

        fpr, tpr, thres = roc_curve(labels, scores)

        accuracy = np.max(1 - (fpr + (1 - tpr)) / 2)
        auc_value = auc(fpr, tpr)

        mems = scores[labels]
        non_mems = scores[~labels]

        r1, r2 = [], []
        for i in range(len(thres)):
            fp_cnt = np.sum(non_mems >= thres[i])
            fn_cnt = np.sum(mems < thres[i])

            _, fp_upper = self._clopper_pearson(fp_cnt, len(non_mems), 1 - 0.95)
            _, fn_upper = self._clopper_pearson(fn_cnt, len(mems), 1 - 0.95)

            r1.append((1 - fn_upper - delta) / (fp_upper + 1e-30))
            r2.append((1 - fp_upper - delta) / (fn_upper + 1e-30))

        max_r1 = np.max(r1)
        max_r2 = np.max(r2)
        max_r = 0

        if max_r1 > max_r2:
            idx = np.argmax(r1)
            max_r = max_r1
            # find rightmost index where fpr <= 0.001 (fpr sorted in increasing order)
            fpr_idx = self._get_indices_of_error_at_thresholds(
                fpr, np.array([0.001]), "fpr"
            )[0]
            low = tpr[fpr_idx]
            logger.info(
                "TNR: {}, "
                "FNR: {}, "
                "emp eps: {}, "
                "tpr@fpr0.001: {}, "
                "eps@fpr0.001: {}, "
                "auc {} accuracy {}, ".format(
                    round(tpr[idx], 5),
                    round(fpr[idx], 5),
                    np.max(round(np.log(max_r1), 5), 0),
                    low.round(5),
                    np.max(round(np.log(r1[fpr_idx]), 5), 0),
                    round(auc_value, 5),
                    round(accuracy, 5),
                )
            )
        else:
            max_r = max_r2
            idx = np.argmax(r2)
            tnr = np.ones_like(fpr) - fpr
            # search for leftmost index where fnr <= 0.001 (fnr sorted in decreasing order)
            fnr = np.ones_like(tpr) - tpr
            fnr_idx = self._get_indices_of_error_at_thresholds(
                fnr, np.array([0.001]), "fnr"
            )[0]
            low = tnr[fnr_idx]
            logger.info(
                "TNR: {}, "
                "FNR: {}, "
                "emp eps: {}, "
                "tnr@fnr0.001: {}, "
                "eps@fnr0.001: {}, "
                "auc {} accuracy {}, ".format(
                    round((1.0 - fpr[idx]), 5),
                    round((1.0 - tpr[idx]), 5),
                    np.max(round(np.log(max_r2), 5), 0),
                    round(low, 5),
                    np.max(round(np.log(r1[fnr_idx]), 5), 0),
                    round(auc_value, 5),
                    round(accuracy, 5),
                )
            )
        emp_eps = np.max(np.log(max_r + 1e-30), 0)

        return accuracy, auc_value, emp_eps

    def compute_metrics_at_error_threshold(
        self,
        delta: float,
        error_threshold: NDArray[float],
        cap_eps: bool = True,
        verbose: bool = False,
        use_fnr_tnr: bool = False,
    ) -> tuple[
        np.float64, np.float64, list[np.float64], list[np.float64], list[np.float64]
    ]:
        """
        Compute accuracy and epsilon metrics at thresholds that are determined by bounds on TPR/FPR according to error_threshold.
        E.g., at error_threshold = 0.01, we compute accuracy and epsilon at a score thresholds that corresponds to the adversary's
        FPR ~ 1% and TPR ~ 1%.

        Args:
            delta: float, privacy parameter in (epsilon, delta) differential privacy
            error_threshold: 1D array of error thresholds
            cap_eps: bool, whether to cap epsilon values to log(num_users)
            verbose: bool, whether to print verbose output
            use_fnr_tnr: bool, whether to use fnr and tnr thresholds in addition to FPR and TPR thresholds

        Returns:
            accuracy: float, max accuracy across all score thresholds
            auc_value: float, area under the curve of tpr and fpr rates for all score thresholds
            eps_fpr_array: list of float, epsilon values using score thresholds corresponding to bounds on FPR
            eps_tpr_array: list of float, epsilon values using score thresholds corresponding to bounds on TPR
            eps_max_array: list of float, element-wise maximum of eps_fpr_array and eps_tpr_array (and eps_fpr_array and eps_fnr_array if use_fnr_tnr is True)
        """

        assert len(error_threshold) > 1

        tpr, fpr = self.get_tpr_fpr()

        accuracy: np.float64 = np.max(1 - (fpr + (1 - tpr)) / 2)
        auc_value = auc(fpr, tpr)

        fnr = 1 - tpr
        tnr = 1 - fpr

        # Divide by zero and invalid value warnings are expectd and occur at certain threshold values
        # We suppress these warnings to avoid disruptive output logs
        with np.errstate(divide="ignore", invalid="ignore"):
            eps1 = np.log(1 - fnr - delta) - np.log(fpr)
            eps2 = np.log(tnr - delta) - np.log(fnr)

        if cap_eps:
            # filter out extreme values in eps1 and eps2
            eps_ub = np.log(self._scores_train.shape[0])
            eps1[eps1 > eps_ub] = eps_ub
            eps2[eps2 > eps_ub] = eps_ub

        if use_fnr_tnr:
            error_threshold = error_threshold[error_threshold < 1.0]

        fpr_indices = self._get_indices_of_error_at_thresholds(
            fpr, error_threshold, "fpr"
        )
        tpr_indices = self._get_indices_of_error_at_thresholds(
            tpr, error_threshold, "tpr"
        )

        eps_fpr_array = eps1[fpr_indices]
        eps_tpr_array = eps1[tpr_indices]
        tpr_array = tpr[tpr_indices]
        fpr_array = fpr[fpr_indices]
        eps_max_array = np.fmax(eps_fpr_array, eps_tpr_array)

        if use_fnr_tnr:
            fnr_indices = self._get_indices_of_error_at_thresholds(
                fnr, error_threshold, "fnr"
            )
            tnr_indices = self._get_indices_of_error_at_thresholds(
                tnr, error_threshold, "tnr"
            )
            eps_fnr_array = eps2[fnr_indices]
            eps_tnr_array = eps2[tnr_indices]
            eps_max_array = np.nanmax(
                [eps_max_array, eps_fnr_array, eps_tnr_array], axis=0
            )

        if verbose:
            logger.info(
                "\n".join(
                    [
                        f"eps@fpr{thre}[tpr={tpr_array[i]:.3f}]: {eps_fpr_array[i]:.3f} eps@max{thre}[fpr={fpr_array[i]:.3f}]: {eps_max_array[i]:.3f}"
                        for i, thre in enumerate(error_threshold)
                    ]
                )
            )
        accuracy = np.float64(accuracy)
        auc_value = np.float64(auc_value)
        eps_fpr_array = [np.float64(x) for x in eps_fpr_array]
        eps_max_array = [np.float64(x) for x in eps_max_array]
        eps_tpr_array = [np.float64(x) for x in eps_tpr_array]

        return accuracy, auc_value, eps_fpr_array, eps_tpr_array, eps_max_array

    def compute_eps_at_tpr_threshold(
        self,
        delta: float,
        tpr_threshold: NDArray[float],
        cap_eps: bool = True,
        verbose: bool = False,
    ) -> list[float]:
        """
        Compute epsilon at error threshold for MIA attack
        """

        assert len(tpr_threshold) > 1

        tpr, fpr = self.get_tpr_fpr()

        fnr = 1 - tpr
        tnr = 1 - fpr

        # Divide by zero and invalid value warnings are expectd and occur at certain threshold values
        # We suppress these warnings to avoid disruptive output logs
        with np.errstate(divide="ignore", invalid="ignore"):
            eps1 = np.log(1 - fnr - delta) - np.log(fpr)
            eps2 = np.log(tnr - delta) - np.log(fnr)

        # filter out extreme values in eps1 and eps2
        if cap_eps:
            eps_ub = np.log(self._scores_train.shape[0])
            eps1[eps1 > eps_ub] = eps_ub
            eps2[eps2 > eps_ub] = eps_ub

        eps_tpr_array = []

        tpr_array = []

        # find leftmost index in tpr that is >= tpr_threshold
        tpr_indices = self._get_indices_of_error_at_thresholds(
            tpr, tpr_threshold, "tpr"
        )

        for i in range(len(tpr_threshold)):
            tpr_idx = tpr_indices[i]
            tpr_level = tpr[tpr_idx]
            eps_tpr = eps1[tpr_idx]
            tpr_array.append(tpr_level)
            eps_tpr_array.append(eps_tpr)

        if verbose:
            logger.info(
                "\n".join(
                    [
                        f"eps@fpr{thre}[tpr={tpr_array[i]:.3f}]: {eps_tpr_array[i]:.3f}"
                        for i, thre in enumerate(tpr_threshold)
                    ]
                )
            )

        return eps_tpr_array

    @staticmethod
    def _clopper_pearson(
        count: int,
        trials: int,
        conf: float,
    ) -> tuple[float, float]:
        """
        Compute clopper pearson CI
        """

        if np.ndim(count) > 0 or np.ndim(trials) > 0 or np.ndim(conf) > 0:
            raise ValueError("_clopper_pearson function only works for scalar values")

        q = count / trials
        ci_low = beta.ppf(conf / 2.0, count, trials - count + 1)
        ci_upp = beta.isf(conf / 2.0, count + 1, trials - count)

        ci_low = ci_low if (q != 0) else 0.0
        ci_upp = ci_upp if (q != 1) else 1.0

        return ci_low, ci_upp

    def _get_scores_and_labels_ordered(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sorts the scores from the highest to the lowest and returns
        the labels sorted by the scores.

        Notes:
            - A train sample is labeled as 1 and a test sample as 0.
        """

        scores: torch.Tensor = torch.cat([self._scores_train, self._scores_test])
        order = torch.argsort(scores, descending=True)
        scores_ordered = scores[order]

        labels = torch.cat(
            [torch.ones_like(self._scores_train), torch.zeros_like(self._scores_test)]
        )
        labels_ordered = labels[order]
        return labels_ordered, scores_ordered
