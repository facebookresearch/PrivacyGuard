# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict
import logging

import numpy as np
import torch
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

    def get_tpr_fpr(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes true positive rate and true negative rate,, useful for plotting
        ROC curves and computing AUC.
        """
        labels_ordered, _ = self._get_scores_and_labels_ordered()

        true_positive_rate = (
            torch.cumsum(labels_ordered == 1, 0) / self._scores_train.shape[0]
        )
        false_positive_rate = (
            torch.cumsum(labels_ordered == 0, 0) / self._scores_test.shape[0]
        )
        return true_positive_rate, false_positive_rate

    def compute_acc_auc_epsilon(self, delta: float) -> tuple[float, float, float]:
        """
        Compute accuracy, AUC and empirical epsilon for MIA attack WITHOUT Copper Pearson CI
        """

        tpr, fpr = self.get_tpr_fpr()
        tpr, fpr = tpr.numpy(), fpr.numpy()

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

    def compute_epsilon_at_error_thresholds(
        self, delta: float, error_thresholds: list[float]
    ) -> tuple[list[float], list[float]]:
        """
        Compute epsilons at error threshold for MIA attack
        """

        assert len(error_thresholds) > 1

        _tpr, _fpr = self.get_tpr_fpr()
        # pyre-fixme[24]: Generic type `np.ndarray` expects 2 type parameters.
        tpr: np.ndarray = _tpr.numpy()
        # pyre-fixme[24]: Generic type `np.ndarray` expects 2 type parameters.
        fpr: np.ndarray = _fpr.numpy()

        fnr = 1 - tpr
        tnr = 1 - fpr

        # Divide by zero and invalid value warnings are expectd and occur at certain threshold values
        # We suppress these warnings to avoid disruptive output logs
        with np.errstate(divide="ignore", invalid="ignore"):
            # generate epsilon values from fnr and tnr
            eps1 = np.log(1 - fnr - delta) - np.log(fpr)
            eps2 = np.log(tnr - delta) - np.log(fnr)

        # filter out extreme values in eps1 and eps2
        eps_ub = np.log(self._scores_train.shape[0])
        eps1[eps1 > eps_ub] = 0.0
        eps2[eps2 > eps_ub] = 0.0

        eps_fpr_array = []
        eps_fnr_array = []

        tpr_array = []
        tnr_array = []

        for threshold in error_thresholds:
            if fpr.min() > threshold:
                # no epsilon value at this threshold
                tpr_level = eps_fpr = 0
            else:
                fpr_idxes = np.where(fpr <= threshold)
                if len(fpr_idxes) == 0:
                    continue
                # take the last index where fpr <= threshold
                fpr_idx = fpr_idxes[0][-1]
                # add the tpr and fnr epsilon to outputs
                tpr_level = tpr[fpr_idx]
                eps_fpr = eps1[fpr_idx]

            if fnr.min() > threshold:
                # no epsilon value at this threshold
                tnr_level = eps_fnr = 0
            else:
                fnr_idxes = np.where(fnr <= threshold)
                if len(fnr_idxes) == 0:
                    continue
                # take the first index where fnr <= threshold
                fnr_idx = fnr_idxes[0][0]
                # add the tnr and tnr epsilon to outputs
                tnr_level = tnr[fnr_idx]
                eps_fnr = eps2[fnr_idx]

            tpr_array.append(tpr_level)
            tnr_array.append(tnr_level)

            eps_fpr_array.append(eps_fpr)
            eps_fnr_array.append(eps_fnr)

        return eps_fpr_array, eps_fnr_array

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
            fpr_idx = np.where(fpr <= 0.001)[0][-1]
            low = tpr[fpr_idx]
            logger.info(
                "TNR: {}, "
                "FNR: {}, "
                "emp eps: {}, "
                "tpr@fpr0.001: {}, "
                "eps@fpr0.001: {}, "
                "auc {} accuracy {}, ".format(
                    tpr[idx].round(5),
                    fpr[idx].round(5),
                    np.max(np.log(max_r1).round(5), 0),
                    low.round(5),
                    np.max(np.log(r1[fpr_idx]).round(5), 0),
                    auc_value.round(5),
                    accuracy.round(5),
                )
            )
        else:
            max_r = max_r2
            idx = np.argmax(r2)
            tnr = 1 - fpr
            fnr = 1 - tpr
            fnr_idx = np.where(fnr <= 0.001)[0][-1]
            # pyre-fixme[16]: `int` has no attribute `__getitem__`.
            low = tnr[fnr_idx]
            logger.info(
                "TNR: {}, "
                "FNR: {}, "
                "emp eps: {}, "
                "tnr@fnr0.001: {}, "
                "eps@fnr0.001: {}, "
                "auc {} accuracy {}, ".format(
                    # pyre-fixme[16]: `int` has no attribute `round`.
                    (1 - fpr[idx]).round(5),
                    (1 - tpr[idx]).round(5),
                    np.max(np.log(max_r2).round(5), 0),
                    low.round(5),
                    np.max(np.log(r1[fnr_idx]).round(5), 0),
                    auc_value.round(5),
                    accuracy.round(5),
                )
            )
        emp_eps = np.max(np.log(max_r + 1e-30), 0)

        return accuracy, auc_value, emp_eps

    def compute_metrics_at_error_threshold(
        self,
        delta: float,
        # pyre-fixme[24]: Generic type `np.ndarray` expects 2 type parameters.
        error_threshold: np.ndarray,
        cap_eps: bool = True,
        verbose: bool = False,
    ) -> tuple[np.float64, np.float64, list[np.float64], list[np.float64]]:
        """
        Compute epsilon at error threshold for MIA attack
        """

        assert len(error_threshold) > 1

        tpr, fpr = self.get_tpr_fpr()
        tpr, fpr = tpr.numpy(), fpr.numpy()

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

        eps_fpr_array = []
        eps_max_array = []

        tpr_array = []
        tnr_array = []

        for threshold in error_threshold:
            if fpr.min() > threshold:
                tpr_level = eps_fpr = 0
            else:
                fpr_idx = np.where(fpr <= threshold)[0][-1]
                tpr_level = tpr[fpr_idx]
                eps_fpr = eps1[fpr_idx]

            if fnr.min() > threshold:
                tnr_level = eps_fnr = 0
            else:
                fnr_idx = np.where(fnr <= threshold)[0][0]
                tnr_level = tnr[fnr_idx]
                eps_fnr = eps2[fnr_idx]

            # pyre-fixme[6]: For 1st argument expected `SupportsRichComparisonT` but
            #  got `Union[ndarray[Any, dtype[Any]], int]`.
            # pyre-fixme[6]: For 2nd argument expected `SupportsRichComparisonT` but
            #  got `Union[ndarray[Any, dtype[Any]], int]`.
            eps_max = max(eps_fnr, eps_fpr)

            tpr_array.append(tpr_level)
            tnr_array.append(tnr_level)

            eps_fpr_array.append(eps_fpr)
            eps_max_array.append(eps_max)

        if verbose:
            logger.info(
                "\n".join(
                    [
                        f"eps@fpr{thre}[tpr={tpr_array[i]:.3f}]: {eps_fpr_array[i]:.3f} eps@max{thre}[tnr={tnr_array[i]:.3f}]: {eps_max_array[i]:.3f}"
                        for i, thre in enumerate(error_threshold)
                    ]
                )
            )
        accuracy = np.float64(accuracy)
        auc_value = np.float64(auc_value)
        eps_fpr_array = [np.float64(x) for x in eps_fpr_array]
        eps_max_array = [np.float64(x) for x in eps_max_array]

        return accuracy, auc_value, eps_fpr_array, eps_max_array

    def compute_eps_at_tpr_threshold(
        self,
        delta: float,
        # pyre-fixme[24]: Generic type `np.ndarray` expects 2 type parameters.
        tpr_threshold: np.ndarray,
        cap_eps: bool = True,
        verbose: bool = False,
    ) -> list[float]:
        """
        Compute epsilon at error threshold for MIA attack
        """

        assert len(tpr_threshold) > 1

        tpr, fpr = self.get_tpr_fpr()
        tpr, fpr = tpr.numpy(), fpr.numpy()

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

        for threshold in tpr_threshold:
            if tpr.min() > threshold:
                tpr_level = eps_tpr = 0
            else:
                tpr_idx = np.where(tpr >= threshold)[0][0]
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
