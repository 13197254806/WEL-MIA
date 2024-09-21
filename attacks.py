import numpy as np
from sklearn.preprocessing import QuantileTransformer
import bisect
from scipy.stats import norm


def loss_attack(probs_list):
    """
    Parameters:
        probs_list: numpy.Array of shape (target_dataset_size, max_length), token-level probabilities in the target model.

    Returns:
        numpy.Array of shape (target_dataset_size, ), membership scores of Loss Attack.
    """
    loss_scores_list = [- np.average(-np.log(x)) for x in probs_list]
    return np.array(loss_scores_list)


def lira(target_probs_list, reference_probs_list):
    """
    Parameters:
        target_probs_list: numpy.Array of shape (target_dataset_size, max_length), token-level probabilities in the target model.
        reference_probs_list: numpy.Array of shape (target_dataset_size, max_length), token-level probabilities in the reference model.

    Returns:
        numpy.Array of shape (target_dataset_size, max_length), membership scores of LiRA.
    """
    lira_scores_list = []
    for target_probs, reference_probs in zip(target_probs_list, reference_probs_list):
        # calculate the loss of each sample
        target_loss = np.average(- np.log(target_probs))
        reference_loss = np.average(- np.log(reference_probs))
        lira_scores_list.append(- (target_loss - reference_loss))
    return np.array(lira_scores_list)


def min_k(probs_list, k):
    """
    Parameters:
        probs_list: numpy.Array of shape (target_dataset_size, max_length), token-level probabilities in the target model.
        k: float, the parameter k in Min-k% Attack

    Returns:
        numpy.Array of shape (target_dataset_size, max_length), membership scores of Min-k% Attack.
    """
    min_k_scores_list = []
    for probs in probs_list:
        # find the index of the k% tokens with minimal prediction probabilities in a sample
        index = np.argpartition(- probs, - round((len(probs) * k)))[- round((len(probs) * k)):]
        min_k_scores_list.append(- np.average(- np.log(probs[index])))
    return np.array(min_k_scores_list)


def neighbour_attack(probs_list, paraphrased_probs_list, neighbour_number):
    """
    Parameters:
        probs_list: numpy.Array of shape (target_dataset_size, max_length), token-level probabilities in the target model.
        paraphrased_probs_list: numpy.Array of shape (target_dataset_size * neighbour_number, max_length), token-level probabilities of paraphrased target dataset in the target model.
        neighbour_number: int, number of neighbours for single target sample

    Returns:
        numpy.Array of shape (target_dataset_size, max_length), membership scores of Min-k% Attack.
    """
    neighbour_scores_list = []
    for idx in range(len(probs_list)):
        score = 0
        target_probs = probs_list[idx]
        paraphrased_probs = paraphrased_probs_list[int(neighbour_number * idx): int(neighbour_number * (idx + 1))]
        for paraphrased in paraphrased_probs:
            score += np.average(- np.log(paraphrased))
        score = score / neighbour_number - np.average(- np.log(target_probs))
        neighbour_scores_list.append(score)
    return np.array(neighbour_scores_list)


def wel_mia(target_probs_list, reference_probs_list, bins_number, temp=2):
    """
    Parameters:
        target_probs_list: numpy.Array of shape (target_dataset_size, max_length), token-level probabilities in the target model.
        reference_probs_list: numpy.Array of shape (target_dataset_size, max_length), token-level probabilities in the reference model.
        bins_number: int, how many bins are used to group tokens.
        temp: float, the smoothing temperature.

    Returns:
        numpy.Array of shape (target_dataset_size, max_length), membership scores of Min-k% Attack.
    """

    # wel_mia_scores_list saves the membership score of each sample
    # difficulty_all saves all token-level negative log probabilities in the target model
    # Y saves all token-level likelihood ratios
    # index_list saves all tokens' position in the target dataset
    # likelihood_scaled_all saves all quantile scaled token-level likelihood
    # difficulty_ave_bins saves the average token difficulty in each bin
    # likelihood_bins saves the quantile-scaled token-level likelihood in each bin
    # likelihood_ave_bins saves the average token likelihood in each bin
    # likelihood_std_bins saves the standard deviation of token likelihood in each bin
    wel_mia_scores_list = []
    difficulty_all = np.array([])
    likelihood_all = np.array([])
    index_list = []
    likelihood_scaled_all = np.zeros_like(target_probs_list)
    difficulty_ave_bins = []
    likelihood_bins = []
    likelihood_ave_bins = []
    likelihood_std_bins = []

    # get all token-level negative log probabilities (difficulty on the target model) and likelihood
    for idx in range(len(target_probs_list)):
        target_nlog_probs = -np.log(target_probs_list[idx])
        reference_nlog_probs = -np.log(reference_probs_list[idx])
        target_nlog_probs[np.isnan(target_nlog_probs)] = 20
        reference_nlog_probs[np.isnan(reference_nlog_probs)] = 20

        length = len(target_nlog_probs)
        index = list(zip([idx] * length, list(range(length))))
        index_list.extend(index)
        difficulty_all = np.concatenate((difficulty_all, target_nlog_probs))
        likelihood_all = np.concatenate((likelihood_all, - (target_nlog_probs - reference_nlog_probs)))

    percentiles = np.linspace(0, 100, bins_number + 1)
    index_list = np.array(index_list)
    difficulty_bins = np.percentile(difficulty_all, percentiles)

    # group all tokens into bins by their difficulty
    for i in range(1, len(difficulty_bins)):
        index_bin = (difficulty_all >= difficulty_bins[i - 1]) & (difficulty_all < difficulty_bins[i])
        index = index_list[(difficulty_all >= difficulty_bins[i - 1]) & (difficulty_all < difficulty_bins[i])]
        likelihood_bin = likelihood_all[index_bin]
        difficulty_ave_bins.append(np.average(difficulty_all[index_bin]))

        if len(likelihood_bin) <= 0:
            likelihood_bins.append(likelihood_bin)
            likelihood_ave_bins.append(np.average(likelihood_bin) if len(likelihood_bin) > 0 else 0)
            likelihood_std_bins.append(np.std(likelihood_bin) if len(likelihood_bin) > 1 else 1e8)
            continue

        # quantile scaling for token-level likelihood
        transformer = QuantileTransformer(output_distribution='normal', random_state=0)
        normalized = transformer.fit_transform(likelihood_bin.reshape(-1, 1)).flatten()

        for j in range(len(normalized)):
            index_x, index_y = index[j][0], index[j][1]
            likelihood_scaled_all[index_x][index_y] = normalized[j]

        likelihood_bins.append(likelihood_bin)
        likelihood_ave_bins.append(np.average(likelihood_bin) if len(likelihood_bin) > 0 else 0)
        likelihood_std_bins.append(np.std(likelihood_bin) if len(likelihood_bin) > 1 else 1e5)
    difficulty_ave_bins = np.array(difficulty_ave_bins)
    likelihood_ave_bins = np.array(likelihood_ave_bins)
    likelihood_std_bins = np.array(likelihood_std_bins)

    # calculate the membership score for each sample
    for idx in range(len(target_probs_list)):
        target_nlog_probs = -np.log(target_probs_list[idx])
        reference_nlog_probs = -np.log(reference_probs_list[idx])
        target_nlog_probs[np.isnan(target_nlog_probs)] = 20
        reference_nlog_probs[np.isnan(reference_nlog_probs)] = 20
        length = len(target_nlog_probs)

        # each token's bin number in a sample
        bins_index = np.array([bisect.bisect_right(difficulty_bins, x) - 1 for x in target_nlog_probs], dtype=int)
        bins_index = np.clip(bins_index, 0, bins_number - 1)
        likehood_scaled_tokens = likelihood_scaled_all[idx][: length]
        likehood_ave_tokens = likelihood_ave_bins[bins_index]
        likehood_std_tokens = likelihood_std_bins[bins_index]


        # the smoothing coefficient
        smoothing_tokens = np.array([1 - norm.cdf(likehood_scaled_tokens[i], likehood_ave_tokens[i], likehood_std_tokens[i] * temp)
                               for i in range(len(likehood_scaled_tokens))])

        weights_tokens = np.ones_like(target_nlog_probs)
        weights_tokens *= smoothing_tokens
        weights_tokens *= difficulty_ave_bins[bins_index]
        score = np.average(likehood_scaled_tokens * weights_tokens)
        wel_mia_scores_list.append(score)
    return np.array(wel_mia_scores_list)
