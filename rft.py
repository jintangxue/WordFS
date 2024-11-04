import numpy as np
from joblib import Parallel, delayed

# import lib_greensr.util.lib_general_tools as lib_gtools

NUM_PARALLEL = 8

'''
To simplify the class, no I/O part involved.

DEFINITION:
    rftloss: the optimal (minimun) total weighted mse (np.float)
'''


class Reletive_Feat_Test():

    def __init__(self):
        pass

    # %%
    # simple RFT
    def get_total_weighted_mse_one_partition(self, input_arr, thre, output_arr):
        sample_indicator_left = input_arr.reshape(-1) <= thre
        sample_num_perc_left = np.sum(sample_indicator_left) * 1.0 / len(sample_indicator_left)
        raw_mse_left = np.var((output_arr.reshape(-1))[sample_indicator_left])
        weighted_mse_left = sample_num_perc_left * raw_mse_left

        sample_indicator_right = input_arr.reshape(-1) > thre
        raw_mse_right = np.var((output_arr.reshape(-1))[sample_indicator_right])
        weighted_mse_right = (1.0 - sample_num_perc_left) * raw_mse_right

        weighted_mse = weighted_mse_left + weighted_mse_right
        return weighted_mse

    def get_rftloss_onedim(self,
                           distribution_arr, target_arr, num_bins,
                           min_flag=True):
        '''
        Function:
            work for a single feature (a single dimension), evaluate its relevancy to the target
        Arg:
            distribution_arr, np.ndarray of shape (n, ) or (n, 1): the feature to evaluate
            target_arr, np.ndarray of shape (n, ): the target as evaluation reference
            min_flag, bool
        Returns:
            min_total_mse, np.float: [when min_flag is True] the minimum total MSE the target_arr can achieve by some certain pseudo binary class splitting
            total_mse, np.ndarray of shape (num_bins + 1, ): the total MSE the target_arr obtains by setting variant pseudo binary class splitting positions
                                                           (from (0, num_bins) to (num_bins, 0) splitting as (num_bins_left_class, num_bins_right_class), totally num_bins + 1 splitting settings)
        '''

        a = np.arange(num_bins + 1)  # the prototype bin edges ready to rescale

        a_min, a_max = np.min(a), np.max(a)
        arr_min, arr_max = np.min(distribution_arr), np.max(distribution_arr)
        bin_edges = (a * 1.0 - a_min) * 1.0 / (a_max - a_min) * (arr_max - arr_min) + arr_min

        weighted_mse_cand = [self.get_total_weighted_mse_one_partition(distribution_arr, thre, target_arr) for thre in
                             bin_edges[1:-1]]

        if min_flag:
            min_total_mse = np.min(weighted_mse_cand)
            return min_total_mse
        else:
            return weighted_mse_cand

    # def get_minTotalMse_multiDim(distribution_arr_2d, target_arr, num_bins, ratio=0.5):
    def get_rftloss_multidim(self,
                             distribution_arr, target_arr, num_bins,
                             ratio=0.5):
        '''
        Work for multiple dimensions
        distribution_arr 2d (num_sample, num_dim), or 3d, or 4d
        '''

        intrinsic_num_thre = 100000
        num_sample = len(distribution_arr)
        num_selection = int(num_sample * ratio)

        if num_sample >= intrinsic_num_thre:
            num_eff = num_selection
        else:
            num_eff = num_sample
        # print('rft num_eff', num_eff)
        cond_samp_idx = np.random.choice(np.arange(len(distribution_arr)),
                                         num_eff,
                                         replace=False)

        distribution_arr_2d = distribution_arr.reshape((len(distribution_arr), -1))
        num_feat = distribution_arr_2d.shape[-1]
        weighted_mse_opt_list = Parallel(n_jobs=NUM_PARALLEL) \
            (delayed(self.get_rftloss_onedim) \
                 (distribution_arr_2d[cond_samp_idx, feat_idx],
                  target_arr[cond_samp_idx],
                  num_bins) \
             for feat_idx in range(num_feat))

        return np.array(weighted_mse_opt_list).reshape(-1)

    def get_rftloss_onechl(self,
                           distribution_arr, target_arr, num_bins,
                           ratio=0.5):
        return self.get_rftloss_multidim(distribution_arr, target_arr, num_bins,
                                         ratio=ratio)

    def get_rftloss_chlwise(self,
                            distribution_arr_4dor3d, target_arr, num_bins,
                            ratio=0.5):
        '''
        distribution_arr_4dor3d
        (num_samples, c, h, w) or (num_samples, c, h*w)
        '''
        num_chl = distribution_arr_4dor3d.shape[1]
        rft_loss_arr_list = \
            [self.get_rftloss_onechl(
                distribution_arr_4dor3d[:, chl_idx:(chl_idx + 1)].reshape((len(distribution_arr_4dor3d), -1)),
                target_arr, num_bins,
                ratio=ratio) for chl_idx in range(num_chl)]
        return rft_loss_arr_list  # 2-nested list, np -> (4, h'*w')

    # %%
    def get_selected_feat_idx_multidim(self, rft_loss_arr, num_feat):
        rft_loss_arr = np.array(rft_loss_arr).reshape(-1)
        dim_idx_sorted = np.argsort(rft_loss_arr)
        dim_idx_sorted = dim_idx_sorted[:num_feat]
        return dim_idx_sorted

    def get_selected_feat_idx_onechl(self, rft_loss_arr, num_feat):
        return self.get_selected_feat_idx_multidim(rft_loss_arr, num_feat)

    def get_selected_feat_idx_chlwise(self, rft_loss_arr, num_feat):
        '''
        len(rft_loss_arr) == num_chl
        num_feat is for each chl
        '''
        dim_idx_sorted_list = \
            [self.get_selected_feat_idx_onechl(i, num_feat)
             for i in rft_loss_arr]
        return dim_idx_sorted_list

    def get_selected_feat_chlwise(self,
                                  distribution_arr_4dor3d,
                                  dim_idx_sorted_multichl):
        '''
        distribution_arr_4dor3d, 3-D or 4-D np.ndarray,
        (num_samples, c, h, w) or (num_samples, c, h*w)

        dim_idx_sorted_multichl, 2-nested list, np -> (4, h'*w')
        '''
        print(distribution_arr_4dor3d.shape, np.array(dim_idx_sorted_multichl).shape)
        num_chl = distribution_arr_4dor3d.shape[1]
        num_samples = len(distribution_arr_4dor3d)

        feat_multichl_3d = distribution_arr_4dor3d.reshape((num_samples, num_chl, -1))
        feat_selected = \
            np.concatenate([(feat_multichl_3d[:, chl_idx])[:, np.array(j).reshape(-1)] for chl_idx, j in
                            zip(np.arange(num_chl), dim_idx_sorted_multichl)], axis=-1)
        return feat_selected


# %%
# shape = (2, 100)
# a = np.arange(np.prod(shape)).reshape(shape)
# b = np.arange(5).reshape((5, 1))
# c = a[:, b]
# print(c.shape)
# %%

if __name__ == "__main__":
    rft = Reletive_Feat_Test()
    feat1 = np.random.rand(100, 4 * 14 * 14)
    # feat2 = np.random.rand(100, 9, 13, 13)
    targ = np.random.rand(100)
    num_feat = 5
    num_bins = 4
    # feat = [feat1, feat2]

    # rft_list = [rft.get_get_rftloss_chlwise(i, targ, num_bins) for i in feat]
    # print([np.array(i).shape for i in rft_list]) #[(4, 196), (9, 169)]
    # dim_idx_sorted_list = [rft.get_selected_feat_idx_chlwiserftloss(i, num_feat) for i in rft_list]
    # print([np.array(i).shape for i in dim_idx_sorted_list]) # [(4, 5), (9, 5)]
    # feat_selected_list = [rft.get_selected_feat_multichl(i,j) for i,j in zip(feat, dim_idx_sorted_list)]

    rft_loss = rft.get_rftloss_multidim(feat1, targ, num_bins)
    print(rft_loss.shape)
    feat_idx_selected = rft.get_selected_feat_idx_multidim(rft_loss, num_feat)
    print(feat_idx_selected)






