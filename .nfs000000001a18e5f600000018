from read_write import *
import os
from ranking import *
from rft import *
from sklearn.model_selection import KFold
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
import argparse
import umap
import multiprocessing


def compute_correlation(k, x_train, y_train):
    dimension_values = x_train[:, k]
    correlation, _ = spearmanr(dimension_values, y_train)
    return correlation


def main():
    parser = argparse.ArgumentParser(description="Configure word vector settings")

    parser.add_argument("--path", type=str, default="/mnt/jintang2/EmbedTextNet-main/Software_EmbedTextNet/Text_similarity/word_vec/glove/glove.6B.300d.txt", help="Path to the word vectors")
    parser.add_argument("--wv_type", type=str, default="Glove", choices=["Glove", "Word2vec", "Fasttext"], help="Word vector type")
    parser.add_argument("--word_sim_dir", type=str, default="../data/word-sim-train/", help="Word similarity directory")
    parser.add_argument("--new_wv_dim", type=int, default=150, help="New word vector dimension")
    parser.add_argument("--method", type=str, default="WordFS-S", choices=["None", "WordFS-P", "WordFS-S", "Algo", "PCA", "EmbedTextNet", "UMAP"], help="Method to use")
    parser.add_argument("--post_processing", type=str, default="PPA", choices=["PPA", "None"], help="Post-processing method")
    parser.add_argument("--fold_count", type=int, default=5, help="Number of folds for cross-validation")
    parser.add_argument("--cross_validation_runs", type=int, default=5, help="Number of cross-validation runs")

    args = parser.parse_args()

    word_vecs, x_train_ori, x_train_names = load_word_vectors(args.path, args.wv_type)
    wv_type = args.wv_type
    word_sim_dir = args.word_sim_dir
    new_wv_dim = args.new_wv_dim
    post_processing = args.post_processing
    method = args.method
    fold_count = args.fold_count
    cross_validation_runs = args.cross_validation_runs

    print("wv_type:", wv_type, ";", "post_processing:", post_processing, ";", "method:", method, "new_wv_dim:", new_wv_dim)

    # Get the list of word similarity datasets in the directory
    file_list = os.listdir(word_sim_dir)
    results = []
    all_datasets_results = []

    if post_processing == "PPA" and (method == "WordFS-P" or "WordFS-S"):
        print("Starting post-processing...")
        x_train = x_train_ori.copy()
        pca = PCA(n_components=300)
        x_train = x_train - np.mean(x_train)
        x_fit = pca.fit_transform(x_train)
        U1 = pca.components_

        z = []

        # Removing Projections on Top Components
        for i, x in enumerate(x_train):
            for u in U1[0:7]:
                x = x - np.dot(u.transpose(), x) * u
            z.append(x)
        z = np.asarray(z)
        x_train = z
        x_train_ori = x_train
        word_vecs = {word: x_train_ori[i] for i, word in enumerate(x_train_names)}

    print("=================================================================================")
    print("%6s" % "Serial", "%20s" % "Dataset", "%15s" % "Num Pairs", "%15s" % "Not found", "%15s" % "Average Rho")
    print("=================================================================================")

    if method == "Algo":
        x_train = x_train_ori.copy()

        # PCA to get Top Components
        pca = PCA(n_components=300)
        x_train = x_train - np.mean(x_train)
        x_fit = pca.fit_transform(x_train)
        U1 = pca.components_

        z = []

        # Removing Projections on Top Components
        for i, x in enumerate(x_train):
            for u in U1[0:7]:
                x = x - np.dot(u.transpose(), x) * u
            z.append(x)

        z = np.asarray(z)

        # PCA Dim Reduction
        pca = PCA(n_components=new_wv_dim)
        x_train = z - np.mean(z)
        x_new_final = pca.fit_transform(x_train)

        # PCA to do Post-Processing Again
        pca = PCA(n_components=new_wv_dim)
        x_new = x_new_final - np.mean(x_new_final)
        x_new = pca.fit_transform(x_new)
        Ufit = pca.components_

        x_new_final = x_new_final - np.mean(x_new_final)

        new_word_vecs = {}
        for k, x in enumerate(x_train_names):
            new_word_vecs[x] = x_new_final[k]
            for u in Ufit[0:7]:
                new_word_vecs[x] = new_word_vecs[x] - np.dot(u.transpose(), new_word_vecs[x]) * u

    elif method == "PCA":
        x_train = x_train_ori.copy()

        # PCA Dim Reduction
        pca = PCA(n_components=new_wv_dim)
        x_train = x_train - np.mean(x_train)
        x_new_final = pca.fit_transform(x_train)

        new_word_vecs = {}
        for k, x in enumerate(x_train_names):
            new_word_vecs[x] = x_new_final[k]

    elif method == "UMAP":
        x_train = x_train_ori.copy()

        # UMAP Dimensionality Reduction
        umap_reducer = umap.UMAP(n_components=new_wv_dim)
        x_new_final = umap_reducer.fit_transform(x_train)

        new_word_vecs = {}
        for k, x in enumerate(x_train_names):
            new_word_vecs[x] = x_new_final[k]

    elif method == "None":
        new_word_vecs = {}
        for k, x in enumerate(x_train_names):
            new_word_vecs[x] = word_vecs[x].copy()

    elif method == "EmbedTextNet":
        EmbedTextNet = {}
        EmbedTextNet_path = "/mnt/jintang2/EmbedTextNet-main/Software_EmbedTextNet/Text_similarity/fasttext_reduced_embedding_300_to_50.txt"
        with open(EmbedTextNet_path, encoding="utf-8") as f:
            for line in f:
                EmbedTextNet_values = line.split()
                EmbedTextNet_word = EmbedTextNet_values[0]
                EmbedTextNet_coefs = np.asarray(EmbedTextNet_values[1:], dtype='float32')
                EmbedTextNet[EmbedTextNet_word] = EmbedTextNet_coefs
            f.close()

        new_word_vecs = EmbedTextNet

    # Train for each dataset
    for i, filename in enumerate(file_list):
        avg_rho_val = 0

        for cv_run in range(cross_validation_runs):
            kf = KFold(n_splits=fold_count, shuffle=True, random_state=42 + cv_run)
            rho_val = 0

            with open(os.path.join(word_sim_dir, filename), 'r', encoding='utf-8') as file:
                lines = file.readlines()

            data = [line.strip().lower().split() for line in lines]

            for fold, (train_idx, test_idx) in enumerate(kf.split(data)):
                manual_dict, auto_dict = ({}, {})
                not_found, total_size = (0, 0)
                x_train = []
                y_train = []
                train_data = [data[idx] for idx in train_idx]
                test_data = [data[idx] for idx in test_idx]

                if method == "WordFS-P" or method == "WordFS-S":
                    for word1, word2, sim_sc in train_data:
                        if word1 in word_vecs and word2 in word_vecs:
                            vec1 = word_vecs[word1].copy()
                            vec2 = word_vecs[word2].copy()
                            norms_word1 = np.linalg.norm(vec1)
                            norms_word2 = np.linalg.norm(vec2)
                            x_train.append(np.multiply(vec1, vec2) / (norms_word1 * norms_word2))
                            y_train.append(float(sim_sc))

                    x_train = np.asarray(x_train)
                    y_train = np.asarray(y_train)
                    num_feat = int(new_wv_dim)

                    if method == "WordFS-P":
                        num_bins = 4
                        rft = Reletive_Feat_Test()
                        rft_loss = rft.get_rftloss_multidim(x_train, y_train, num_bins)
                        RFT_feat_idx_selected = rft.get_selected_feat_idx_multidim(rft_loss, x_train.shape[-1])
                        feat_idx_selected = RFT_feat_idx_selected[:int(new_wv_dim)]

                    if method == "WordFS-S":
                        dimension_correlations = Parallel(n_jobs=multiprocessing.cpu_count())(
                            delayed(compute_correlation)(k, x_train, y_train) for k in range(x_train.shape[-1])
                        )

                        # Rank dimensions based on correlation
                        sorted_dimensions = np.argsort(dimension_correlations)[::-1]  # Sim
                        dimension_correlations.sort()
                        feat_idx_selected = sorted_dimensions[:int(new_wv_dim)]

                    unique_elements, indices, counts = np.unique(feat_idx_selected, return_index=True,
                                                                 return_counts=True)

                    sorted_indices = np.argsort(indices)
                    unique_elements_sorted = unique_elements[sorted_indices]
                    unique_elements_sorted = unique_elements_sorted[:num_feat]
                    x_new_final = x_train_ori[:, unique_elements_sorted]

                    new_word_vecs = {}

                    for k, x in enumerate(x_train_names):
                        new_word_vecs[x] = x_new_final[k]

                # evaluate
                for word1, word2, sim_sc in test_data:
                    if word1 in word_vecs and word2 in word_vecs:
                        manual_dict[(word1, word2)] = float(sim_sc)
                        auto_dict[(word1, word2)] = cosine_sim(new_word_vecs[word1], new_word_vecs[word2])
                    else:
                        not_found += 1
                    total_size += 1
                rho_val += spearmans_rho(assign_ranks(manual_dict), assign_ranks(auto_dict))

            avg_rho_val += rho_val / fold_count

        # Compute and store the final average Rho for the current file
        final_avg_rho = avg_rho_val / float(cross_validation_runs)
        results.append((i + 1, filename, total_size, not_found, final_avg_rho))

        all_datasets_results.append(final_avg_rho)

        print("%6s" % str(i + 1), "%20s" % filename, "%15s" % str(total_size), "%15s" % str(not_found),
              "%15.4f" % final_avg_rho)

    print("overall averaged:", np.mean(all_datasets_results))
    print("Output dimension:", len(new_word_vecs[word1]))
    print("post_processing:", post_processing, "method:", method)


if __name__ == "__main__":
    main()
