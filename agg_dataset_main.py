from read_write import *
from ranking import *
from rft import *
from scipy.stats import spearmanr
import os
import argparse
import umap
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import pandas as pd
import multiprocessing
import time


# Function to normalize scores within a dataset
def normalize_scores(scores):
    scaler = MinMaxScaler(feature_range=(0, 1))
    return scaler.fit_transform(np.array(scores).reshape(-1, 1)).flatten()


def compute_correlation(k, x_train, y_train):
    dimension_values = x_train[:, k]
    correlation, _ = spearmanr(dimension_values, y_train)
    return correlation


def main():
    parser = argparse.ArgumentParser(description="Configure word vector settings")

    parser.add_argument("--path", type=str, default="/mnt/jintang2/EmbedTextNet-main/Software_EmbedTextNet/Text_similarity/word_vec/glove/glove.6B.300d.txt", help="Path to the word vectors")
    parser.add_argument("--wv_type", type=str, default="Glove", choices=["Glove", "Word2vec", "Fasttext"], help="Word vector type")
    parser.add_argument("--word_sim_dir", type=str, default="../data/word-sim-combined/", help="Word similarity directory")
    parser.add_argument("--new_wv_dim", type=int, default=150, help="New word vector dimension")
    parser.add_argument("--method", type=str, default="WordFS-S", choices=["None", "WordFS-P", "WordFS-S", "Algo", "PCA", "EmbedTextNet", "UMAP"], help="Method to use")
    parser.add_argument("--post_processing", type=str, default="PPA", choices=["PPA", "None"], help="Post-processing method")
    parser.add_argument("--fold_count", type=int, default=5, help="Number of folds for cross-validation")
    parser.add_argument("--n_iterations", type=int, default=5, help="Number of iterations")
    parser.add_argument("--save_vectors", type=bool, default=True, help="Save the vectors or not")
    parser.add_argument("--evaluation", type=bool, default=True, help="Evaluate on the aggregated dataset")

    args = parser.parse_args()

    # args.path = "/mnt/jintang2/EmbedTextNet-main/Software_EmbedTextNet/Text_similarity/word_vec/GoogleNews-vectors-negative300.bin"
    # args.path = "/mnt/jintang2/EmbedTextNet-main/Software_EmbedTextNet/Text_similarity/word_vec/wiki-news-300d-1M.vec"

    word_vecs, x_train_ori, x_train_names = load_word_vectors(args.path, args.wv_type)
    wv_type = args.wv_type
    word_sim_dir = args.word_sim_dir
    new_wv_dim = args.new_wv_dim
    post_processing = args.post_processing
    method = args.method
    fold_count = args.fold_count
    save_vectors = args.save_vectors
    evaluation = args.evaluation
    n_iterations = args.n_iterations

    print("wv_type:", wv_type, ";", "post_processing:", post_processing, ";", "method:", method, "new_wv_dim:", new_wv_dim)

    agg_dataset_path = os.path.join(word_sim_dir, 'combined_normalized_dataset.txt')
    df = pd.read_csv(agg_dataset_path, sep=' ', header=None, names=['word1', 'word2', 'normalized_score', 'source'])

    print('Learning Started!')
    start_time = time.time()

    if post_processing == "PPA" and (method == "WordFS-P" or method == "WordFS-S"):
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

    elif method == "Algo":
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

    elif method == "UMAP":
        print("Starting processing...")
        x_train = x_train_ori.copy()

        # UMAP Dimensionality Reduction
        umap_reducer = umap.UMAP(n_components=new_wv_dim)
        x_new_final = umap_reducer.fit_transform(x_train)

        new_word_vecs = {}
        for k, x in enumerate(x_train_names):
            new_word_vecs[x] = x_new_final[k]

        print('Learning Finished!')
        print('Building time: {:.2f} seconds'.format(time.time() - start_time))

    elif method == "None":
        new_word_vecs = {}
        for k, x in enumerate(x_train_names):
            new_word_vecs[x] = word_vecs[x].copy()

    elif method == "EmbedTextNet":
        EmbedTextNet = {}
        EmbedTextNet_path = "/mnt/jintang2/EmbedTextNet-main/Software_EmbedTextNet/Text_similarity/glove_reduced_embedding_300_to_50.txt"
        with open(EmbedTextNet_path, encoding="utf-8") as f:
            for line in f:
                EmbedTextNet_values = line.split()
                EmbedTextNet_word = EmbedTextNet_values[0]
                EmbedTextNet_coefs = np.asarray(EmbedTextNet_values[1:], dtype='float32')
                EmbedTextNet[EmbedTextNet_word] = EmbedTextNet_coefs
            f.close()

        new_word_vecs = EmbedTextNet

    if evaluation == 1:
        print("=================================================================================")
        print("%6s" % "Serial", "%20s" % "Dataset", "%15s" % "Num Pairs", "%15s" % "Not found", "%15s" % "Rho")
        print("=================================================================================")
        final_rho_values = []
        for iteration in range(n_iterations):
            print("-------------Start iteration:", iteration, "-------------")
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42+iteration)
            data_tuples = list(df.itertuples(index=False, name=None))
            Rho_val = 0
            for fold, (train_idx, test_idx) in enumerate(skf.split(df, df['source'])):
                train_data = [(word1, word2, sim_sc) for idx in train_idx for word1, word2, sim_sc, _ in [data_tuples[idx]]]
                test_data = [(word1, word2, sim_sc) for idx in test_idx for word1, word2, sim_sc, _ in [data_tuples[idx]]]
                manual_dict, auto_dict = ({}, {})
                not_found, total_size = (0, 0)
                x_train = []
                y_train = []

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

                        sorted_dimensions = np.argsort(dimension_correlations)[::-1]
                        dimension_correlations.sort()
                        feat_idx_selected = sorted_dimensions[:int(new_wv_dim)]

                    # Get unique elements along with their indices and counts
                    unique_elements, indices, counts = np.unique(feat_idx_selected, return_index=True, return_counts=True)

                    # Sort by indices to maintain original order
                    sorted_indices = np.argsort(indices)
                    unique_elements_sorted = unique_elements[sorted_indices]
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

                print("%6s" % str(fold + 1), "%20s" % "combined_dataset", "%15s" % str(total_size), "%15s" % str(not_found),
                      "%15.4f" % spearmans_rho(assign_ranks(manual_dict), assign_ranks(auto_dict)))
                Rho_val += spearmans_rho(assign_ranks(manual_dict), assign_ranks(auto_dict))

                # print(f"Fold {fold + 1}: Training on {len(train_data)} samples, Testing on {len(test_data)} samples")

            print("Final Rho:", Rho_val / fold_count)
            final_rho_values.append(Rho_val / fold_count)

        average_rho = np.mean(final_rho_values)
        print("Average Final Rho over iterations:", average_rho)
        print("wv_type:", args.wv_type, ";", "post_processing:", args.post_processing, ";", "method:", args.method)
        print(len(new_word_vecs[word1]))
        print(new_wv_dim)

    if save_vectors == 1:
        x_train = []
        y_train = []

        data_tuples = list(df.itertuples(index=False, name=None))
        train_data = [(word1, word2, sim_sc) for idx in np.arange(len(data_tuples)) for word1, word2, sim_sc, _ in
                      [data_tuples[idx]]]

        if method == "WordFS-P" or method == "WordFS-S":
            print('Start feature selection!')
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

            if method == "WordFS-S":
                dimension_correlations = Parallel(n_jobs=multiprocessing.cpu_count())(
                    delayed(compute_correlation)(k, x_train, y_train) for k in range(x_train.shape[-1])
                )

                # Rank dimensions based on correlation
                sorted_dimensions = np.argsort(dimension_correlations)[::-1]  # Sim
                dimension_correlations.sort()
                feat_idx_selected = sorted_dimensions[:int(new_wv_dim)]

                print('Finish Sim!')
                print('Sim time {:.2f} seconds:'.format(time.time() - start_time))

            if method == "WordFS-P":
                num_bins = 4
                rft = Reletive_Feat_Test()
                rft_loss = rft.get_rftloss_multidim(x_train, y_train, num_bins)
                RFT_feat_idx_selected = rft.get_selected_feat_idx_multidim(rft_loss, x_train.shape[-1])
                feat_idx_selected = RFT_feat_idx_selected[:int(new_wv_dim)]
    
                print('Finish RFT!')
                print('RFT time {:.2f} seconds:'.format(time.time() - start_time))

            unique_elements, indices, counts = np.unique(feat_idx_selected, return_index=True, return_counts=True)
            sorted_indices = np.argsort(indices)
            unique_elements_sorted = unique_elements[sorted_indices]
            x_new_final = x_train_ori[:, unique_elements_sorted]

            # print("feature_final:", X_new_final.shape)

            new_word_vecs = {}
            for k, x in enumerate(x_train_names):
                new_word_vecs[x] = x_new_final[k]

        if not evaluation:
            print('Learning Finished!')
            print('Building time: {:.2f} seconds'.format(time.time() - start_time))

        print('Saving vectors...')
        file_name = f"./test_wv/{post_processing}_{method}_{new_wv_dim}_{wv_type}.txt"  # This creates the file name dynamically

        with open(file_name, 'w', encoding='utf-8') as new_vectors_file:
            for word, vector in new_word_vecs.items():
                # Write the word to the file
                new_vectors_file.write("%s\t" % word)

                # Iterate through each component of the vector and write it to the file
                for component in vector:
                    new_vectors_file.write("%f\t" % component)

                # End the line for the current word vector
                new_vectors_file.write("\n")

        print(f"saved to ./test_wv/{post_processing}_{method}_{new_wv_dim}_{wv_type}.txt")


if __name__ == "__main__":
    main()
