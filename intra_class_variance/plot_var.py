import os
import matplotlib.pyplot as plt


class PlotVar:
    def get_shape(self, i):
        dict_ = {'*': 'star',
                 '+': 'plus',
                 ',': 'pixel',
                 '.': 'point',
                 '3': 'tri_left',
                 '<': 'triangle_left',
                 'D': 'diamond',
                 'H': 'hexagon2',
                 'P': 'plus_filled',
                 'X': 'x_filled',
                 '_': 'hline',
                 'd': 'thin_diamond',
                 'h': 'hexagon1',
                 'o': 'circle',
                 'p': 'pentagon',
                 's': 'square',
                 'x': 'x',
                 '|': 'vline',
                 '>': 'triangle_right',
                 '8': 'octagon',
                 0: 'tickleft',
                 1: 'tickright',
                 10: 'caretupbase',
                 11: 'caretdownbase',
                 4: 'caretleft',
                 5: 'caretright',
                 8: 'caretleftbase',
                 9: 'caretrightbase',
                 '2': 'tri_up',
                 '1': 'tri_down',
                 '4': 'tri_right',
                 '^': 'triangle_up',
                 'v': 'triangle_down',
                 2: 'tickup',
                 3: 'tickdown',
                 6: 'caretup',
                 7: 'caretdown',
                 'None': 'nothing',
                 'none': 'nothing',
                 }
        return [k for k, v in dict_.items()][i]

    def get_color(self, split, label):
        dict_ = {
            "train_0": "red",
            "train_1": "green",
            "test_0": "blue",
            "test_1": "magenta",
            "val_0":  "cyan",
            "val_1": "black",
            # "red","green","blue","yellow","pink","black","orange","purple","beige","brown","gray","cyan","magenta"
        }
        return dict_["_".join([split, str(label)])]

    def plot_intra_class_var(self, df):

        path = "results_intra_class_variance"
        os.makedirs(path, exist_ok=True)
        df = self.normalize(df)
        # according to dataset
        for metric_name in df["metric_name"].unique():  # NOTE plot 따로
            metric_df = df[(df["metric_name"] == metric_name)]
            self.plot_inter_dataset(metric_df, metric_name, path)
            self.plot_intra_dataset(metric_df, metric_name, path)

    def normalize(self, df):
        from sklearn.preprocessing import QuantileTransformer, PowerTransformer
        # normalization Quantile and normal dist
        # df["mean"] = QuantileTransformer().fit_transform( df["mean"].values.reshape(-1, 1))
        # normalization PowerTransform
        for metric_name in df["metric_name"].unique():
            for level in df["level"].unique():
                for label in df["label"].unique():
                    condition = (df.metric_name == metric_name) & (
                        df.level == level) & (df.label == label)
                    df.loc[condition, "mean"] = PowerTransformer().fit_transform(
                        df[condition]["mean"].values.reshape(-1, 1))
        return df

    def save_fig(self, path, metric_name):
        # set legend
        plt.legend(loc='center left', bbox_to_anchor=(
            1, 0.5))  # 으래 왼쪽이 그림의 y 축 중앙 (0.5) 에 오도록
        plt.title(f"{metric_name}")
        plt.xlabel("image-level")
        plt.ylabel("patch-level")
        # set resolution of png
        plt.savefig(os.path.join(
            path, f"{metric_name}.png"), dpi=300, bbox_inches='tight')

        plt.clf()
        plt.cla()

    def plot_inter_dataset(self, metric_df, metric_name, path):
        """
        Dataset 간 관계를 보기 위한 plot
        """
        # 문자를 다르게
        for i, dataset in enumerate(sorted(metric_df["dataset"].unique())):
            dataset_df = metric_df[metric_df["dataset"] == dataset]
            # for split in dataset_df["split"].unique():  # split_label 합쳐서 색을 다르게
            #    split_df = dataset_df[dataset_df["split"] == split]
            split = "train"
            label_df = dataset_df
            for label in label_df["label"].unique():
                img_df = label_df[(label_df["label"] == label) & (
                    label_df["level"] == "image-level")]
                img_val = img_df["mean"].mean().item()
                patch_df = label_df[(label_df["label"] == label) & (
                    label_df["level"] == "patch-level")]
                patch_val = patch_df["mean"].mean().item()

                plt.scatter(img_val, patch_val,
                            c=self.get_color(split, label), marker=self.get_shape(i), label=f"{dataset}_{label}")
        self.save_fig(path, metric_name)

    def plot_intra_dataset(self, metric_df, metric_name, path):
        """
        Dataset 내부 split, label 간 관계를 보기 위한 plot
        """
        # 문자를 다르게
        for i, dataset in enumerate(sorted(metric_df["dataset"].unique())):
            dataset_df = metric_df[metric_df["dataset"] == dataset]
            for split in dataset_df["split"].unique():  # split_label 합쳐서 색을 다르게
                split_df = dataset_df[dataset_df["split"] == split]
                label_df = split_df
                for label in label_df["label"].unique():
                    img_df = label_df[(label_df["label"] == label) & (
                        label_df["level"] == "image-level")]
                    img_val = img_df["mean"].mean().item()
                    patch_df = label_df[(label_df["label"] == label) & (
                        label_df["level"] == "patch-level")]
                    patch_val = patch_df["mean"].mean().item()

                    plt.scatter(img_val, patch_val,
                                c=self.get_color(split, label), marker=self.get_shape(i), label=f"{dataset}_{split}_{label}")

            self.save_fig(path, f"{metric_name}_{dataset}")
