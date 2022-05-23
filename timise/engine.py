import os
import kimimaro
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.io import imread

from .mAP_3Dvolume.mAP_engine import mAP_computation, print_mAP_stats
from .associations import calculate_associations, print_association_stats, association_plot_2d, association_plot_3d
from .utils import Namespace, prepare_files, cable_length, mAP_out_to_dataframe

class TIMISE:
    """TIMISE main class """
    def __init__(self, split_categories=None, split_property=None, split_ths=None, map_chunk_size=10):
        """TIMISE class initialization.

           Parameters
           ----------
           split_categories : array of str, optional
               Categories to divide the instances to be analized. E.g. ['small', 'medium', 'large'].

           split_property : str, optional
               Property of the instances to consider when spliting into categories. Possible values 'volume',
               'skel_size' and 'cable_length'.

           split_ths : array of float/int, optional
               Thresholds to divide the instances into categories. Its length is 1 less than split_categories.
               E.g. [1000, 4000] means: 0 < small < 1000, 1000 <= medium < 4000, 4000 <= large.

           map_chunk_size : int, optional
               How many slices to load (for memory-efficient computation).
        """

        if not split_categories is None:
            if not split_property is None:
                assert split_property in ['volume', 'skel_size', 'cable_length']
            else:
                raise ValueError("'split_property' can not be None while setting 'split_categories'")
            if not split_ths is None:
                if len(split_ths)+1 != len(split_categories):
                    raise ValueError("'split_ths' needs to be one less in length than 'split_categories'")
            else:
                raise ValueError("'split_ths' can not be None while setting 'split_categories'")

        self.split_categories = split_categories
        self.split_property = split_property
        self.split_ths = split_ths
        if split_ths is not None:
            s = ''
            for th in split_ths:
                s += str(th)+','
            s = s[:-1] # Remove the last comma
            self.map_th = s
        else:
            self.map_th = "5000,30000"
        self.map_chunk_size = map_chunk_size

        # mAP
        self.map_out_filename = "map_match_p.txt"
        self.map_out_csv = "map.csv"
        self.map_stats_file = "map_map.txt"

        # Statistic
        self.stats_pred_out_filename = "prediction_stats.csv"
        self.stats_gt_out_filename = "gt_stats.csv"

        # Association
        self.association_file = "associations.csv"
        self.association_stats_file = "associations_stats.csv"

        # Final gt errors
        self.final_errors_file = "gt_final.csv"

        self.pred_out_dirs = []

    def evaluate(self, pred_dir, gt_dir, out_dir, data_resolution=[30,8,8], verbose=True):
        self.data_resolution = data_resolution
        self.verbose = verbose
        self.pred_out_dirs = []

        print("*** Preliminary checks . . . ")
        if not os.path.isdir(pred_dir):
            raise FileNotFoundError("{} directory does not exist".format(pred_dir))
        if not os.path.isdir(gt_dir):
            raise FileNotFoundError("{} directory does not exist".format(pred_dir))
        else:
            self.gt_h5_file, self.gt_tif_file = prepare_files(gt_dir, verbose=verbose)

        pfolder_ids = sorted(next(os.walk(pred_dir))[1])
        pfolder_ids = [os.path.join(pred_dir, p) for p in pfolder_ids]
        pfolder_ids = [p for p in pfolder_ids if os.path.normpath(p) != gt_dir]
        self.multiple_preds = True if len(pfolder_ids) > 1 else False
        if self.multiple_preds or len(pfolder_ids) == 1:
            if verbose: print("Found {} predictions: {}".format(len(pfolder_ids), pfolder_ids))
        else:
            if verbose: print("No subfolders found, only considering files in {}".format(pred_dir))
            pfolder_ids = [pred_dir]

        print("*** [DONE] Preliminary checks . . .")

        os.makedirs(out_dir, exist_ok=True)

        #################
        # GT statistics #
        #################
        gt_stats_out_file = os.path.join(out_dir, self.stats_gt_out_filename)
        if not os.path.exists(gt_stats_out_file):
            print("Calculating GT statistics . . .")
            self._get_file_statistics(self.gt_tif_file, gt_stats_out_file)
        else:
            print("Skipping GT statistics calculation (seems to be done here: {} )".format(gt_stats_out_file))


        print("*** Evaluating . . .")
        for n, id_ in enumerate(pfolder_ids):
            print("Processing folder {}".format(id_))
            pred_files = sorted(next(os.walk(id_))[2])

            pred_out_dir = os.path.join(out_dir, os.path.basename(os.path.normpath(id_)))
            self.pred_out_dirs.append(pred_out_dir)
            map_out_file = os.path.join(pred_out_dir, self.map_out_filename)
            folder_association_file = os.path.join(pred_out_dir, self.association_file)
            stats_out_file = os.path.join(pred_out_dir, self.stats_pred_out_filename)
            final_error_file = os.path.join(pred_out_dir, self.final_errors_file)
            os.makedirs(pred_out_dir, exist_ok=True)

            # Ensure .tif/.h5 files are created
            pred_h5_file, pred_tif_file = prepare_files(id_, verbose=verbose)


            #######
            # mAP #
            #######
            if not os.path.exists(map_out_file):
                print("Run mAP code . . .")
                args = Namespace(gt_seg=self.gt_h5_file, predict_seg=pred_h5_file, predict_score='',
                                 predict_heatmap_channel=-1, threshold=self.map_th, threshold_crumb=0,
                                 chunk_size=self.map_chunk_size, output_name=os.path.join(pred_out_dir, "map"),
                                 do_txt=1, do_eval=1, slices=-1, verbose=verbose)
                mAP_computation(args)

                mAP_out_to_dataframe(map_out_file, os.path.join(pred_out_dir, self.map_out_csv), self.verbose)
            else:
                print("Skipping mAP calculation (seems to be done here: {} )".format(map_out_file))


            ##########################
            # Predictions statistics #
            ##########################
            if not os.path.exists(stats_out_file):
                print("Calculating predictions statistics . . .")
                self._get_file_statistics(pred_tif_file, stats_out_file)
            else:
                print("Skipping predictions statistics calculation (seems to be done here: {} )".format(stats_out_file))


            ################
            # Associations #
            ################
            if not os.path.exists(folder_association_file):
                print("Calculating associations . . .")
                calculate_associations(pred_tif_file, self.gt_tif_file, gt_stats_out_file, final_error_file,
                                       self.verbose)
            else:
                print(pred_tif_file)
                print("Skipping association calculation (seems to be done here: {} )".format(folder_association_file))

        print("*** [DONE] Evaluating . . .")


    def summary(self):
        if len(self.pred_out_dirs) == 0:
            raise ValueError("No data found. Did you call TIMISE.evaluate()?")

        for f in self.pred_out_dirs:
            print("Stats in {}".format(f))
            print('')
            print_mAP_stats(os.path.join(f, self.map_stats_file))
            print('')
            print_association_stats(os.path.join(f, self.association_stats_file))


    def plot(self, plot_type='error_2d', individual_plots=False, nbins=30, draw_std=True, color_by="association_type",
             symbol="tag", draw_plane=True, log_x=True, log_y=True):
        """Plot errors in different formats. When multiple predictions are available a common plot is created.

           Parameters
           ----------
           plot_type : str, optional
               Type of plot to be visualized/created. Options are ['error_2d', 'error_3d'].

           individual_plots: bool, optional
               Force the creation of individual error plots (2D and 3D) apart of the common plot when multiple
               predictions are available.

           nbins : int, optional
               Number of bins to gather the values into. Only applied when plot_type is 'error_2d'.

           draw_std : bool, optional
               Whether to draw the standard deviation. Only applied when plot_type is 'error_2d'.

           color_by : str, optional
               Property to be used for coloring. Only applied when plot_type is 'error_3d'.

           symbol : str, optional
               Property to be used for make the symbols. Only applied when plot_type is 'error_3d'.

           draw_plane : bool, optional
               Whether to draw the plane on Z=0. Only applied when plot_type is 'error_3d'.

           log_x : bool, optional
               Wheter to apply log into x axis. Applied when plot_type is 'error_2d' or 'error_3d'.

           log_y : bool, optional
               Wheter to apply log into x axis. Applied when plot_type is 'error_2d' or 'error_3d'.
        """

        assert plot_type in ['error_2d', 'error_3d']
        assert color_by in ['association_type', 'tag']
        assert symbol in ['association_type', 'tag']

        if self.multiple_preds:
            if not split_categories is None:
                for f in self.pred_out_dirs:
                    print("Processing folder {}".format(f))
            else:
                print("??")

        if individual_plots or not self.multiple_preds:
            if not self.split_categories is None:
                assoc_file = os.path.join(self.pred_out_dirs[0], self.final_errors_file)
                if plot_type == 'error_3d':
                    association_plot_3d(assoc_file, self.pred_out_dirs[0], draw_plane=draw_plane, log_x=log_x,
                                        log_y=log_y, color=color_by, symbol=symbol)
                    # color = tag , symbol = association_type
                elif plot_type == 'error_2d':
                    association_plot_2d(assoc_file, self.pred_out_dirs[0], log_x=log_x, log_y=log_y, bins=nbins,
                                        draw_std=draw_std)
            else:
                print("??")


    def _get_file_statistics(self, input_file, out_csv_file):
        """Calculate instances statistics such as volume, skeleton size and cable length."""
        if self.verbose: print("Reading file {} . . .".format(input_file))
        img = imread(input_file)

        if self.verbose: print("Calculating volumes . . .")
        values, volumes = np.unique(img, return_counts=True)
        values=values[1:].tolist()
        volumes=volumes[1:]

        if self.verbose: print("Skeletonizing . . .")
        skels = kimimaro.skeletonize(img, parallel=0, parallel_chunk_size=100, dust_threshold=0)
        keys = list(skels.keys())
        self.verbose: print("Create skeleton image . . .")
        s = img.shape
        del img
        out = np.zeros(s, dtype=np.uint16)
        c_length = []
        for label in keys:
            ind_skel = skels[label]
            vertices = ind_skel.vertices

            # Fill skeleton image
            for i in range(len(vertices)):
                v = vertices[i]
                z, x, y = int(v[0]), int(v[1]), int(v[2])
                out[z,x,y] = label

            # Cable length
            l = cable_length(ind_skel.vertices, ind_skel.edges, res = self.data_resolution)
            c_length.append(l)

        self.verbose: print("Obtaining skeleton size . . .")
        _, skel_sizes = np.unique(out, return_counts=True)
        skel_size=skel_sizes[1:]

        data_tuples = list(zip(values,volumes,skel_size,c_length))
        dataframe = pd.DataFrame(data_tuples, columns=['label','volume','skel_size','cable_length'])
        if not self.split_categories is None:
            dataframe['tag'] = self.split_categories[0]
            for i in range(len(self.split_ths)):
                dataframe.loc[dataframe[self.split_property] >= self.split_ths[i], "tag"] = self.split_categories[i+1]
            dataframe = dataframe.sort_values(by=[self.split_property])
        else:
            dataframe = dataframe.sort_values(by=['volume'])
        dataframe.to_csv(out_csv_file, index=False)

