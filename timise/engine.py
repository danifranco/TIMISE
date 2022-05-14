import os
import kimimaro
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.io import imread

from .mAP_3Dvolume.mAP_engine import mAP_computation
from .associations import calculate_associations, print_association_stats
from .utils import Namespace, prepare_files, cable_length, mAP_out_to_dataframe

class TIMISE:
    """TIMISE main class """
    def __init__(self, map_th="5000,30000", map_th_crumb=2000, map_chunk_size=10):
        self.map_th = map_th
        self.map_th_crumb = map_th_crumb
        self.map_chunk_size = map_chunk_size

        # mAP
        self.map_out_filename = "map_match_p.txt"
        self.map_out_csv = "map.csv"

        # Statistic
        self.stats_pred_out_filename = "prediction_stats.csv"
        self.stats_gt_out_filename = "gt_stats.csv"

        # Association
        self.association_file = "associations.csv"
        self.association_stats_file = "associations_stats.csv"

        self.pred_out_dirs = []

    def evaluate(self, pred_dir, gt_dir, out_dir, data_resolution=[30,8,8], multiple_preds=False, verbose=True):
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

        if multiple_preds:
            pfolder_ids = sorted(next(os.walk(pred_dir))[1])
            pfolder_ids = [os.path.join(pred_dir, p) for p in pfolder_ids]
            pfolder_ids = [p for p in pfolder_ids if os.path.normpath(p) != gt_dir]
            if verbose: print("Found {} predictions: {}".format(len(pfolder_ids), pfolder_ids))
        else:
            pfolder_ids = [pred_dir]
        print("*** [DONE] Preliminary checks . . .")


        print("*** Evaluating . . .")
        for n, id_ in enumerate(pfolder_ids):
            print("Processing folder {}".format(id_))
            pred_files = sorted(next(os.walk(id_))[2])

            pred_out_dir = os.path.join(out_dir, os.path.basename(os.path.normpath(id_)))
            self.pred_out_dirs.append(pred_out_dir)
            gt_stats_out_file = os.path.join(out_dir, self.stats_gt_out_filename)
            map_out_file = os.path.join(pred_out_dir, self.map_out_filename)
            folder_association_file = os.path.join(pred_out_dir, self.association_file)
            stats_out_file = os.path.join(pred_out_dir, self.stats_pred_out_filename)
            os.makedirs(pred_out_dir, exist_ok=True)

            # Ensure .tif/.h5 files are created
            pred_h5_file, pred_tif_file = prepare_files(id_, verbose=verbose)


            #################
            # GT statistics #
            #################
            if not os.path.exists(gt_stats_out_file):
                print("Calculating GT statistics . . .")
                self._get_file_statistics(self.gt_tif_file, gt_stats_out_file)
            else:
                print("Skipping GT statistics calculation (seems to be done here: {} )".format(gt_stats_out_file))


            #######
            # mAP #
            #######
            if not os.path.exists(map_out_file):
                print("Run mAP code . . .")
                args = Namespace(gt_seg=self.gt_h5_file, predict_seg=pred_h5_file, predict_score='',
                                 predict_heatmap_channel=-1, threshold=self.map_th, threshold_crumb=self.map_th_crumb,
                                 chunk_size=self.map_chunk_size, output_name=os.path.join(pred_out_dir, "map"),
                                 do_txt=1, do_eval=1, slices=-1, verbose=verbose)
                mAP_computation(args)

                mAP_out_to_dataframe(map_out_file, os.path.join(pred_out_dir, self.map_out_csv), self.verbose)
            else:
                print("Skipping mAP calculation (seems to be done here: {} )".format(map_out_file))


            ################
            # Associations #
            ################
            if not os.path.exists(folder_association_file):
                print("Calculating associations . . .")
                calculate_associations(pred_tif_file, self.gt_tif_file, pred_out_dir, self.verbose)
            else:
                print(pred_tif_file)
                print("Skipping association calculation (seems to be done here: {} )".format(folder_association_file))


            ##########################
            # Predictions statistics #
            ##########################
            if not os.path.exists(stats_out_file):
                print("Calculating predictions statistics . . .")
                self._get_file_statistics(pred_tif_file, stats_out_file)
            else:
                print("Skipping predictions statistics calculation (seems to be done here: {} )".format(stats_out_file))

        print("*** [DONE] Evaluating . . .")


    def summary(self):
        if len(self.pred_out_dirs) == 0:
            raise ValueError("No data found. Did you call TIMISE.evaluate()?")

        for f in self.pred_out_dirs:
            print("Stats in {}".format(f))
            print('')
            print_association_stats(os.path.join(f, self.association_stats_file))


    def _get_file_statistics(self, input_file, out_csv_file):
        """Calculate instances statistics such as volume, skeleton size and cable length."""
        if self.verbose: print("Reading file {} . . .".format(input_file))
        img = imread(input_file)

        if self.verbose: print("Calculating volumes . . .")
        values, volumes = np.unique(img, return_counts=True)
        values=values[1:]
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
        dataframe = dataframe.sort_values(by=['volume'])
        dataframe.to_csv(out_csv_file, index=False)

