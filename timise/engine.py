import os
import sys
import kimimaro
import h5py
import shutil
import fileinput
import numpy as np
import pandas as pd
from skimage.io import imread
import networkx as nx

from .mAP_3Dvolume.mAP_engine import print_mAP_stats, mAP_computation_fast
from .associations import calculate_associations_from_map, print_association_stats
from .matching import calculate_matching_metrics, print_matching_stats
from .utils import (Namespace, check_files, cable_length, create_map_aux_file_from_stats, create_map_groups_from_associations)
from .plotter import Plotter

class TIMISE:
    """TIMISE main class """
    def __init__(self, metrics=['mAP', 'associations', 'matching', 'pred_stats'], split_categories=['small', 'medium', 'large'],
                 split_property='cable_length', split_ths=[1000,4000], map_chunk_size=10, matching_stats_ths=[0.3, 0.5, 0.75],
                 log_prefix_str=''):
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

           matching_stats_ths : List of floats, optional
               Thresholds to be used when calculation matching metrics. Each one must be in [0,1) range 
               and refers to the minimum IoU to consider a TP. 

           log_prefix_str : str, optional
               Prefix to be prepended to all prints. 
        """
        if not isinstance(metrics, list):
            raise ValueError("'metrics' needs to be a list'")
        if len(metrics) == 0:
            raise ValueError("You need to select at least one metric to calculate")
        metrics = [x.lower() for x in metrics]
        for val in metrics:
            if val not in ['map', 'associations', 'matching', 'pred_stats']:
                raise ValueError("Available metrics: 'mAP', 'associations', 'matching' and 'pred_stats'. {} is unknown".format(val))

        if not split_categories is None:
            self.show_categories = True
            if not split_property is None:
                assert split_property in ['volume', 'cable_length']
            else:
                raise ValueError("'split_property' can not be None while setting 'split_categories'")
            if not split_ths is None:
                if len(split_ths)+1 != len(split_categories):
                    raise ValueError("'split_ths' needs to be one less in length than 'split_categories'")
            else:
                raise ValueError("'split_ths' can not be None while setting 'split_categories'")
        else:
            self.show_categories = False

        for val in matching_stats_ths:
            if not 0<= val < 1:
                raise ValueError("'matching_stats_ths' values need to be in [0,1) range.")

        self.log_prefix_str = log_prefix_str 
        self.metrics = metrics
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
        
        self.grouping_file = "group_file.txt"

        # mAP
        self.map_out_filename = "map_match_p.txt"
        self.precomputed_matching_file = "pred_gt_matching_info.csv"
        self.map_stats_file = "map_map.txt"
        self.map_aux_dir = "map_aux_files"

        # Statistic
        self.stats_pred_out_filename = "prediction_stats.csv"
        self.stats_gt_out_filename = "gt_stats.csv"

        # Association
        self.association_file = "associations.csv"
        self.association_stats_file = "associations_stats.csv"

        # Matching metrics
        self.matching_file = "matching_metrics.csv"
        self.matching_stats_ths = matching_stats_ths

        # Final gt errors
        self.final_errors_file = "gt_final.csv"

        self.pred_out_dirs = []
        self.pred_files = []
        self.method_names = []


    def evaluate(self, pred_dir, gt_dir, out_dir, data_resolution=[30,8,8]):
        self.data_resolution = data_resolution

        print(self.log_prefix_str+"####### Preliminary checks #######")
        if not os.path.isdir(pred_dir):
            raise FileNotFoundError("{} directory does not exist".format(pred_dir))
        if not os.path.isdir(gt_dir):
            raise FileNotFoundError("{} directory does not exist".format(gt_dir))
        else:
            self.gt_file = check_files(gt_dir)

        pfolder_ids = sorted(next(os.walk(pred_dir))[1])
        pfolder_ids = [os.path.join(pred_dir, p) for p in pfolder_ids]
        pfolder_ids = [p for p in pfolder_ids if os.path.normpath(p) != gt_dir]
        self.multiple_preds = True if len(pfolder_ids) > 1 else False
        if self.multiple_preds or len(pfolder_ids) == 1:
            print(self.log_prefix_str+"\tFound {} predictions: {}".format(len(pfolder_ids), pfolder_ids))
        else:
            print(self.log_prefix_str+"\tNo subfolders found, only considering files in {}".format(pred_dir))
            pfolder_ids = [pred_dir]

        os.makedirs(out_dir, exist_ok=True)

        #################
        # GT statistics #
        #################
        print(self.log_prefix_str+"####### GT statistics #######")
        gt_stats_out_file = os.path.join(out_dir, self.stats_gt_out_filename)
        print(self.log_prefix_str+"\tCalculating GT statistics . . .")
        self._get_file_statistics(self.gt_file, gt_stats_out_file)
        if not self.split_categories is None and self.split_property == 'cable_length' and 'map' in self.metrics:
            print(self.log_prefix_str+"\tCreating grouping aux file . . .")
            gt_map_th_aux_file = os.path.join(out_dir, "gt_"+self.grouping_file)
            create_map_aux_file_from_stats(gt_stats_out_file, gt_map_th_aux_file, cat=self.split_categories,
                log_prefix_str=self.log_prefix_str)
        else:
            gt_map_th_aux_file = ''

        print("\n\n"+self.log_prefix_str+"--- Start evaluation ---")
        for n, id_ in enumerate(pfolder_ids):
            print(self.log_prefix_str+"Processing folder {}".format(id_))

            pred_out_dir = os.path.join(out_dir, os.path.basename(os.path.normpath(id_)))
            self.pred_out_dirs.append(pred_out_dir)
            self.method_names.append(os.path.basename(os.path.normpath(id_)))
            map_out_file = os.path.join(pred_out_dir, self.map_out_filename)
            pred_grouping_file = os.path.join(pred_out_dir, "pred_"+self.grouping_file)
            map_aux_dir = os.path.join(pred_out_dir, self.map_aux_dir)
            precomputed_matching_file=os.path.join(pred_out_dir, self.precomputed_matching_file)
            matching_file = os.path.join(pred_out_dir, self.matching_file)
            stats_out_file = os.path.join(pred_out_dir, self.stats_pred_out_filename)
            final_error_file = os.path.join(pred_out_dir, self.final_errors_file)
            assoc_file = os.path.join(pred_out_dir, self.association_file)
            assoc_stats_file = os.path.join(pred_out_dir, self.association_stats_file)
            os.makedirs(pred_out_dir, exist_ok=True)

            # Ensure .tif/.h5 files are created
            pred_file = check_files(id_)
            self.pred_files.append(pred_file)


            ##########################
            # Predictions statistics #
            ##########################
            if 'pred_stats' in self.metrics or 'associations' in self.metrics:
                print(self.log_prefix_str+"####### Predictions statistics #######")
                if not os.path.exists(stats_out_file):        
                    print(self.log_prefix_str+"\tCalculating predictions statistics . . .")
                    self._get_file_statistics(pred_file, stats_out_file)
                else:
                    print(self.log_prefix_str+"\tSkipping predictions statistics calculation (seems to be done here: {} )".format(stats_out_file))
                

            ################
            # Associations #
            ################
            if not os.path.exists(precomputed_matching_file) or not os.path.exists(assoc_file):
                print(self.log_prefix_str+"####### Computing matching between predictions and ground truth #######")
                args = Namespace(gt_seg=self.gt_file, predict_seg=pred_file, gt_bbox='', predict_score='', predict_bbox='',
                                predict_heatmap_channel=-1, threshold=self.map_th, threshold_crumb=0,
                                chunk_size=self.map_chunk_size, output_name=os.path.join(pred_out_dir, "map"),
                                group_gt='', group_pred='', do_txt=1, do_eval=1, slices=-1, 
                                matching_out_file=precomputed_matching_file, associations_file=assoc_file, 
                                aux_dir=map_aux_dir, log_prefix_str=self.log_prefix_str, associations=True)
                mAP_computation_fast(args)         

            if 'associations' in self.metrics:
                print(self.log_prefix_str+"####### Computing association metrics #######")
                if not os.path.exists(final_error_file): 
                    print(self.log_prefix_str+"\tCalculating associations . . .")
                    calculate_associations_from_map(assoc_file, gt_stats_out_file, assoc_stats_file, final_error_file,
                        self.log_prefix_str)
                else:
                    print(self.log_prefix_str+"\tSkipping association calculation (seems to be done here: {} )".format(final_error_file))

            # Creating the grouping file
            print(self.log_prefix_str+"####### Grouping file #######")
            cat = self.split_categories if self.split_categories is not None else ['all']
            if not os.path.exists(pred_grouping_file):
                create_map_groups_from_associations(map_aux_dir, gt_stats_out_file, assoc_file, pred_grouping_file,
                    cat=cat, log_prefix_str=self.log_prefix_str)   

            #######
            # mAP #
            #######
            if 'map' in self.metrics:
                print(self.log_prefix_str+"####### Computing mAP #######")
                if not os.path.exists(map_out_file):
                    print(self.log_prefix_str+"\tRun mAP code . . .")
                    args = Namespace(gt_seg=self.gt_file, predict_seg=pred_file, gt_bbox='', predict_score='', predict_bbox='',
                                predict_heatmap_channel=-1, threshold=self.map_th, threshold_crumb=0,
                                chunk_size=self.map_chunk_size, output_name=os.path.join(pred_out_dir, "map"),
                                group_gt=gt_map_th_aux_file, group_pred=pred_grouping_file, do_txt=1, do_eval=1, 
                                slices=-1, matching_out_file=precomputed_matching_file, 
                                associations_file=assoc_file, aux_dir=map_aux_dir, log_prefix_str=self.log_prefix_str, 
                                associations=False)
                    mAP_computation_fast(args)  
                else:
                    print(self.log_prefix_str+"\tSkipping mAP calculation (seems to be done here: {} )".format(map_out_file))

            ####################
            # Matching metrics #
            ####################
            if 'matching' in self.metrics:
                print(self.log_prefix_str+"####### Computing matching metrics #######")
                if not os.path.exists(matching_file):
                    print(self.log_prefix_str+"\tCalculating matching metrics . . .")
                    calculate_matching_metrics(matching_file, pred_grouping_file, self.split_categories, 
                        precomputed_matching_file=precomputed_matching_file, gt_stats_file=gt_stats_out_file,  
                        thresh=self.matching_stats_ths, log_prefix_str=self.log_prefix_str)
                else:
                    print(self.log_prefix_str+"\tSkipping matching metrics calculation (seems to be done here: {} )".format(matching_file))

        print("\n"+self.log_prefix_str+"--- End evaluation ---")


    def summary(self):
        if len(self.pred_out_dirs) == 0:
            raise ValueError("No data found. Did you call TIMISE.evaluate()?")

        for f in self.pred_out_dirs:
            print(self.log_prefix_str+"Stats in {}".format(f))
            print('')
            if 'map' in self.metrics:
                print_mAP_stats(os.path.join(f, self.map_stats_file))
                print('')
            if 'associations' in self.metrics:
                print_association_stats(os.path.join(f, self.association_stats_file), 
                                        os.path.join(f,self.map_aux_dir), self.show_categories)
                print('')
            if 'matching' in self.metrics:
                print_matching_stats(os.path.join(f, self.matching_file), self.show_categories)


    def plot(self, plot_type='error_pie', show=True, individual_plots=False, nbins=30, draw_std=True,
             color_by="association_type", symbol="category", draw_plane=True, xaxis_range=None,
             yaxis_range=None, log_x=False, log_y=False, font_size=25, order=[], plot_shape=[1100,500]):
        """Plot errors in different formats. When multiple predictions are available a common plot is created.

           Parameters
           ----------
           plot_type : str, optional
               Type of plot to be visualized/created. Options are ['error_2d', 'error_3d'].

           individual_plots: bool, optional
               Force the creation of individual error plots (2D and 3D) apart of the common plot when multiple
               predictions are available.

           show : bool, optional
               Wheter to show or not the plot after saving.

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

           xaxis_range : array of 2 floats, optional
               Range of x axis.

           yaxis_range : array of 2 floats, optional
               Range of x axis.

           log_x : bool, optional
               Wheter to apply log into x axis. Applied when plot_type is 'error_2d' or 'error_3d'.

           log_y : bool, optional
               Wheter to apply log into x axis. Applied when plot_type is 'error_2d' or 'error_3d'.

           font_size : int, optional
               Size of the font to be used in the plots. 

           order : list of str, optional
               Order each prediction based on a given list. The names need to match the names used for
               each prediction folder. E.g ['prediction1', 'prediction2'].

           plot_shape : 2d array of ints, optional
               Defines the shape of the plot.
        """
        if not 'associations' in self.metrics:
            raise ValueError("You need to compute associations to create the neuroglancer file. "
                             "TIMISE(metrics=['associations'])")

        assert plot_type in ['assoc_error_2d', 'assoc_error_3d', 'assoc_stats', 'error_bar', 'error_pie']
        assert color_by in ['association_type', 'category']
        assert symbol in ['association_type', 'category']
        if len(plot_shape) != 2:
            raise ValueError("'plot_shape' needs to have 2 values: [width, height]")

        plotter = Plotter(xaxis_range=xaxis_range, yaxis_range=yaxis_range, match_th=self.matching_stats_ths[-1])

        final_file = os.path.join(self.pred_out_dirs[0], self.final_errors_file)
        assoc_file = os.path.join(self.pred_out_dirs[0], self.association_stats_file) 
        matching_file = os.path.join(self.pred_out_dirs[0], self.matching_file) 
        plot_dir = os.path.join(self.pred_out_dirs[0], "plots")

        if self.multiple_preds:
            if plot_type == 'error_pie':
                plotter.error_pie_multiple_predictions(self.pred_out_dirs, self.association_stats_file, self.matching_file,
                    show=show)
            elif plot_type == 'error_bar':
                plotter.error_bar_multiple_predictions(self.pred_out_dirs, self.association_stats_file, self.matching_file, 
                    show=show, order=order, shape=plot_shape)
            elif plot_type == 'assoc_stats':
                plotter.association_multiple_predictions(self.pred_out_dirs, self.association_stats_file,
                    show=show, show_categories=self.show_categories, order=order, shape=plot_shape)

        if individual_plots or not self.multiple_preds:
            os.makedirs(plot_dir, exist_ok=True)

            if not self.split_categories is None:
                if plot_type == 'assoc_error_3d':
                    plotter.association_plot_3d(final_file, plot_dir, show=show, draw_plane=draw_plane,
                        log_x=log_x, log_y=log_y, color=color_by, symbol=symbol, font_size=font_size, shape=plot_shape)
            if plot_type == 'assoc_error_2d':
                plotter.association_plot_2d(final_file, plot_dir, show=show, log_x=log_x, log_y=log_y, bins=nbins,
                    draw_std=draw_std, font_size=font_size, shape=plot_shape)
            elif plot_type == 'error_pie':
                plotter.error_pie(assoc_file, matching_file, plot_dir, show=show, hide_correct=True)

    def create_neuroglancer_file(self, method_name, categories=['all']):
        """Create a python script to visualize a method prediction in neuroglancer. 

           Parameters
           ----------
           method_name : str
               Name of the method to visualize. Set it to "GT" to visualize the ground truth. 
               
           categories : list of str, optional
               Categories of instances to be selected for the visualization.
        """
        if not 'pred_stats' in self.metrics:
            raise ValueError("You need to compute prediction stats to create the neuroglancer file. "
                             "TIMISE(metrics=['pred_stats'])")
        if len(self.pred_out_dirs) == 0:
            raise ValueError("No data found. Did you call TIMISE.evaluate()?")
        if method_name.lower() != "gt":
            if method_name not in self.method_names :
                raise ValueError("{} method not recognized. Available: {}".format(method_name, self.method_names))

            method_id = self.method_names.index(method_name)
            input_file = self.pred_files[method_id]
            df_file = os.path.join(self.pred_out_dirs[method_id], self.stats_pred_out_filename)
        else:
            input_file = self.gt_file
            df_file = os.path.join(os.path.dirname(self.pred_out_dirs[0]), self.stats_gt_out_filename)

        df = pd.read_csv(df_file, index_col=False) 
        if categories == ['all']:      
            label_ids = df['label'].tolist()
            cat_names = '_all'
        else:
            cat_names = ''
            label_ids = []
            for cat in categories:
                label_ids += df[df['category']==cat]['label'].tolist()
                cat_names+='_'+cat
        label_ids = str(label_ids)[1:-1]

        f = os.path.join(os.path.dirname(self.pred_out_dirs[0]), "neuroglancer_"+str(method_name)+cat_names+".py")
        shutil.copyfile("examples/neuroglancer/template.py", f)

        # Replace input file entry, resolution and selected instance labels
        for line in fileinput.input([f], inplace=True):
            if line.strip().startswith('input_file='):
                line = "input_file=\""+input_file+"\"\n"
            elif line.strip().startswith('scales=SCALE)'):
                line = "        scales="+str(self.data_resolution)+")\n"
            elif line.strip().startswith("s.layers[\"segmentation\"].layer.segments ="):
                line = "    s.layers[\"segmentation\"].layer.segments = {"+label_ids+"}\n"
            sys.stdout.write(line)
        
        print(self.log_prefix_str+"Neuroglancer script created in {}".format(f))
        

    def _get_file_statistics(self, input_file, out_csv_file):
        """Calculate instances statistics such as volume, skeleton size and cable length."""
        if not os.path.exists(out_csv_file):
            print(self.log_prefix_str+"\tReading file {} . . .".format(input_file))
            if str(input_file).endswith('.h5'):
                h5f = h5py.File(input_file, 'r')
                k = list(h5f.keys())
                img = np.array(h5f[k[0]])
            else:
                img = imread(input_file)

            print(self.log_prefix_str+"\tCalculating volumes . . .")
            values, volumes = np.unique(img, return_counts=True)
            values=values[1:].tolist()
            volumes=volumes[1:].tolist()

            print(self.log_prefix_str+"\tSkeletonizing . . .")
            skels = kimimaro.skeletonize(img, parallel=0, parallel_chunk_size=100, dust_threshold=0)
            keys = list(skels.keys())

            print(self.log_prefix_str+"\tCalculating cable length . . .")
            del img
            c_length = []
            skel_size = []
            vol = []
            bifurcations = []
            holes = []
            for label in keys:
                ind_skel = skels[label]
                vertices = ind_skel.vertices

                # Cable length
                l = cable_length(ind_skel.vertices, ind_skel.edges, res = self.data_resolution)
                c_length.append(l)
                skel_size.append(ind_skel.vertices.shape[0])
                vol.append(volumes[values.index(label)])

                G = nx.Graph()
                G.add_edges_from(ind_skel.edges)
                
                # Bifurcations
                nbifurcations = 0
                for node in range(len(ind_skel.vertices)):
                    if len(list(G.neighbors(node))) > 2:
                        nbifurcations += 1
                bifurcations.append(nbifurcations)

                # Holes
                holes.append(len(nx.cycle_basis(G)))

            # Kimimaro drops very tiny instances (e.g. 1 pixel instances). To be coherent later on we need those also
            # so we check which instances where not processed
            for v in values:
                if not v in keys:
                   keys.append(v)
                   vol.append(volumes[values.index(v)])
                   c_length.append(1)
                   skel_size.append(1)

            data_tuples = list(zip(keys,vol,skel_size,c_length,bifurcations,holes))
            dataframe = pd.DataFrame(data_tuples, columns=['label','volume','skel_size','cable_length','bifurcations','holes'])
        else:
            print(self.log_prefix_str+"\tSkipping GT statistics calculation (seems to be done here: {} )".format(out_csv_file))
            dataframe = pd.read_csv(out_csv_file, index_col=False)

        if not self.split_categories is None and not 'categories' in dataframe.columns:
            print(self.log_prefix_str+"\tAdding categories information . . .")
            dataframe['category'] = self.split_categories[0]
            for i in range(len(self.split_ths)):
                dataframe.loc[dataframe[self.split_property] >= self.split_ths[i], "category"] = self.split_categories[i+1]
            dataframe = dataframe.sort_values(by=[self.split_property])
        else:
            dataframe = dataframe.sort_values(by=['volume'])
        print(self.log_prefix_str+"\tSaving statistics in {}".format(out_csv_file))
        dataframe.to_csv(out_csv_file, index=False)

