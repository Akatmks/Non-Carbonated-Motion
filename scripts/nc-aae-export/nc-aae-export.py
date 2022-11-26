# nc-aae-export.py
# Copyright (c) Akatsumekusa

#  
#        ::::    :::  ::::::::  ::::    :::                                                                            
#       :+:+:   :+: :+:    :+: :+:+:   :+:                                                                             
#      :+:+:+  +:+ +:+    +:+ :+:+:+  +:+                                                                              
#     +#+ +:+ +#+ +#+    +:+ +#+ +:+ +#+                                                                               
#    +#+  +#+#+# +#+    +#+ +#+  +#+#+#                                                                                
#   #+#   #+#+# #+#    #+# #+#   #+#+#                                                                                 
#  ###    ####  ########  ###    ####                                                                                  
#        ::::::::      :::     :::::::::  :::::::::   ::::::::  ::::    :::     ::: ::::::::::: :::::::::: :::::::::   
#      :+:    :+:   :+: :+:   :+:    :+: :+:    :+: :+:    :+: :+:+:   :+:   :+: :+:   :+:     :+:        :+:    :+:   
#     +:+         +:+   +:+  +:+    +:+ +:+    +:+ +:+    +:+ :+:+:+  +:+  +:+   +:+  +:+     +:+        +:+    +:+    
#    +#+        +#++:++#++: +#++:++#:  +#++:++#+  +#+    +:+ +#+ +:+ +#+ +#++:++#++: +#+     +#++:++#   +#+    +:+     
#   +#+        +#+     +#+ +#+    +#+ +#+    +#+ +#+    +#+ +#+  +#+#+# +#+     +#+ +#+     +#+        +#+    +#+      
#  #+#    #+# #+#     #+# #+#    #+# #+#    #+# #+#    #+# #+#   #+#+# #+#     #+# #+#     #+#        #+#    #+#       
#  ########  ###     ### ###    ### #########   ########  ###    #### ###     ### ###     ########## #########         
#            :::         :::     ::::::::::         :::::::::: :::    ::: :::::::::   ::::::::  ::::::::: ::::::::::: 
#         :+: :+:     :+: :+:   :+:                :+:        :+:    :+: :+:    :+: :+:    :+: :+:    :+:    :+:      
#       +:+   +:+   +:+   +:+  +:+                +:+         +:+  +:+  +:+    +:+ +:+    +:+ +:+    +:+    +:+       
#     +#++:++#++: +#++:++#++: +#++:++#           +#++:++#     +#++:+   +#++:++#+  +#+    +:+ +#++:++#:     +#+        
#    +#+     +#+ +#+     +#+ +#+                +#+         +#+  +#+  +#+        +#+    +#+ +#+    +#+    +#+         
#   #+#     #+# #+#     #+# #+#                #+#        #+#    #+# #+#        #+#    #+# #+#    #+#    #+#          
#  ###     ### ###     ### ##########         ########## ###    ### ###         ########  ###    ###    ###           
#  

# ---------------------------------------------------------------------
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
# 
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
# BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ---------------------------------------------------------------------
# Title font: Alligator by Simon Bradley
# ---------------------------------------------------------------------

bl_info = {
    "name": "Non Carbonated AAE Export",
    "description": "Export Aegisub-Motion compatible AAE data to file",
    "author": "Akatsumekusa",
    "support": "COMMUNITY",
    "category": "Video Tools",
    "blender": (2, 93, 0),
    "location": "Clip Editor > Tools > Solve > NC AAE Export",
    "warning": "",
    "doc_url": "https://github.com/Akatmks/Non-Carbonated-Motion",
    "tracker_url": "https://github.com/Akatmks/Non-Carbonated-Motion/issues"
}

import bpy
from dataclasses import dataclass
import enum
from enum import Enum

# ("import name", "PyPI name", "minimum version")
modules = (("numpy", "numpy", "1.9.0"), ("matplotlib", "matplotlib", ""), ("sklearn", "scikit-learn", "0.22"))

is_dependencies_ready = False

class NCAAEExportSettings(bpy.types.PropertyGroup):
    bl_label = "NCAAEExportSettings"
    bl_idname = "NCAAEExportSettings"
    
    do_koma_uchi: bpy.props.BoolProperty(name="Enable",
                                         description="Enable コマ打ち awareness.\nThis ensures the tracking data to be consistent with the コマ打ち of the clip.\nコマ打ち is also known as ones, twos, threes, fours, et cetera in English.\nDisable this option if you are tracking a non-anime clip",
                                         default=True)
    max_koma_uchi: bpy.props.IntProperty(name="Maxコマ打ち",
                                         description="The maximum amount of コマ打ち expected.\nThe default value for a 23.976 fps anime is 4. Increase this value if you are working with a 60 fps clip",
                                         default=4)
    do_smoothing: bpy.props.BoolProperty(name="Smoothing",
                                         description="Perform final smoothing.\nDo not enable this option if you are tracking a shaking scene",
                                         default=False)
    do_statistics: bpy.props.BoolProperty(name="Statistics",
                                          description="Generate statistics images alongside AAE data",
                                          default=False)
    do_do_not_overwrite: bpy.props.BoolProperty(name="Do not overwrite",
                                                description="Generate a unique file every time",
                                                default=False)
    do_copy_to_clipboard: bpy.props.BoolProperty(name="Copy to clipboard",
                                                 description="Copy the result to clipboard",
                                                 default=False)

class NCAAEExportExport(bpy.types.Operator):
    bl_label = "Export"
    bl_description = "Export AAE data as txt files next to the original movie clip"
    bl_idname = "movieclip.nc_aae_export_export"

    class _triangular_number:
        # https://en.wikipedia.org/wiki/Triangular_number
        # +-------------+-----------------------------------------------------------------+
        # |             | Track 1 (0)  Track 2 (1)  Track 3 (2)  Track 4 (3)  Track 5 (4) |
        # +-------------+-----------------------------------------------------------------+
        # | Track 1 (0) |                                                                 |
        # | Track 2 (1) |           0                                                     |
        # | Track 3 (2) |           1            2                                        |
        # | Track 4 (3) |           3            4            5                           |
        # | Track 5 (4) |           6            7            8            9              |
        # +-------------+-----------------------------------------------------------------+

        def __init__(self, original_size):
            """
            Parameters
            ----------
            original_size : int
                The size of the array you want to pair from.

            """
            self.full_size = (original_size - 1) * original_size // 2

        def get_complete_pairs(self):
            """
            Returns
            ------
            first_indexes : npt.NDArray[int]
            second_indexes : npt.NDArray[int]
            
            """
            import numpy as np
            from numpy import floor, sqrt

            # Using np.arange() and then running the array through sqrt() is
            # 12 to 15 times faster than using np.empty() and two fors to
            # assign values.
            # This is Python.
            choices = np.arange(self.full_size)

            first_indexes = floor(sqrt(2 * choices + 0.25) + 0.5).astype(int)
            second_indexes = choices - (first_indexes - 1) * first_indexes // 2

            return first_indexes, second_indexes

        def get_sample_pairs(self, sample_size):
            """
            Parameters
            ----------
            sample_size : int

            Returns
            ------
            first_indexes : npt.NDArray[int]
            second_indexes : npt.NDArray[int]

            """
            import numpy as np
            from numpy import floor, sqrt

            if self._random == None:
                self._init_random()

            if sample_size <= self.full_size:
                first_indexes, second_indexes = self.get_complete_pairs()
            else:
                choices = self._random.choice(self.full_size, sample_size)
                first_indexes = floor(sqrt(2 * choices + 0.25) + 0.5).astype(int)
                second_indexes = choices - (first_indexes - 1) * first_indexes // 2

            return first_indexes, second_indexes

        full_size = -1
        """
        The number of possible pairs.
        """

        def _init_random(self):
            import numpy.random as npr

            self._random = npr.default_rng()
        
        _random = None

    class _frame_type(Enum):
        EMPTY = enum
        ONE = enum.auto()
        TWO = enum.auto()
        TWO_NO_ORIGIN = enum.auto()
        MANY_ONLY_PURE_X_Y = enum.auto()
        MANY_LIKELY_SCALE_X_Y = enum.auto()
        MANY_POSSIBLE_SCALE_X_Y = enum.auto()

    class _section_type(Enum):
        # UNDETERMINED is only used in case of _frame_type.ONE and
        # _frame_type.TWO. Koma uchi detection has active lookahead and won't
        # use this option.
        UNDETERMINED = enum.auto()
        EMPTY = enum.auto()
        PURE_X_Y = enum.auto()
        SCALE_X_Y = enum.auto()
        SCALE_X_Y_KOMA_UCHI = enum.auto()

    @dataclass
    class _section:
        # Info
        start_frame: int = -1
        end_frame: int = -1 # not included
        type: object = None
        koma_uchi: int = 1

        # Data
        # npt.NDArray[npt.NDArray[float64]]
        reduced_movement: object = None
        # npt.NDArray[float64]
        scale_origin: object = None
        # npt.NDArray[float64]
        scaling: object = None

    def execute(self, context):
        clip = context.edit_movieclip
        settings = context.screen.NCAAEExportSettings

        ratio \
            = NCAAEExportExport._step_03_calculate_aspect_ratio( \
                  clip)
        position, movement \
            = NCAAEExportExport._step_04_create_position_and_movement_array_from_tracking_markers( \
                  clip, ratio)
        
        reduced_position, reduced_movement, origin, scaling \
            = NCAAEExportExport._step_08_reduce_position_and_movement_array_and_calculate_origin_and_scaling_array( \
                  position, movement)

        print(clip.filepath)
        
        return {'FINISHED'}

    @staticmethod
    def _step_03_calculate_aspect_ratio(clip):
        """
        Calculate aspect ratio. [Step 03]

        Parameters
        ----------
        clip : bpy.types.MovieClip

        Returns
        -------
        ratio : npt.NDArray[float]

        """
        ar = clip.size[0] / clip.size[1]
        # As of 2021/2022
        if ar < 1 / 1.35: # 9:16, 9:19 and higher videos
            return np.array([1 / 1.35, 1 / 1.35 / ar], dtype=np.float64)
        elif ar < 1: # vertical videos from 1:1, 3:4, up to 1:1.35
            return np.array([ar, 1], dtype=np.float64)
        elif ar <= 1.81: # 1:1, 4:3, 16:9, up to 1920 x 1061
            return np.array([ar, 1], dtype=np.float64)
        else: # Ultrawide
            return np.array([1.81, 1.81 / ar], dtype=np.float64)

    @staticmethod
    def _step_04_create_position_and_movement_array_from_tracking_markers(clip, ratio):
        """
        Create position and movement array from tracking markers. [Step 04]

        Parameters
        ----------
        clip : bpy.types.MovieClip
        ratio : npt.NDArray[float]
            ratio likely from Step 03

        Returns
        -------
        position : npt.NDArray[float64]
        movement : npt.NDArray[float64]
            As explained below.

        """
        import numpy as np

        if not clip.frame_duration >= 1:
            raise ValueError("clip.frame_duration must be greater than or equal to 1")

        # position array structure
        # +---------+------------------------------------------+
        # |         |           Track 0  Track 1  Track 2      |
        # +---------+------------------------------------------+
        # | Frame 0 | array([[  [x, y],  [x, y],  [x, y]   ],  |
        # | Frame 1 |        [  [x, y],  [x, y],  [x, y]   ],  |
        # | Frame 2 |        [  [x, y],  [x, y],  [x, y]   ],  |
        # | Frame 3 |        [  [x, y],  [x, y],  [x, y]   ],  |
        # | Frame 4 |        [  [x, y],  [x, y],  [x, y]   ]]) |
        # +---------+------------------------------------------+
        # There should be more calculations intraframe than interframe so
        # position and movement arrays put the frame as the first axis.
        #
        # The origin will be located at the upper left corner of the video,
        # contrary to Blender's usual lower left corner.
        #
        # The x and y value will have a pixel aspect ratio of 1:1. See Step 03
        # for the range where x and y value is on screen.
        #
        # The start frame of a video will be frame 0, instead of Blender's
        # usual frame 1.
        # 
        # Also, on the topic of precision, NC AAE Export uses float64 across
        # the whole script instead of Blender's float32.
        position = np.full((clip.frame_duration, len(clip.tracking.tracks), 2), np.nan, dtype=np.float64)
        
        # This will become the slowest step of all the procedures
        for i, track in enumerate(clip.tracking.tracks):
            for marker in track.markers[1:] if not track.markers[0].is_keyed else track.markers:
                if not 0 < marker.frame <= clip.frame_duration:
                    continue
                if marker.mute:
                    continue
                position[marker.frame - 1][i] = [marker.co[0], 1 - marker.co[1]]

        position *= ratio

        # movement array structure
        # +-----------+------------------------------------------+
        # |           |           Track 0  Track 1  Track 2      |
        # +-----------+------------------------------------------+
        # | Frame 0/1 | array([[  [x, y],  [x, y],  [x, y]   ],  |
        # | Frame 1/2 |        [  [x, y],  [x, y],  [x, y]   ],  |
        # | Frame 2/3 |        [  [x, y],  [x, y],  [x, y]   ],  |
        # | Frame 3/4 |        [  [x, y],  [x, y],  [x, y]   ],  |
        # | Frame 4/  |        [  [x, y],  [x, y],  [x, y]   ]]) |
        # +-----------+                                          |
        # | size -= 1 |                                          |
        # +-----------+------------------------------------------+
        movement = np.diff(position, axis=0)
        movement = np.vstack((movement, movement[-1]))

        return position, movement

    @staticmethod
    def _step_08_reduce_position_and_movement_array_and_calculate_origin_and_scaling_array(position, movement):
        """
        Remove the nans in the position and movement array and calculate the
        origin array. [Step 08]

        Parameters
        ----------
        position : npt.NDArray[float64]
        movement : npt.NDArray[float64]
            The position and movement arrays likely coming from Step 04.

        Returns
        -------
        reduced_position : npt.NDArray[npt.NDArray[float64]]
        reduced_movement : npt.NDArray[npt.NDArray[float64]]
            Position and movement array without nans.
        origin : npt.NDArray[npt.NDArray[float64]]
            Origin array.
        scaling : npt.NDArray[npt.NDArray[float64]]
            Scaling array.

        """
        import numpy as np

        frames = movement.shape[0]
        reduced_position = np.empty(frames, dtype=object)
        reduced_movement = np.empty(frames, dtype=object)
        origin = np.empty(frames, dtype=object)
        scaling = np.empty(frames, dtype=object)

        for frame in range(frames):
            reduced_position[frame], reduced_movement[frame] \
                = NCAAEExportExport._step_08__reduce_position_and_movement_array_per_frame( \
                      position[frame], movement[frame])
            
            if reduced_movement[frame].shape[0] >= 2:
                origin[frame], scaling[frame] \
                    = NCAAEExportExport._step_08__calculate_origin_array_per_frame( \
                          reduced_position[frame], reduced_movement[frame])
            else:
                origin[frame] \
                    = np.empty((0, 2), dtype=np.float64)
                scaling[frame] \
                    = np.empty(0, dtype=np.float64)

        return reduced_position, reduced_movement, origin, scaling

    @staticmethod
    def _step_08__reduce_position_and_movement_array_per_frame(position, movement):
        """
        Parameters
        ----------
        position : npt.NDArray[float64]
        movement : npt.NDArray[float64]
            The position and movement arrays of the frame.

        Returns
        -------
        reduced_position : npt.NDArray[float64]
        reduced_movement : npt.NDArray[float64]
            Position and movement array without nans.

        """
        import numpy as np

        available_mask = ~np.isnan(movement[:, 0])

        return position[available_mask], movement[available_mask]

    @staticmethod
    def _step_08__calculate_origin_and_scaling_array_per_frame(reduced_position, reduced_movement):
        """
        Parameters
        ----------
        reduced_position : npt.NDArray[float64]
        reduced_movement : npt.NDArray[float64]
            Position and movement array for the frame without nans and not
            empty.

        Returns
        -------
        origin : npt.NDArray[float64]
            Origin array.
        scaling : npt.NDArray[float64]
            Scaling array.

        """
        import numpy as np
        import numpy.linalg as LA

        select = NCAAEExportExport._triangular_number(reduced_movement.shape[0])
        first_indexes, second_indexes = select.get_complete_pairs()

        pizza = np.column_stack(((first_reduced_position := reduced_position[first_indexes]),
                                 (first_reduced_movement := reduced_movement[first_indexes]),
                                 (second_reduced_position := reduced_position[second_indexes]),
                                 (second_reduced_movement := reduced_movement[second_indexes])))

        # https://stackoverflow.com/questions/563198/
        def eat(slice):
            if (j := np.cross(slice[2:4], slice[6:8])) == 0: # := requires Python 3.8 (Blender 2.93)
                return np.array([np.nan, np.nan], dtype=np.float64)
            else:
                return slice[0:2] + np.cross(slice[4:6] - slice[0:2], slice[6:8]) / j * slice[2:4]

        origin = np.apply_along_axis(eat, 1, pizza)

        distance = LA.norm((diff_reduced_position := first_reduced_position - second_reduced_position), axis=1) # axis requires numpy version 1.8.0
        distance_next = LA.norm(diff_reduced_position + first_reduced_movement - second_reduced_movement, axis=1)

        scaling = distance_next / distance

        return origin, scaling

    @staticmethod
    def _step_18_decide_proper_method_and_count_scale_koma_uchi(reduced_movement, origin, scaling, do_koma_uchi, max_koma_uchi, do_statistics, ratio):
        """
        Decide if the scale x/y method or the pure x/y method is suitable for
        each frame and count the scale コマ打ち. Position コマ打ち will be calculated
        in later functions. [Step 18]

        Parameters
        ----------
        reduced_movement : npt.NDArray[npt.NDArray[float64]]
        origin : npt.NDArray[npt.NDArray[float64]]
        scaling : npt.NDArray[npt.NDArray[float64]]
            Reduced movement array, origin array and scaling array likely
            coming from step 08.
        do_smoothing : bool
            NCAAEExportSettings.do_smoothing

        do_koma_uchi : bool
            NCAAEExportSettings.do_koma_uchi.
        max_koma_uchi : int
            NCAAEExportSettings.max_koma_uchi.
        do_statistics : bool
            NCAAEExportSettings.do_statistics
        ratio : npt.NDArray[float]
            ratio likely from Step 03

        Returns
        -------
        method : npt.NDArray[NCAAEExportExport._method]
        filtered_origin : npt.NDArray[npt.NDArray[float64]]
        
        """
        import numpy as np
        
        frames = reduced_movement.shape[0]

        frame_type, origin_cluster_sizes, origin_cluster_centroids, cluster_scaling \
            = NCAAEExportExport._step_18__check_origin_array( \
                  reduced_movement, origin, scaling)

        sections \
            = NCAAEExportExport._step_18__detect_section( \
                  reduced_movement, \
                  frame_type, origin_cluster_sizes, origin_cluster_centroids, cluster_scaling, \
                  do_smoothing)

    @staticmethod
    def _step_18__check_origin_array(reduced_movement, origin, scaling):
        """
        Parameters
        ----------
        reduced_movement : npt.NDArray[npt.NDArray[float64]]
            Reduced movement array.
        origin : npt.NDArray[npt.NDArray[float64]]
            Origin array.
        scaling : npt.NDArray[npt.NDArray[float64]]
            Scaling array.

        Returns
        -------
        frame_type : npt.NDArray[NCAAEExportExport._frame_type]
        origin_cluster_sizes : npt.NDArray[npt.NDArray[int]]
        origin_cluster_centroids : npt.NDArray[npt.NDArray[float64]]
        cluster_scaling : npt.NDArray[npt.NDArray[float64]]
        
        """
        import numpy as np

        frames = reduced_movement.shape[0]
        frame_type = np.empty(frames, dtype=object)
        origin_cluster_sizes = np.empty(frames, dtype=object)
        origin_cluster_centroids = np.empty(frames, dtype=object)
        cluster_scaling = np.empty(frames, dtype=object)

        for frame in range(frames):
            if reduced_movement[frame].shape[0] == 0:
                frame_type[frame] \
                    = NCAAEExportExport._frame_type.EMPTY
                origin_cluster_sizes[frame] \
                    = np.empty(0, dtype=int)
                origin_cluster_centroids[frame] \
                    = np.empty((0, 2), dtype=np.float64)
                cluster_scaling[frame] \
                    = np.empty(0, dtype=np.float64)

            elif reduced_movement[frame].shape[0] == 1:
                frame_type[frame] \
                    = NCAAEExportExport._frame_type.ONE
                origin_cluster_sizes[frame] \
                    = np.empty(0, dtype=int)
                origin_cluster_centroids[frame] \
                    = np.empty((0, 2), dtype=np.float64)
                cluster_scaling[frame] \
                    = np.empty(0, dtype=np.float64)

            elif reduced_movement[frame].shape[0] == 2:
                if origin[frame][0, 0] != np.nan:
                    frame_type[frame] \
                        = NCAAEExportExport._frame_type.TWO
                    origin_cluster_sizes[frame] \
                        = np.array([1], dtype=int)
                    origin_cluster_centroids[frame] \
                        = origin[frame][0].reshape((1, 2))
                    cluster_scaling[frame] \
                        = scaling[frame][0].reshape((1))
                else:
                    frame_type[frame] \
                        = NCAAEExportExport._frame_type.TWO_NO_ORIGIN
                    origin_cluster_sizes[frame] \
                        = np.empty(0, dtype=int)
                    origin_cluster_centroids[frame] \
                        = np.empty((0, 2), dtype=np.float64)
                    cluster_scaling[frame] \
                        = np.empty(0, dtype=np.float64)
            
            else: # reduced_movement[frame].shape[0] >= 3
                frame_type[frame], \
                origin_cluster_sizes[frame], \
                origin_cluster_centroids[frame], \
                cluster_scaling[frame] \
                    = NCAAEExportExport._step_18__check_origin_array_for_possible_scale_origin_by_frame( \
                          origin[frame], scaling[frame])
            
        return frame_type, origin_cluster_sizes, origin_cluster_centroids, cluster_scaling
                    
    @staticmethod
    def _step_18__check_origin_array_for_possible_scale_origin_by_frame(origin, scaling):
        """
        Parameters
        ----------
        origin : npt.NDArray[float64]
            Origin array of the frame. Not empty.
        scaling : npt.NDArray[npt.NDArray[float64]]
            Scaling array of the frame.

        Returns
        -------
        frame_type : NCAAEExportExport._frame_type
        cluster_sizes : npt.NDArray[int]
        cluster_centroids : npt.NDArray[float64]
        cluster_scaling : npt.NDArray[float64]

        """
        import numpy as np
        from sklearn.cluster import DBSCAN
        from sklearn.neighbors import NearestCentroid, \
                                      LocalOutlierFactor

        reducing_mask = ~np.isnan(origin[:, 0])
        reduced_origin = origin[reducing_mask]
        reduced_scaling = scaling[reducing_mask]

        clustering = DBSCAN(min_samples=np.ceil(origin.shape[0] * 0.20).astype(int), eps=0.20)
        estimation = clustering.fit_predict(reduced_origin)
        
        inlier_mask = estimation != -1
        inlier_origin = reduced_origin[inlier_mask]
        inlier_estimation = estimation[inlier_mask]

        clf = NearestCentroid()
        clf.fit(inlier_origin, inlier_estimation)

        cluster_sizes = np.bincount(inlier_estimation)
        cluster_centroids = clf.centroids_

        cluster_scaling = np.empty((cluster_sizes.shape[0]), dtype=np.float64)
        for cluster in range(cluster_sizes.shape[0]):
            clf = LocalOutlierFactor(n_neighbors=np.floor(cluster_sizes[cluster] * 0.80).astype(int))
            estimation = clf.fit_predict(reduced_scaling_cluster := reduced_scaling[estimation == cluster])

            cluster_scaling[cluster] = np.mean(reduced_scaling_cluster[estimation == 1])

        if cluster_sizes.shape[0] == 0:
            return NCAAEExportExport._frame_type.MANY_ONLY_PURE_X_Y, cluster_sizes, cluster_centroids, cluster_scaling
        elif cluster_sizes.shape[0] == 1:
            return NCAAEExportExport._frame_type.MANY_LIKELY_SCALE_X_Y, cluster_sizes, cluster_centroids, cluster_scaling
        else: # cluster_sizes.shape[0] > 1:
            sorted_cluster_sizes = np.sort(cluster_sizes)
            if sorted_cluster_sizes[-1] > sorted_cluster_sizes[-2] * 1.50:
                return NCAAEExportExport._frame_type.MANY_LIKELY_SCALE_X_Y, cluster_sizes, cluster_centroids, cluster_scaling
            else:
                return NCAAEExportExport._frame_type.MANY_POSSIBLE_SCALE_X_Y, cluster_sizes, cluster_centroids, cluster_scaling

    @staticmethod
    def _step_18__detect_section(reduced_position, reduced_movement, frame_type, origin_cluster_sizes, origin_cluster_centroids, cluster_scaling, do_smoothing, do_koma_uchi, max_koma_uchi):
        """
        Parameters
        ----------
        reduced_position : npt.NDArray[npt.NDArray[int]]
        reduced_movement : npt.NDArray[npt.NDArray[int]]
        frame_type : npt.NDArray[NCAAEExportExport._frame_type]
        origin_cluster_sizes : npt.NDArray[npt.NDArray[int]]
        origin_cluster_centroids : npt.NDArray[npt.NDArray[float64]]
        cluster_scaling : npt.NDArray[npt.NDArray[float64]]
        do_smoothing : bool
            NCAAEExportSettings.do_smoothing
        do_koma_uchi : bool
            NCAAEExportSettings.do_koma_uchi
        max_koma_uchi : int
            NCAAEExportSettings.max_koma_uchi

        Returns
        -------
        sections : list[NCAAEExportExport._section]
        """
        from collections import deque
        import numpy as np

        frames = reduced_movement.shape[0]
        early_smoothing_error_margin = 2
        koma_uchi_keyframe_margin = 1 if do_koma_uchi else 0 # Probably doesn't support value other than 1
        koma_uchi_koma_margin = max_koma_uchi - koma_uchi_keyframe_margin if do_koma_uchi else 0
        koma_uchi_error_margin = 1 # Probably also doesn't support any other value than 1.
        sections = deque()

        f = 0
        trace_backward = 0
        while f < frames:
            section = NCAAEExportExport._section()
            section.start_frame = f

            # If current section ...
            if frame_type[f] == NCAAEExportExport._frame_type.EMPTY:
                section.type = NCAAEExportExport._section_type.EMPTY

                while f < frames:
                    f += 1
                    if frame_type[f] != NCAAEExportExport._frame_type.EMPTY:
                        break
                section.end_frame = f
                sections.append(section)

                continue
            # If current section ...
            elif frame_type[f] in (NCAAEExportExport._frame_type.ONE, \
                                   NCAAEExportExport._frame_type.TWO, \
                                   NCAAEExportExport._frame_type.TWO_NO_ORIGIN):
                start_frame_type = frame_type[f]
                while f < frames:
                    f += 1
                    if frame_type[f] != start_frame_type:
                        break
                # If next section ...
                else:
                    section.end_frame = f

                    # If previous section ...
                    if trace_backward == 0 and \
                       (section.start_frame == 0 or \
                        sections[-1].type in (NCAAEExportExport._section_type.EMPTY, \
                                              NCAAEExportExport._section_type.PURE_X_Y, \
                                              NCAAEExportExport._section_type.SCALE_X_Y_KOMA_UCHI)) or \
                       trace_backward != 0 and \
                       (sections[-trace_backward].start_frame == 0 or \
                        sections[-trace_backward-1].type in (NCAAEExportExport._section_type.EMPTY, \
                                                             NCAAEExportExport._section_type.PURE_X_Y, \
                                                             NCAAEExportExport._section_type.SCALE_X_Y_KOMA_UCHI)):
                        if trace_backward != 0:
                            section.start_frame = sections[-trace_backward].start_frame
                        
                        section.type = NCAAEExportExport._section_type.PURE_X_Y
                        section.reduced_movement = reduced_movement[section.start_frame:section.end_frame]
                    # If previous section ...
                    elif trace_backward == 0 and \
                         sections[-1].type == NCAAEExportExport._section_type.SCALE_X_Y or \
                         trace_backward != 0 and \
                         sections[-trace_backward-1].type == NCAAEExportExport._section_type.SCALE_X_Y:
                        section.type = NCAAEExportExport._section_type.SCALE_X_Y

                        def func_(start_frame, end_frame, \
                                  reduced_position, reduced_movement, \
                                  frame_type, origin_cluster_centroids, cluster_scaling, \
                                  previous_frame_scale_origin):
                            import numpy as np
                            import numpy.linalg as LA

                            if frame_type[start_frame] in (NCAAEExportExport._frame_type.ONE, \
                                                           NCAAEExportExport._frame_type.TWO_NO_ORIGIN):
                                if do_smoothing:
                                    scale_origin = np.full((start_frame - end_frame, 2), np.nan, dtype=np.float64)
                                    scaling = np.full((start_frame - end_frame), np.nan, dtype=np.float64)
                                else:
                                    scale_origin = np.full((start_frame - end_frame, 2), previous_frame_scale_origin, dtype=np.float64)
                                    scaling = np.mean(LA.norm((d_ := reduced_position[start_frame:end_frame] - previous_frame_scale_origin) + reduced_movement[start_frame:end_frame], axis=1) / \
                                                      LA.norm(d_, axis=1), axis=1)
                            elif frame_type[start_frame] == NCAAEExportExport._frame_type.TWO:
                                scale_origin = np.stack(origin_cluster_centroids[start_frame:end_frame])[:, 0]
                                scaling = np.stack(cluster_scaling[start_frame:end_frame])[:, 0]
                            else: raise ValueError

                            return scale_origin, scaling
                        
                        if trace_backward == 0:
                            section.scale_origin, \
                            section.scaling \
                                = func_( \
                                      section.start_frame, section.end_frame, \
                                      reduced_position, reduced_movement, \
                                      frame_type, origin_cluster_centroids, cluster_scaling, \
                                      sections[-1].scale_origin[-1])
                        else: # trace_backward != 0
                            section.scale_origin, \
                            section.scaling \
                                = func_( \
                                      sections[-trace_backward].start_frame, sections[-trace_backward].end_frame, \
                                      reduced_position, reduced_movement, \
                                      frame_type, origin_cluster_centroids, cluster_scaling, \
                                      sections[-trace_backward-1].scale_origin[-1])
                            for t_ in range(trace_backward-1, 0, -1):
                                section.scale_origin = np.vstack(((func_return_ := func_( \
                                                                                       sections[-t_].start_frame, sections[-t_].end_frame, \
                                                                                       reduced_position, reduced_movement, \
                                                                                       frame_type, origin_cluster_centroids, cluster_scaling, \
                                                                                       section.scale_origin[-1]))[0], \
                                                                  section.scale_origin))
                                section.scaling = np.vstack((func_return_[1], section.scaling))
                            section.scale_origin = np.vstack(((func_return_ := func_(\
                                                                                   section.start_frame, section.end_frame, \
                                                                                   reduced_position, reduced_movement, \
                                                                                   frame_type, origin_cluster_centroids, cluster_scaling, \
                                                                                   section.scale_origin[-1]))[0], \
                                                              section.scale_origin))
                            section.scaling = np.vstack((func_return_[1], section.scaling))
                            section.start_frame = sections[-trace_backward].start_frame
                    # If previous section ...
                    else: raise ValueError

                    for _ in range(0, trace_backward):
                        sections.pop()
                    sections.append(section)
                    trace_backward = 0

                    continue
                # If next section ...
                # We don't know if the next section will be koma uchi or not.
                # The handling will be happening in the next section instead.
                section.type[f] == NCAAEExportExport._section_type.UNDETERMINED
                section.end_frame = f
                trace_backward += 1
                sections.append(section)

                continue
            # If current section ...
            elif frame_type[f] == NCAAEExportExport._frame_type.MANY_ONLY_PURE_X_Y:
                section.type = NCAAEExportExport._section_type.PURE_X_Y

                while f < frames:
                    f += 1
                    if frame_type[f] != NCAAEExportExport._frame_type.MANY_ONLY_PURE_X_Y
                        break
                if trace_backward != 0:
                    section.start_frame = sections[-trace_backward].start_frame
                section.end_frame = f

                section.reduced_movement = reduced_movement[section.start_frame:section.end_frame]
                
                for _ in range(0, trace_backward):
                    sections.pop()
                sections.append(section)
                trace_backward = 0

                continue
            # If current section ...
            elif frame_type[f] in (NCAAEExportExport._frame_type.MANY_LIKELY_SCALE_X_Y, \
                                   NCAAEExportExport._frame_type.MANY_POSSIBLE_SCALE_X_Y): 
                while f < frames:
                    f += 1
                    if not frame_type[f] in (NCAAEExportExport._frame_type.MANY_LIKELY_SCALE_X_Y, \
                                             NCAAEExportExport._frame_type.MANY_POSSIBLE_SCALE_X_Y):
                        break
                else:
                    # Do something
                    continue

                if do_koma_uchi and max_koma_uchi >= 2 and \
                   f - section.frame_start <= koma_uchi_keyframe_margin + koma_uchi_error_margin and \
                   frame_type[f] == NCAAEExportExport._frame_type.MANY_ONLY_PURE_X_Y:
                    k_ = 2
                    # A koma_uchi should at least be 4 cycles long, and
                    # All 4 cycles shouldn't contain any EMPTY, and
                    # The first 3 cycles shouldn't contain any ONE, TWO or TWO_NO_ORIGIN, and
                    # The first frame of each cycle (keyframe) must be MANY_PROBABLY_SCALE_X_Y or MANY_LIKELY_SCALE_X_Y, and
                    # Non keyframes should contain equal or less SCALE_X_Y frames than error_margin per cycle, and
                    # The sum of the error count should be less than 2 within the 4 cycles.
                    while not (section.frame_start + k_ * 4 <= frames and \
                               np.all((t_ := frame_type[section.frame_start:section.frame_start + k_ * 4]) != NCAAEExportExport._frame_type.EMPTY) and \
                               np.all(~np.in1d(t_[:k_ * 3], [NCAAEExportExport._frame_type.ONE, \
                                                             NCAAEExportExport._frame_type.TWO, \
                                                             NCAAEExportExport._frame_type.TWO_NO_ORIGIN])) and \
                               np.all((b_ := np.in1d(t_, [NCAAEExportExport._frame_type.MANY_LIKELY_SCALE_X_Y, \
                                                          NCAAEExportExport._frame_type.MANY_POSSIBLE_SCALE_X_Y]).reshape((4, k_)))[:, 0]) and \
                               np.amax((e_ := np.count_nonzero(b_[:, 1:], axis=1))) <= koma_uchi_error_margin and \
                               np.sum(e_) <= 2):
                        
                        if k_ == koma_uchi_keyframe_margin + koma_uchi_koma_margin:
                            break
                        k_ += 1
                    else:
                        # Keyframe:
                        #     Non SCALE_X_Y will terminal koma_uchi at the cycle before.
                        # Non keyframe:
                        #     EMPTY or frames will terminal koma_uchi at the cycle before.
                        #     ONE, TWO or TWO_NO_ORIGIN will terminal koma_uchi after the current cycle.
                        #     error count more than error_margin will terminal koma_uchi at the cycle before.
                        c_ = np.amin([(cm_ := (frames - section.frame_start) // k_), \
                                      ci_ if (ci_ := np.argmax(~((b_ := np.in1d((t_ := frame_type[section.frame_start:section.frame_start + k_ * cm_].reshape((-1, k_))), \
                                                                                [NCAAEExportExport._frame_type.MANY_LIKELY_SCALE_X_Y, \
                                                                                 NCAAEExportExport._frame_type.MANY_POSSIBLE_SCALE_X_Y]))[:, 0]))) != 0 else cm_, \
                                      cj_ // (k_ - 1) if (cj_ := np.argmax((tk_ := t_[:, 1:]) == NCAAEExportExport._frame_type.EMPTY)[1]) != 0 else cm_, \
                                      ck_ // (k_ - 1) + 1 if (ck_ := np.argmax(np.in1d(tk_, \
                                                                                       [NCAAEExportExport._frame_type.ONE, \
                                                                                        NCAAEExportExport._frame_type.TWO, \
                                                                                        NCAAEExportExport._frame_type.TWO_NO_ORIGIN]))) != 0 else cm_, \
                                      cl_ if (cl_ := np.argmax(np.count_nonzero(b_[:, 1:], axis=1) > koma_uchi_error_margin)) != 0 else cm_])

                        section.end_frame = section.start_frame + c_ * k_
                        section.koma_uchi = k_
                        section.reduced_movement = reduced_movement[section.start_frame:section.end_frame].reshape((-1, k_))[:, 1:]
                        # TODO DECIDE MANY_POSSIBLE_SCALE_X_Y
                        # TODO DEAL WITH UNDETERMINED

                        sections.append(section)

                        continue






                # If next section ...
                if f - section.frame_start <= early_smoothing_error_margin and \
                : # next


            # If current section ...
            else: raise ValueError









        # early_smoothing_error_margin = 2

        # edges = np.nonzero(method[:-1] == method[1:] | \
        #                    is_scale_x_y_possible[:-1] == is_scale_x_y_possible[1:])[0] + 1
        # # multiple insertions requires numpy version 1.8.0
        # edges = np.insert(edges, [0, edges.shape[0]], [0, frames]) # Note that the frames here (not frames-1) is out of range

        # i = -1
        # trace_backward = -1
        # while i < edges.shape[0] - 1:
        #     if method[edges[i]] == NCAAEExportExport._method.NOTHING:
        #         trace_backward = i+1
        #         i = trace_backward
        #         continue

        #     elif method[edges[i]] == NCAAEExportExport._method.UNDETERMINED:
        #         if i == 0 or \
        #            method[edges[i-1] == NCAAEExportExport._method.NOTHING]:
        #             j = i+1
        #             while j < edges.shape[0]:
        #                 if method[edges[j]] in (NCAAEExportExport._method.PURE_X_Y, \
        #                                         NCAAEExportExport._method.SCALE_X_Y):
        #                     break

        #                 j = j+1
        #             else:
        #                 method[method == NCAAEExportExport._method.UNDETERMINED] \
        #                     = NCAAEExportExport._method.PURE_X_Y
                        
        #                 trace_backward = edges.shape[0] - 1
        #                 i = trace_backward
        #                 continue

        #             trace_backward = i
        #             i = j
        #             continue
        #         else: # i != 0
        #             prev_method = method[edges[i-1]]
                    
        #             j = i+1
        #             while j < edges.shape[0]:
        #                 if method[edges[j]] in (NCAAEExportExport._method.EMPTY, \
        #                                         NCAAEExportExport._method.PURE_X_Y, \
        #                                         NCAAEExportExport._method.SCALE_X_Y)
        #                     break
                        
        #                 j = j+1
                    
        #             if j == edges.shape[0]:
        #                 # SINGLE TRACKER COULD ALSO WORK FOR SCALE X Y IF THE ORIGIN DON'T MOVE

        #     elif method[edges[i]] == NCAAEExportExport._method.PURE_X_Y:


        #             if edges[i+1] - edges[i] <= early_smoothing_error_margin and \
        #                i+1 != edges.shape[0] - 1 and \
        #                edges[i+2] - edges[i+1] > early_smoothing_error_margin:
        #                 if method[edges[i+1]] == NCAAEExportExport._method.UNDETERMINED:
        #                     j = i+1
        #                     while j < edges.shape[0] - 1:
        #                         j += 1

        #                         if method[edges[j]] in (NCAAEExportExport._method.PURE_X_Y, \
        #                                                 NCAAEExportExport._method.SCALE_X_Y):
        #                             trace_backward = i
        #                             i = j
        #                             continue
        #                     else:
        #                         method[method == NCAAEExportExport._method.UNDETERMINED] \
        #                             = NCAAEExportExport._method.PURE_X_Y
                                
        #                         i = edges.shape[0] - 2
        #                         continue
                            




        #                                 NCAAEExportExport._method.PURE_X_Y):
        #             if (edges[i+1] - edges[i] <= early_smoothing_error_margin or \
        #                 edges[i+1] - edges[i] == 1 if do_koma_uchi else False) and \
        #                i+1 != edges.shape[0] - 1:
        #                 if not do_koma_uchi and \
        #                    edges[i+2] - edges[i+1] > early_smoothing_error_margin:
        #                     do()
        #                 elif do_koma_uchi
        #             else:
        #                 continue
            
        
        # else: # not do_koma_uchi
        #     early_smoothing_error_margin = 2

        #     for i in range(edges.shape[0] - 1):
        #         if method[edge[i]] in (NCAAEExportExport._method.PURE_X_Y_UNSURE):

        #         if edges[i+1] - edges[i] <= early_smoothing_error_margin and \
        #             if i == 0:
        #                 if i+1 != edges.shape[0] - 1:
        #                     if edges[i+2] - edges[i+1] > early_smoothing_error_margin:
        #                         if method[edges[i]] in (NCAAEExportExport._method.UNDETERMINED, \
        #                                                 NCAAEExportExport._method.SCALE_X_Y_UNSURE, \
        #                                                 NCAAEExportExport._method.SCALE_X_Y) and \
        #                            method[edges[i+1]] in (NCAAEExportExport._method.PURE_X_Y_UNSURE, \
        #                                                   NCAAEExportExport._method.PURE_X_Y):
        #                     else: # edges[i+2] - edges[i+1] <= 2
        #                         continue
        #                 else: # i+1 == edges.shape[0] - 1
        #                     continue


        #             if edges[i+2] - edges[i+1] > 2 if i+1 == edges.shape[0] - 1 else True:
        #                 if method[edges[i]] == NCAAEExportExport._method.SCALE_X_Y or \
        #                    method[edges[i]] == NCAAEExportExport._method.SCALE_X_Y_UNSURE:
        #                     if method[edges[i-1]]
        #                     method[edges[i]:edge[i+1]] = NCAAEExportExport._method.PURE_X_Y
        #                 elif method[edges[i]] = NCAAEExportExport._method.PURE_X_Y:
        #                     method[edges[i]:edge[i+1]] = NCAAEExportExport._method.SCALE_X_Y
        #             else 
        #         else: # edges[i+1] - edges[i] > 2
        #             continue

    # @staticmethod
    # def _step_18__reduce_origin_array_and_remove_outliers_per_frame(origin):
    #     """
    #     Parameters
    #     ----------
    #     origin : npt.NDArray[float64]
    #         Origin array of the frame. Not empty.

    #     Returns
    #     -------
    #     origin_estimator : sklearn.base.BaseEstimator
    #         Origin estimator.

    #     """
    #     import numpy as np
    #     from sklearn.neighbors import LocalOutlierFactor

    #     reduced_origin = origin[np.nonzero(~np.isnan(origin[:, 0]))]

    #     clf = LocalOutlierFactor(n_neighbors=np.ceil(reduced_origin.shape[0] / 3).astype(int)) # LocalOutlierFactor contamination parameter requires sklearn version 0.22
    #     estimation = clf.fit_predict(reduced_origin)

    #     return reduced_origin[np.nonzero(estimation == 1)]

    # @staticmethod
    # def _step_18__check_if_scale_origin_is_clear(origin, filtered_origin):
    #     """
    #     Parameters
    #     ----------
    #     origin : npt.NDArray[float64]
    #     filtered_origin : npt.NDArray[float64]
    #         Origin and filtered origin array of the frame. Not empty.

    #     Returns
    #     -------
    #     is_existing : bool

    #     """
    #     import numpy as np

    #     if np.ceil(origin.shape[0] * 0.20) <= filtered_origin.shape[0]:
    #         if np.all(np.amax(filtered_origin[frame], axis=0) - np.amin(filtered_origin[frame], axis=0) < [0.25, 0.25]):


    # @staticmethod
    # def _step_18_try_to_find_scale_origin_and_count_scale_koma_uchi(position_x, position_y, movement_x, movement_y, ratio_x, ratio_y, max_koma_uchi, do_plot):
    #     """
    #     Pick random pairs of movements from the position array and try to find
    #     the scale origin for every frame.
    #     Decide if the scale method or the pure x/y method is suitable for the
    #     clip. It will slice the clip into multiple sections if different
    #     methods are suitable for difference sections of the clip.
    #     It will also count the scale コマ打ち, while position コマ打ち will be
    #     calculated in other functions.
    #     [Step 18]

    #     Parameters
    #     ----------
    #     position_x : npt.NDArray[float64]
    #     position_y : npt.NDArray[float64]
    #     movement_x : npt.NDArray[float64]
    #     movement_y : npt.NDArray[float64]
    #         The position and movement arrays likely coming from Step 04.
    #     ratio_x : float
    #     ratio_y : float
    #     max_koma_uchi : int
    #         NCAAEExportSettings.max_koma_uchi if NCAAEExportSettings.do_koma_uchi else 1.
    #     do_plot : bool
    #         NCAAEExportSettings.do_statistics.

    #     Returns
    #     -------
    #     sections : list[tuple[int, NCAAEExportExport._method]]
    #         A list of (frame, method)
    #         This list only records the edges, or the frame when the method
    #         changes from one to another.
    #     scale_koma_uchi : list[tuple[int, int]]
    #         A list of (frame, koma_uchi)
    #         This list only records the edges.
    #     origins_x : npt.NDArray[npt.NDArray[float64] or None]
    #     origins_y : npt.NDArray[npt.NDArray[float64] or None]

    #     """
    #     import numpy as np
        
    #     frames = movement_x.shape[0]
    #     origins_x # TODO
    #     origins_y # TODO 
    #     prev = NCAAEExportExport._method.UNDEFINED
    #     for i in range(frames // 2, frames):
    #         prev, origins_x, origins_y, is_complete, nan_percent, scalars_x, scalars_y \
    #             = NCAAEExportExport._step_18__call_search_scale_origins(position_x[i], position_y[i], movement_x[i], movement_y[i], prev, do_plot)


    #     # scale_koma_uchi is for return, just return the current stats
    #     # TODO finish this for
    #     # TODO matplotlib
        
    
    @staticmethod
    def _step_2C_find_scale_origin_and_calculate_scale():
        pass

    @staticmethod
    def _step_48_scale_origin_smoothing():
        pass

    @staticmethod
    def _step_80_scale_smoothing():
        pass

    @staticmethod
    def _step_E0_generate_pseudo_scale_point():
        pass

    @staticmethod
    def _step_E2_export_for_scale():
        pass

class NCAAEExport(bpy.types.Panel):
    bl_label = "NC AAE Export"
    bl_idname = "SOLVE_PT_nc_aae_export"
    bl_space_type = "CLIP_EDITOR"
    bl_region_type = "TOOLS"
    bl_category = "Solve"

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        
        settings = context.screen.NCAAEExportSettings
        
        column = layout.column(heading="コマ打ち")
        column.prop(settings, "do_koma_uchi")
        column.prop(settings, "max_koma_uchi")
        
        column = layout.column(heading="Functions")
        column.prop(settings, "do_smoothing")
        column.prop(settings, "do_statistics")
        
        column = layout.column(heading="Result")
        column.prop(settings, "do_do_not_overwrite")
        column.prop(settings, "do_copy_to_clipboard")
        
        row = layout.row()
        row.scale_y = 2
        row.enabled = len(context.edit_movieclip.tracking.tracks) >= 1
        row.operator("movieclip.nc_aae_export_export")

    @classmethod
    def poll(cls, context):
        return context.edit_movieclip is not None

classes = (NCAAEExportSettings,
           NCAAEExportExport,
           NCAAEExport)

class NCAAEExportRegisterInstallDependencies(bpy.types.Operator):
    bl_label = "Install dependencies"
    bl_description = "NC AAE Export requires additional packages to be installed.\nBy clicking this button, NC AAE Export will download and install " + \
                     (" and ".join([", ".join(["pip"] + [module[1] for module in modules[:-1]]), modules[-1][1]]) if len(modules) != 0 else "pip") + \
                     " into your Blender distribution.\nThis process might take up to 3 minutes. Your Blender will freeze during the process"
    bl_idname = "preference.nc_aae_export_register_install_dependencies"
    bl_options = {"REGISTER", "INTERNAL"}

    def execute(self, context):
        import importlib.util
        import os
        import subprocess
        import sys

        if os.name == "nt":
            self._execute_nt(context)
        else:
            subprocess.run([sys.executable, "-m", "ensurepip"], check=True) # sys.executable requires Blender 2.93
            subprocess.run([sys.executable, "-m", "pip", "install"] + [module[1] + ">=" + module[2] if module[2] != "" else module[1] for module in modules], check=True)
            
        for module in modules:
            if importlib.util.find_spec(module[0]) == None:
                return {'FINISHED'}

        global is_dependencies_ready      
        is_dependencies_ready = True

        register_main_classes()
        unregister_register_class()
        
        self.report({"INFO"}, "Dependencies installed successfully.")
            
        return {'FINISHED'}

    def _execute_nt(self, context):
        # Python, in a Python, in a PowerShell, in a Python
        import importlib.util
        import os
        from pathlib import PurePath
        import subprocess
        import sys
        from tempfile import NamedTemporaryFile

        with NamedTemporaryFile(mode="w+", encoding="utf-8", suffix=".py", delete=False) as f:
            f.write("import os, subprocess, sys, traceback\n")
            f.write("if __name__ == \"__main__\":\n")
            f.write("\ttry:\n")

            f.write("\t\tsubprocess.run([\"" + PurePath(sys.executable).as_posix() + "\", \"-m\", \"ensurepip\"], check=True)\n")
            f.write("\t\tsubprocess.run([\"" + PurePath(sys.executable).as_posix() + "\", \"-m\", \"pip\", \"install\", \"" + \
                                        "\", \"".join([module[1] + ">=" + module[2] if module[2] != "" else module[1] for module in modules]) + \
                                        "\"], check=True)\n")

            f.write("\texcept:\n")
            f.write("\t\ttraceback.print_exc()\n")
            f.write("\t\tos.system(\"pause\")\n")

        print("nc-aae-export: " + "PowerShell -Command \"& {Start-Process \\\"" + sys.executable + "\\\" \\\"" + PurePath(f.name).as_posix() + "\\\" -Verb runAs -Wait}\"")
        os.system("PowerShell -Command \"& {Start-Process \\\"" + sys.executable + "\\\" \\\"" + PurePath(f.name).as_posix() + "\\\" -Verb runAs -Wait}\"")

class NCAAEExportRegisterPreferencePanel(bpy.types.AddonPreferences):
    bl_idname = __name__
    
    def draw(self, context):
        layout = self.layout
        
        layout.operator("preference.nc_aae_export_register_install_dependencies", icon="CONSOLE")

register_classes = (NCAAEExportRegisterInstallDependencies,
                    NCAAEExportRegisterPreferencePanel)
           
def register():
    import importlib.util
    if importlib.util.find_spec("packaging") != None:
        import packaging.version
    elif importlib.util.find_spec("distutils") != None: # distutils deprecated in Python 3.12
        import distutils.version
    
    for module in modules:
        if importlib.util.find_spec(module[0]) == None:
            register_register_classes()
            return

        if module[1]:
            exec("import " + module[0])
            module_version = eval(module[0] + ".__version__")
            if "packaging" in locals():
                if packaging.version.parse(module_version) < packaging.version.parse(module[2]):
                    register_register_classes()
                    return
            elif "distutils" in locals(): # distutils deprecated in Python 3.12
                if distutils.version.LooseVersion(module_version) < distutils.version.LooseVersion(module[2]):
                    register_register_classes()
                    return
    else:
        global is_dependencies_ready
        is_dependencies_ready = True
        
        register_main_classes()

def register_main_classes():
    for class_ in classes:
        bpy.utils.register_class(class_)
        
    bpy.types.Screen.NCAAEExportSettings = bpy.props.PointerProperty(type=NCAAEExportSettings)

def register_register_classes():
    for class_ in register_classes:
        bpy.utils.register_class(class_)
    
def unregister():
    if not is_dependencies_ready:
        unregister_register_class()
    else:
        unregister_main_class()

def unregister_main_class():
    del bpy.types.Screen.NCAAEExportSettings
    
    for class_ in classes:
        bpy.utils.unregister_class(class_)

def unregister_register_class():
    for class_ in register_classes:
        bpy.utils.unregister_class(class_)

if __name__ == "__main__":
    register()
#    unregister() 















# import bpy, mathutils, math

# def write_files(prefix, context):
#     scene = context.scene
#     fps = scene.render.fps / scene.render.fps_base

#     clipno = 0

#     for clip in bpy.data.movieclips:
#         trackno = 0

#         for track in clip.tracking.tracks:
#             with open("{0}_c{1:02d}_t{2:02d}.txt".format(prefix, clipno, trackno), "w") as f:

#                 frameno = clip.frame_start
#                 startarea = None
#                 startwidth = None
#                 startheight = None
#                 startrot = None

#                 data = []

#                 f.write("Adobe After Effects 6.0 Keyframe Data\r\n\r\n")
#                 f.write("\tUnits Per Second\t{0:.3f}\r\n".format(fps))
#                 f.write("\tSource Width\t{0}\r\n".format(clip.size[0]))
#                 f.write("\tSource Height\t{0}\r\n".format(clip.size[1]))
#                 f.write("\tSource Pixel Aspect Ratio\t1\r\n")
#                 f.write("\tComp Pixel Aspect Ratio\t1\r\n\r\n")

#                 while frameno <= clip.frame_duration:
#                     marker = track.markers.find_frame(frameno)
#                     frameno += 1

#                     if not marker or marker.mute:
#                         continue

#                     coords = marker.co
#                     corners = marker.pattern_corners

#                     area = 0
#                     width = math.sqrt((corners[1][0] - corners[0][0]) * (corners[1][0] - corners[0][0]) + (corners[1][1] - corners[0][1]) * (corners[1][1] - corners[0][1]))
#                     height = math.sqrt((corners[3][0] - corners[0][0]) * (corners[3][0] - corners[0][0]) + (corners[3][1] - corners[0][1]) * (corners[3][1] - corners[0][1]))
#                     for i in range(1,3):
#                         x1 = corners[i][0] - corners[0][0]
#                         y1 = corners[i][1] - corners[0][1]
#                         x2 = corners[i+1][0] - corners[0][0]
#                         y2 = corners[i+1][1] - corners[0][1]
#                         area += x1 * y2 - x2 * y1

#                     area = abs(area / 2)

#                     if startarea == None:
#                         startarea = area
                        
#                     if startwidth == None:
#                         startwidth = width
#                     if startheight == None:
#                         startheight = height

#                     zoom = math.sqrt(area / startarea) * 100
                    
#                     xscale = width / startwidth * 100
#                     yscale = height / startheight * 100

#                     p1 = mathutils.Vector(corners[0])
#                     p2 = mathutils.Vector(corners[1])
#                     mid = (p1 + p2) / 2
#                     diff = mid - mathutils.Vector((0,0))

#                     rotation = math.atan2(diff[0], diff[1]) * 180 / math.pi

#                     if startrot == None:
#                         startrot = rotation
#                         rotation = 0
#                     else:
#                         rotation -= startrot - 360

#                     x = coords[0] * clip.size[0]
#                     y = (1 - coords[1]) * clip.size[1]

#                     data.append([marker.frame, x, y, xscale, yscale, rotation])

#                 posline = "\t{0}\t{1:.3f}\t{2:.3f}\t0"
#                 scaleline = "\t{0}\t{1:.3f}\t{2:.3f}\t100"
#                 rotline = "\t{0}\t{1:.3f}"

#                 positions = "\r\n".join([posline.format(d[0], d[1], d[2]) for d in data]) + "\r\n\r\n"
#                 scales = "\r\n".join([scaleline.format(d[0], d[3], d[4]) for d in data]) + "\r\n\r\n"
#                 rotations = "\r\n".join([rotline.format(d[0], d[5]) for d in data]) + "\r\n\r\n"

#                 f.write("Anchor Point\r\n")
#                 f.write("\tFrame\tX pixels\tY pixels\tZ pixels\r\n")
#                 f.write(positions)

#                 f.write("Position\r\n")
#                 f.write("\tFrame\tX pixels\tY pixels\tZ pixels\r\n")
#                 f.write(positions)

#                 f.write("Scale\r\n")
#                 f.write("\tFrame\tX percent\tY percent\tZ percent\r\n")
#                 f.write(scales)

#                 f.write("Rotation\r\n")
#                 f.write("\tFrame Degrees\r\n")
#                 f.write(rotations)

#                 f.write("End of Keyframe Data\r\n")

#                 trackno += 1

#             clipno += 1
#     return {'FINISHED'}

# from bpy_extras.io_utils import ExportHelper
# from bpy.props import StringProperty

# class ExportAFXKey(bpy.types.Operator, ExportHelper):
#     """Export motion tracking markers to Adobe After Effects 6.0 compatible files"""
#     bl_idname = "export.afxkey"
#     bl_label = "Export to Adobe After Effects 6.0 Keyframe Data"
#     filename_ext = ""
#     filter_glob = StringProperty(default="*", options={'HIDDEN'})

#     def execute(self, context):
#         return write_files(self.filepath, context)

# classes = (
#     ExportAFXKey,
# )

# def menu_func_export(self, context):
#     self.layout.operator(ExportAFXKey.bl_idname, text="Adobe After Effects 6.0 Keyframe Data")


# def register():
#     from bpy.utils import register_class
#     for cls in classes:
#         register_class(cls)
#     bpy.types.TOPBAR_MT_file_export.append(menu_func_export)

# def unregister():
#     from bpy.utils import unregister_class
#     for cls in reversed(classes):
#         unregister_class(cls)
#     bpy.types.TOPBAR_MT_file_export.remove(menu_func_export)

# if __name__ == "__main__":
#     register()
