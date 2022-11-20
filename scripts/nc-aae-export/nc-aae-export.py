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
    do_predictive_smoothing: bpy.props.BoolProperty(name="Predictive smoothing",
                                                    description="Enable predictive final smoothing.\nThis allows final smoothing to artificially construct new data when it suspects existing data to be imprecise or existing data is missing. It will be helpful in cases such as when the tracking target fades or blurs out. It will also improve overall precision, especially when the movement per frame is small",
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
    
    class _method(Enum):
        # The initial value. ValueError should be raised if a frame is still
        # UNDEFINTED after Step 18
        UNDEFINED = -1
        # The frame type cannot be determined. Final smoothing is required to
        # generate the frame
        UNDETERMINED = 0
        # The frame doesn't have any tracking markers
        NOTHING = 1
        # Pure x/y, including still frames
        PURE_X_Y = 2
        # Scale x/y
        SCALE_X_Y = 4
        SCLAE_X_Y_UNSURE = 5

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

    def execute(self, context):
        clip = context.edit_movieclip
        settings = context.screen.NCAAEExportSettings

        ratio \
            = NCAAEExportExport._step_03_calculate_aspect_ratio( \
                  clip)
        position, movement \
            = NCAAEExportExport._step_04_create_position_and_movement_array_from_tracking_markers( \
                  clip, ratio)
        
        reduced_position, reduced_movement, origin, filtered_origin \
            = NCAAEExportExport._step_08_reduce_position_and_movement_array_and_calculate_origin_array_and_filtered_origin_array( \
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
        ratio : tuple[float]

        """
        ar = clip.size[0] / clip.size[1]
        # As of 2021/2022
        if ar < 1 / 1.35: # 9:16, 9:19 and higher videos
            return (1 / 1.35, 1 / 1.35 / ar)
        elif ar < 1: # vertical videos from 1:1, 3:4, up to 1:1.35
            return (ar, 1)
        elif ar <= 1.81: # 1:1, 4:3, 16:9, up to 1920 x 1061
            return (ar, 1)
        else: # Ultrawide
            return (1.81, 1.81 / ar)

    @staticmethod
    def _step_04_create_position_and_movement_array_from_tracking_markers(clip, ratio):
        """
        Create position and movement array from tracking markers. [Step 04]

        Parameters
        ----------
        clip : bpy.types.MovieClip
        ratio : tuple[float]
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
        # |         |           Track 1  Track 2  Track 3      |
        # +---------+------------------------------------------+
        # | Frame 1 | array([[  [x, y],  [x, y],  [x, y]   ],  |
        # | Frame 2 |        [  [x, y],  [x, y],  [x, y]   ],  |
        # | Frame 3 |        [  [x, y],  [x, y],  [x, y]   ],  |
        # | Frame 4 |        [  [x, y],  [x, y],  [x, y]   ],  |
        # | Frame 5 |        [  [x, y],  [x, y],  [x, y]   ]]) |
        # +---------+------------------------------------------+
        # There should be more calculations intraframe than interframe so
        # position and movement arrays put the frame as the first axis.
        #
        # The x and y value will have a pixel aspect ratio of 1:1. See Step 03
        # for the range where x and y value is on screen.
        # 
        # Also, on the topic of precision, NC AAE Export uses float64 across
        # the whole script
        position = np.full((clip.frame_duration, len(clip.tracking.tracks), 2), np.nan, dtype=np.float64)
        
        # This will become the slowest step of all the procedures
        for i, track in enumerate(clip.tracking.tracks):
            for marker in track.markers[1:] if not track.markers[0].is_keyed else track.markers:
                if not 0 < marker.frame <= clip.frame_duration:
                    continue
                if marker.mute:
                    continue
                position[marker.frame - 1][i] = marker.co

        position *= ratio

        # movement array structure
        # +-----------+------------------------------------------+
        # |           |           Track 1  Track 2  Track 3      |
        # +-----------+------------------------------------------+
        # | Frame 1/2 | array([[  [x ,y],  [x, y],  [x, y]   ],  |
        # | Frame 2/3 |        [  [x ,y],  [x, y],  [x, y]   ],  |
        # | Frame 3/4 |        [  [x ,y],  [x, y],  [x, y]   ],  |
        # | Frame 4/5 |        [  [x ,y],  [x, y],  [x, y]   ]]) |
        # +-----------+                                          |
        # | size -= 1 |                                          |
        # +-----------+------------------------------------------+
        return position, np.diff(position, axis=0)

    @staticmethod
    def _step_08_reduce_position_and_movement_array_and_calculate_origin_array_and_filtered_origin_array(position, movement):
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

        """
        import numpy as np

        frames = movement.shape[0]
        reduced_position = np.empty(frames, dtype=object)
        reduced_movement = np.empty(frames, dtype=object)
        origin = np.empty(frames, dtype=object)

        for frame in range(0, frames):
            reduced_position[frame], reduced_movement[frame] \
                = NCAAEExportExport._step_08__reduce_position_and_movement_array_per_frame( \
                      position[frame], movement[frame])
            
            if reduced_movement[frame].shape[0] >= 2:
                origin[frame] \
                    = NCAAEExportExport._step_08__calculate_origin_array_per_frame( \
                          reduced_position[frame], reduced_movement[frame])
            else:
                origin[frame] \
                    = np.empty((0, 2), dtype=np.float64)

        return reduced_position, reduced_movement, origin

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

        available_indexes = np.nonzero(~np.isnan(movement[:, 0]))

        return position[available_indexes], movement[available_indexes]

    @staticmethod
    def _step_08__calculate_origin_array_per_frame(reduced_position, reduced_movement):
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

        """
        import numpy as np

        select = NCAAEExportExport._triangular_number(reduced_movement.shape[0])
        first_indexes, second_indexes = select.get_complete_pairs()

        pizza = np.column_stack((reduced_position[first_indexes],
                                 reduced_movement[first_indexes],
                                 reduced_position[second_indexes],
                                 reduced_movement[second_indexes]))

        # https://stackoverflow.com/questions/563198/
        def eat(slice):
            if (j := np.cross(slice[2:4], slice[6:8])) == 0: # := requires Python 3.8 (Blender 2.93)
                return np.array([np.nan, np.nan], dtype=np.float64)
            else:
                return slice[0:2] + np.cross(slice[4:6] - slice[0:2], slice[6:8]) / j * slice[2:4]

        return np.apply_along_axis(eat, 1, pizza)

    @staticmethod
    def _step_18_decide_proper_method_and_count_scale_koma_uchi(reduced_movement, origin, max_koma_uchi):
        """
        Decide if the scale x/y method or the pure x/y method is suitable for
        each frame and count the scale コマ打ち. Position コマ打ち will be calculated
        in later functions. [Step 18]

        Parameters
        ----------
        reduced_movement : npt.NDArray[npt.NDArray[float64]]
        origin : npt.NDArray[npt.NDArray[float64]]
            Reduced movement array and origin array likely coming from step
            08.
        max_koma_uchi : bool
            NCAAEExportSettings.max_koma_uchi.
        
        """
        import numpy as np
        
        frames = movement.shape[0]
        method = np.empty(frames, dtype=object)
        filtered_origin = np.empty(frames, dtype=object)

        for frame in range(0, frames):
            if reduced_movement[frame].shape[0] == 0:
                filtered_origin[frame] \
                    = np.empty((0, 2), dtype=np.float64)
                method[frame] \
                    = NCAAEExportExport._method.NOTHING

            elif reduced_movement[frame].shape[0] == 1:
                filtered_origin[frame] \
                    = np.empty((0, 2), dtype=np.float64)
                method[frame] \
                    = NCAAEExportExport._method.UNDETERMINED

            elif reduced_movement[frame].shape[0] == 2:
                filtered_origin[frame] \
                    = np.empty((0, 2), dtype=np.float64)
                method[frame] \
                    = NCAAEExportExport._method.UNDETERMINED
            
            elif reduced_movement[frame].shape[0] == 3:
                if (function_return \
                        := NCAAEExportExport._step_18__check_origin_array_for_possible_scale_origin( \
                               origin[frame]))[0]:
                    filtered_origin[frame] \
                        = function_return[1]
                    method[frame] \
                        = NCAAEExportExport._method.SCLAE_X_Y_UNSURE
                else:
                    filtered_origin[frame] \
                        = np.empty((0, 2), dtype=np.float64)
                    method[frame] \
                        = NCAAEExportExport._method.UNDETERMINED
                
            else: # movement[frame].shape[0] >= 3
                filtered_origin[frame] \
                    = NCAAEExportExport._step_18__reduce_origin_array_and_remove_outliers_per_frame( \
                          origin[frame])

                if (function_return \
                        := NCAAEExportExport._step_18__check_origin_array_for_possible_scale_origin(\
                               origin[frame]))[0]:
                    filtered_origin[frame] \
                        = function_return[1]
                    method[frame] \
                        = NCAAEExportExport._method.SCLAE_X_Y
                        
                else:
                    filtered_origin[frame] \
                        = function_return[1]
                    method[frame] \
                        = NCAAEExportExport._method.PURE_X_Y
                        
        # koma_uchi

    @staticmethod
    def _step_18__check_origin_array_for_possible_scale_origin(origin):
        """
        Parameters
        ----------
        origin : npt.NDArray[float64]
            Origin array of the frame. Not empty.

        Returns
        -------
        is_exisiting : bool
            True if the scale origin is likely exisiting.
        filtered_origin : npt.NDArray[float64]
            The points in the origin array that is in the same cluster as the
            possible scale origin.

        """
        import numpy as np
        from sklearn.cluster import DBSCAN

        clustering = DBSCAN(min_samples=np.ceil(origin.shape[0] * 0.20).astype(int), eps=0.20)
        estimation = clustering.fix_predict(origin)

        cluster_sizes = np.bincount(estimation + 1)[1:]
        if cluster_sizes.shape[0] == 0:
            return False, np.empty((0, 2), dtype=np.float64)
        elif cluster_sizes.shape[0] == 1:
            return True, origin[np.nonzero(estimation == 0)]
        else:
            index_max = np.argmax(cluster_sizes)
            for index in range(cluster_sizes.shape[0]):
                if index == index_max:
                    continue

                if not cluster_size[index_max] > cluster_sizes[index] * 4/3:
                    return False, origin[np.nonzero(estimation == index_max)]
            else:
                return True, origin[np.nonzero(estimation == index__max)]




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
        row = column.row()
        row.enabled = settings.do_smoothing
        row.prop(settings, "do_predictive_smoothing")
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
