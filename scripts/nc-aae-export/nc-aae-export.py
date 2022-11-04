# nc-aae-export.py
# Copyright (c) Akatsumekusa

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

bl_info = {
    "name": "Number Crunching AAE Export",
    "description": "Export Aegisub-Motion compatible AAE data to file",
    "author": "Akatsumekusa",
    "support": "COMMUNITY",
    "category": "Video Tools",
    "blender": (2, 93, 0),
    "location": "Clip Editor > Tools > Solve > NC AAE Export",
    "warning": "",
    "doc_url": "https://github.com/Akatmks/Number-Crunching-Motion",
    "tracker_url": "https://github.com/Akatmks/Number-Crunching-Motion/issues"
}

import bpy

# ("import name", "PyPI name")
modules = (("numpy", "numpy"), ("scipy", "scipy"), ("matplotlib", "matplotlib"))

is_dependencies_ready = False

class NCAAEExportSettings(bpy.types.PropertyGroup):
    bl_label = "NCAAEExportSettings"
    bl_idname = "NCAAEExportSettings"
    
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
    bl_description = "Export AAE data as txt file next to the original movie clip"
    bl_idname = "movieclip.nc_aae_export_export"

    def execute(self, context):
        _export(context.edit_movieclip, context.screen.NCAAEExportSettings)
        return {'FINISHED'}
    
    def _export(self, clip, settings):
        print(clip.filepath)

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
        
        column = layout.column()
        column.prop(settings, "do_smoothing")
        column.prop(settings, "do_statistics")
        
        column = layout.column(heading="Result")
        column.prop(settings, "do_do_not_overwrite")
        column.prop(settings, "do_copy_to_clipboard")
        
        row = layout.row()
        row.scale_y = 2
        row.operator("movieclip.nc_aae_export_export")

classes = (NCAAEExportSettings,
           NCAAEExportExport,
           NCAAEExport)

class NCAAEExportRegisterInstallDependencies(bpy.types.Operator):
    bl_label = "Install dependencies"
    bl_description = "NC AAE Export requires additional packages to be installed.\nBy clicking this button, NC AAE Export will download and install " + \
                     (" and ".join([", ".join(["pip"] + [module[0] for module in modules[:-1]]), modules[-1][0]]) if len(modules) != 0 else "pip") + \
                     " into your Blender distribution"
    bl_idname = "preference.nc_aae_export_register_install_dependencies"
    bl_options = {"REGISTER", "INTERNAL"}

    def execute(self, context):
        import importlib.util
        import os
        import subprocess
        import sys

        if os.name == "nt":
            self._execute_nt(context)
            
            for module in modules:
                if importlib.util.find_spec(module[0]) == None:
                    return {'FINISHED'}
        else:
            subprocess.run([sys.executable, "-m", "ensurepip"], check=True) # "blender": (2, 93, 0)
            subprocess.run([sys.executable, "-m", "pip", "install"] + [module[1] for module in modules], check=True)

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
                                        "\", \"".join([module[1] for module in modules]) + \
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
    for module in modules:
        if importlib.util.find_spec(module[0]) == None:
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
