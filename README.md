## nc-aae-export.py *[Not ready]*

Non Carbonated AAE Export is an [Aegisub-Motion](https://github.com/TypesettingTools/Aegisub-Motion/) compatible script to export tracking data from Blender.  

Instead of meticulously creating and tracking a single marker or plane track, Non Carbonated AAE Export accepts as many markers as you can possibly put on the screen. With the power of machine learning, Non Carbonated AAE Export can automatically deal with tracking errors and inaccuracies, merge all the tracks together for the best precision and output tracking data for Aegisub-Motion effortlessly, mostly.

Non Carbonated AAE Export supports:  
* `\pos`  
* `\fscx` and `\fscy`  

Akatsumekusa is looking for potential methods to support `\frx`, `\fry` and `\frz`.  

## aka.NonCarbonatedMotion.lua & nc-mo.py *[Not ready]*

Non Carbonated Motion is a Python-based program with Aegisub interface that takes raw tracking data from Blender and applies it to the subtitle in a similar manner as [Aegisub-Motion](https://github.com/TypesettingTools/Aegisub-Motion/).  

Non Carbonated Motion supports: 
* `\pos`  
* `\fscx` and `\fscy`  
* `\frx`, `\fry` and `\frz`  
* `\frz` and `\fax`  
