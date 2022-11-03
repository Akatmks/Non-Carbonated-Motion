## nc-aae-export.py *[Not ready]*

Number Crunching AAE Export is an [Aegisub-Motion](https://github.com/TypesettingTools/Aegisub-Motion/) compatible script to export tracking data from Blender.  

Instead of using a single tracking marker or plane, Number Crunching Motion uses as many markers as your computer could possibly handle. It will check the markers fbf and remove the markers that are probably not working very well. Then it will take the average of all the remaining markers and apply it to the subtitle. That's where the „Number Crunching“ in Number Crunching Motion comes from.  

Number Crunching AAE Export supports:  
* `\pos`  
* `\fscx` and `\fscy`  

## aka.NumberCrunchingMotion.lua & nc-mo-export.py *[Not ready]*

Number Crunching Motion is an Aegisub automation script that takes raw tracking data from Blender and applies it to the subtitle in a similar manner as [Aegisub-Motion](https://github.com/TypesettingTools/Aegisub-Motion/).  

Instead of using a single tracking marker or plane, Number Crunching Motion uses as many markers as your computer could possibly handle. It will check the markers fbf and remove the markers that are probably not working very well. Then it will take the average of all the remaining markers and apply it to the subtitle. That's where the „Number Crunching“ in Number Crunching Motion comes from.  

*Number Crunching Motion will be looking for maintainers to help build the binaries.*  

Number Crunching Motion supports: 
* `\pos`  
* `\fscx` and `\fscy`  
* `\frz` and `\fax`  

Akatsumekusa is looking for algorithms to archive `\frx`, `\fry` and `\frz`.  
