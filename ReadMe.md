TODO:  
Implement Data Augemntation 

PREPROCESSING:
    Total files removed: 61281
    ALL labels has 41188 images after removing all 2K pixel images. (Nothing was done on the MIXED folder).
    TODO: Check why each label has exactly the same amount of pictures with 2K pixels.
    NOTE: Total number of images (without MIXED) in train is  329504

Mean and Std on training data with 2K pixel images removed:
    Mean: [0.1887909322977066, 0.1887909322977066, 0.1887909322977066]
    Std:  [0.29729199409484863, 0.29729199409484863, 0.29729199409484863]

TODO: Implement padding

Better controll run so that it is fair 

==============================
      DIMENSION STATISTICS
==============================
Total Images: 383976
Width:  Mean = 156.4 | Std = 110.8 | Median = 136.0
Height: Mean = 70.3 | Std = 32.4 | Median = 68.0
Avg Aspect Ratio (W/H): 2.23

Suggested Input Size (Multiple of 32): 128