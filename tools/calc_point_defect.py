from ovito.io import *
from ovito.data import *
from ovito.modifiers import *
from ovito.pipeline import *
import numpy as np

# Load the input file
pipeline = import_file('/Volumes/Elements SE/辐照损伤工作/100keV/Cubic/0/dump.xyz')

# Print the total number of frames in the trajectory
print("Total number of frames: %d" % pipeline.source.num_frames)

# Add the Wigner-Seitz analysis modifier to the pipeline
ws = WignerSeitzAnalysisModifier()
pipeline.modifiers.append(ws)

# Loop over all frames in the trajectory
for frame in range(pipeline.source.num_frames):
    # Compute the Wigner-Seitz analysis for the current frame
    data = pipeline.compute(frame)
    
    # Get the simulation time of the current frame
    time = data.attributes['Time']
    
    # Get the number of vacancies in the current frame
    vacancy_count = data.attributes['WignerSeitz.vacancy_count']
    
    # Print the simulation time and vacancy count for the current frame
    print("Frame %d: time = %f, vacancy count = %d" % (frame, time, vacancy_count))