from ovito.io import *
from ovito.data import *
from ovito.modifiers import *
from ovito.pipeline import *

pipeline = import_file('dump.xyz')
print("Total number of frames: %d" % pipeline.source.num_frames)

ws = WignerSeitzAnalysisModifier()
pipeline.modifiers.append(ws)

for frame in range(pipeline.source.num_frames):
    data = pipeline.compute(frame)
    time = data.attributes['Time']
    vacancy_count = data.attributes['WignerSeitz.vacancy_count']
    print("Frame %d: time = %f, point defect count = %d" % (frame, time, vacancy_count))