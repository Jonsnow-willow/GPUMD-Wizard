from ovito.io import import_file
from ovito.modifiers import DislocationAnalysisModifier

pipeline = import_file('dump.xyz')

modifier = DislocationAnalysisModifier(trial_circuit_length = 14,circuit_stretchability = 9)
modifier.input_crystal_structure = DislocationAnalysisModifier.Lattice.BCC
pipeline.modifiers.append(modifier)

data = pipeline.compute(pipeline.source.num_frames-1)
time = data.attributes['Time'] 
print("time: %s" % time)
cell_volume = data.attributes['DislocationAnalysis.cell_volume']
print("Cell volume: %f" % cell_volume)
total_line_length = data.attributes['DislocationAnalysis.total_line_length']
print("Total dislocation line length: %f" % total_line_length)
print("Found %i dislocation segments" % len(data.dislocations.segments))
for segment in data.dislocations.segments:
    print("Segment %i: length=%f, Burgers vector=%s" % (segment.id, segment.length, segment.true_burgers_vector))