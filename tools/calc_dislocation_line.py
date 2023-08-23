from ovito.io import import_file
from ovito.modifiers import DislocationAnalysisModifier

pipeline = import_file('/Volumes/Elements SE/辐照损伤工作/100keV/Cubic/0/dump.xyz')

modifier = DislocationAnalysisModifier(trial_circuit_length = 14,circuit_stretchability = 9)
modifier.input_crystal_structure = DislocationAnalysisModifier.Lattice.BCC
modifier.output_type = 'all'
pipeline.modifiers.append(modifier)
data = pipeline.compute(pipeline.source.num_frames-1)
time = data.attributes['Time'] 
print("time: %s" % time)

# Get the total length of all dislocation lines
total_line_length = data.attributes['DislocationAnalysis.total_line_length']
print("Total dislocation line length: %f" % total_line_length)

# Get the length of each type of dislocation line
dislocations = modifier.output_dislocations
for dislocation_type in dislocations:
    length = dislocations[dislocation_type]
    print("Dislocation type %s length: %f" % (dislocation_type, length))

# Compute the dislocation density
cell_volume = data.attributes['DislocationAnalysis.cell_volume']
dislocation_density = total_line_length / cell_volume
print("Dislocation density: %f" % dislocation_density)