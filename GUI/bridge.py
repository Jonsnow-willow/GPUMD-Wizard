from PySide6.QtCore import QObject, Slot
from wizard.calculator import MaterialCalculator
from wizard.atoms import SymbolInfo
from calorine.calculators import CPUNEP
from calorine.nep import read_model
import threading
import os

class Bridge(QObject):
	def __init__(self):
		self.elements = []
		self.historical = []
		self.lock = threading.Lock()
		self.result = ""
		QObject.__init__(self)

	@Slot(str, result=list)
	def read_force_field(self, s):
		with open(s, 'r') as f:
			force_field = f.readlines()
		self.elements = force_field[0].split()[2:]
		return self.elements

	@Slot(str, result=list)
	def get_force_field(self, s):
		if s:
			return str(read_model(s)).split('\n')

	def run(self):
		now_path = os.getcwd()
		results_path = os.path.dirname(self.args[0])
		counter = 0
		while os.path.exists(os.path.join(results_path, self.args[1], str(counter))):
			counter += 1
		results_name = os.path.join(results_path, self.args[1], str(counter))
		os.makedirs(results_name)
		os.chdir(results_name)

		symbol_info = SymbolInfo(self.args[1], ['fcc', 'bcc', 'hcp'][self.args[2]], float(self.args[3]), float(self.args[4]) if self.args[4] else 0)
		atoms = symbol_info.create_bulk_atoms()
		calc = CPUNEP(self.args[0])
		material_calculator = MaterialCalculator(atoms, calc, symbol_info)
		#lattice_constant
		material_calculator.lattice_constant()
		#elastic_constants
		material_calculator.elastic_constant()
		#eos_curve
		material_calculator.eos_curve()
		#phonon_dispersion
		material_calculator.phonon_dispersion()
		#formation_energy_vacancy
		material_calculator.formation_energy_vacancy()
		#migration_energy_vacancy
		material_calculator.migration_energy_vacancy()
		#formation_energy_divacancies
		for nth in [1,2,3]:
			material_calculator.formation_energy_divacancies(nth)
		#formation_energy_surface
		for miller in [(1,1,0),(0,0,1),(1,1,1),(1,1,2),(2,1,0),(2,2,1),(3,1,1),(3,1,0),(3,2,1),(3,2,0)]:
			material_calculator.formation_energy_surface(miller)
		#stacking_fault
		if symbol_info.structure == 'fcc':
			material_calculator.stacking_fault(a = (1,0,1), b = (1,2,-1), miller='0', distance = symbol_info.lattice_constant[0]/2, nlayers= 12)
			material_calculator.stacking_fault(a = (1,1,0), b = (-1,1,0), miller='1', distance = symbol_info.lattice_constant[0]/2, nlayers= 12)
		elif symbol_info.structure == 'bcc':
			material_calculator.stacking_fault(a = (-1,1,1), b = (-1,1,-2), miller='0', distance = symbol_info.lattice_constant[0]/2, nlayers= 12)
			material_calculator.stacking_fault(a = (1,1,-1), b = (1,-1,0), miller='1', distance = symbol_info.lattice_constant[0]/2, nlayers= 12)
		else:
			material_calculator.stacking_fault(a = (1,-1,0), b = (1,0,0), miller='0', distance = 1.1 * symbol_info.lattice_constant[0], nlayers= 14)
			material_calculator.stacking_fault(a = (1,-2,0), b = (1,1,0), miller='1', distance = 0.45 * symbol_info.lattice_constant[0], nlayers= 14)

		#screw_dipole
		material_calculator.bcc_metal_screw_dipole_move()
		
		#screw_one
		material_calculator.bcc_metal_screw_one_move()
		os.chdir(now_path)

		with self.lock:
			self.result = os.path.join(results_name, f"{self.args[1]}_phono.png")
			self.historical.append({"name": self.args[1], "data": self.result})
		print("phonon_dispersion done", self, self.result)
		
	@Slot(list, result=bool)
	def calculate_properties(self, args):
		print("calculation start")
		with self.lock:
			self.result = ""
		self.args = args
		self.thread = threading.Thread(target=self.run)
		self.thread.start()

	@Slot(result=str)
	def get_result(self):
		print(self, self.result)
		with self.lock:
			return self.result

	@Slot(result=list)
	def get_historical(self):
		with self.lock:
			return self.historical