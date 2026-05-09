from ase import Atoms, Atom
from ase.build import bulk
from wizard.utils.io import dump_xyz, write_run
import numpy as np
import random, os, re, shutil

SUPPORTED_LATTICE_TYPES = {"bcc", "fcc", "hcp"}
INTERSTITIAL_SITES = {
    "bcc": {
        "oct": [
            (0.5, 0.0, 0.0),
            (0.0, 0.5, 0.0),
            (0.0, 0.0, 0.5),
            (0.5, 0.5, 0.0),
            (0.5, 0.0, 0.5),
            (0.0, 0.5, 0.5),
        ],
        "tet": [
            (0.5, 0.25, 0.0),
            (0.5, 0.75, 0.0),
            (0.25, 0.5, 0.0),
            (0.75, 0.5, 0.0),
            (0.5, 0.0, 0.25),
            (0.5, 0.0, 0.75),
            (0.25, 0.0, 0.5),
            (0.75, 0.0, 0.5),
            (0.0, 0.5, 0.25),
            (0.0, 0.5, 0.75),
            (0.0, 0.25, 0.5),
            (0.0, 0.75, 0.5),
        ],
    },
    "fcc": {
        "oct": [
            (0.5, 0.5, 0.5),
            (0.5, 0.0, 0.0),
            (0.0, 0.5, 0.0),
            (0.0, 0.0, 0.5),
        ],
        "tet": [
            (0.25, 0.25, 0.25),
            (0.25, 0.25, 0.75),
            (0.25, 0.75, 0.25),
            (0.25, 0.75, 0.75),
            (0.75, 0.25, 0.25),
            (0.75, 0.25, 0.75),
            (0.75, 0.75, 0.25),
            (0.75, 0.75, 0.75),
        ],
    },
    "hcp": {
        "tet": [
            (1.0/3.0, 2.0/3.0, 0.125),
            (1.0/3.0, 2.0/3.0, 0.875),
            (0.0, 0.0, 0.375),
            (0.0, 0.0, 0.625),
        ],
        "oct": [
            (2.0/3.0, 1.0/3.0, 0.25),
            (2.0/3.0, 1.0/3.0, 0.75),
        ]
    },
}

SCREW_Z_DISPLACEMENTS = {
    "initial": np.array([
        [0.4383058734, -0.5714240022, 0.4151336435, 0.3961204643, -0.6309063217, 0.3322946896, 0.2883053039, -0.7537754388, 0.2151956583, 0.1982659655, -0.8050183462, 0.2050780078, 0.2292011390, -0.7333834610, 0.3110325001],
        [0.4344421505, -0.5798901201, 0.3975878411, 0.3610489064, -0.6949430574, 0.2387570089, 0.1864030372, -0.8432689054, 0.1453517187, 0.1490039011, -0.8311326886, 0.2097556482, 0.2717639117, -0.6644765268, 0.3814921364],
        [0.7787977309, -0.2345665212, -0.2593515042, 0.6885798318, -0.4133024290, -0.5124043151, 0.4400319242, -0.5785892723, -0.5818869915, 0.4280450138, -0.5416154296, -0.4692720367, 0.6437723297, -0.2805461098, -0.2450863980],
        [0.1312221737, 0.1242487725, 0.1073637160, 0.0366938199, -0.2340569474, -0.2987931633, -0.3137580716, -0.3181724220, -0.3170757312, -0.3090284456, -0.2804667143, -0.1007861417, 0.0865237820, 0.1173850825, 0.1281006320],
        [0.1722089804, 0.2034954394, 0.3939635901, 0.5709831644, 0.5991727560, 0.6071424949, 0.6081717677, 0.6036492767, 0.5884590428, 0.5225426277, 0.2511914104, 0.1817756459, 0.1650385479, 0.1581167904, 0.1545254544],
        [0.5706175747, -0.3530679936, 0.7601596169, 0.8320770482, -0.1378472244, 0.8719118310, 0.8684135390, -0.1504853587, 0.8014512119, 0.7017445505, -0.3998555688, 0.5484061973, 0.5237924857, -0.4894539748, 0.5026086621],
        [0.9547878071, 0.0187843014, 0.0807409755, 1.1214238637, 0.1410695179, 0.1444896156, 1.1328485171, 0.1028634855, 0.0501782834, 0.9837723664, -0.0720921626, -0.1084517293, 0.8691911289, -0.1449883909, -0.1544480419],
        [1.0237688431, 0.0610845188, 0.0850166502, 1.0948888287, 0.0913597276, 0.0741833997, 1.0429195704, 0.0007012102, -0.0432816464, 0.9200338150, -0.1068365315, -0.1256876419, 0.8610257581, -0.1485691039, -0.1557084126],
        [0.3495735044, 0.3726135444, 0.3858424561, 0.3884236680, 0.3802821731, 0.3617705152, 0.3345959072, 0.3028481838, 0.2719091335, 0.2455669179, 0.2248627411, 0.2091133662, 0.1971866364, 0.1880657203, 0.1809738499],
    ]),
    "one_move": np.array([
        [0.4479569984, -0.5588271941, 0.4320193461, 0.4193036567, -0.5987885891, 0.3753670155, 0.3400594095, -0.7021284560, 0.2580454569, 0.2301784997, -0.7819830492, 0.2218639168, 0.2417304120, -0.7237793187, 0.3185798515],
        [0.4464331172, -0.5626457752, 0.4236939185, 0.4020724853, -0.6332255906, 0.3123564801, 0.2477039491, -0.8026182088, 0.1712333181, 0.1661198843, -0.8192185129, 0.2184407948, 0.2783411638, -0.6593383734, 0.3856092650],
        [0.7885816540, -0.2190750813, -0.2319077259, 0.7441311336, -0.3064612532, -0.4065076370, 0.4947457771, -0.5514896609, -0.5665464171, 0.4377529324, -0.5349630393, -0.4644438253, 0.6474302910, -0.2776816471, -0.2427837151],
        [0.1356737111, 0.1322029740, 0.1254520427, 0.1090842856, 0.0414604782, -0.2291226663, -0.2960290446, -0.3103200421, -0.3126656587, -0.3062082393, -0.2785091702, -0.0993483456, 0.0876244758, 0.1182548500, 0.1288052863],
        [0.1607399862, 0.1711595711, 0.2012242673, 0.3846034508, 0.5677030761, 0.5958564515, 0.6024237412, 0.6001735443, 0.5861322488, 0.5208765537, 0.2499399412, 0.1808014198, 0.1642588316, 0.1574787159, 0.1539937073],
        [0.5321593366, -0.4332506661, 0.6410264295, 0.7530186912, -0.1757623203, 0.8518608506, 0.8563584559, -0.1584513826, 0.7958204260, 0.6975627836, -0.4030797690, 0.5458464686, 0.5217120889, -0.4911775863, 0.5011575861],
        [0.9036844148, -0.0515779206, 0.0106486707, 1.0707535881, 0.1087011151, 0.1235761105, 1.1186585293, 0.0927459089, 0.0426538641, 0.9779804958, -0.0766770653, -0.1121658272, 0.8661243164, -0.1475618812, -0.1566374317],
        [0.9707047189, 0.0131105734, 0.0477320289, 1.0677560581, 0.0717433145, 0.0597289219, 1.0319864489, -0.0077864721, -0.0500275483, 0.9145610856, -0.1113558937, -0.1294773123, 0.8578054433, -0.1513374297, -0.1581124157],
        [0.3088309652, 0.3364686530, 0.3563547308, 0.3653358698, 0.3623872890, 0.3478079517, 0.3235495044, 0.2939670906, 0.2646539915, 0.2395510706, 0.2198064931, 0.2048118974, 0.1934874308, 0.1848536369, 0.1781606670],
    ]),
    "dipole_move": np.array([
        [0.4456132135, -0.5615261241, 0.4288798139, 0.4156091034, -0.6031943968, 0.3700319247, 0.3334833667, -0.7104022148, 0.2473883867, 0.2160897260, -0.8011020001, 0.1954205085, 0.2053930308, -0.7705385783, 0.2668603744],
        [0.4442983267, -0.5651550685, 0.4207036810, 0.3984512141, -0.6376958322, 0.3067096631, 0.2403683213, -0.8124814743, 0.1574013976, 0.1457371065, -0.8507620825, 0.1690603475, 0.2100058123, -0.7279170328, 0.3358381985],
        [0.7871663084, -0.2207562312, -0.2339368138, 0.7416345810, -0.3096058281, -0.4105861126, 0.4892543809, -0.5592577967, -0.5782999365, 0.4182102635, -0.5718964044, -0.5414672292, 0.5311516162, -0.3557205006, -0.2802039671],
        [0.1351542482, 0.1315796579, 0.1246903337, 0.1081325431, 0.0402378464, -0.2307504627, -0.2983023424, -0.3137155175, -0.3182788711, -0.3172199974, -0.3091202073, -0.2803205913, -0.0992020058, 0.0868740664, 0.1176322760],
        [0.1614250994, 0.1720051327, 0.2022942549, 0.3860009982, 0.5696055901, 0.5985970786, 0.6067085818, 0.6077994815, 0.6033298358, 0.5881637813, 0.5218010593, 0.2505264948, 0.1817818002, 0.1651986002, 0.1583170583],
        [0.5344017930, -0.4304612237, 0.6445884249, 0.7577200226, -0.1692850212, 0.8613127615, 0.8712925683, -0.1320723805, 0.8490964574, 0.8009820177, -0.2987964244, 0.5998831938, 0.5484029766, -0.4761065906, 0.5106783130],
        [0.9076951584, -0.0465726834, 0.0170558349, 1.0792140580, 0.1203073230, 0.1402496517, 1.1438733958, 0.1323618793, 0.1024438571, 1.0498063772, -0.0165072001, -0.0722113500, 0.8915421367, -0.1307725407, -0.1449621816],
        [0.9780582789, 0.0224683294, 0.0599402170, 1.0841125925, 0.0941914548, 0.0908341220, 1.0737667461, 0.0425876060, 0.0004443772, 0.9565443355, -0.0800645585, -0.1068957536, 0.8742518581, -0.1390686642, -0.1487125781],
        [0.3184641822, 0.3485476151, 0.3717358761, 0.3851416462, 0.3878872075, 0.3798728495, 0.3614548915, 0.3343447793, 0.3026465771, 0.2717399390, 0.2454229905, 0.2247102556, 0.2089377928, 0.1969837404, 0.1878319406],
    ]),
}

class AlloyInfo():
    def __init__(self, formula, lattice_type, *lattice_constant):
        lattice_type = lattice_type.lower()
        if lattice_type not in SUPPORTED_LATTICE_TYPES:
            raise ValueError(
                f"Unsupported lattice type: {lattice_type}. Supported: {', '.join(sorted(SUPPORTED_LATTICE_TYPES))}"
            )
        self.formula = formula
        self.lattice_type = lattice_type
        self.lattice_constant = lattice_constant
        self.symbols = []
        self.compositions = []
        for symbol, composition in re.findall(r'([A-Z][a-z]*)(\d*)', formula):
            self.symbols.append(symbol)
            self.compositions.append(int(composition) if composition else 1)
    
    def create_bulk_atoms(self, supercell = (3, 3, 3)) -> Atoms:
        symbol, crystalstructure, lc = self.symbols[0], self.lattice_type, self.lattice_constant
        if crystalstructure == 'hcp':
            atoms = bulk(symbol, crystalstructure, a = lc[0], c = lc[1]) * supercell
        else:
            atoms = bulk(symbol, crystalstructure, a = lc[0], cubic = True) * supercell
        if len(self.symbols) > 1:
            if len(atoms) < sum(self.compositions):
                raise ValueError('The number of atoms in the unit cell is less than the number of symbols.')
            element_ratio = np.array(self.compositions) / sum(self.compositions)
            element_counts = np.ceil(element_ratio * len(atoms)).astype(int)
            symbols = np.repeat(self.symbols, element_counts)
            np.random.shuffle(symbols)
            atoms.set_chemical_symbols(symbols[:len(atoms)])
        atoms.info['formula'] = self.formula
        atoms.info['config_type'] = f'{self.formula}_{self.lattice_type}_bulk'
        return atoms

    def create_interstitial_atoms(self, supercell = (3, 3, 3), 
                                  intersitials = [
                                        {'symbol': 'C', 'type': 'oct', 'num': 5},
                                        {'symbol': 'H', 'type': 'tet', 'num': 10},
                                  ]) -> Atoms:
        atoms = self.create_bulk_atoms(supercell)
        cell = atoms.get_cell()
        used_sites = set()

        for intersitial in intersitials:
            symbol = intersitial['symbol']
            intersitial_type = intersitial['type']
            num = intersitial['num']
            sites = np.array(INTERSTITIAL_SITES[self.lattice_type][intersitial_type])
            interstitial_sites = []
            for i in range(supercell[0]):
                for j in range(supercell[1]):
                    for k in range(supercell[2]):
                        offset = np.array([i, j, k])
                        interstitial_sites.extend((sites + offset) / supercell)
            interstitial_sites = [site for site in interstitial_sites if tuple(site) not in used_sites]
            indices = np.random.choice(len(interstitial_sites), num, replace = False)

            for index in indices:
                scaled_position = interstitial_sites[index]
                used_sites.add(tuple(scaled_position))
                atoms.append(Atom(symbol = symbol, position = np.dot(scaled_position, cell)))
        atoms.wrap()
        return atoms
    
    def create_screw_atoms(self, model = 'bulk') -> Atoms:
        if self.lattice_type != 'bcc':
            raise ValueError('Screw dislocation atoms are only supported for bcc.')
        model = model.lower()
        if model not in {'bulk', 'initial', 'one_move', 'dipole_move'}:
            raise ValueError("model should be one of 'bulk', 'initial', 'one_move', or 'dipole_move'.")

        lc = self.lattice_constant[0]
        cell = np.array([
            [12.244558081272086, 0.0, 0.0],
            [6.122279038869259, 6.372678522968197, -0.43286219081272087],
            [0.0, 0.0, 0.8658175123674912],
        ])
        glide = cell[0, 0] / 15
        dy = cell[1, 1] / 9
        burgers = cell[2, 2]

        positions = []
        for j in range(9):
            x0 = (0.513562 + 0.5 * j + (j + 2) // 3) * glide
            y = (j + 0.341963) * dy
            for i in range(15):
                x = x0 + i * glide
                z = ((-i) % 3) * burgers / 3
                positions.append([x, y, z])
        positions = np.array(positions)

        if model in SCREW_Z_DISPLACEMENTS:
            positions[:, 2] += SCREW_Z_DISPLACEMENTS[model].reshape(-1) * burgers
        positions *= lc
        cell *= lc

        if len(self.symbols) > 1:
            element_ratio = np.array(self.compositions) / sum(self.compositions)
            element_counts = np.ceil(element_ratio * len(positions)).astype(int)
            symbols = np.repeat(self.symbols, element_counts)
            np.random.shuffle(symbols)
            symbols = symbols[:len(positions)]
        else:
            symbols = [self.symbols[0] for _ in range(len(positions))]

        atoms = Atoms(symbols = symbols, positions = positions, cell = cell, pbc = True)
        atoms.info['formula'] = self.formula
        atoms.info['config_type'] = f'{self.formula}_bcc_{model}_screw'
        return atoms

    def __str__(self):
        return f"Formula: {self.formula}, Lattice Type: {self.lattice_type}, Lattice Constant: {self.lattice_constant}"

class Morph():
    def __init__(self, atoms):
        if not isinstance(atoms, Atoms):
            raise TypeError("atoms must be an instance of ase.Atoms")
        self.atoms = atoms
        
    def gpumd(self, dirname = 'relax', run_in = ['potential nep.txt', 'velocity 300', 'time_step 1', 
             'ensemble npt_scr 300 300 200 0 500 2000', 'dump_thermo 1000', 'dump_restart 30000', 
             'dump_exyz 10000','run 30000'], nep_path = 'nep.txt', gpumd_path = 'gpumd',
              electron_stopping_path = 'electron_stopping_fit.txt', run = True):
        atoms = self.atoms
        if os.path.exists(dirname):
            raise FileExistsError('Directory already exists')
        os.makedirs(dirname)
        if os.path.exists(nep_path):
            shutil.copy(nep_path, dirname)
        else:
            raise FileNotFoundError('nep.txt does not exist')
        if os.path.exists(electron_stopping_path):
            shutil.copy(electron_stopping_path, dirname)
        original_directory = os.getcwd()
        os.chdir(dirname)
        write_run(run_in)
        dump_xyz('model.xyz', atoms)
        if run:
            os.system(gpumd_path)
        os.chdir(original_directory)

    def set_pka(self, energy, direction, index = None, symbol = None):
        atoms = self.atoms
        direction = np.asarray(direction)
        if atoms.has('momenta') is None:
            raise ValueError('The velocities of atoms are not set.')
        velocities = atoms.get_velocities()
        
        if index is None:
            center = np.dot([0.5, 0.5, 0.5], atoms.get_cell())
            if symbol is None:
                index = np.argmin(np.sum((atoms.positions - center)**2, axis=1))
            else:
                element_indices = [i for i, atom in enumerate(atoms) if atom.symbol == symbol]
                element_positions = atoms.positions[element_indices]
                index = element_indices[np.argmin(np.sum((element_positions - center)**2, axis=1))]

        mass = atoms[index].mass
        vx = pow(2 * energy / mass , 0.5) * direction[0] / pow(np.sum(direction ** 2), 0.5) / 10.18051
        vy = pow(2 * energy / mass , 0.5) * direction[1] / pow(np.sum(direction ** 2), 0.5) / 10.18051
        vz = pow(2 * energy / mass , 0.5) * direction[2] / pow(np.sum(direction ** 2), 0.5) / 10.18051
        velocities[index] = [vx, vy, vz]
        
        atoms_masses = atoms.get_masses() 
        total_mass = np.sum(atoms_masses)
        momentum = np.sum(velocities * atoms_masses[:, np.newaxis], axis=0) 
        velocities -= momentum / total_mass
        atoms.set_velocities(velocities)

        print(f'Index: {index}')
        print(f'Symbol: {atoms[index].symbol}')
        print(f'Position: {atoms[index].position[0]:.2f}, {atoms[index].position[1]:.2f}, {atoms[index].position[2]:.2f}')
        print(f'Mass: {atoms[index].mass:.2f}')
        print(f'Velocity: {vx:.4f}, {vy:.4f}, {vz:.4f} (Angstrom/fs)')
       
    def velocity(self, vx, vy, vz, group = 0):
        atoms = self.atoms
        if atoms.has('momenta') is None:
            raise ValueError('The velocities of atoms are not set.')
        velocities = atoms.get_velocities()
        for index in range(len(atoms)):
            if int(atoms.info['group'][index]) == group:
                velocities[index] = [vx, vy, vz]
        
        atoms_masses = atoms.get_masses() 
        total_mass = np.sum(atoms_masses)
        momentum = np.sum(velocities * atoms_masses[:, np.newaxis], axis=0) 
        velocities -= momentum / total_mass
        atoms.set_velocities(velocities)

    def zero_momentum(self):
        atoms = self.atoms
        if atoms.has('momenta') is None:
            raise ValueError('The velocities of atoms are not set.')
        masses = atoms.get_masses()[:, np.newaxis]
        total_mass = np.sum(masses)
        momentum = atoms.get_momenta()
        total_momentum = np.sum(momentum, axis=0)
        momentum -= masses * (total_momentum / total_mass)
        atoms.set_momenta(momentum)
    
    def shuffle_symbols(self):
        atoms = self.atoms
        s = atoms.get_chemical_symbols()
        random.shuffle(s)
        atoms.set_chemical_symbols(s)

    def coord_element_set(self, coord, symbol):
        atoms = self.atoms
        for atom in atoms:
            if np.allclose(atom.position, coord):
                atom.symbol = symbol
                break

    def random_center(self, index = None):
        atoms = self.atoms
        if index is None:
            index = np.random.randint(0, len(atoms))
        center = atoms.cell.diagonal() / 2
        diff = center - atoms[index].position
        for atom in atoms:
            atom.position += diff

        for atom in atoms:
            atom.position %= atoms.cell.diagonal()

    def scale_lattice(self, scale):
        atoms = self.atoms
        origin_cell = atoms.cell.copy()
        atoms.set_cell(scale * origin_cell, scale_atoms=True)

    def create_self_interstitial_atom(self, vector, symbol = None, index = 0):
        atoms = self.atoms
        if symbol is not None:
            atom = Atom(symbol, atoms[index].position - vector)
        else:
            atom = Atom(atoms[index].symbol, atoms[index].position - vector)
        atoms[index].position += vector
        atoms.append(atom)

    def create_random_interstitial(self, symbols, num=1):
        atoms_to_insert = []
        for _ in range(num):
            symbol = random.choice(symbols)
            atom = Atom(symbol=symbol)
            atoms_to_insert.append(atom)
        self.insert_atoms(atoms_to_insert)

    def create_vacancy(self, index = 0):
        del self.atoms[index]

    def create_vacancies(self, num_vacancies):
        atoms = self.atoms
        if num_vacancies > len(atoms):
            raise ValueError("num_vacancies should be less than or equal to the total number of atoms.")
        indices_to_remove = np.random.choice(len(atoms), num_vacancies, replace=False)
        removed_atoms = atoms[indices_to_remove]
        indices_to_remove = sorted(indices_to_remove, reverse=True)
        for index in indices_to_remove:
            del self.atoms[index]
        return removed_atoms
    
    def insert_atoms(self, atoms_to_insert, distance=1.2):
        indices_to_insert = np.random.choice(len(self.atoms), len(atoms_to_insert), replace=False)
        target_atoms = self.atoms[indices_to_insert]
        for atom_to_insert, target_atom in zip(atoms_to_insert, target_atoms):
            unit_vector = np.random.randn(3)  
            unit_vector /= np.linalg.norm(unit_vector)  
            displacement_vector = distance * unit_vector  
            new_position = target_atom.position + displacement_vector 
            atom_to_insert.position = new_position
            self.atoms.append(atom_to_insert)
            
    def create_fks(self, num_vacancies):
        removed_atoms = self.create_vacancies(num_vacancies)
        self.insert_atoms(removed_atoms)
