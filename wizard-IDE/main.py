import sys
from pathlib import Path
from PySide6.QtGui import QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine, qmlRegisterType
from PySide6.QtQuickControls2 import QQuickStyle
from bridge import Bridge

QML_IMPORT_NAME = "io.qt.textproperties"
QML_IMPORT_MAJOR_VERSION = 1

class Convert:
    __ElementToAtomicNumber = {"H":1,"He":2,"Li":3,"Be":4,"B":5,"C":6,"N":7,"O":8,"F":9,"Ne":10,"Na":11,"Mg":12,"Al":13,"Si":14,"P":15,"S":16,"Cl":17,"Ar":18,"K":19,"Ca":20,"Sc":21,"Ti":22,"V":23,"Cr":24,"Mn":25,"Fe":26,"Co":27,"Ni":28,"Cu":29,"Zn":30,"Ga":31,"Ge":32,"As":33,"Se":34,"Br":35,"Kr":36,"Rb":37,"Sr":38,"Y":39,"Zr":40,"Nb":41,"Mo":42,"Tc":43,"Ru":44,"Rh":45,"Pd":46,"Ag":47,"Cd":48,"In":49,"Sn":50,"Sb":51,"Te":52,"I":53,"Xe":54,"Cs":55,"Ba":56,"La":57,"Ce":58,"Pr":59,"Nd":60,"Pm":61,"Sm":62,"Eu":63,"Gd":64,"Tb":65,"Dy":66,"Ho":67,"Er":68,"Tm":69,"Yb":70,"Lu":71,"Hf":72,"Ta":73,"W":74,"Re":75,"Os":76,"Ir":77,"Pt":78,"Au":79,"Hg":80,"Tl":81,"Pb":82,"Bi":83,"Po":84,"At":85,"Rn":86,"Fr":87,"Ra":88,"Ac":89,"Th":90,"Pa":91,"U":92,"Np":93,"Pu":94,"Am":95,"Cm":96,"Bk":97,"Cf":98,"Es":99,"Fm":100,"Md":101,"No":102,"Lr":103,"Rf":104,"Db":105,"Sg":106,"Bh":107,"Hs":108,"Mt":109,"Ds":110,"Rg":111,"Cn":112,"Nh":113,"Fl":114,"Mc":115,"Lv":116,"Ts":117,"Og":118}
    __AtomicNumberToAtomicMass = [0.000,1.008,4.003,6.941,9.012,10.81,12.01,14.01,16.00,19.00,20.18,22.99,24.30,26.98,28.09,30.97,32.04,35.45,39.95,39.10,40.08,44.96,47.87,50.94,52.00,54.94,55.84,58.93,58.69,63.55,65.38,69.72,72.63,74.92,78.97,79.90,83.80,85.47,87.62,88.91,91.22,92.91,95.95,97,101.1,102.9,106.4,107.9,112.4,114.8,118.7,121.8,127.6,126.9,131.3,132.9,137.3,138.9,140.1,140.9,144.2,145,150.4,152.0,157.2,158.9,162.5,164.9,167.3,168.9,173.0,175.0,178.5,180.9,183.8,186.2,190.2,192.2,195.1,197.0,200.6,204.4,207.2,209.0,209,210,222,223,226,227,232.0,231.0,238.0,237,244,243,247,247,251,252,257,258,259,266,267,268,269,270,269,278,281,282,285,286,289,290,293,294,294]
    __AtomicNumberToValenceElectron = ["","s","p","s","s","p","p","p","p","p","p","s","s","p","p","p","p","p","p","s","s","d","d","d","d","d","d","d","d","d","d","p","p","p","p","p","p","s","s","d","d","d","d","d","d","d","d","d","d","p","p","p","p","p","p","s","s","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","d","d","d","d","d","d","d","d","d","p","p","p","p","p","p","s","s","f","f","f","f","f","f","f","f","f","f","f","f","f","f","f","d","d","d","d","d","d","d","d","d","p","p","p","p","p","p"]
    __AtomicNumberToName = ["","Hydrogen","Helium","Lithium","Beryllium","Boron","Carbon","Nitrogen","Oxygen","Fluorine","Neon","Sodium","Magnesium","Aluminum","Silicon","Phosphorus","Sulfur","Chlorine","Argon","Potassium","Calcium","Scandium","Titanium","Vanadium","Chromium","Manganese","Iron","Cobalt","Nickel","Copper","Zinc","Gallium","Germanium","Arsenic","Selenium","Bromine","Krypton","Rubidium","Strontium","Yttrium","Zirconium","Niobium","Molybdenum","Technetium","Ruthenium","Rhodium","Palladium","Silver","Cadmium","Indium","Tin","Antimony","Tellurium","Iodine","Xenon","Cesium","Barium","Lanthanum","Cerium","Praseodymium","Neodymium","Promethium","Samarium","Europium","Gadolinium","Terbium","Dysprosium","Holmium","Erbium","Thulium","Ytterbium","Lutetium","Hafnium","Tantalum","Tungsten","Rhenium","Osmium","Iridium","Platinum","Gold","Mercury","Thallium","Lead","Bismuth","Polonium","Astatine","Radon","Francium","Radium","Actinium","Thorium","Protactinium","Uranium","Neptunium","Plutonium","Americium","Curium","Berkelium","Californium","Einsteinium","Fermium","Mendelevium","Nobelium","Lawrencium","Rutherfordium","Dubnium","Seaborgium","Bohrium","Hassium","Meitnerium","Darmstadtium","Roentgenium","Copernicium","Nihonium","Flerovium","Moscovium","Livermorium","Tennessine","Oganesson"]
    __AtomicNumberToElement = ["","H","He","Li","Be","B","C","N","O","F","Ne","Na","Mg","Al","Si","P","S","Cl","Ar","K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Ge","As","Se","Br","Kr","Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","In","Sn","Sb","Te","I","Xe","Cs","Ba","La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu","Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg","Tl","Pb","Bi","Po","At","Rn","Fr","Ra","Ac","Th","Pa","U","Np","Pu","Am","Cm","Bk","Cf","Es","Fm","Md","No","Lr","Rf","Db","Sg","Bh","Hs","Mt","Ds","Rg","Cn","Nh","Fl","Mc","Lv","Ts","Og"]
    def atomic_number_to_mass(atomic_number:int):
        return Convert.__AtomicNumberToAtomicMass[atomic_number]
    def atomic_number_to_valence_electron(atomic_number:int):
        return Convert.__AtomicNumberToValenceElectron[atomic_number]
    def atomic_number_to_name(atomic_number:int):
        return Convert.__AtomicNumberToName[atomic_number]
    def atomic_number_to_element(atomic_number:int):
        return Convert.__AtomicNumberToElement[atomic_number]
    def element_to_atomic_number(element:str):
        return Convert.__ElementToAtomicNumber[element]
    def element_to_atomic_mass(element:str):
        return Convert.__AtomicNumberToAtomicMass[Convert.element_to_atomic_number(element)]
    def element_to_atomic_valence_electron(element:str):
        return Convert.__AtomicNumberToValenceElectron[Convert.element_to_atomic_number(element)]
    def element_to_atomic_name(element:str):
        return Convert.__AtomicNumberToName[Convert.element_to_atomic_number(element)]

if __name__ == '__main__':
	app = QGuiApplication(sys.argv)
	QQuickStyle.setStyle("Fusion")
	engine = QQmlApplicationEngine()
	qmlRegisterType(Bridge, QML_IMPORT_NAME, QML_IMPORT_MAJOR_VERSION, 0, "Bridge")
	qml_file = Path(__file__).parent / 'qml/Main.qml'
	engine.load(qml_file)
	if not engine.rootObjects():
		sys.exit(-1)
	sys.exit(app.exec())