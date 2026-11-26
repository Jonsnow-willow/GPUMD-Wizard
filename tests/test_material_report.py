import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
from ase.build import bulk
from ase.calculators.emt import EMT

from wizard.calc.calculator import AlloyCalculator, MaterialCalculator
from wizard.structure.atoms import AlloyInfo


class MaterialReportTests(unittest.TestCase):
    def setUp(self):
        temporary_directory = TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        self.addCleanup(os.chdir, Path.cwd())
        os.chdir(temporary_directory.name)
        self.report = Path("MaterialProperties.out")

    def test_lattice_and_elastic_share_one_heading_and_keep_public_results(self):
        atoms = AlloyInfo("Cu", "fcc", 3.6).create_bulk_atoms((1, 1, 1))
        calculator = MaterialCalculator(atoms, EMT(), clamped=True)
        lattice = calculator.lattice_constant()
        stiffness = np.arange(36, dtype=float).reshape(6, 6)
        with patch("wizard.calc.calculator.get_elastic_stiffness_tensor", return_value=stiffness):
            elastic = calculator.elastic_constant()

        self.assertIsInstance(lattice, list)
        self.assertIn("Cell lengths", "\n".join(lattice))
        self.assertIsInstance(elastic["output"], list)
        np.testing.assert_array_equal(elastic["Cij"], stiffness)
        report = self.report.read_text(encoding="utf-8")
        self.assertEqual(report.count("Cu · FCC bulk · 4 atoms"), 1)
        self.assertIn(f"{calculator.atom_energy:.4f} eV/atom", report)
        self.assertIn("Elastic constants (GPa)", report)
        self.assertIn("C66", report)
        self.assertNotIn("Actual composition", report)

    def test_generated_nominal_formula_and_original_eos_filename(self):
        calculator = AlloyCalculator(AlloyInfo("CuAu", "fcc", 3.8), EMT(),
                                     supercell=(1, 1, 1), clamped=True)
        calculator.lattice_constant()
        report = self.report.read_text(encoding="utf-8")
        self.assertIn("CuAu · FCC bulk · 4 atoms", report)
        self.assertNotIn("Actual composition", report)

        stem = f"{calculator.atoms.get_chemical_formula()}_fcc_bulk"
        self.assertEqual(calculator.info, stem)
        image = Path(calculator.eos_curve())
        self.assertEqual(image.name, f"{stem}_eos_curve.png")
        self.assertTrue(image.is_file())
        self.assertTrue((Path("eos_curve_out") / f"{stem}_eos_curve.out").is_file())

    def test_generated_off_ratio_supercell_displays_its_actual_composition_once(self):
        atoms = AlloyInfo("Cu2Au", "fcc", 3.8).create_bulk_atoms((1, 1, 1))
        calculator = MaterialCalculator(atoms, EMT(), clamped=True)
        calculator.lattice_constant()
        calculator.lattice_constant()
        report = self.report.read_text(encoding="utf-8")
        self.assertEqual(report.count("Cu2Au · FCC bulk · 4 atoms"), 1)
        composition_lines = [line for line in report.splitlines() if "Actual composition" in line]
        self.assertEqual(len(composition_lines), 1)
        self.assertIn(atoms.get_chemical_formula(), composition_lines[0])

    def test_explicit_formula_and_missing_structure_metadata(self):
        atoms = bulk("Cu", "fcc", a=3.6, cubic=True)
        atoms.set_chemical_symbols(["Au", "Cu", "Au", "Cu"])
        calculator = MaterialCalculator(atoms, EMT(), clamped=True)
        calculator.lattice_constant()
        report = self.report.read_text(encoding="utf-8")
        self.assertRegex(report, r"(AuCu|CuAu) · 4 atoms")
        self.assertNotIn("Actual composition", report)

        atoms.info["formula"] = "CuAu"
        calculator = MaterialCalculator(atoms, EMT(), clamped=True)
        self.assertEqual(calculator.info, "CuAu")
        self.report.unlink()
        calculator.lattice_constant()
        report = self.report.read_text(encoding="utf-8")
        self.assertIn("CuAu · 4 atoms", report)
        self.assertNotIn("Actual composition", report)

    def test_interleaved_calculators_and_recreated_report(self):
        cu = MaterialCalculator(bulk("Cu", "fcc", a=3.6, cubic=True), EMT(), clamped=True)
        au = MaterialCalculator(bulk("Au", "fcc", a=4.1, cubic=True), EMT(), clamped=True)
        for calculator in (cu, au, cu):
            calculator.lattice_constant()

        report = self.report.read_text(encoding="utf-8")
        cu_heading = "Cu · 4 atoms"
        au_heading = "Au · 4 atoms"
        self.assertEqual(report.count(cu_heading), 2)
        self.assertLess(report.index(cu_heading), report.index(au_heading))
        self.assertLess(report.index(au_heading), report.rindex(cu_heading))

        self.report.unlink()
        cu.lattice_constant()
        self.assertIn(cu_heading, self.report.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
