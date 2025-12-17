# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

import openmm
from openmmtools.testsystems import TestSystem

DIRECTORY = Path(__file__).parent


class AlanineTetrapeptideImplicit(TestSystem):
    """Alanine tetrapeptide in implicit solvent."""

    def __init__(self, constraints=None, hydrogenMass=None, **kwargs):
        TestSystem.__init__(self, **kwargs)
        pdbfile = openmm.app.PDBFile(str(DIRECTORY / "ala4_ref.pdb"))
        forcefield = openmm.app.ForceField("amber99sbildn.xml", "amber99_obc.xml")
        system = forcefield.createSystem(
            pdbfile.topology,
            nonbondedMethod=openmm.app.NoCutoff,
            constraints=constraints,
            hydrogenMass=hydrogenMass,
        )

        self.topology = pdbfile.topology
        self.system = system
