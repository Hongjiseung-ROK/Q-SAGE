import os
import subprocess
import re
from rdkit import Chem

class OrcaManager:
    def __init__(self, working_dir, orca_cmd="orca"):
        self.working_dir = working_dir
        self.orca_cmd = orca_cmd
        
        # Check for local ORCA installation
        local_orca = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "orca_bin", "orca.exe"))
        if os.path.exists(local_orca):
            print(f"Found local ORCA at: {local_orca}")
            self.orca_cmd = local_orca
            
        if not os.path.exists(self.working_dir):
            os.makedirs(self.working_dir)

    def write_input(self, mol, name):
        """
        Writes an ORCA input file (.inp) for the given molecule.
        Standard: ! B3LYP 6-31G(d) Opt
        """
        inp_path = os.path.join(self.working_dir, f"{name}.inp")
        xyz_block = Chem.MolToXYZBlock(mol)
        
        # Remove the first two lines of XYZ block (count and comment)
        xyz_lines = xyz_block.strip().split("\n")[2:]
        coords = "\n".join(xyz_lines)
        
        content = f"""! B3LYP 6-31G(d) Opt
%pal nprocs 1 end
* xyz 0 1
{coords}
*
"""
        with open(inp_path, "w") as f:
            f.write(content)
        
        return inp_path

    def run_orca(self, name):
        """
        Executes ORCA on the generated input file.
        Returns the path to the output file.
        """
        inp_file = f"{name}.inp"
        out_file = f"{name}.out"
        inp_path = os.path.join(self.working_dir, inp_file)
        out_path = os.path.join(self.working_dir, out_file)
        
        if not os.path.exists(inp_path):
            print(f"Error: Input file {inp_path} not found.")
            return None
            
        # Check if output already exists (skip computation if done)
        if os.path.exists(out_path):
            print(f"Output for {name} already exists. Skipping ORCA execution.")
            return out_path
            
        cmd = [self.orca_cmd, inp_path]
        
        print(f"Running ORCA for {name}...")
        try:
            with open(out_path, "w") as outfile:
                subprocess.run(cmd, stdout=outfile, stderr=subprocess.STDOUT, check=True)
            return out_path
        except subprocess.CalledProcessError as e:
            print(f"ORCA execution failed for {name}: {e}")
            return None
        except FileNotFoundError:
            print(f"ORCA command '{self.orca_cmd}' not found.")
            return None

    def parse_output(self, name):
        """
        Parses the ORCA output file to extract HOMO and LUMO energies.
        Returns dictionary: {'homo': float, 'lumo': float, 'gap': float} or None on failure.
        """
        out_path = os.path.join(self.working_dir, f"{name}.out")
        if not os.path.exists(out_path):
             return None
             
        homo = None
        lumo = None
        
        try:
            with open(out_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
                
            # Key Pro Check: Did the SCF converge?
            if "SCF CONVERGED AFTER" not in content and "SCF CONVERGED" not in content:
                print(f"Warning: {name} did NOT converge. Skipping.")
                return None

            lines = content.split('\n')
            
            # Naive parsing for Orbital Energies
            # Looking for "OCC" and "VIR" (Occupied and Virtual)
            # ORCA output format varies, but usually looks like:
            # NO   OCC          E(Eh)            E(eV)
            # 0   2.0000      -10.0000        -272.11
            # ...
            
            # Better strategy: Regex for specific lines
            # "HOMO-LUMO gap" might be directly printed in newer versions, but let's parse orbitals.
            
            orbital_energies = []
            parsing_orbitals = False
            
            for line in lines:
                if "ORBITAL ENERGIES" in line:
                    parsing_orbitals = True
                    orbital_energies = [] # Reset if multiple SCF cycles
                    continue
                
                if parsing_orbitals:
                    if "--------" in line or "NO" in line or line.strip() == "":
                        continue
                    
                    # End of section (usually next header or empty lines)
                    if "MULLIKEN" in line or "LOEWDIN" in line: 
                        parsing_orbitals = False
                        continue
                        
                    parts = line.split()
                    if len(parts) >= 4:
                        try:
                            occ = float(parts[1])
                            energy_ev = float(parts[3]) # Usually Eh is col 2, eV is col 3
                            orbital_energies.append({'occ': occ, 'energy': energy_ev})
                        except ValueError:
                            pass

            if not orbital_energies:
                print(f"Could not parse orbital energies for {name}")
                return None
                
            # Identify HOMO (last occupied) and LUMO (first virtual)
            # This presumes sorted order which ORCA provides
            prev_occ = -1
            for orb in orbital_energies:
                if orb['occ'] == 0.0 and prev_occ > 0.0:
                    lumo = orb['energy']
                    break
                if orb['occ'] > 0.0:
                    homo = orb['energy']
                    prev_occ = orb['occ']
            
            if homo is not None and lumo is not None:
                gap = lumo - homo
                return {'homo': homo, 'lumo': lumo, 'gap': gap}
            else:
                return None

        except Exception as e:
             print(f"Parsing error for {name}: {e}")
             return None

if __name__ == "__main__":
    # Test script
    manager = OrcaManager("tests/orca_test")
    # For testing, we won't actually run ORCA unless the user has it.
    # We can mock the run or just check file generation.
    print("OrcaManager initialized. Use manager.write_input(mol, name) to generate inputs.")
