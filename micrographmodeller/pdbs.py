import os
from micrographmodeller import utils


def call_chimera(filepath, output_folder):
    """
    Run chimera for pdb file in order to add hydrogens and add symmetry units. The resulting structure is stored as a
    new pdb file with the extension {id}_symmetry.pdb. The function calls chimera on command line to execute.

    Reference Chimera: UCSF Chimera--a visualization system for exploratory research and analysis. Pettersen EF,
    Goddard TD, Huang CC, Couch GS, Greenblatt DM, Meng EC, Ferrin TE. J Comput Chem. 2004 Oct;25(13):1605-12.

    @param pdb_folder: path to a folder where pdb files are stored
    @type  pdb_folder: L{string}
    @param pdb_id: ID of pdb file
    @type  pdb_id: L{string}

    @return: name of file where the new pdb stucture is stored in, this can differ for pdb depending on
    symmetry thus we need to return it
    @rtype: L{string}

    @author: Marten Chaillet
    """
    print(f' - Calling chimera for {filepath}')

    input_folder, filename = os.path.split(filepath)
    pdb_id, extension = os.path.splitext(filename)
    # pdb_id = file.split('.')[0]
    # extension = file.split('.')[-1]

    # Skip if output files from this function already exist
    if os.path.exists(os.path.join(output_folder, f'{pdb_id}_rem-solvent_sym_addh.pdb')):
        output_filepath = os.path.join(output_folder, f'{pdb_id}_rem-solvent_sym_addh.pdb')
        print(f'File already exists: {output_filepath}')
        # return f'{pdb_id}_rem-solvent_sym_addh'
        return output_filepath
    elif os.path.exists(os.path.join(output_folder, f'{pdb_id}_rem-solvent_addh.pdb')):
        output_filepath = os.path.join(output_folder, f'{pdb_id}_rem-solvent_addh.pdb')
        print(f'File already exists: {output_filepath}')
        # return  f'{pdb_id}_rem-solvent_addh'
        return output_filepath

    if extension == '.pdb':
        # Command 'sym' in chimera crashes when there is no BIOMT symmetry in the pdb file. We need to make sure
        # sym is only executed when the BIOMT information specifies symmetrical units.
        symmetry = []
        try:
            with open(filepath,'r') as pdb:
                line = pdb.readline().split()
                while line and line[0] != 'REMARK':
                    line = pdb.readline().split()
                while line and line[0] == 'REMARK':
                    if line[1] == '350' and len(line) > 3:
                        if 'BIOMT' in line[2]:
                            symmetry.append(int(line[3]))
                    line = pdb.readline().split()
        except FileNotFoundError as e:
            print(e)
            raise utils.StructureModificationError('Could not read pdb file.')

        print(f'{pdb_id} has {len(set(symmetry))} symmetrical {"unit" if len(set(symmetry)) == 1 else "units"}.')

        if len(set(symmetry)) > 1:
            scriptpath = os.path.join(output_folder, f'_rem-solvent_sym_addh_{pdb_id}.py')
            output_filepath = os.path.join(output_folder, f'{pdb_id}_rem-solvent_sym_addh.pdb')
            try:
                with open(scriptpath, 'w') as chimera_script:
                    chimera_script.write(f'# Open chimera for {pdb_id} then execute following command:\n'
                                         f'# (i) remove solvent (ii) add hydrogens (iii) add symmetry units\n'
                                         f'from chimerax.core.commands import run\n'
                                         f'run(session, "open {filepath}")\n'
                                         f'run(session, "delete solvent")\n'
                                         f'run(session, "delete ions")\n'
                                         f'run(session, "addh")\n' # If using pdb2pqr do not add hydrogens here.
                                         f'run(session, "sym #1 biomt")\n'             # group biomt is also the default
                                         f'run(session, "save {output_filepath} #2")\n'
                                         f'run(session, "exit")\n')
            except utils.StructureModificationError as e:
                print(e)
                raise Exception('Could not create chimera script.')
        else:
            scriptpath = os.path.join(output_folder, f'_rem-solvent_addh_{pdb_id}.py')
            output_filepath = os.path.join(output_folder, f'{pdb_id}_rem-solvent_addh.pdb')
            try:
                with open(scriptpath, 'w') as chimera_script:
                    chimera_script.write(f'# Open chimera for {pdb_id} then execute following command:\n'
                                         f'# (i) remove solvent (ii) add hydrogens (iii) add symmetry units\n'
                                         f'from chimerax.core.commands import run\n'
                                         f'run(session, "open {filepath}")\n'
                                         f'run(session, "delete solvent")\n'
                                         f'run(session, "delete ions")\n'
                                         f'run(session, "addh")\n' # If using pdb2pqr do not add hydrogens here.
                                         f'run(session, "save {output_filepath} #1")\n'
                                         f'run(session, "exit")\n')
            except FileNotFoundError as e:
                print(e)
                raise utils.StructureModificationError('Could not create chimera script.')
        # module chimera should be loaded here...
        if os.system(f'chimerax --nogui --script {scriptpath}') != 0:
            raise utils.StructureModificationError('Chimera is likely not on your current path.')

        if len(set(symmetry)) > 1:
            return output_filepath
            # return f'{pdb_id}_rem-solvent_sym_addh' # returns new pdb name
        else:
            return output_filepath
            # return f'{pdb_id}_rem-solvent_addh'

    elif extension == '.cif':
        # If cif, assume the file does not contain any structural symmetry information as this is usually not the case
        # for large complexes.
        # Do transformations with chimera, and store as pdb. Chimera cannot write mmCIF files. ChimeraX can.
        # Also not that opening mmCIF files in chimera takes significantly longer.
        scriptpath = os.path.join(output_folder,f'_rem-solvent_addh_{pdb_id}.py')
        output_filepath = os.path.join(output_folder, f'{pdb_id}_rem-solvent_addh.pdb')
        try:
            with open(scriptpath, 'w') as chimera_script:
                chimera_script.write(f'# Open chimera for {pdb_id} then execute following command:\n'
                                     f'# (i) remove solvent (ii) add hydrogens (iii) add symmetry units\n'
                                     f'from chimerax.core.commands import run\n'
                                     f'run(session, "open {filepath}")\n'
                                     f'run(session, "delete solvent")\n'
                                     f'run(session, "delete ions")\n'
                                     f'run(session, "addh")\n' # If using pdb2pqr do not add hydrogens here.
                                     f'run(session, "save {output_filepath} #1")\n'
                                     f'run(session, "exit")\n')
        except FileNotFoundError as e:
            print(e)
            raise utils.StructureModificationError('Could not create chimera script.')
        # module chimera should be loaded here...
        if os.system(f'chimerax --nogui --script {scriptpath}') != 0:  # 0 is succes
            raise utils.StructureModificationError('Chimera is likely not on your current path.')
        return output_filepath
    else:
        print('non-valid structure file extension')
        return 0


def read_structure(filepath):
    """
    Read pdb, cif, or pqr file and return atom data in lists.

    @param filepath: full path to the file, either .pdb, .cif, or .pqr
    @type  filepath: L{str}

    @return: a tuple of 6 lists (x_coordinates, y_coordinates, z_coordinates, elements, b_factors, occupancies)
    @rtype: L{tuple} -> (L{list},) * 6 with types (float, float, float, str, float, float)

    @author: Marten Chaillet
    """
    x_coordinates, y_coordinates, z_coordinates, elements, b_factors, occupancies = [], [], [], [], [], []

    _, extension = os.path.splitext(filepath)

    if extension == '.pdb':
        try:
            with open(filepath, 'r') as pdb:
                lines = pdb.readlines()
                atoms = [line for line in lines if line[:4] == 'ATOM']
                for line in atoms:
                    '''
        PDB example
        ATOM   4366  OXT SER I 456      10.602  32.380  -1.590  1.00 53.05           O
                    '''
                    if line[76:78].strip() != '':
                        x_coordinates.append(float(line[30:38]))
                        y_coordinates.append(float(line[38:46]))
                        z_coordinates.append(float(line[46:54]))
                        elements.append(line[76:78].strip())
                        b_factors.append(float(line[60:66]))
                        occupancies.append(float(line[54:60]))
                    else:  # for bare pdb files without b_factor/occup/element
                        x_coordinates.append(float(line[30:38]))
                        y_coordinates.append(float(line[38:46]))
                        z_coordinates.append(float(line[46:54]))
                        elements.append(line[13])
                        b_factors.append(1.)
                        occupancies.append(1.)
                hetatms = [line for line in lines if line[:6] == 'HETATM']
                for line in hetatms:
                    '''
        PDB example
        HETATM13897 MG    MG A 501     120.846  94.563  17.347  1.00 79.97          MG
                    '''
                    x_coordinates.append(float(line[30:38]))
                    y_coordinates.append(float(line[38:46]))
                    z_coordinates.append(float(line[46:54]))
                    elements.append(line[76:78].strip())
                    b_factors.append(float(line[60:66]))
                    occupancies.append(float(line[54:60]))

        except Exception as e:
            print(e)
            raise Exception('Could not read pdb file.')
    elif extension == '.cif':
        try:
            with open(filepath, 'r') as pdb:
                lines = pdb.readlines()
                for line in lines:
                    if line.strip():
                        split_line = line.split()
                        if split_line[0] == 'ATOM':
                            '''
            PDBx/mmCIF example
            ATOM   171293 O  OP1   . G   WB 75 255  ? 252.783 279.861 251.593 1.00 50.94  ? 255  G   aa OP1   1
                            '''
                            x_coordinates.append(float(split_line[10]))
                            y_coordinates.append(float(split_line[11]))
                            z_coordinates.append(float(split_line[12]))
                            elements.append(split_line[2].strip())
                            b_factors.append(float(split_line[14]))
                            occupancies.append(float(split_line[13]))
                        elif split_line[0] == 'HETATM':
                            '''
            PDBx/mmCIF example
            HETATM 201164 MG MG    . MG  FD 79 .    ? 290.730 254.190 214.591 1.00 30.13  ? 3332 MG  A  MG    1
                            '''
                            x_coordinates.append(float(split_line[10]))
                            y_coordinates.append(float(split_line[11]))
                            z_coordinates.append(float(split_line[12]))
                            elements.append(split_line[2].strip())
                            b_factors.append(float(split_line[14]))
                            occupancies.append(float(split_line[13]))
        except Exception as e:
            print(e)
            raise Exception('Could not read cif file.')
    elif extension == '.pqr':
        try:
            with open(filepath, 'r') as pqr:
                lines = pqr.readlines()
                for line in lines:
                    if not line.strip():
                        split_line = line.split()
                        # TODO Whay about HETATM lines?
                        if split_line[0] == 'ATOM':
                            '''
                PQR example
                ATOM   5860  HA  ILE   379      26.536  13.128  -3.443  0.0869 1.3870
                            '''
                            x_coordinates.append(float(split_line[5]))
                            y_coordinates.append(float(split_line[6]))
                            z_coordinates.append(float(split_line[7]))
                            elements.append(split_line[2][0])  # first letter of long atom id is the element
                            b_factors.append(0.0)  # not avalaible in PQR format
                            occupancies.append(1.0)  # not avalaible in PQR format
                        # HETATM not working here because extracting element type from double letter
                        # elements, like MG, does not work properly. Should be tested though.
        except Exception as e:
            print(e)
            raise Exception('Could not read pqr file.')
    else:
        print(f'invalid filetype in read_structure() for {filepath}, return 0')
        return 0

    return x_coordinates, y_coordinates, z_coordinates, elements, b_factors, occupancies
