import pandas as pd
import numpy as np

def xyz_reader(filename, skiprows=2):
    return pd.read_table('17070801_CHD_opt_CCSD_aug-cc-pVDZ.xyz',
                         sep='[\^\s]+', engine='python',
                         skiprows = skiprows, index_col=False,
                         names=['atom','x','y','z'])

def calc_distance(r0,r1):
    return np.linalg.norm(r1 - r0)

def make_r(xyz_row):
    return np.array([xyz_row['x'],xyz_row['y'],xyz_row['z']])

def atom_atom_distances(xyz_table):
    aacolumns=['A1','A2','r']
    aa_df = pd.DataFrame(columns = aacolumns)

    for index0, row0 in xyz_table.iterrows():
        for index1, row1 in xyz_table.iterrows():
            r0 = make_r(row0)
            r1 = make_r(row1)
            r = calc_distance(r0,r1)

            A1 = row0['atom']+str(index0)
            A2 = row1['atom']+str(index1)

            aa_df0 = pd.DataFrame([[A1,A2,r]],columns = aacolumns)
            aa_df = aa_df.append(aa_df0)

    return aa_df

def justThisAtom(aaDF,atomCode):
    idx1 = aaDF['A1'].apply(lambda A: A[0] == atomCode)
    idx2 = aaDF['A2'].apply(lambda A: A[0] == atomCode)
    return aaDF[ idx1&idx2 ]

def generateBleachBounds( aaDF, Rs, wmax=100, width=1 ):
    bounds = [(0,wmax) for r in Rs]
    for r in aaDF['r']:
        idx = np.abs(Rs - r).argmin()
        if width == 0:
            bounds[idx] = (-wmax, 0)
        else:
            llim = (idx-width)*int((idx - width)>0)

            for new_idx in xrange(llim, idx+width+1):
                if new_idx > Rs.shape[0]:
                    break
                bounds[new_idx] = (-wmax, 0)
    return bounds


def generateAtomBoundsFromXYZ( filename, Rs, atomCode, wmax=100, width=1 ):
    xyzDF = xyz_reader(filename)
    atomAtomDistances = atom_atom_distances(xyzDF)
    a1A1Distance     = justThisAtom(atomAtomDistances, atomCode)

    return generateBleachBounds( a1A1Distance, Rs, wmax, width )




def generateAtomDistancesFromXYZ( filename, Rs, atomCode ):
    xyzDF = xyz_reader(filename)
    atomAtomDistances = atom_atom_distances(xyzDF)
    return justThisAtom(atomAtomDistances, atomCode)
