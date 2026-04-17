from Bio import PDB
import numpy as np
import time
import os
import argparse
from biopandas.mol2 import PandasMol2
import warnings
import featurization as featurization
import pickle

warnings.filterwarnings("ignore")

# elements in ligand to be considered
el_l = ["C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "H"]
el_p = ["C", "N", "O", "S"]  # elements in protein to be considered

PDB_file_name = "protein"


########################################################## element specific
class get_cloudpoint:
    def __init__(self, pdb_path, cut=12):
        self.cut = cut
        self.pdb_path = pdb_path

    def get_protein_atom_coordinate(self, pdbid):
        parser = PDB.PDBParser()
        pdbfile_path = f"{self.pdb_path}/{pdbid}/{pdbid}_{PDB_file_name}.pdb"
        struct = parser.get_structure(pdbid, pdbfile_path)

        coor = []
        for model in struct:
            for chain in model:
                for residue in chain:
                    if residue.id[0] == " ":
                        res_name = residue.resname  # Get amino acid name
                        for atom in residue:
                            if atom.element != "H":  # Exclude hydrogen atoms
                                XYZ = atom.get_coord()
                                coor.append(
                                    [XYZ[0], XYZ[1], XYZ[2], atom.element, res_name]
                                )
            break

        return np.array(coor)

    def get_ligand_atom_coordinate(self, pdbid):
        mol2file = f"{self.pdb_path}/{pdbid}/{pdbid}_ligand.mol2"
        pmol = PandasMol2().read_mol2(mol2file)
        x = pmol.df["x"].values.reshape(-1, 1)
        y = pmol.df["y"].values.reshape(-1, 1)
        z = pmol.df["z"].values.reshape(-1, 1)
        atom_type = np.array(
            [x.split(".")[0] for x in pmol.df["atom_type"].values]
        ).reshape(-1, 1)

        return np.concatenate((x, y, z, atom_type), axis=1)

    def fit_cutoff(self, pdbid):
        # eliminate points in protein whose distance with points in ligands > 12A.
        p_coor = self.get_protein_atom_coordinate(pdbid)
        l_coor = self.get_ligand_atom_coordinate(pdbid)

        remainder = []
        for elm1 in p_coor:
            for elm2 in l_coor:
                dis = np.linalg.norm(
                    elm1[:3].astype(np.float32) - elm2[:3].astype(np.float32)
                )

                if dis <= self.cut:
                    remainder.append(elm1)
                    break

        PRO = np.array(remainder)
        LIG = l_coor

        return PRO, LIG


class PLL_Feature:
    def __init__(self, pdb_path, pdbid, filtration_dists, max_dim):
        self.sigma = 0
        self.pdb_path = pdb_path
        self.pdbid = pdbid
        self.num_statistics = 9
        self.filtration_dists = filtration_dists
        self.max_dim = max_dim

    def call_eig(self, A):

        eigens = np.linalg.eigvalsh(A)
        return np.real(eigens)

    def setup_pointcloud(self, cutoff):
        cloudpoint_generator = get_cloudpoint(self.pdb_path, cut=cutoff)
        self.PRO, self.LIG = cloudpoint_generator.fit_cutoff(self.pdbid)

    def generate_xyz_element(self, e_1, e_2):

        PRO_xyz = self.PRO[self.PRO[:, 3] == e_1][:, :3]
        LIG_xyz = self.LIG[self.LIG[:, 3] == e_2][:, :3]

        return np.array(PRO_xyz).astype(float), np.array(LIG_xyz).astype(float)

    def _ensure_feature_directory(self, args):
        """Helper method to ensure feature directory exists."""
        feature_path = f"{args.feature_path}/{self.pdbid}"
        if not os.path.exists(feature_path):
            os.makedirs(feature_path)
        return feature_path

    def _compute_features_for_combination(self, e1, e2, method_type):
        """
        Helper method to compute features for a single element/category combination.

        Args:
            e1: First element
            e2: Second element
            method_type: "element" or "category"

        Returns:
            Tuple of (L0_features, L1_features)
        """

        # Generate coordinates based on method type
        if method_type == "element":
            print(f"Processing: {e1} - {e2}")
            PRO_xyz, LIG_xyz = self.generate_xyz_element(e1, e2)

        if len(LIG_xyz) == 0 or len(PRO_xyz) == 0:

            stats_cloud = [
                "sum",
                "mean",
                "median",
                "std",
                "max",
                "min",
                "var",
            ]  # 7 statistics
            num_filtration_dists = len(self.filtration_dists)

            features_pointcloud_dict = {}
            for dim in range(self.max_dim):
                features_temp = [0.0] * (self.num_statistics * num_filtration_dists)
                features_pointcloud_dict[dim] = {s: features_temp for s in stats_cloud}

            return features_pointcloud_dict, features_pointcloud_dict

        # Compute features
        # stores a dictionary, key: 0,1,..,max_dim-1
        # for each key, it stores a dictionary with keys ["sum", "mean", "median", "std", "max", "min", "var"], where each key gives a list of features of number = (num_statistics * num_filtration_dists)
        features_pointcloud_dict_PRO, features_pointcloud_dict_LIG = (
            featurization.features_pointcloud(
                PRO_xyz, LIG_xyz, self.filtration_dists, max_dim=self.max_dim
            )
        )

        return features_pointcloud_dict_PRO, features_pointcloud_dict_LIG

    def _get_element_combinations(self):
        """
        Args:
            self
        Returns:
            List of tuples containing element combinations
        """

        combinations = [(e_p, e_l) for e_p in el_p for e_l in el_l]

        return combinations

    def PLL_L01_features_ES(self, args):
        """
        Element-specific features computation.

        Args:
            args: Arguments object

        Returns:
            Tuple of (L0_features, L1_features)
        """
        self._ensure_feature_directory(args)

        ES_features_pointcloud_dict_PRO = {dim: {} for dim in range(self.max_dim)}
        ES_features_pointcloud_dict_LIG = {dim: {} for dim in range(self.max_dim)}

        combinations = self._get_element_combinations()

        # Compute features for each combination
        for e1, e2 in combinations:
            features_pointcloud_dict_PRO, features_pointcloud_dict_LIG = (
                self._compute_features_for_combination(e1, e2, "element")
            )
            for dim in range(self.max_dim):
                ES_features_pointcloud_dict_PRO[dim][(e1, e2)] = (
                    features_pointcloud_dict_PRO[dim]
                )
                ES_features_pointcloud_dict_LIG[dim][(e1, e2)] = (
                    features_pointcloud_dict_LIG[dim]
                )

        return ES_features_pointcloud_dict_PRO, ES_features_pointcloud_dict_LIG


def save_feature_PLL(args, cutoff, pdbid, filtration_dists):

    print(f"Processing: {pdbid}")
    PLL_object = PLL_Feature(args.pdb_path, pdbid, filtration_dists, args.max_dim)
    PLL_object.setup_pointcloud(cutoff)

    # Element-specific persistent Laplacian

    ES_features_pointcloud_dict_PRO, ES_features_pointcloud_dict_LIG = (
        PLL_object.PLL_L01_features_ES(args)
    )

    for dim in range(args.max_dim):
        save_path = f"{args.feature_path}/{pdbid}/{pdbid}_L{dim}_ES_PRO_r{cutoff:.2f}-fil{args.filtration_upper}.pkl"
        with open(save_path, "wb") as f:
            pickle.dump(ES_features_pointcloud_dict_PRO[dim], f)
        save_path = f"{args.feature_path}/{pdbid}/{pdbid}_L{dim}_ES_LIG_r{cutoff:.2f}-fil{args.filtration_upper}.pkl"
        with open(save_path, "wb") as f:
            pickle.dump(ES_features_pointcloud_dict_LIG[dim], f)


def main(args):

    t0 = time.time()

    cutoff = 12
    filtration_radius = np.arange(2, args.filtration_upper + args.dr, args.dr)

    pdbid = args.pdbid
    save_feature_PLL(args, cutoff, pdbid, filtration_radius)

    t1 = time.time()

    print(f"Time cost: {t1 - t0} seconds")


def parse_args():

    parser = argparse.ArgumentParser(description="Get PL features for pdbbind")
    parser.add_argument("--feature_path", type=str, default="features")
    parser.add_argument("--pdb_path", type=str, default="PDBs")
    parser.add_argument("--pdbid", type=str)
    parser.add_argument("--filtration_upper", type=int, default=6)
    parser.add_argument("--dr", type=float, default=0.5)
    parser.add_argument("--max_dim", type=int, default=2)
    args = parser.parse_args()

    print(args)
    main(args)


if __name__ == "__main__":
    parse_args()
    print("End!")
