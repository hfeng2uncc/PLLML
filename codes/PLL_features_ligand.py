from Bio import PDB
from VR_complex import PointCloudRips
import numpy as np
import time
import os
import argparse
from biopandas.mol2 import PandasMol2
import warnings
import featurization_ligand as featurization
import pickle

warnings.filterwarnings("ignore")


# elements in ligand to be considered
el_l = ["C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "H"]

element_groups = [
    ("C",),
    ("N",),
    ("O",),
    ("S",),
    ("C", "N"),
    ("C", "O"),
    ("C", "S"),
    ("N", "O"),
    ("N", "S"),
    ("O", "S"),
    ("N", "P"),
    ("F", "Cl", "Br", "I"),
    ("C", "O", "N", "S", "F", "P", "Cl", "Br", "I"),
]
element_groups_dict = {}
for id, elements in enumerate(element_groups):
    element_groups_dict[id] = elements


class PLL_Feature:
    def __init__(self, mol2file, filtration_dists, max_dim):
        self.sigma = 0
        self.mol2file = mol2file
        self.num_statistics = 9
        self.filtration_dists = filtration_dists
        self.max_dim = max_dim
        self.get_ligand_atom_coordinate()

    def get_ligand_atom_coordinate(self):
        pmol = PandasMol2().read_mol2(self.mol2file)
        x = pmol.df["x"].values.reshape(-1, 1)
        y = pmol.df["y"].values.reshape(-1, 1)
        z = pmol.df["z"].values.reshape(-1, 1)
        atom_type = np.array(
            [x.split(".")[0] for x in pmol.df["atom_type"].values]
        ).reshape(-1, 1)

        self.xyz_corr = np.concatenate((x, y, z, atom_type), axis=1)

    def generate_xyz_element(self, elements):

        xyz_list = []
        for e in elements:
            xyz_e = self.xyz_corr[self.xyz_corr[:, 3] == e][:, :3]
            xyz_list.append(xyz_e)

        pointcloud = np.concatenate(xyz_list, axis=0).astype(float)

        return pointcloud

    def _compute_features_for_combination(self, pc, rips_dict, elements):

        pointcloud = self.generate_xyz_element(elements)

        if len(pointcloud) == 0:

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

            return features_pointcloud_dict

        features_pointcloud_dict = featurization.features_pointcloud(
            pointcloud, pc, rips_dict, self.filtration_dists, max_dim=self.max_dim
        )

        return features_pointcloud_dict

    def PLL_features_ES(self):

        ES_features_pointcloud_dict = {dim: {} for dim in range(self.max_dim)}

        pointcloud_all = self.generate_xyz_element(el_l)
        # Initialize point cloud
        pc = PointCloudRips(pointcloud_all)
        # Generate Rips complexes for all distances with maximum 3-dimensional simplices
        rips_dict = pc.generate_rips_dict_by_all_distances(
            self.filtration_dists, max_dim=self.max_dim
        )

        # Compute features for each combination
        for key, elements in element_groups_dict.items():
            features_pointcloud_dict = self._compute_features_for_combination(
                pc, rips_dict, elements
            )
            for dim in range(self.max_dim):
                ES_features_pointcloud_dict[dim][key] = features_pointcloud_dict[dim]

        return ES_features_pointcloud_dict


def save_feature_PLL(args, pdbid, filtration_dists):

    print(f"Processing: {pdbid}")
    mol2file = f"{args.pdb_path}/{pdbid}/{pdbid}_ligand.mol2"
    PLL_object = PLL_Feature(mol2file, filtration_dists, args.max_dim)

    ES_features_pointcloud_dict = PLL_object.PLL_features_ES()

    for dim in range(args.max_dim):
        save_path = f"{args.feature_path}/{pdbid}/{pdbid}_L{dim}-fil{args.filtration_upper}_ligand.pkl"
        with open(save_path, "wb") as f:
            pickle.dump(ES_features_pointcloud_dict[dim], f)


def main(args):

    t0 = time.time()

    filtration_radius = np.arange(2, args.filtration_upper + args.dr, args.dr)

    pdbid = args.pdbid
    save_feature_PLL(args, pdbid, filtration_radius)

    t1 = time.time()

    print(f"Time cost: {t1 - t0} seconds")


def parse_args():

    parser = argparse.ArgumentParser(description="Get PL features for pdbbind")
    parser.add_argument("--feature_path", type=str, default="features")
    parser.add_argument("--pdb_path", type=str, default="PDBs")
    parser.add_argument("--pdbid", type=str)
    parser.add_argument("--filtration_upper", type=int, default=10)
    parser.add_argument("--dr", type=float, default=0.5)
    parser.add_argument("--max_dim", type=int, default=2)
    args = parser.parse_args()

    print(args)
    main(args)


if __name__ == "__main__":
    parse_args()
    print("End!")
