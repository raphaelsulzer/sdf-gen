from pathlib import Path
import trimesh
from argparse import ArgumentParser
import subprocess
import numpy as np
import marching_cubes as mc
import math
from tqdm import tqdm
import os


dims = [16, 24, 32, 40, 48, 64, 80, 96, 128]

paddings = [2, ]
paddings = paddings + [int(math.ceil(x / dims[0]) * paddings[0]) for x in dims[1:]]
voxel_resolutions = [1.01 * (1 / (dim - 2 * pad)) for dim, pad in zip(dims, paddings)]


def iso(dim):
    max_iso = 1
    min_iso = 1
    m = (max_iso - min_iso) / (dims[-1] - dims[0])
    c = max_iso - m * dims[-1]
    return m * dim + c


print('Dims:', dims)
print('Pads:', paddings)
print('Voxs:', voxel_resolutions)

sdf_gen_cmd = lambda inpath, outpath, voxres, pad: f"bin/sdf_gen_shapenet {inpath} {outpath} {voxres} {pad}"


def visualize_distance_field(df_path, vox_res, dim):
    df = np.load(df_path)
    vertices, triangles = mc.marching_cubes(df, vox_res * iso(dim))
    mc.export_obj(vertices, triangles, df_path.parent / (df_path.stem + ".obj"))
    mesh = trimesh.load(df_path.parent / (df_path.stem + ".obj"), process=False)
    mesh.apply_scale(dims[-1]/ dim)
    mesh.export(df_path.parent / (df_path.stem + ".obj"))
    os.remove(df_path)
    os.remove(df_path.parent / "material.mtl")


def export_distance_field(mesh_path, visualize=False):
    for dim, res, pad in zip(dims, voxel_resolutions, paddings):
        failure_lr = subprocess.call(sdf_gen_cmd(str(mesh_path), str(mesh_path.parent / f"{dim:03d}"), res, pad), shell=True)
        os.remove(str(mesh_path.parent / f"{dim:03d}") + "_if.npy")
        if visualize:
            visualize_distance_field(mesh_path.parent / f"{dim:03d}.npy", res, dim)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_folder', type=str)
    parser.add_argument('-n', '--num_proc', default=1, type=int)
    parser.add_argument('-p', '--proc', default=0, type=int)

    args = parser.parse_args()
    files = sorted([x for x in Path(args.input_folder).iterdir()])
    files = [x for i, x in enumerate(files) if i % args.num_proc == args.proc]
    for f in tqdm(files):
        export_distance_field(f / "model_normalized.obj", True)
