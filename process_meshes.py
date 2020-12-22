from argparse import ArgumentParser
from pathlib import Path
import subprocess
import numpy as np
import marching_cubes as mc
import trimesh

sdf_gen_highres_cmd = lambda filename: f"bin/sdf_gen {filename} 0.0625 8"
sdf_gen_lowres_cmd = lambda filename: f"bin/sdf_gen {filename} 0.5 1"


def read_df(path):
    with open(path, "r") as file:
        lines = file.readlines()
        nx, ny, nz = map(int, lines[0].split(' '))
        print(nx, ny, nz)
        x0, y0, z0 = map(float, lines[1].split(' '))
        print(x0, y0, z0)
        delta = float(lines[2].strip())
        print(delta)
        data = np.zeros([nx, ny, nz])
        for i, line in enumerate(lines[3:]):
            idx = i % nx
            idy = int(i / nx) % ny
            idz = int(i / (nx * ny))
            val = float(line.strip())
            data[idx, idy, idz] = np.abs(val)
    return data


def read_idf(path):
    with open(path, "r") as file:
        lines = file.readlines()
        nx, ny, nz = map(int, lines[0].split(' '))
        print(nx, ny, nz)
        x0, y0, z0 = map(float, lines[1].split(' '))
        print(x0, y0, z0)
        delta = float(lines[2].strip())
        print(delta)
        data = np.zeros([nx, ny, nz, 3])
        vals = []
        for i, line in enumerate(lines[3:]):
            idx = i % nx
            idy = int(i / nx) % ny
            idz = int(i / (nx * ny))
            val = float(line.strip())
            vals.append(val / 64)
            data[idx, idy, idz, 0] = (val % 256) / 255
            data[idx, idy, idz, 1] = ((val // 256) % 256) / 255
            data[idx, idy, idz, 2] = ((val // 256 // 256) % 256) /255
        print(list(set(vals)))
    return data


def get_valid_rooms(mesh_dir, limit=None):
    list_of_scenes = Path(mesh_dir).iterdir()
    list_of_rooms = []
    for scene in list_of_scenes:
        for room in scene.iterdir():
            if (room / "mesh.obj").exists():
                list_of_rooms.append(room)
    return list_of_rooms[:limit]


def export_distance_field(mesh_dir, output_path):
    output_path.parents[0].mkdir(exist_ok=True, parents=True)
    mesh = trimesh.load(mesh_dir / "mesh.obj")
    bbox = mesh.bounding_box.bounds
    loc = (bbox[0] + bbox[1]) / 2
    scale = (bbox[1] - bbox[0])[1]
    mesh.apply_translation(-loc)
    mesh.apply_scale(3 / scale)
    bbox = mesh.bounding_box.bounds.copy()
    mesh.apply_translation(-bbox[0])
    mesh.export(output_path)
    failure = subprocess.call(sdf_gen_highres_cmd(output_path), shell=True)


def visualize_highres(df_path, output_vis_path):
    df = read_df(str(df_path)+".df")
    idf = read_idf(str(df_path)+".if")
    vertices, triangles = mc.marching_cubes_color(df, idf, 0.0625 * 0.75)
    mc.export_obj(vertices, triangles, str(output_vis_path))


def visualize_lowres(df_path, output_vis_path):
    to_point_list = lambda s: np.concatenate([c[:, np.newaxis] for c in np.where(s == True)], axis=1)
    df = read_df(df_path)
    point_list = to_point_list(df <= 0.5 * 0.5)
    if point_list.shape[0] > 0:
        base_mesh = trimesh.voxel.ops.multibox(centers=point_list, pitch=1)
        base_mesh.export(output_vis_path)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--mesh_dir", type=str, default='outputs')
    parser.add_argument("--df_lowres_dir", type=str, default='df_lowres')
    parser.add_argument("--df_highres_dir", type=str, default='df_highres')
    args = parser.parse_args()
    valid_rooms = get_valid_rooms(args.mesh_dir)
    for room in valid_rooms:
        room_id = f"{room.parents[0].name}__{room.name}"
        # export_distance_field(room, Path(args.df_highres_dir) / f"{room_id}.obj")
        visualize_highres(Path(args.df_highres_dir) / f"{room_id}", Path(args.df_highres_dir) / f"vis_{room_id}.obj")