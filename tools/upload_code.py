import json
import shutil
from pathlib import Path
from typing import Any

import click
from kaggle.api.kaggle_api_extended import KaggleApi


def copy_all_files(source_dirs: Path, dest_dir: Path):
    """
    source_dir: Source directory
    dest_dir: Destination directory
    """
    for source_dir in source_dirs:
        # Search for all file paths in source_dir
        for source_path in source_dir.rglob("*"):
            if source_path.is_file():
                # Calculate the relative path in dest_dir
                relative_path = source_path.relative_to(source_dir)
                dest_path = dest_dir / source_dir.name / relative_path

                # Create the destination directory if necessary
                dest_path.parent.mkdir(parents=True, exist_ok=True)

                # Copy the file
                shutil.copy2(source_path, dest_path)
                print(f"Copied {source_path} to {dest_path}")


@click.command()
@click.option("--title", "-t", default="kuto-CMI-code")
@click.option("--dirs", "-d", type=list[Path], default=[Path("./src"), Path("./run")])
@click.option("--user_name", "-u", default="kuto0633")
@click.option("--new", "-n", is_flag=True)
def main(
    title: str,
    dirs: list[Path],
    user_name: str = "kuto0633",
    new: bool = False,
):
    """extentionを指定して、dir以下のファイルをzipに圧縮し、kaggleにアップロードする。

    Args:
        title (str): kaggleにアップロードするときのタイトル
        dir (Path): アップロードするファイルがあるディレクトリ
        extentions (list[str], optional): アップロードするファイルの拡張子.
        user_name (str, optional): kaggleのユーザー名.
        new (bool, optional): 新規データセットとしてアップロードするかどうか.
    """
    tmp_dir = Path("./tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    copy_all_files(dirs, tmp_dir)

    # dataset-metadata.jsonを作成
    dataset_metadata: dict[str, Any] = {}
    dataset_metadata["id"] = f"{user_name}/{title}"
    dataset_metadata["licenses"] = [{"name": "CC0-1.0"}]
    dataset_metadata["title"] = title
    with open(tmp_dir / "dataset-metadata.json", "w") as f:
        json.dump(dataset_metadata, f, indent=4)

    # api認証
    api = KaggleApi()
    api.authenticate()

    if new:
        api.dataset_create_new(
            folder=tmp_dir,
            dir_mode="tar",
            convert_to_csv=False,
            public=False,
        )
    else:
        api.dataset_create_version(
            folder=tmp_dir,
            version_notes="",
            dir_mode="tar",
            convert_to_csv=False,
        )

    # delete tmp dir
    shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    main()
