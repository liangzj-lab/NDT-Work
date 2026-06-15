from pathlib import Path


"""
Example:
    Update input_dir in main(), then run:
    python data_processing/rename_dcn_to_dcm.py
"""


def rename_dcn_to_dcm(root_dir: Path) -> int:
    if not root_dir.exists():
        raise FileNotFoundError(f"Folder not found: {root_dir}")

    if not root_dir.is_dir():
        raise NotADirectoryError(f"Not a folder: {root_dir}")

    renamed_count = 0
    for file_path in root_dir.rglob("*.dcn"):
        target_path = file_path.with_suffix(".dcm")
        if target_path.exists():
            print(f"Skip: {target_path} already exists")
            continue

        file_path.rename(target_path)
        renamed_count += 1
        # print(f"Renamed: {file_path} -> {target_path}")

    return renamed_count


def main() -> None:
    input_dir = Path(r"D:\项目文件\执行项目文件\PG25-LX19 浙江锅检所铝焊缝智能超声检测技术研究\执行过程文件\线夹图像\2026.3.28日天津检测220kV东范一线")

    renamed_count = rename_dcn_to_dcm(input_dir)
    print(f"Done. Renamed {renamed_count} file(s).")


if __name__ == "__main__":
    main()
    print("All done.")
