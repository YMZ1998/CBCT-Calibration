import pydicom, glob

files = sorted(glob.glob(r"D:\code\CBCT-Calibration\Scripts\starage\uploads\78687576529999\dicomV1\1760442503\*.dcm"))
uids = set()
orientations = set()
positions = []
spacings = set()

for f in files:
    ds = pydicom.dcmread(f, force=True)

    # 打印文件名
    print(f"\n📄 File: {f}")

    # 打印 spacing 信息
    px_spacing = getattr(ds, "PixelSpacing", None)
    slice_thickness = getattr(ds, "SliceThickness", None)
    spacing_between_slices = getattr(ds, "SpacingBetweenSlices", None)
    pos = getattr(ds, "ImagePositionPatient", [None, None, None])

    print(f"  PixelSpacing (XY): {px_spacing}")
    print(f"  SliceThickness (Z): {slice_thickness}")
    print(f"  SpacingBetweenSlices: {spacing_between_slices}")
    print(f"  ImagePositionPatient(Z): {pos[2] if pos[2] is not None else 'N/A'}")

    # 收集统计信息
    spacings.add(tuple(px_spacing))
    positions.append(pos[2])
    uids.add(getattr(ds, "SeriesInstanceUID", "None"))
    orientations.add(tuple(getattr(ds, "ImageOrientationPatient", [0] * 6)))

print("\n🧩 统计结果：")
print("Series UID 数量：", len(uids))
print("方向数：", len(orientations))
print("PixelSpacing 数量：", len(spacings))
print("Z 范围：", min(positions), "→", max(positions))
