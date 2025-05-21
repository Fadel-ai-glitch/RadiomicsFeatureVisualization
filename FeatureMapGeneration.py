import os
import six
import SimpleITK as sitk
from radiomics import featureextractor as originalradiomics
import logging
from tqdm import tqdm
logger = logging.getLogger('radiomics.featureextractor')
logger.setLevel(logging.ERROR)  # Set to ERROR to suppress warnings
def GetFeatureMap(params_path, store_path, image_path, roi_path, voxelBasedSet=True):
    extractor = originalradiomics.RadiomicsFeatureExtractor(params_path)
    result = extractor.execute(image_path, roi_path, voxelBased=voxelBasedSet)

    for key, val in six.iteritems(result):
        if isinstance(val, sitk.Image):
            shape = (sitk.GetArrayFromImage(val)).shape
            #print(f'✅ Feature map {key} shape: {shape}')
            output_path = os.path.join(store_path, f'{key}.nrrd')
            sitk.WriteImage(val, output_path, True)
        else:
            logger.info(f"ℹ️ {key}: {val}")

if __name__ == "__main__":
    # === Paramètres à adapter ===
    base_dir = "/home/kpegouni/Documents/AI-projects/radiomics/data/interim/ICM-DATA/PELVIS/T2WI"
    param_path = "/home/kpegouni/Documents/AI-projects/radiomics/notebooks/radiomics.yaml"
    out_base = "/home/kpegouni/Documents/AI-projects/radiomics/RadiomicsFeatureVisualization/ICM-DATA/FeatureMap/T2WI"

    vol_dir = os.path.join(base_dir, "volumes")
    mask_dir = os.path.join(base_dir, "segmentations")
    os.makedirs(out_base, exist_ok=True)

    volume_files = sorted([f for f in os.listdir(vol_dir) if f.endswith(".nii.gz")])

    for vol_file in tqdm(volume_files, desc="📊 Génération des feature maps"):
        subject_id = vol_file.replace(".nii.gz", "")
        vol_path = os.path.join(vol_dir, vol_file)
        mask_path = os.path.join(mask_dir, f"{subject_id}_mask.nii.gz")

        if not os.path.exists(mask_path):
            print(f"⚠️ Masque manquant pour {subject_id} — ignoré")
            continue

        out_dir = os.path.join(out_base, subject_id)
        os.makedirs(out_dir, exist_ok=True)

        try:
            GetFeatureMap(param_path, out_dir, vol_path, mask_path)
        except Exception as e:
            print(f"❌ Erreur pour {subject_id} : {e}")

