
import os
import time
import six

import SimpleITK as sitk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

#from SKMRradiomics import featureextractor
from radiomics import featureextractor
from FeatureMapShow import FeatureMapVisualizition


class FeatureMapper:
    """

        This class would found a candidate case image to generate radiomics feature map based voxel. /

    """

    def __init__(self):
        self.feature_pd = pd.DataFrame()
        self.selected_feature_list = []
        self.store_path = ''
        self.kernelRadius = ''
        self.sub_img_array = np.array([])
        self.sub_roi_array = np.array([])
    # def load(self, feature_csv_path, selected_feature_list):
    #     self.feature_pd = pd.read_csv(feature_csv_path, index_col=0)
    #     self.selected_feature_list = selected_feature_list

    def load(self, feature_csv_path, selected_feature_list):
            self.feature_pd = pd.read_csv(feature_csv_path, index_col=1)
            self.selected_feature_list = selected_feature_list

            if 'label' in self.feature_pd.columns:
                # Remplacer les NaN par une valeur par défaut (ex: -1 pour "inconnu")
                # Choisissez une valeur qui n'entre pas en conflit avec vos vrais labels (0, 1, etc.)
                self.feature_pd['label'].fillna(-1, inplace=True)
                print("Attention: 'label' manquants (NaN) ont été remplacés par -1.")
                self.feature_pd['label'] = self.feature_pd['label'].astype(int) # Convertir en int après

    def seek_single_candidate_case(self, feature_name, case_num):
        sub_feature_pd = self.feature_pd[['label', feature_name]].copy()
        sub_feature_pd.sort_values(by=feature_name, inplace=True)
        sorted_index_list = sub_feature_pd.axes[0]
        max_case, min_case = sorted_index_list[-1], sorted_index_list[0]

        max_info = max_case + '(' + str(int(sub_feature_pd.at[max_case, 'label'])) + ')'
        min_info = min_case + '(' + str(int(sub_feature_pd.at[min_case, 'label'])) + ')'
        print('{} value maximum case : {}, minimum case : {}'.format(feature_name, max_info, min_info))
        top_case_list = list(sorted_index_list)[-case_num:]
        last_case_list = list(sorted_index_list)[:case_num]
        return top_case_list, last_case_list

    def seek_candidate_case(self, feature_csv_path, selected_feature_list, case_num):
        """
            seek a candidate image by feature value from feature csv， print the candidate case.

        Parameters
        ----------
        feature_csv_path : str, radiomics feature csv;
        selected_feature_list : list, selected feature name list;


        """
        self.load(feature_csv_path, selected_feature_list)
        candidate_case_list = []
        candidate_case_dict = {}
        for sub_feature in selected_feature_list:
            top_case_list, last_case_list = self.seek_single_candidate_case(sub_feature, case_num)
            candidate_case_dict[sub_feature] = {'top': top_case_list, 'last': last_case_list}

        # seek common case
        for sub_feature in list(candidate_case_dict.keys()):
            all_features = candidate_case_dict[sub_feature]['top'] + candidate_case_dict[sub_feature]['last']
            if len(candidate_case_list) == 0:
                candidate_case_list = all_features

            else:
                candidate_case_list = list(set(candidate_case_list).intersection(set(all_features)))

        # check common case

        for sub_feature in list(candidate_case_dict.keys()):
            sub_checked_case = list(set(candidate_case_dict[sub_feature]['top']).
                                    intersection(set(candidate_case_list)))

            candidate_case_dict[sub_feature]['top'] = [index + "(" + str(int(self.feature_pd.at[index, 'label'])) + ")"
                                                       for index in sub_checked_case]

            sub_checked_case = list(set(candidate_case_dict[sub_feature]['last']).
                                    intersection(set(candidate_case_list)))
            candidate_case_dict[sub_feature]['last'] = [index + "(" + str(int(self.feature_pd.at[index, 'label'])) + ")"
                                                        for index in sub_checked_case]

        df = pd.DataFrame.from_dict(candidate_case_dict, orient='index')
        print(df)

    @staticmethod
    def decode_feature_name(feature_name_list):
        sub_filter_name = ''
        img_setting = {'imageType': 'Original'}
        feature_dict = {}
        for sub_feature in feature_name_list:

            # big feature class
            if sub_feature in ['firstorder', 'glcm', 'glrlm', 'ngtdm', 'glszm']:
                # extract all features
                sub_feature_setting = {sub_feature: []}
                feature_dict.update(sub_feature_setting)

            else:
                img_type = sub_feature.split('_')[-3]
                if img_type.rfind('wavelet') != -1:

                # if img_type in ['LLL', 'HLL','LHL', 'LLH', 'HHL', 'HHH','HLH','LHH']:
                    img_setting['imageType'] = 'Wavelet'
                    sub_filter_name = img_type.split('-')[-1]
                elif img_type.rfind('LOG') != -1:
                    img_setting['imageType'] = 'LoG'
                    sub_filter_name = img_type

                else:
                    img_setting['imageType'] = 'Original'
                # if img_type not in img_setting['imageType']:
                #     img_setting['imageType'].append(img_type)

                feature_class = sub_feature.split('_')[-2]
                feature_name = sub_feature.split('_')[-1]

                if feature_class not in feature_dict.keys():
                    feature_dict[feature_class] = []
                    feature_dict[feature_class].append(feature_name)
                else:
                    feature_dict[feature_class].append(feature_name)
        print(img_setting)
        print(feature_dict)
        return img_setting, feature_dict, sub_filter_name

    # crop img by kernelRadius, remove redundancy slice to speed up,
    def crop_img(self, original_roi_path, original_img_path, store_key=''):
        roi = sitk.ReadImage(original_roi_path)

        roi_array = sitk.GetArrayFromImage(roi)
        max_roi_slice_index = np.argmax(np.sum(roi_array, axis=(1, 2)))

        z_range = [max_roi_slice_index - self.kernelRadius, max_roi_slice_index + self.kernelRadius + 1]
        x_index = np.where(np.sum(roi_array[max_roi_slice_index], axis=0) > 0)[0]
        x_range = [min(x_index) - self.kernelRadius, max(x_index) + self.kernelRadius + 1]
        y_index = np.where(np.sum(roi_array[max_roi_slice_index], axis=1) > 0)[0]
        y_range = [min(y_index) - self.kernelRadius, max(y_index) + self.kernelRadius + 1]


        cropped_roi_array = roi_array[z_range[0]:z_range[1]]
        cropped_roi = sitk.GetImageFromArray(cropped_roi_array)
        cropped_roi.SetDirection(roi.GetDirection())
        cropped_roi.SetOrigin(roi.GetOrigin())
        cropped_roi.SetSpacing(roi.GetSpacing())

        img = sitk.ReadImage(original_img_path)
        img_array = sitk.GetArrayFromImage(img)
        cropped_img_array = img_array[z_range[0]:z_range[1]]
        cropped_img = sitk.GetImageFromArray(cropped_img_array)
        cropped_img.SetDirection(img.GetDirection())
        cropped_img.SetOrigin(img.GetOrigin())
        cropped_img.SetSpacing(img.GetSpacing())

        roi_info = [roi.GetDirection(), roi.GetOrigin(), roi.GetSpacing()]
        img_info = [img.GetDirection(), img.GetOrigin(), img.GetSpacing()]
        index_dict = {0:'direction', 1:'origin', 2:'spacing'}
        start = 0
        for sub_roi_info, sub_img_info in zip(roi_info, img_info):
            if sub_roi_info != sub_img_info:
                print(index_dict[start], 'failed')
                print('roi:', sub_roi_info)
                print('img:', sub_img_info)

        sitk.WriteImage(cropped_img, os.path.join(self.store_path, store_key + '_cropped_img.nii.gz'))
        sitk.WriteImage(cropped_roi, os.path.join(self.store_path, store_key + '_cropped_roi.nii.gz'))
        self.sub_img_array = np.transpose(cropped_img_array, (1, 2, 0))
        self.sub_roi_array = np.transpose(cropped_roi_array, (1, 2, 0))
        print('ROI size: ', np.sum(cropped_roi))
        return cropped_img, cropped_roi

    def generate_feature_map(self, candidate_img_path, candidate_roi_path, kernelRadius, feature_name_list, store_path):
        """
            Generate specific feature map based on kernel Radius.

        Parameters
        ----------
        candidate_img_path: str, candidate image path;
        candidate_roi_path: str, candidate ROI path;
        kernelRadius: integer, specifies the size of the kernel to use as the radius from the center voxel. \
                    Therefore the actual size is 2 * kernelRadius + 1. E.g. a value of 1 yields a 3x3x3 kernel, \
                    a value of 2 5x5x5, etc. In case of 2D extraction, the generated kernel will also be a 2D shape
                    (square instead of cube).
        feature_name_list: [str], [feature_name1, feature_name2,...] or ['glcm', 'glrlm']
        store_path: str;

        Returns
        -------

        """

        start_time = time.time()
        self.kernelRadius = kernelRadius
        self.store_path = store_path # self.store_path is a string here
        parameter_path = r'D:\MyScript\RadiomicsVisualization\RadiomicsFeatureVisualization\RadiomicsParams.yaml'
        setting_dict = {'label': 1, 'interpolator': 'sitkBSpline', 'correctMask': True,
                        'geometryTolerance': 10, 'kernelRadius': self.kernelRadius,
                        'maskedKernel': True, 'voxelBatch': 50}

        # Fix: Changed to RadiomicsFeatureExtractor (singular)
        extractor = featureextractor.RadiomicsFeatureExtractor(parameter_path, self.store_path, **setting_dict)
        extractor.disableAllImageTypes()
        extractor.disableAllFeatures()

        img_setting, feature_dict, sub_filter_name = self.decode_feature_name(feature_name_list)
        extractor.enableImageTypeByName(**img_setting)
        extractor.enableFeaturesByName(**feature_dict)

        cropped_original_img, cropped_original_roi = self.crop_img(candidate_roi_path, candidate_img_path,
                                                                   store_key='original')

        if sub_filter_name:
            # generate filter image firstly for speeding up
            extractor.execute(candidate_img_path, candidate_roi_path, voxelBased=False)
            candidate_img_path = os.path.join(self.store_path, sub_filter_name+'.nii.gz')
            cropped_filter_img, cropped_filter_roi = self.crop_img(candidate_roi_path, candidate_img_path,
                                                                   store_key=sub_filter_name)
            result = extractor.execute(cropped_filter_img, cropped_filter_roi, voxelBased=True)
        #
        #
        else:
            result = extractor.execute(cropped_original_img, cropped_original_roi, voxelBased=True)
        # without parameters, glcm ,kr=5 ,646s ,cropped img, map shape (5, 132, 128)
        # without parameters, glcm ,kr=1 ,386s ,cropped img, map shape (3, 122, 128)

        # without parameters, glcm ,kr=1 ,566s ,without cropped img, map shape (5, 132, 128)

        # extract original image

        for key, val in six.iteritems(result):
            if isinstance(val, sitk.Image):
                shape = (sitk.GetArrayFromImage(val)).shape
                # Feature map
                sitk.WriteImage(val, os.path.join(store_path, key + '.nrrd'), True)


    def show_feature_map(self, show_img_path, show_roi_path, show_feature_map_path, store_path):
        feature_map_img = sitk.ReadImage(show_feature_map_path)
        feature_map_array = sitk.GetArrayFromImage(feature_map_img)
        feature_map_array.transpose(1, 2, 0)
        feature_map_visualization = FeatureMapVisualizition()
        feature_map_visualization.LoadData(show_img_path, show_roi_path, show_feature_map_path)

        # hsv/jet/gist_rainbow
        feature_map_visualization.Show(color_map='rainbow', store_path=store_path)


def main():
    feature_mapper = FeatureMapper()

    # --- 1. Définition des chemins et des paramètres ---

    # Chemin vers le fichier CSV de vos caractéristiques agrégées par patient (non pas les feature maps)
    # C'est le fichier que vous avez montré en capture d'écran.
    feature_csv_path = '/home/kpegouni/Documents/AI-projects/radiomics/notebooks/radiomics_features_T2WI.csv' # <--- METTEZ VOTRE CHEMIN CSV ICI

    # Liste des noms de caractéristiques que vous souhaitez générer en tant que feature maps.
    # Ces noms doivent correspondre exactement aux noms des colonnes dans votre CSV
    # (par exemple, 'original_glcm_JointEntropy').
    features_name_list = ['original_glcm_DifferenceEntropy', 'original_glcm_JointEntropy']
    # Si vous voulez générer TOUTES les GLCM (ou autres classes), vous pourriez faire :
    # features_name_list = ['glcm'] # ou ['firstorder', 'glcm', 'glrlm', 'ngtdm', 'glszm']

    # Chemin du dossier RACINE où se trouvent VOS IMAGES ET ROIs ORIGINALES
    # C'est là que le script ira chercher 'sub-0400554.nii.gz' et 'sub-0400554_mask.nii.gz'
    base_data_dir = Path('/home/kpegouni/Documents/AI-projects/radiomics/data/interim/ICM-DATA/PELVIS/T2WI/') # <-- VÉRIFIEZ CE CHEMIN !

    # Chemin du dossier où les FEATURE MAPS GÉNÉRÉES et les IMAGES RECARDÉES seront stockées.
    # Chaque patient aura idéalement son propre sous-dossier ici.
    base_feature_map_output_dir = Path('/home/kpegouni/Documents/AI-projects/radiomics/RadiomicsFeatureVisualization/ICM-DATA/FeatureMap/T2WI/') # <-- VÉRIFIEZ CE CHEMIN !

    # Le rayon du noyau pour l'extraction voxel par voxel.
    # 1 -> noyau 3x3x3, 2 -> noyau 5x5x5, etc.
    kernel_radius_for_feature_maps = 1

    # --- 2. Optionnel : Recherche de cas candidats (si vous avez un fichier CSV de features agrégées) ---
    # Si vous n'avez pas de fichier CSV global des features et que vous voulez juste générer pour un patient spécifique,
    # vous pouvez sauter cette étape.

    feature_mapper.load(feature_csv_path, features_name_list)
    print("\nRecherche des cas candidats pour les caractéristiques sélectionnées:")
    top_cases, last_cases = feature_mapper.seek_single_candidate_case('original_glcm_JointEntropy', 3) # Exemple: chercher 3 cas extrêmes pour JointEntropy
    print(f"Top cases for JointEntropy: {top_cases}")
    print(f"Last cases for JointEntropy: {last_cases}")
    # Vous pouvez décommenter et adapter cela si vous avez un CSV global et voulez cette fonctionnalité.

    # --- 3. Génération des cartographies de caractéristiques pour un patient spécifique ---

    # Définissez l'ID du patient que vous voulez traiter et visualiser.
    # C'est le 'subject-ID' de votre CSV et le préfixe de vos fichiers NIfTI/NRRD.
    patient_id_to_process = 'sub-2304155' # Remplacez par l'ID du patient actuel

    # Construction des chemins complets pour les fichiers d'entrée de CE patient
    current_patient_img_path = base_data_dir / 'volumes' / f'{patient_id_to_process}.nii.gz'
    current_patient_roi_path = base_data_dir / 'segmentations' / f'{patient_id_to_process}_mask.nii.gz'

    # Création du dossier de sortie SPÉCIFIQUE à ce patient pour les feature maps
    patient_feature_map_output_dir = base_feature_map_output_dir / patient_id_to_process
    patient_feature_map_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n--- Génération des Feature Maps pour le patient: {patient_id_to_process} ---")
    print(f"Image source: {current_patient_img_path}")
    print(f"ROI source: {current_patient_roi_path}")
    print(f"Dossier de sortie des Feature Maps: {patient_feature_map_output_dir}")

    # Vérifiez si les fichiers d'entrée existent avant de continuer
    if not current_patient_img_path.exists():
        print(f"Erreur: Fichier image introuvable à {current_patient_img_path}")
        return
    if not current_patient_roi_path.exists():
        print(f"Erreur: Fichier ROI introuvable à {current_patient_roi_path}")
        return

    # Appel de la méthode de génération des feature maps
    # Le chemin de stockage doit être une chaîne pour pyradiomics, d'où le str()
    feature_mapper.generate_feature_map(
        str(current_patient_img_path),
        str(current_patient_roi_path),
        kernel_radius_for_feature_maps,
        features_name_list,
        str(patient_feature_map_output_dir) # Stockage spécifique au patient
    )
    print(f"Génération des Feature Maps terminée pour {patient_id_to_process}.")


    # --- 4. Visualisation des cartographies générées ---

    # Les chemins des images et ROIs recadrées qui ont été SAUVEGARDÉES par generate_feature_map
    # dans le dossier 'patient_feature_map_output_dir'
    cropped_img_path = patient_feature_map_output_dir / 'original_cropped_img.nii.gz'
    cropped_roi_path = patient_feature_map_output_dir / 'original_cropped_roi.nii.gz'

    # Sélection de la caractéristique SPÉCIFIQUE à visualiser parmi celles qui viennent d'être générées
    feature_to_visualize_name = 'original_glcm_JointEntropy' # Choisissez une caractéristique de features_name_list
    feature_map_file = patient_feature_map_output_dir / f'{feature_to_visualize_name}.nrrd'

    # Chemin pour sauvegarder les figures de visualisation
    patient_figures_output_dir = base_feature_map_output_dir / 'figures' / patient_id_to_process
    patient_figures_output_dir.mkdir(parents=True, exist_ok=True) # Crée le sous-dossier patient si nécessaire

    fig_save_path_base = patient_figures_output_dir / feature_to_visualize_name


    print(f"\n--- Visualisation de la Feature Map: {feature_to_visualize_name} ---")
    print(f"Image recadrée: {cropped_img_path}")
    print(f"ROI recadrée: {cropped_roi_path}")
    print(f"Feature Map à visualiser: {feature_map_file}")
    print(f"Figures sauvegardées dans: {patient_figures_output_dir}")

    # Vérifiez si les fichiers nécessaires à la visualisation existent
    if not cropped_img_path.exists():
        print(f"Erreur: Fichier image recadrée introuvable à {cropped_img_path}")
        return
    if not cropped_roi_path.exists():
        print(f"Erreur: Fichier ROI recadrée introuvable à {cropped_roi_path}")
        return
    if not feature_map_file.exists():
        print(f"Erreur: Fichier Feature Map introuvable à {feature_map_file}")
        print("Veuillez vous assurer que la feature map a été générée correctement.")
        return

    # Appel de la méthode de visualisation
    feature_mapper.show_feature_map(
        str(cropped_img_path),
        str(cropped_roi_path),
        str(feature_map_file),
        str(fig_save_path_base)
    )
    print("Visualisation terminée.")

if __name__ == '__main__':
    main()