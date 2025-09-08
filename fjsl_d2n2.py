import os
import numpy as np
from multiprocessing import Pool
import re
from datetime import datetime
import SimpleITK as sitk
import pydicom
import math
import tempfile

# 配置路径
input_path = "/server02_data/_RAWDATA/PET_CT/FuJianShengLi/图像2"  # 请替换为您的输入路径
output_path = "/server02_data/_RAWDATA/PET_CT/FuJianShengLi/img2_nii"  # 请替换为您的输出路径

# 定义序列名映射
sequence_mapping = {
    "WB Stand": "WB_Stand",
    "WB 3D MAC": "WB_3D_MAC",
    "Brain Stand": "Brain_Stand",
    "Static Brain 3D MAC": "Static_Brain_3D_MAC",
    "+sl": "WB_3D_MAC_2"
}

# PET序列关键词
suv_keywords = ["MAC", "+sl"]


def get_patient_directories(root_dir):
    """获取所有病人目录"""
    patient_dirs = []
    for item in os.listdir(root_dir):
        item_path = os.path.join(root_dir, item)
        if os.path.isdir(item_path):
            patient_dirs.append(item_path)
    return patient_dirs


def process_patient(patient_dir):
    """处理单个病人的所有DICOM文件"""
    try:
        patient_id = os.path.basename(patient_dir)
        print(f"处理病人: {patient_id}")

        # 获取病人目录下的所有DICOM文件
        dicom_files = []
        for root, dirs, files in os.walk(patient_dir):
            for file in files:
                if file.lower().endswith('.dcm') or re.search(r'\.(dcm|img)$', file.lower()):
                    dicom_files.append(os.path.join(root, file))

        if not dicom_files:
            print(f"在 {patient_dir} 中未找到DICOM文件")
            return

        # 获取Patient ID和Study Date
        first_dicom = pydicom.dcmread(dicom_files[0])
        patient_id_tag = getattr(first_dicom, 'PatientID', '')  # Patient ID
        study_date_tag = getattr(first_dicom, 'StudyDate', '')  # Study Date

        # 创建输出目录名
        if patient_id_tag and study_date_tag:
            output_dir_name = f"{patient_id_tag}_{study_date_tag}"
        else:
            output_dir_name = patient_id  # 如果无法获取DICOM标签，使用文件夹名

        # 按序列分组DICOM文件
        series_dict = group_dicom_files_by_series(dicom_files)

        # 处理每个序列
        for series_description, series_files in series_dict.items():
            process_series(series_files, series_description, output_dir_name)

    except Exception as e:
        print(f"处理病人 {patient_dir} 时出错: {str(e)}")
        import traceback
        traceback.print_exc()


def group_dicom_files_by_series(dicom_files):
    """按序列描述将DICOM文件分组"""
    series_dict = {}

    for file_path in dicom_files:
        try:
            # 读取DICOM文件的序列描述
            ds = pydicom.dcmread(file_path)
            series_description = getattr(ds, 'SeriesDescription', '')
            if not series_description:
                # 如果无法读取序列描述，使用文件名作为备用
                series_description = os.path.basename(os.path.dirname(file_path))

            # 去掉首尾空格
            series_description = series_description.strip()

            # 添加到对应的序列组
            if series_description not in series_dict:
                series_dict[series_description] = []
            series_dict[series_description].append(file_path)

        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {str(e)}")

    return series_dict


def process_series(series_files, series_description, output_dir_name):
    """处理单个DICOM序列并将其转换为NIfTI"""
    try:
        # 确保文件按切片位置排序
        series_files.sort(key=lambda x: get_slice_position(x))

        # 映射序列名
        if series_description in sequence_mapping:
            nii_name = sequence_mapping[series_description]
        else:
            # 替换中间空格为下划线，但保留首尾无空格
            nii_name = series_description.strip().replace(" ", "_")

        # 检查是否为PET图像
        is_pet = any(kw.lower() in series_description.lower() for kw in suv_keywords)

        # 创建输出目录
        output_patient_dir = os.path.join(output_path, output_dir_name)
        os.makedirs(output_patient_dir, exist_ok=True)

        # 输出文件路径
        output_file = os.path.join(output_patient_dir, f"{nii_name}.nii.gz")

        # 使用SimpleITK读取DICOM系列
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(series_files)
        image = reader.Execute()

        # 如果是PET图像，计算SUV值
        if is_pet:
            # 获取患者体重
            patient_weight = get_patient_weight(series_files[0])

            if patient_weight is not None:
                # 计算SUV值
                suv_image = calculate_suv(image, series_files, patient_weight)

                # 保存SUV图像
                sitk.WriteImage(suv_image, output_file)
                print(f"✅ 成功转换PET序列 '{series_description}' -> SUV图像: {os.path.basename(output_file)}")
            else:
                # 如果没有体重信息，直接保存不计算SUV
                sitk.WriteImage(image, output_file)
                print(f"⚠️  PET序列 '{series_description}' 无体重信息，未计算SUV: {os.path.basename(output_file)}")
        else:
            # 非PET图像直接保存
            sitk.WriteImage(image, output_file)
            print(f"已保存: {output_file}")

    except Exception as e:
        print(f"处理序列 {series_description} 时出错: {str(e)}")
        import traceback
        traceback.print_exc()


def get_patient_weight(dicom_file):
    """获取患者体重"""
    try:
        ds = pydicom.dcmread(dicom_file)
        weight = getattr(ds, 'PatientWeight', None)
        if weight:
            return float(weight)
    except:
        pass
    return None


def calculate_suv(sitk_image, dicom_files, weight_kg):
    """
    计算SUV值

    参数:
        sitk_image: SimpleITK图像对象
        dicom_files: DICOM文件列表
        weight_kg: 患者体重(kg)

    返回:
        suv_image: SUV值的SimpleITK图像
    """
    # 获取参考DICOM文件（通常是第一个文件）
    ref_dcm = pydicom.dcmread(dicom_files[0])

    # 获取必要参数
    try:
        # 放射性药物总剂量 (单位Bq)
        radiopharmaceutical_info = getattr(ref_dcm, 'RadiopharmaceuticalInformationSequence', [])
        if not radiopharmaceutical_info:
            print("⚠️ 缺少放射性药物信息序列")
            return sitk_image

        radiopharmaceutical_info = radiopharmaceutical_info[0]
        total_dose = float(getattr(radiopharmaceutical_info, 'RadionuclideTotalDose', 0))

        # 注射时间
        injection_time_str = getattr(radiopharmaceutical_info, 'RadiopharmaceuticalStartTime', "")
        if not injection_time_str:
            injection_time_str = getattr(ref_dcm, 'RadiopharmaceuticalStartTime', "")

        # 半衰期 (单位秒)
        half_life = float(getattr(radiopharmaceutical_info, 'RadionuclideHalfLife', 0))

        # 采集时间
        acquisition_time_str = getattr(ref_dcm, 'AcquisitionTime', "")

        # 检查是否所有必要参数都存在
        if not all([total_dose, injection_time_str, half_life, acquisition_time_str]):
            print("⚠️ 缺少必要的DICOM参数")
            return sitk_image

    except (AttributeError, IndexError, ValueError) as e:
        print(f"⚠️ 缺少必要的DICOM参数: {str(e)}")
        return sitk_image

    # 计算时间差 (秒)
    try:
        # 处理不同格式的时间字符串
        injection_time = parse_dicom_time(injection_time_str)
        acquisition_time = parse_dicom_time(acquisition_time_str)

        # 计算时间差
        time_diff_seconds = (datetime.combine(datetime.min, acquisition_time) -
                             datetime.combine(datetime.min, injection_time)).total_seconds()

        # 处理跨天的情况（如果采集时间在注射时间之前）
        if time_diff_seconds < 0:
            time_diff_seconds += 24 * 3600  # 加上一天的秒数

    except ValueError as e:
        print(f"⚠️ 时间格式错误: {e}")
        return sitk_image

    # 计算衰减因子
    decay_factor = math.exp(-math.log(2) * time_diff_seconds / half_life)

    # 计算衰减后剂量
    decayed_dose = total_dose * decay_factor

    # 计算SUV缩放因子
    suv_factor = (weight_kg * 1000) / decayed_dose if decayed_dose > 0 else 1.0

    # 转换为SUV
    image_array = sitk.GetArrayFromImage(sitk_image)
    suv_array = image_array * suv_factor

    # 创建新的SimpleITK图像
    suv_image = sitk.GetImageFromArray(suv_array)
    suv_image.CopyInformation(sitk_image)

    return suv_image


def parse_dicom_time(time_str):
    """
    解析DICOM时间字符串，支持多种格式
    """
    # 移除可能存在的空格
    time_str = time_str.strip()

    # 处理不同长度的时间字符串
    if len(time_str) == 6:  # HHMMSS
        return datetime.strptime(time_str, "%H%M%S").time()
    elif len(time_str) > 6 and '.' in time_str:  # HHMMSS.ffffff
        # 分离整数部分和小数部分
        parts = time_str.split('.')
        time_part = parts[0]
        # 确保时间部分是6位
        if len(time_part) < 6:
            time_part = time_part.zfill(6)
        return datetime.strptime(f"{time_part}.{parts[1]}", "%H%M%S.%f").time()
    elif len(time_str) == 4:  # HHMM
        return datetime.strptime(time_str, "%H%M").time()
    else:
        # 尝试通用解析
        try:
            return datetime.strptime(time_str, "%H%M%S.%f").time()
        except ValueError:
            try:
                return datetime.strptime(time_str, "%H%M%S").time()
            except ValueError:
                try:
                    return datetime.strptime(time_str, "%H%M").time()
                except ValueError:
                    raise ValueError(f"无法解析时间字符串: {time_str}")


def get_slice_position(dicom_file):
    """获取DICOM文件的切片位置"""
    try:
        ds = pydicom.dcmread(dicom_file)
        position = getattr(ds, 'SliceLocation', None)
        if position:
            return float(position)
        else:
            # 如果无法获取切片位置，使用实例编号
            instance_number = getattr(ds, 'InstanceNumber', 0)
            return float(instance_number)
    except:
        return 0


if __name__ == "__main__":
    # 获取所有病人目录
    patient_dirs = get_patient_directories(input_path)
    print(f"找到 {len(patient_dirs)} 个病人目录")

    # 使用8个进程处理
    with Pool(8) as p:
        p.map(process_patient, patient_dirs)

    print("所有处理完成!")