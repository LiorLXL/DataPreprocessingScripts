# DICOM转nii.gz并且处理其中的PET序列的SUV值，以及同名文件加入后缀，转SUV值时从xlsx文件中读取

import os
import shutil
import tempfile
import pydicom
import dicom2nifti
import multiprocessing
import pandas as pd
import numpy as np
import nibabel as nib
from dicom2nifti import convert_dicom
from tqdm import tqdm
from datetime import datetime
import math


def load_weight_data(weight_file):
    """
    从Excel文件加载体重数据

    参数:
        weight_file: Excel文件路径

    返回:
        weight_dict: 字典，键为病人ID，值为体重(kg)
    """
    if not os.path.exists(weight_file):
        print(f"⚠️ 体重文件不存在: {weight_file}")
        return {}

    try:
        # 读取Excel文件
        df = pd.read_excel(weight_file)

        # 检查必要的列
        if '影像号' not in df.columns or '体重' not in df.columns:
            print("⚠️ Excel文件中缺少'影像号'或'体重'列")
            return {}

        # 创建字典：影像号 -> 体重
        weight_dict = {}
        for _, row in df.iterrows():
            patient_id = str(row['影像号']).strip()
            weight = float(row['体重'])
            weight_dict[patient_id] = weight

        print(f"✅ 成功加载 {len(weight_dict)} 位病人的体重数据")
        return weight_dict

    except Exception as e:
        print(f"❌ 加载体重数据失败: {str(e)}")
        return {}


def calculate_suv(pixel_array, dicom_files, weight_kg):
    """
    计算SUV值

    参数:
        pixel_array: 原始像素值数组
        dicom_files: DICOM文件列表
        weight_kg: 患者体重(kg)

    返回:
        suv_array: SUV值数组
    """
    # 获取参考DICOM文件（通常是第一个文件）
    ref_dcm = pydicom.dcmread(dicom_files[0])

    # 获取必要参数
    try:
        # 放射性药物总剂量 (单位Bq)
        total_dose = float(ref_dcm.RadiopharmaceuticalInformationSequence[0].RadionuclideTotalDose)

        # 注射时间
        injection_time_str = ref_dcm.RadiopharmaceuticalInformationSequence[0].RadiopharmaceuticalStartTime
        injection_time = datetime.strptime(injection_time_str, "%H%M%S.%f").time()

        # 半衰期 (单位秒)
        half_life = float(ref_dcm.RadiopharmaceuticalInformationSequence[0].RadionuclideHalfLife)

        # 采集时间
        acquisition_time_str = ref_dcm.AcquisitionTime
        acquisition_time = datetime.strptime(acquisition_time_str, "%H%M%S.%f").time()

    except (AttributeError, IndexError, ValueError) as e:
        print(f"⚠️ 缺少必要的DICOM参数: {str(e)}")
        return pixel_array

    # 计算时间差 (秒)
    time_diff_seconds = (datetime.combine(datetime.min, acquisition_time) -
                         datetime.combine(datetime.min, injection_time)).total_seconds()

    # 计算衰减因子
    decay_factor = math.exp(-math.log(2) * time_diff_seconds / half_life)

    # 计算衰减后剂量
    decayed_dose = total_dose * decay_factor

    # 计算SUV缩放因子
    suv_factor = (weight_kg * 1000) / decayed_dose

    # 转换为SUV
    suv_array = pixel_array * suv_factor

    return suv_array


def convert_dicomdir_series(dicomdir_path, keywords, output_dir, patient_id, weight_dict, suv_keywords=['PET']):
    """
    通过DICOMDIR文件转换序列名称中包含指定关键字的DICOM序列为NIfTI格式

    参数:
        dicomdir_path: DICOMDIR文件路径
        keywords: 需要匹配的关键字列表
        output_dir: 输出目录路径
        patient_id: 病人ID
        weight_dict: 体重数据字典
        suv_keywords: 需要进行SUV转换的关键字列表
    """
    try:
        # 读取DICOMDIR文件
        dicomdir = pydicom.dcmread(dicomdir_path)
        base_path = os.path.dirname(dicomdir_path)

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 获取病人体重 (如果可用)
        patient_weight = weight_dict.get(patient_id)
        if patient_weight is not None:
            print(f"📊 病人 {patient_id} 体重: {patient_weight} kg")

        # 遍历DICOMDIR记录
        series_converted = 0
        current_series = None
        series_map = {}

        # 第一次遍历：建立序列映射
        for record in dicomdir.DirectoryRecordSequence:
            if record.DirectoryRecordType == 'SERIES':
                series_desc = getattr(record, 'SeriesDescription', '').lower()
                series_uid = getattr(record, 'SeriesInstanceUID', 'unknown')
                series_map[series_uid] = {
                    'desc': series_desc,
                    'images': []
                }
                current_series = series_uid
            elif record.DirectoryRecordType == 'IMAGE' and current_series:
                series_map[current_series]['images'].append(record)

        # 用于跟踪序列名称出现次数的字典
        series_name_count = {}

        # 第二次遍历：处理匹配的序列
        for series_uid, series_data in series_map.items():
            series_desc = series_data['desc']

            # 检查序列描述是否包含任何关键字
            if any(keyword.lower() in series_desc for keyword in keywords):
                # 为每个序列创建临时目录
                with tempfile.TemporaryDirectory() as series_temp_dir:
                    # 复制DICOM文件到临时目录
                    dicom_files = []
                    files_copied = 0
                    for img_record in series_data['images']:
                        rel_path = img_record.ReferencedFileID
                        # 处理可能的空字节问题
                        rel_path = [p for p in rel_path if p]
                        src_path = os.path.join(base_path, *rel_path)

                        # 确保源文件存在
                        if os.path.exists(src_path):
                            dst_path = os.path.join(series_temp_dir, os.path.basename(src_path))
                            shutil.copy2(src_path, dst_path)
                            dicom_files.append(dst_path)
                            files_copied += 1
                        else:
                            # 使用相对路径减少日志长度
                            rel_src_path = os.path.relpath(src_path, base_path)
                            print(f"⚠️ 文件不存在: {rel_src_path}")

                    # 检查是否有文件被复制
                    if files_copied == 0:
                        print(f"⚠️ 序列 '{series_desc}' 没有找到任何DICOM文件")
                        continue

                    # 生成安全序列名称
                    safe_desc = ''.join(c if c.isalnum() else '_' for c in series_desc)[:30]

                    # 移除所有数字后缀（如果有的话）
                    if safe_desc.endswith('_'):
                        safe_desc = safe_desc.rstrip('_')

                    # 处理同名序列 - 添加序号后缀
                    if safe_desc in series_name_count:
                        series_name_count[safe_desc] += 1
                        output_filename = f"{safe_desc}_{series_name_count[safe_desc]}.nii.gz"
                    else:
                        series_name_count[safe_desc] = 1
                        output_filename = f"{safe_desc}.nii.gz"

                    output_path = os.path.join(output_dir, output_filename)

                    # 检查文件是否已存在（防止覆盖）
                    if os.path.exists(output_path):
                        # 如果文件已存在，添加唯一后缀
                        unique_suffix = 1
                        while True:
                            new_filename = f"{safe_desc}_{unique_suffix}.nii.gz"
                            new_output_path = os.path.join(output_dir, new_filename)
                            if not os.path.exists(new_output_path):
                                output_path = new_output_path
                                break
                            unique_suffix += 1

                    try:
                        # 检查是否为PET序列且需要SUV转换
                        is_pet = any(kw.lower() in series_desc for kw in suv_keywords)
                        perform_suv = is_pet and patient_weight is not None

                        if perform_suv:
                            # 使用正确的转换方法
                            # 1. 首先转换到临时NIfTI文件
                            temp_nifti_path = os.path.join(series_temp_dir, "temp.nii.gz")
                            dicom2nifti.dicom_series_to_nifti(series_temp_dir, temp_nifti_path)

                            # 2. 加载NIfTI文件
                            nifti_img = nib.load(temp_nifti_path)
                            pixel_data = nifti_img.get_fdata()

                            # 3. 计算SUV值
                            suv_data = calculate_suv(pixel_data, dicom_files, patient_weight)

                            # 4. 创建新的NIfTI图像
                            suv_img = nib.Nifti1Image(suv_data, nifti_img.affine, nifti_img.header)

                            # 5. 保存SUV图像
                            nib.save(suv_img, output_path)

                            # 6. 删除临时文件
                            if os.path.exists(temp_nifti_path):
                                os.remove(temp_nifti_path)

                            print(f"✅ 成功转换PET序列 '{series_desc}' -> SUV图像: {os.path.basename(output_path)}")
                        else:
                            # 普通转换
                            dicom2nifti.dicom_series_to_nifti(series_temp_dir, output_path)
                            if is_pet and patient_weight is None:
                                print(
                                    f"⚠️ 跳过SUV转换: 病人 {patient_id} 体重数据缺失 -> {os.path.basename(output_path)}")
                            else:
                                print(f"✅ 成功转换序列 '{series_desc}' -> {os.path.basename(output_path)}")

                        series_converted += 1
                    except Exception as e:
                        print(f"❌ 转换序列 '{series_desc}' 失败: {str(e)}")

        return series_converted

    except Exception as e:
        print(f"❌ 处理病人 {patient_id} 时发生错误: {str(e)}")
        return 0


def process_single_patient(patient_info):
    """
    处理单个病人的包装函数，用于多进程

    参数:
        patient_info: 包含(patient_dir, patient_path, keywords, output_root, weight_dict)的元组
    """
    patient_dir, patient_path, keywords, output_root, weight_dict = patient_info

    # 查找DICOMDIR文件
    dicomdir_path = os.path.join(patient_path, 'DICOMDIR')

    if not os.path.exists(dicomdir_path):
        return patient_dir, 0, f"⚠️ 未找到 DICOMDIR 文件"

    # 创建病人特定的输出目录
    patient_output_dir = os.path.join(output_root, patient_dir)
    os.makedirs(patient_output_dir, exist_ok=True)

    # 处理当前病人
    series_converted = convert_dicomdir_series(
        dicomdir_path=dicomdir_path,
        keywords=keywords,
        output_dir=patient_output_dir,
        patient_id=patient_dir,
        weight_dict=weight_dict
    )

    return patient_dir, series_converted, "✅ 完成"


def process_all_patients(data_root, keywords, output_root, weight_file, num_processes=None):
    """
    使用多进程处理数据集中的所有病人

    参数:
        data_root: 包含所有病人数据的根目录
        keywords: 需要匹配的关键字列表
        output_root: 输出文件的根目录
        weight_file: 体重数据Excel文件路径
        num_processes: 使用的进程数，默认使用CPU核心数的75%
    """
    # 确保输出根目录存在
    os.makedirs(output_root, exist_ok=True)

    # 加载体重数据
    weight_dict = load_weight_data(weight_file)

    # 收集所有病人目录
    patient_dirs = []
    for patient_dir in os.listdir(data_root):
        patient_path = os.path.join(data_root, patient_dir)
        if os.path.isdir(patient_path):
            patient_dirs.append((patient_dir, patient_path, keywords, output_root, weight_dict))

    total_patients = len(patient_dirs)

    if total_patients == 0:
        print("⚠️ 未找到任何病人目录")
        return

    print(f"找到 {total_patients} 位病人需要处理")

    # 设置进程数（默认使用CPU核心数的75%）
    if num_processes is None:
        cpu_count = os.cpu_count() or 1
        num_processes = max(1, int(cpu_count * 0.75))
        print(f"自动设置进程数: {num_processes} (基于 {cpu_count} 个CPU核心)")

    # 使用多进程池处理
    with multiprocessing.Pool(processes=num_processes) as pool:
        # 使用tqdm显示进度条
        results = []
        with tqdm(total=total_patients, desc="处理病人") as pbar:
            for result in pool.imap_unordered(process_single_patient, patient_dirs):
                patient_id, series_converted, status = result
                results.append((patient_id, series_converted))
                pbar.update(1)
                pbar.set_postfix_str(f"当前: {patient_id}, 序列: {series_converted}")

    # 汇总结果
    total_series = sum(series for _, series in results)
    processed_patients = len([p for p in results if p[1] > 0])

    print("\n" + "=" * 50)
    print(f"处理完成! 共处理 {processed_patients}/{total_patients} 位病人，转换 {total_series} 个序列")
    print("=" * 50)

    # 打印失败病人列表
    failed_patients = [p for p, s in results if s == 0]
    if failed_patients:
        print("\n⚠️ 以下病人处理失败:")
        for patient in failed_patients:
            print(f"  - {patient}")


# 使用示例
if __name__ == "__main__":
    # 设置数据集根目录和输出根目录
    data_root = "/server02_data/_RAWDATA/PET_CT/YunTai"
    output_root = "/server02_data/_RAWDATA/PET_CT/YunTaiTest"

    # 体重数据文件路径
    weight_file = "/server02_data/_RAWDATA/PET_CT/YunTai_reports_hasWeight.xlsx"  # 替换为实际的Excel文件路径

    # 要匹配的关键字
    keywords = ['CT', 'PET', 'Thorax', 'Head']

    # 处理所有病人
    process_all_patients(
        data_root=data_root,
        keywords=keywords,
        output_root=output_root,
        weight_file=weight_file,
        num_processes=8  # 可以手动设置进程数
    )