# DataPreprocessingScripts
Some useful data preprocessing scripts

---

**petct_dicom2nii2.py**:DICOM图像转为nii.gz格式并且处理其中的PET序列的SUV值，以及同一检测中的同名序列在处理完后加入后缀以区分，在计算转化SUV值时体重数据从xlsx文件中读取
**petct_d2n_finaledition.py**:在所有的数据中查找有使用PSMA显像剂的病人并且在使用PSMA检查的所有序列前加上PSMA标识，此外，该脚本还可以处理非不同显像剂序列同名的问题，若序列同名，保存SeriesNumber较大的序列

脚本可直接运行，只需更改其中的data_root,output_root等路径
