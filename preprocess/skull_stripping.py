"""
# 去颅骨
# nipype和fsl需要在linux系统安装！
"""

import glob
import os
import SimpleITK as sitk
import tqdm
from nipype.interfaces import fsl



if __name__ == "__main__":
	btr = fsl.BET()  # BET是FSL软件中颅骨剥离（提取大脑）的算法
	fsl.FSLCommand.set_default_output_type('NIFTI')
	btr.inputs.frac = 0.05  # Fractional intensity threshold; smaller values give larger brain outline estimates

	in_dir = '/data2/datasets/IXI/original/test/image'

	for file in tqdm.tqdm(glob.glob(os.path.join(in_dir, '*.nii.gz'))):
		btr.inputs.in_file = os.path.join(in_dir, file)
		btr.inputs.out_file = os.path.join(in_dir, file) 
		res = btr.run()	 # 需要在linux端运行,并安装FSL软件。		
