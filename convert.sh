#!/bin/bash
#$ -S /bin/bash
#$ -o logs -j y
#$ -N cp
#$ -V
#$ -cwd
#$ -l h_vmem=6G


if [ -z "$SGE_TASK_ID" ] || [ "$SGE_TASK_ID" == "undefined" ]; then
  SubjID=$1
	echo SubjID="$SubjID"
else
 SubjID=`sed "${SGE_TASK_ID}p;d" $1`
fi;



data=/ifs/tmp/bwade/OASIS/
newData=/ifs/tmp/yeunkim/OASIS/

export PATH=/usr/local/fsl/bin:$PATH

subjdir=${newData}/${SubjID}/

install -d ${subjdir}

# fpath=${data}/${SubjID}/PROCESSED/MPRAGE/T88_111/
fpath=${data}/${SubjID}/RAW/

# fslchfiletype NIFTI_GZ ${fpath}/*_t88_gfc.img ${subjdir}/${SubjID}.gfc.nii.gz 
fslchfiletype NIFTI_GZ ${fpath}/*-1_anon.img ${subjdir}/${SubjID}.raw1.nii.gz 
