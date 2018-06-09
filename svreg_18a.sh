#!/bin/bash
#$ -S /bin/bash
#$ -o logs 
#$ -j y
#$ -N svreg18a
#$ -V
#$ -cwd
#$ -l h_vmem=23G

export TMPDIR=/ifs/tmp/
export TMP=/ifs/tmp/
export LC_ALL="C"

if [ -z "$SGE_TASK_ID" ] || [ "$SGE_TASK_ID" == "undefined" ]; then
SubjID=$1
echo SubjID="$SubjID"
else

SubjID=`sed "${SGE_TASK_ID}p;d" $1`
fi;

SVREG=/ifs/tmp/yeunkim/BrainSuite18a/svreg/bin/svreg.sh
# ATLAS=/ifshome/ajoshi/AnandJoshi/svreg_16a_build2229_linux/BrainSuiteAtlas1/mri
ATLAS=/ifshome/yeunkim/BrainSuite16a/svreg/BrainSuiteAtlas1/mri

cd ${SubjID}
$SVREG ${SubjID}.gfc ${ATLAS} -U

