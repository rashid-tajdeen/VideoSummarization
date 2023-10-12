#!/bin/bash
#SBATCH --job-name=dia-ret-data
#SBATCH --partition=gpu #set to GPU for GPU usage
#SBATCH --nodes=1              # number of nodes
#SBATCH --mem=600MB               # memory per node in MB (different units with$
#SBATCH --ntasks-per-node=64    # number of cores
#SBATCH --time=72:00:00           # HH-MM-SS
#SBATCH --output /home/nmathi2s/perl5/job_tf.%N.%j.out # filename for STDOUT (%N: nodename, %j: j$
#SBATCH --error /home/nmathi2s/perl5/job_tf.%N.%j.err  # filename for STDERR


# locate to your root directory
cd /home/rtajde2s/VideoSummarization/src

# run the script
python3 main.py --frame_selection uniform --train --valid

#Find your batch status in https://wr0.wr.inf.h-brs.de/wr/stat/batch.xhtml
#To start training run "sbatch execution.sh"