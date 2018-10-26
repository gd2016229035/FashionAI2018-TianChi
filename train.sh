num_gpus=1
num_workers=8
model=resnet50_v2 #resnet152_v2
epoch=15
batch_size=16
task_file_1=train_task_design.py

#No randomcrop augmentation in 'length' task
task_file_2=train_task_length.py

#python2 prepare_data.py
python2 $task_file_1 --task collar_design_labels --model $model --batch-size $batch_size --num-gpus $num_gpus -j $num_workers --epochs $epoch --lr-steps 7,12,15
python2 $task_file_1 --task neckline_design_labels --model $model --batch-size $batch_size --num-gpus $num_gpus -j $num_workers --epochs $epoch --lr-steps 7,12,15
python2 $task_file_2 --task skirt_length_labels --model $model --batch-size $batch_size --num-gpus $num_gpus -j $num_workers --epochs $epoch --lr-steps 7,12,15 
python2 $task_file_2 --task sleeve_length_labels --model $model --batch-size $batch_size --num-gpus $num_gpus -j $num_workers --epochs $epoch --lr-steps 7,12,15 
python2 $task_file_1 --task neck_design_labels --model $model --batch-size $batch_size --num-gpus $num_gpus -j $num_workers --epochs $epoch --lr-steps 7,12,15
python2 $task_file_2 --task coat_length_labels --model $model --batch-size $batch_size --num-gpus $num_gpus -j $num_workers --epochs $epoch --lr-steps 7,12,15 
python2 $task_file_1 --task lapel_design_labels --model $model --batch-size $batch_size --num-gpus $num_gpus -j $num_workers --epochs $epoch --lr-steps 7,12,15 
python2 $task_file_2 --task pant_length_labels --model $model --batch-size $batch_size --num-gpus $num_gpus -j $num_workers --epochs $epoch --lr-steps 7,12,15

cd submission
result_dir=submission_${model}_${times}_${batch_size}_${epoch}_5.20
mkdir $result_dir
mv collar_design_labels.csv neckline_design_labels.csv skirt_length_labels.csv sleeve_length_labels.csv neck_design_labels.csv coat_length_labels.csv lapel_design_labels.csv pant_length_labels.csv  $result_dir/
cd $result_dir
cat collar_design_labels.csv neckline_design_labels.csv skirt_length_labels.csv sleeve_length_labels.csv neck_design_labels.csv coat_length_labels.csv lapel_design_labels.csv pant_length_labels.csv > submission.csv
zip submission.zip submission.csv

#rm -rf /usr/data/fashionai/models/*
#cd ../..
#mv *.npy $result_dir/
#echo "000000" | sudo -S shutdown -h 5


