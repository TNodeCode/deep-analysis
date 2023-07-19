for model in resnet152 fasterrcnn_resnet50_v2 maskrcnnv2 fcos retinanetv2
do
   for layer in f1 f2 f3 f4
   do
      for components in 1 2 3 4 5
      do
         echo "$model $layer $components"
         python app.py dff $model $layer images/containers $components
      done
   done
done