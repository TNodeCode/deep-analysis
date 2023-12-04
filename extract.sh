for model in wae_mmd
do
   for layer in e1 e2 e3 e4
   do
      for components in 1 2 3 4 5
      do
         echo "$model $layer $components"
         python app.py dff $model $layer images/containers $components
      done
   done
done