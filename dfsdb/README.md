## Description
For this research project, a projective testing technique used in psychology, the House-Tree-Person Test (H-T-P) was adapted in a simplified version as a means to assess one's personality trait and possible mental health issue through drawing. The designed system is aimed to increase user engagement during the assessment, as opposed to a traditional survey approach.

The solution makes use of image classifiers written in Python using PyTorch. For each element of the test (House, Tree, and Person) 3 possible outcomes are generated

## Important setup information
The pre-trained models that are used in the code for inference are too large to be pushed in this repository. To obtain the 9 models that the application uses for inference follow these steps: <br>
<br>
<em>(In the following steps `"House"` is to be replaced with either `"House", "Tree", "Person"` depending on the model you wish to obtain.)</em>

 1. Un-comment the lines 167-176 in `HouseClassifier.py` depending on the model you want to obtain.
 2. Adjsut the model parameters here
 `checkpoint_losses = training(model, device, 0.0001, num_epochs, train_loader)`
 3. Replace `num_epochs` with the number of desired epochs for training
 4. For each of the elements (House, Tree, Person) three separate models were trained with 10, 12, and 15 epochs. The learning rate was 0.0001 for every model.
 5. The snippet in lines 167-176 will store the model  for example: `house_model_12.tar'`. This represents a model from the house classifier trained with 12 epochs.
 6. Run `HouseClassifier.py`
 7. Repeat the steps for each of the elements to obtain all 9 models.
