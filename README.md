#Optical Character Recognition using naiive bayes.

This project uses handwritten digits from the MNIST database to make predictions using a naiive bayes model.

The images are 28x28 pixels and they are mapped to a matrix of 0, 1, 2 depending on the level of black.
0 = White, 1 = Black, 2 = Somewhere in between.

For each label 0 - 9, a probability density is stored in a matrix that is used to compare test data against. The probability is calcualted in a logarithmic scale in order to avoid float underflow. 
And the final score is calculated by multiplying and adding each respective pixel. The highest score is selected as a match.
