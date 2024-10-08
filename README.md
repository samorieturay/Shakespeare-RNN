# Shakespeare Text Generator

This project is a simple Recurrent Neural Network (RNN) built using TensorFlow and Keras to generate text in the style of William Shakespeare. The model is trained on a portion of Shakespeare's work and can produce new text based on learned patterns.

## Project Overview

The script uses an LSTM (Long Short-Term Memory) network, a type of RNN that is well-suited for sequential data like text. The model is trained on a slice of Shakespeare's text, learning to predict the next character in a sequence. Once trained, the model can generate new text by sampling from the learned probability distributions of characters.

## Key Features
**Data Preprocessing:** The input text is preprocessed by converting all characters to lowercase and extracting a subset of the text for training.
Model Architecture: The model consists of an LSTM layer followed by a Dense layer with a softmax activation function to predict the next character.
**Sampling Strategy:** A custom sampling function is used to generate text based on the probability distribution of the next character, allowing control over the "creativity" of the model via a temperature parameter.
Output Logging: The generated text is logged to an output.txt file, with each run of the script appending new output, along with a timestamp, to the file.

## Prerequisites
* Python 3.x
* TensorFlow and Keras (pip install tensorflow)
* Numpy (pip install numpy)

## Output example.
* For more examples check out output.txt


Generated on: 2024-08-24 12:33:39
......0.2.....
g edward iv:
why, then, thy husband's lands in the seas of surpers,
the brother to the father son the brother,
the father with the son the brother shall.

warwick:
what say the son the brother to the send
the father of the broken of the father,
the father of the broken of the send the sir.
the sires the sir the seas the crain to the broth
......0.4.....
th age and endless night;
my inch of taper to the grave the love to his love.

queen margaret:
no, what seep the father for the brother,
the disprownds, to the base the lands mants;
and i was the need with the man and the brother fiertess.

first mursence:
now we say the sinser speak of the nother,
in the lady men mays to the servent the 
......0.6.....
amus:
wherein our entertainment shall shall see see?

butchat:
then is the kinging in the missans men in bold
that thou that we we seas the seas what to the rose
that i see the earless know the king and in the bold.

first let:
the ward be my seam of the excolous in comes
manch the prince hath lancent his strust
and the name, the fear us 
......0.8.....
liet:

lady capulet:
that is, because the set the lands of the drayh,
and what i ream but duke of lost it we mast
he suppise but with sirfils of the subming his
he will in tell to leting of sir.

king yenry vig:
what was for our yonce both not with this king.

juliet:
he is fairs shall speak these with thy canish?
murse doth cannlirgarnin
......1.0.....
madam, madam!
ay, let the county take yout in feitong,
to the in is which, for which i, comely out plante
to have on seek in her two iil in nother
camb-darn, to-norbickl, graven: the staining you, sweet.

duchoo: marc, boneming sinstal finder steep wilt
with his comes
the likes far like me. at i: and, good thanks,
that to the brovers the 
