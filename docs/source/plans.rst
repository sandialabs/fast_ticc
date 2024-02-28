Plans For Further Development
=============================

Time, funding, and brains permitting, we have a list of things we'd like
to do with TICC after the first release.  These are in roughly descending
order according to priority.

1. Examples of how to use TICC.  We're actually working on this already --
   we'd rather get the code out there and then update it rather than
   wait any longer.

2. Prediction.  Given an already-trained TICC model and a new data set
   with the same sensors, predict what cluster labels would be assigned
   to the new data points.

3. Prediction with model update.  Just like the above, but also update
   the Markov random fields for each cluster to incorporate the new data.

4. Full-on streaming capability.  We'll maintain a window of data.  As
   new points come in, we'll add them to the end of the window and expire
   the oldest points, then update the model and assign new labels to the
   new points (and possibly older-but-still-current ones).

Everything that has to do with prediction and streaming also raises questions
of how much you let the old model change and what to do (if anything) about
the number of clusters.  We'll discuss those in depth when we release code
that implements the changes.

If one or more of these is interesting to you, by all means fork the
repository, dive in, and open a pull request!

