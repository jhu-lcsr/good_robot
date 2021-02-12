# Key Components

**`demo.py`**

This script defines the `Demonstration` class, which is a data structure to store task demonstrations, reading them from their parent folder (expected to contain `txt` file with actions and color and depth heightmaps before/after each action was executed). It is compatible with any task demonstration generated with the `touch.py` script.

**`touch.py`**

This is a utility to generate task demonstrations by specifying actions with key presses and specifying action location by clicking on images of the simulated workspace. The saved demonstrations can be directly loaded in downstream scripts using the interface provided by `demo.py`.

**`evaluate_demo_correspondence.py`**

This is a script that compares 2 demonstrations, assuming 1 is the example to be imitated while the other is treated as the execution environment. This is where all learning from demonstration methods are actually implemented (getting demo action embeddings for row-making and stacking, running these policies on the current execution environment, corresponding demo action to execution embeddings (L2 distance, also options to incorporate history), and visualizing the computed distance at each pixel on the execution env frame).

**`robot_with_imitation.py`**

This is an extension of `main.py` to incorporate the imitation signal during execution. Currently does not change the way actions are taken (that is still either the optimal stacking action or the optimal row-making action), but computes forward pass on the passed in demonstration.

**`model.py`**

Modified existing forward method in the old `model.py` to return the final output probabilities as well as the features preceding the output probabilities. Returns None for `action_feat` if the `keep_action_feat` isn't set.

**`trainer.py`**

Modification of the old `trainer.py` that provides the option to keep the features from which q-values are calculated for each policy. Needed the modification to deal with indexing issues that occurred when stripping off the padding from the outputs of `model.forward()`. Additional flags during either initialization or the forward pass include `keep_action_feat`, `use_demo`, `demo_mask`, and `place_common_sense`. This allows the trainer class to be used for computing forward passes in all the different use-cases. 

**`evaluate_imitation_signal.py`**

This is deprecated by `evaluate_demo_correspondence.py`.

# To Run

*Generating demonstrations*

`python touch.py`

*Running evaluation script comparing demonstrations*

For anything other than unstacking:

`python evaluate_demo_correspondence.py -e demos/__template_demo_folder__ -d demos/__imitation_demo_folder__ -r logs/best_row_making/snapshot.reinforcement_action_efficiency_best_value.pth -s logs/best_stacking/snapshot.reinforcement_action_efficiency_best_value.pth -t custom -v`

For unstacking, pass `unstack` for the `-t` flag. Trained models and generated demonstrations are linked in the release.
