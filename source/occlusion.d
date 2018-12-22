module occlusion;

import discretemdp;
import discretefunctions;
import std.typecons;
import std.algorithm.comparison;
import std.random;


public Set!State randomOccludedStates(Model m, int num_occluded_states) {

    Set!State returnval = new Set!(State)(new Tuple!(State)[0]);

    auto arr = m.S().toArray().randomShuffle();

    return new Set!(State)(arr[0..num_occluded_states]);
    
}

public Sequence!(State, Action) removeOccludedTimesteps (Sequence!(State, Action) trajectory, Set!State occluded_states) {

    Sequence!(State, Action) returnval = new Sequence!(State, Action)(trajectory.length);

    foreach(i, timestep ; trajectory) {
        if (! occluded_states.contains(timestep[0])) {
            returnval[i] = timestep;
        } else {
            returnval[i] = Tuple!(State, Action)(null, null);
        }
    }
    
    return returnval;
}

public Sequence!(State, Action)[] removeOccludedTimesteps (Sequence!(State, Action)[] trajectory, Set!State occluded_states) {

    Sequence!(State, Action)[] returnval = new Sequence!(State, Action)[trajectory.length];

    foreach (i ; 0 .. trajectory.length) {
        returnval[i] = removeOccludedTimesteps(trajectory[i], occluded_states);
    }

    return returnval;   
}

